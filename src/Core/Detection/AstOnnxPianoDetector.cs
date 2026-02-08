using System.Text.Json;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using PianoActivityTracker.Core.Audio;

namespace PianoActivityTracker.Core.Detection;

public sealed class AstOnnxPianoDetector : IPianoDetector, IDisposable
{
    private readonly InferenceSession _session;
    private readonly AstPreprocessor _preprocessor;
    private readonly int[] _targetLabelIndices;
    private readonly float _threshold;
    private readonly float[] _ringBuffer;
    private int _writeIndex;
    private int _totalWritten;
    private readonly string _inputName;
    private readonly string _outputName;
    private readonly int[] _inputShape;
    private float _smoothedScore;

    public AstOnnxPianoDetector(string modelDirectory, AstOnnxOptions? options = null)
    {
        if (string.IsNullOrWhiteSpace(modelDirectory))
        {
            throw new ArgumentException("Model directory is required.", nameof(modelDirectory));
        }

        options ??= new AstOnnxOptions();

        var preprocessorPath = Path.Combine(modelDirectory, "preprocessor_config.json");
        var labelCsvPath = Path.Combine(modelDirectory, "class_labels_indices.csv");

        if (!File.Exists(preprocessorPath))
        {
            throw new FileNotFoundException("AST preprocessor_config.json not found.", preprocessorPath);
        }

        if (!File.Exists(labelCsvPath))
        {
            throw new FileNotFoundException("AudioSet class_labels_indices.csv not found.", labelCsvPath);
        }

        var preprocessorConfig = AstPreprocessorConfig.Load(preprocessorPath);
        _preprocessor = new AstPreprocessor(preprocessorConfig);

        _threshold = options.Threshold;

        var labelMap = LoadLabelMap(labelCsvPath);
        var targetLabels = options.TargetLabels?.Length > 0
            ? options.TargetLabels
            : new[] { "Piano", "Electric piano" };

        _targetLabelIndices = ResolveTargetIndices(labelMap, targetLabels);

        _ringBuffer = new float[_preprocessor.RequiredSamples];

        _session = CreateSession(modelDirectory, options);

        _inputName = _session.InputMetadata.Keys.First();
        _outputName = _session.OutputMetadata.Keys.First();
        _inputShape = _session.InputMetadata[_inputName].Dimensions
            .Select(d => d <= 0 ? 1 : d)
            .ToArray();
    }

    public DetectionResult Process(AudioFrame frame)
    {
        if (frame.SampleRate != _preprocessor.SampleRate)
        {
            throw new ArgumentException($"AudioFrame.SampleRate must be {_preprocessor.SampleRate}Hz.", nameof(frame));
        }

        AppendSamples(frame.Samples);

        var orderedSamples = ReadBufferedSamples();
        var inputTensor = _preprocessor.CreateInputTensor(orderedSamples, _inputShape);

        var input = NamedOnnxValue.CreateFromTensor(_inputName, inputTensor);
        using var results = _session.Run(new[] { input });
        var logits = results.First(r => r.Name == _outputName).AsEnumerable<float>().ToArray();

        var best = 0f;
        foreach (var index in _targetLabelIndices)
        {
            if (index < 0 || index >= logits.Length)
            {
                continue;
            }

            var score = Sigmoid(logits[index]);
            if (score > best)
            {
                best = score;
            }
        }

        _smoothedScore = (_smoothedScore * 0.70f) + (best * 0.30f);
        var confidence = Math.Max(best, _smoothedScore);
        var isPiano = confidence >= _threshold;
        return new DetectionResult(isPiano, confidence, frame.StartTime);
    }

    public void Dispose()
    {
        _session.Dispose();
    }

    private void AppendSamples(ReadOnlySpan<float> samples)
    {
        for (var i = 0; i < samples.Length; i++)
        {
            _ringBuffer[_writeIndex] = samples[i];
            _writeIndex = (_writeIndex + 1) % _ringBuffer.Length;
            if (_totalWritten < _ringBuffer.Length)
            {
                _totalWritten++;
            }
        }
    }

    private float[] ReadBufferedSamples()
    {
        if (_totalWritten < _ringBuffer.Length)
        {
            var buffered = new float[_ringBuffer.Length];
            var valid = _totalWritten;
            var start = (_writeIndex - valid + _ringBuffer.Length) % _ringBuffer.Length;

            for (var i = 0; i < valid; i++)
            {
                buffered[i] = _ringBuffer[(start + i) % _ringBuffer.Length];
            }

            return buffered;
        }

        var ordered = new float[_ringBuffer.Length];
        var tail = _ringBuffer.Length - _writeIndex;
        Array.Copy(_ringBuffer, _writeIndex, ordered, 0, tail);
        Array.Copy(_ringBuffer, 0, ordered, tail, _writeIndex);
        return ordered;
    }

    private static InferenceSession CreateSession(string modelDirectory, AstOnnxOptions options)
    {
        if (!string.IsNullOrWhiteSpace(options.ModelFileName))
        {
            var explicitPath = Path.Combine(modelDirectory, options.ModelFileName);
            if (!File.Exists(explicitPath))
            {
                throw new FileNotFoundException("Requested ONNX model file not found.", explicitPath);
            }

            return new InferenceSession(explicitPath, new SessionOptions());
        }

        var candidates = new[]
        {
            Path.Combine(modelDirectory, "model.onnx"),
            Path.Combine(modelDirectory, "model_int8.onnx")
        };

        var errors = new List<string>();
        foreach (var path in candidates)
        {
            if (!File.Exists(path))
            {
                continue;
            }

            try
            {
                return new InferenceSession(path, new SessionOptions());
            }
            catch (Exception ex)
            {
                errors.Add($"{Path.GetFileName(path)}: {ex.Message}");
            }
        }

        if (errors.Count > 0)
        {
            throw new InvalidOperationException($"Unable to load any AST ONNX model. {string.Join(" | ", errors)}");
        }

        throw new FileNotFoundException("No ONNX model file found. Expected model.onnx or model_int8.onnx.");
    }

    private static Dictionary<string, int> LoadLabelMap(string csvPath)
    {
        var map = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);
        using var reader = new StreamReader(csvPath);
        var header = reader.ReadLine();
        if (header is null)
        {
            return map;
        }

        while (!reader.EndOfStream)
        {
            var line = reader.ReadLine();
            if (string.IsNullOrWhiteSpace(line))
            {
                continue;
            }

            var fields = ParseCsvLine(line);
            if (fields.Length < 3)
            {
                continue;
            }

            if (!int.TryParse(fields[0], out var index))
            {
                continue;
            }

            var displayName = fields[2].Trim();
            if (!string.IsNullOrWhiteSpace(displayName) && !map.ContainsKey(displayName))
            {
                map.Add(displayName, index);
            }
        }

        return map;
    }

    private static int[] ResolveTargetIndices(Dictionary<string, int> labelMap, IReadOnlyList<string> targets)
    {
        var indices = new List<int>();
        foreach (var label in targets)
        {
            if (labelMap.TryGetValue(label, out var index))
            {
                indices.Add(index);
            }
        }

        if (indices.Count == 0)
        {
            throw new InvalidOperationException("None of the requested labels were found in class_labels_indices.csv.");
        }

        return indices.ToArray();
    }

    private static string[] ParseCsvLine(string line)
    {
        var values = new List<string>();
        var current = new System.Text.StringBuilder();
        var inQuotes = false;

        for (var i = 0; i < line.Length; i++)
        {
            var c = line[i];
            if (c == '"')
            {
                if (inQuotes && i + 1 < line.Length && line[i + 1] == '"')
                {
                    current.Append('"');
                    i++;
                    continue;
                }

                inQuotes = !inQuotes;
                continue;
            }

            if (c == ',' && !inQuotes)
            {
                values.Add(current.ToString());
                current.Clear();
                continue;
            }

            current.Append(c);
        }

        values.Add(current.ToString());
        return values.ToArray();
    }

    private static float Sigmoid(float value)
    {
        return 1f / (1f + MathF.Exp(-value));
    }
}

public sealed class AstOnnxOptions
{
    public float Threshold { get; init; } = 0.18f;

    public string? ModelFileName { get; init; }

    public string[] TargetLabels { get; init; } = { "Piano", "Electric piano", "Keyboard (musical)" };
}

internal sealed class AstPreprocessorConfig
{
    public int SamplingRate { get; init; }
    public int MaxLength { get; init; }
    public int NumMelBins { get; init; }
    public float Mean { get; init; }
    public float Std { get; init; }
    public float PaddingValue { get; init; }

    public static AstPreprocessorConfig Load(string path)
    {
        using var stream = File.OpenRead(path);
        using var document = JsonDocument.Parse(stream);
        var root = document.RootElement;

        return new AstPreprocessorConfig
        {
            SamplingRate = root.GetProperty("sampling_rate").GetInt32(),
            MaxLength = root.GetProperty("max_length").GetInt32(),
            NumMelBins = root.GetProperty("num_mel_bins").GetInt32(),
            Mean = root.GetProperty("mean").GetSingle(),
            Std = root.GetProperty("std").GetSingle(),
            PaddingValue = root.TryGetProperty("padding_value", out var padding) ? padding.GetSingle() : 0f
        };
    }
}
