using NAudio.Wave;
using NAudio.Wave.SampleProviders;
using PianoActivityTracker.Core.Audio;
using PianoActivityTracker.Core.Detection;

const int targetSampleRate = 16_000;
const int frameSamples = 16_000;
const int hopSamples = 4_000;

var root = FindRepoRoot();
var inputPath = args.Length > 0
    ? Path.GetFullPath(args[0], Environment.CurrentDirectory)
    : Path.Combine(root, "samples");
var maxSeconds = args.Length > 1 && int.TryParse(args[1], out var parsedSeconds)
    ? Math.Max(5, parsedSeconds)
    : 40;

var modelDir = Path.Combine(root, "assets", "models", "ast");
var detector = new AstOnnxPianoDetector(modelDir, new AstOnnxOptions
{
    Threshold = 0.18f,
    TargetLabels = new[] { "Piano", "Electric piano", "Keyboard (musical)" }
});

var files = ResolveFiles(inputPath);

if (files.Length == 0)
{
    Console.WriteLine($"No files found in: {inputPath}");
    return;
}

Console.WriteLine($"Input: {inputPath}");
Console.WriteLine($"Model directory: {modelDir}");
Console.WriteLine($"Max seconds per file: {maxSeconds}");
Console.WriteLine();

foreach (var file in files)
{
    try
    {
        Console.WriteLine($"Processing: {Path.GetFileName(file)}");
        var mono16k = LoadAsMono16k(file, maxSeconds);
        if (mono16k.Length < frameSamples)
        {
            Console.WriteLine($"{Path.GetFileName(file)}: too short ({mono16k.Length} samples)");
            continue;
        }

        var confidences = new List<float>();
        var hits = 0;
        var time = DateTime.UtcNow;
        for (var start = 0; start + frameSamples <= mono16k.Length; start += hopSamples)
        {
            var frame = new float[frameSamples];
            Array.Copy(mono16k, start, frame, 0, frameSamples);
            var result = detector.Process(new AudioFrame(frame, targetSampleRate, time));
            confidences.Add(result.Confidence);
            if (result.IsPiano)
            {
                hits++;
            }
            time = time.AddMilliseconds(250);
        }

        var avg = confidences.Average();
        var max = confidences.Max();
        var min = confidences.Min();
        var p90 = Percentile(confidences, 0.9);
        Console.WriteLine(
            $"{Path.GetFileName(file)} | frames={confidences.Count} | hitRate={(double)hits / confidences.Count:0.000} | min={min:0.000} avg={avg:0.000} p90={p90:0.000} max={max:0.000}");
    }
    catch (Exception ex)
    {
        Console.WriteLine($"{Path.GetFileName(file)}: failed - {ex.Message}");
    }
}

static string FindRepoRoot()
{
    var dir = Environment.CurrentDirectory;
    while (!string.IsNullOrWhiteSpace(dir))
    {
        if (Directory.Exists(Path.Combine(dir, ".git")))
        {
            return dir;
        }
        dir = Path.GetDirectoryName(dir) ?? string.Empty;
    }

    return Environment.CurrentDirectory;
}

static string[] ResolveFiles(string inputPath)
{
    if (File.Exists(inputPath))
    {
        return new[] { inputPath };
    }

    if (Directory.Exists(inputPath))
    {
        return Directory.GetFiles(inputPath);
    }

    return Array.Empty<string>();
}

static float[] LoadAsMono16k(string path, int maxSeconds)
{
    using var reader = new MediaFoundationReader(path);
    ISampleProvider sampleProvider = reader.ToSampleProvider();

    if (sampleProvider.WaveFormat.Channels > 1)
    {
        if (sampleProvider.WaveFormat.Channels == 2)
        {
            var stereo = new StereoToMonoSampleProvider(sampleProvider)
            {
                LeftVolume = 0.5f,
                RightVolume = 0.5f
            };
            sampleProvider = stereo;
        }
        else
        {
            throw new NotSupportedException($"Unsupported channel count: {sampleProvider.WaveFormat.Channels}");
        }
    }

    if (sampleProvider.WaveFormat.SampleRate != targetSampleRate)
    {
        sampleProvider = new WdlResamplingSampleProvider(sampleProvider, targetSampleRate);
    }

    var maxSamples = targetSampleRate * maxSeconds;
    var result = new List<float>(maxSamples);
    var buffer = new float[targetSampleRate];
    while (result.Count < maxSamples)
    {
        var read = sampleProvider.Read(buffer, 0, buffer.Length);
        if (read <= 0)
        {
            break;
        }

        for (var i = 0; i < read; i++)
        {
            result.Add(buffer[i]);
        }
    }

    return result.ToArray();
}

static float Percentile(List<float> values, double percentile)
{
    if (values.Count == 0)
    {
        return 0f;
    }

    var sorted = values.OrderBy(v => v).ToArray();
    var index = (int)Math.Round((sorted.Length - 1) * percentile);
    return sorted[Math.Clamp(index, 0, sorted.Length - 1)];
}
