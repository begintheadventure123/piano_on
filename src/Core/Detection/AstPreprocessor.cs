using Microsoft.ML.OnnxRuntime.Tensors;

namespace PianoActivityTracker.Core.Detection;

internal sealed class AstPreprocessor
{
    private readonly AstPreprocessorConfig _config;
    private readonly LogMelSpectrogram _spectrogram;

    public AstPreprocessor(AstPreprocessorConfig config)
    {
        _config = config ?? throw new ArgumentNullException(nameof(config));
        _spectrogram = new LogMelSpectrogram(
            config.SamplingRate,
            config.NumMelBins,
            frameLengthMs: 25,
            frameShiftMs: 10);
    }

    public int SampleRate => _config.SamplingRate;

    public int RequiredSamples => _config.SamplingRate * 10;

    public DenseTensor<float> CreateInputTensor(float[] samples, int[] inputShape)
    {
        var features = _spectrogram.Compute(samples);
        var padded = PadOrTrim(features, _config.MaxLength, _config.NumMelBins, _config.PaddingValue);
        Normalize(padded, _config.Mean, _config.Std);

        return BuildTensor(padded, _config.MaxLength, _config.NumMelBins, inputShape);
    }

    private static float[] PadOrTrim(float[] features, int maxLength, int melBins, float paddingValue)
    {
        var frameCount = features.Length / melBins;
        if (frameCount == maxLength)
        {
            return features;
        }

        var padded = new float[maxLength * melBins];
        var framesToCopy = Math.Min(frameCount, maxLength);
        Array.Copy(features, padded, framesToCopy * melBins);

        if (framesToCopy < maxLength && paddingValue != 0f)
        {
            var start = framesToCopy * melBins;
            for (var i = start; i < padded.Length; i++)
            {
                padded[i] = paddingValue;
            }
        }

        return padded;
    }

    private static void Normalize(float[] features, float mean, float std)
    {
        var denom = Math.Abs(std) < 1e-6f ? 1f : std;
        for (var i = 0; i < features.Length; i++)
        {
            features[i] = (features[i] - mean) / denom;
        }
    }

    private static DenseTensor<float> BuildTensor(float[] features, int frames, int melBins, int[] inputShape)
    {
        var shape = inputShape.Length switch
        {
            4 => new[] { 1, 1, frames, melBins },
            _ => new[] { 1, frames, melBins }
        };

        var tensor = new DenseTensor<float>(shape);
        var offset = 0;
        if (shape.Length == 4)
        {
            for (var t = 0; t < frames; t++)
            {
                for (var m = 0; m < melBins; m++)
                {
                    tensor[0, 0, t, m] = features[offset++];
                }
            }
        }
        else
        {
            for (var t = 0; t < frames; t++)
            {
                for (var m = 0; m < melBins; m++)
                {
                    tensor[0, t, m] = features[offset++];
                }
            }
        }

        return tensor;
    }
}
