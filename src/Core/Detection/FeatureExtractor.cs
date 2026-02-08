using System.Numerics;
using PianoActivityTracker.Core.Audio;

namespace PianoActivityTracker.Core.Detection;

public sealed class FeatureExtractor
{
    public FeatureVector Extract(AudioFrame frame)
    {
        if (frame.SampleRate != 16_000)
        {
            throw new ArgumentException("AudioFrame.SampleRate must be 16000Hz.", nameof(frame));
        }

        if (frame.Samples.Length == 0)
        {
            return default;
        }

        var rms = ComputeRms(frame.Samples);
        var zcr = ComputeZeroCrossingRate(frame.Samples);

        var spectrum = ComputeMagnitudeSpectrum(frame.Samples, frame.SampleRate, out var freqResolution);
        var centroid = ComputeCentroid(spectrum, freqResolution);
        var bandwidth = ComputeBandwidth(spectrum, freqResolution, centroid);
        var rolloff = ComputeRolloff(spectrum, freqResolution, 0.85f);
        var flatness = ComputeSpectralFlatness(spectrum);
        ComputeBandEnergies(spectrum, freqResolution, out var lowEnergy, out var midEnergy, out var highEnergy);

        var mfcc = ComputePlaceholderMfcc(spectrum);

        return new FeatureVector(
            rms,
            zcr,
            centroid,
            bandwidth,
            rolloff,
            flatness,
            lowEnergy,
            midEnergy,
            highEnergy,
            mfcc[0],
            mfcc[1],
            mfcc[2],
            mfcc[3],
            mfcc[4]);
    }

    private static float ComputeRms(ReadOnlySpan<float> samples)
    {
        double sum = 0;
        foreach (var sample in samples)
        {
            sum += sample * sample;
        }

        return (float)Math.Sqrt(sum / samples.Length);
    }

    private static float ComputeZeroCrossingRate(ReadOnlySpan<float> samples)
    {
        if (samples.Length < 2)
        {
            return 0;
        }

        var crossings = 0;
        for (var i = 1; i < samples.Length; i++)
        {
            var prev = samples[i - 1];
            var current = samples[i];
            if ((prev >= 0 && current < 0) || (prev < 0 && current >= 0))
            {
                crossings++;
            }
        }

        return (float)crossings / (samples.Length - 1);
    }

    private static float[] ComputeMagnitudeSpectrum(ReadOnlySpan<float> samples, int sampleRate, out float freqResolution)
    {
        var fftSize = NextPowerOfTwo(Math.Min(samples.Length, 4096));
        if (fftSize < 256)
        {
            fftSize = 256;
        }

        var buffer = new Complex[fftSize];
        var usable = Math.Min(samples.Length, fftSize);

        for (var i = 0; i < usable; i++)
        {
            var hann = 0.5f - 0.5f * MathF.Cos((2f * MathF.PI * i) / (usable - 1));
            buffer[i] = new Complex(samples[i] * hann, 0);
        }

        FftInPlace(buffer);

        var magnitudes = new float[(fftSize / 2) + 1];
        for (var i = 0; i < magnitudes.Length; i++)
        {
            magnitudes[i] = (float)buffer[i].Magnitude;
        }

        freqResolution = (float)sampleRate / fftSize;
        return magnitudes;
    }

    private static float ComputeCentroid(ReadOnlySpan<float> magnitudes, float freqResolution)
    {
        double weighted = 0;
        double total = 0;

        for (var i = 0; i < magnitudes.Length; i++)
        {
            var magnitude = magnitudes[i];
            weighted += magnitude * (i * freqResolution);
            total += magnitude;
        }

        return total <= 0 ? 0 : (float)(weighted / total);
    }

    private static float ComputeBandwidth(ReadOnlySpan<float> magnitudes, float freqResolution, float centroid)
    {
        double weighted = 0;
        double total = 0;

        for (var i = 0; i < magnitudes.Length; i++)
        {
            var freq = i * freqResolution;
            var diff = freq - centroid;
            var magnitude = magnitudes[i];
            weighted += diff * diff * magnitude;
            total += magnitude;
        }

        return total <= 0 ? 0 : (float)Math.Sqrt(weighted / total);
    }

    private static float ComputeRolloff(ReadOnlySpan<float> magnitudes, float freqResolution, float percentile)
    {
        double totalEnergy = 0;
        for (var i = 0; i < magnitudes.Length; i++)
        {
            totalEnergy += magnitudes[i] * magnitudes[i];
        }

        if (totalEnergy <= 0)
        {
            return 0;
        }

        var threshold = totalEnergy * percentile;
        double cumulative = 0;
        for (var i = 0; i < magnitudes.Length; i++)
        {
            cumulative += magnitudes[i] * magnitudes[i];
            if (cumulative >= threshold)
            {
                return i * freqResolution;
            }
        }

        return (magnitudes.Length - 1) * freqResolution;
    }

    private static float ComputeSpectralFlatness(ReadOnlySpan<float> magnitudes)
    {
        if (magnitudes.Length == 0)
        {
            return 0;
        }

        double geoMeanLog = 0;
        double arithMean = 0;
        var count = 0;

        for (var i = 0; i < magnitudes.Length; i++)
        {
            var mag = Math.Max(magnitudes[i], 1e-12f);
            geoMeanLog += Math.Log(mag);
            arithMean += mag;
            count++;
        }

        if (count == 0 || arithMean <= 0)
        {
            return 0;
        }

        var geoMean = Math.Exp(geoMeanLog / count);
        var flatness = geoMean / (arithMean / count);
        return (float)Math.Clamp(flatness, 0.0, 1.0);
    }

    private static void ComputeBandEnergies(ReadOnlySpan<float> magnitudes, float freqResolution, out float low, out float mid, out float high)
    {
        double lowSum = 0;
        double midSum = 0;
        double highSum = 0;

        for (var i = 0; i < magnitudes.Length; i++)
        {
            var freq = i * freqResolution;
            var mag = magnitudes[i];
            var energy = mag * mag;

            if (freq < 300f)
            {
                lowSum += energy;
            }
            else if (freq < 2000f)
            {
                midSum += energy;
            }
            else
            {
                highSum += energy;
            }
        }

        low = (float)Math.Log10(1 + lowSum);
        mid = (float)Math.Log10(1 + midSum);
        high = (float)Math.Log10(1 + highSum);
    }

    private static float[] ComputePlaceholderMfcc(ReadOnlySpan<float> magnitudes)
    {
        // TODO: Replace with proper mel filter bank + DCT MFCC implementation.
        // MVP uses five log-energy bands to keep interface and scoring stable.
        var coefficients = new float[5];
        if (magnitudes.Length == 0)
        {
            return coefficients;
        }

        var bandSize = Math.Max(1, magnitudes.Length / coefficients.Length);
        for (var i = 0; i < coefficients.Length; i++)
        {
            var start = i * bandSize;
            var end = i == coefficients.Length - 1 ? magnitudes.Length : Math.Min(magnitudes.Length, start + bandSize);

            double energy = 0;
            for (var j = start; j < end; j++)
            {
                energy += magnitudes[j] * magnitudes[j];
            }

            coefficients[i] = (float)Math.Log10(1 + energy);
        }

        return coefficients;
    }

    private static int NextPowerOfTwo(int value)
    {
        var n = 1;
        while (n < value)
        {
            n <<= 1;
        }

        return n;
    }

    private static void FftInPlace(Complex[] buffer)
    {
        var n = buffer.Length;

        var j = 0;
        for (var i = 1; i < n; i++)
        {
            var bit = n >> 1;
            while ((j & bit) != 0)
            {
                j ^= bit;
                bit >>= 1;
            }

            j ^= bit;
            if (i < j)
            {
                (buffer[i], buffer[j]) = (buffer[j], buffer[i]);
            }
        }

        for (var len = 2; len <= n; len <<= 1)
        {
            var angle = -2 * Math.PI / len;
            var wLen = new Complex(Math.Cos(angle), Math.Sin(angle));

            for (var i = 0; i < n; i += len)
            {
                var w = Complex.One;
                var half = len / 2;
                for (var k = 0; k < half; k++)
                {
                    var u = buffer[i + k];
                    var v = buffer[i + k + half] * w;
                    buffer[i + k] = u + v;
                    buffer[i + k + half] = u - v;
                    w *= wLen;
                }
            }
        }
    }
}
