using System.Numerics;

namespace PianoActivityTracker.Core.Detection;

internal sealed class LogMelSpectrogram
{
    private readonly int _sampleRate;
    private readonly int _melBins;
    private readonly int _frameLength;
    private readonly int _frameShift;
    private readonly int _fftSize;
    private readonly float[] _window;
    private readonly float[][] _melFilterBank;

    public LogMelSpectrogram(int sampleRate, int melBins, int frameLengthMs, int frameShiftMs)
    {
        if (sampleRate <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(sampleRate));
        }

        if (melBins <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(melBins));
        }

        _sampleRate = sampleRate;
        _melBins = melBins;
        _frameLength = (int)Math.Round(sampleRate * (frameLengthMs / 1000.0));
        _frameShift = (int)Math.Round(sampleRate * (frameShiftMs / 1000.0));
        _fftSize = NextPowerOfTwo(_frameLength);

        _window = BuildPoveyWindow(_frameLength);
        _melFilterBank = BuildMelFilterBank(_melBins, _fftSize, _sampleRate);
    }

    public float[] Compute(ReadOnlySpan<float> samples)
    {
        if (samples.Length < _frameLength)
        {
            return Array.Empty<float>();
        }

        var frameCount = 1 + (samples.Length - _frameLength) / _frameShift;
        var features = new float[frameCount * _melBins];

        var frameBuffer = new float[_frameLength];
        var fftBuffer = new Complex[_fftSize];
        var spectrum = new float[(_fftSize / 2) + 1];

        for (var frame = 0; frame < frameCount; frame++)
        {
            var start = frame * _frameShift;
            var mean = 0f;

            for (var i = 0; i < _frameLength; i++)
            {
                var sample = samples[start + i];
                mean += sample;
                frameBuffer[i] = sample;
            }

            mean /= _frameLength;

            var prev = 0f;
            for (var i = 0; i < _frameLength; i++)
            {
                var sample = frameBuffer[i] - mean;
                var emphasized = sample - (0.97f * prev);
                prev = sample;
                frameBuffer[i] = emphasized * _window[i];
            }

            Array.Clear(fftBuffer, 0, fftBuffer.Length);
            for (var i = 0; i < _frameLength; i++)
            {
                fftBuffer[i] = new Complex(frameBuffer[i], 0);
            }

            FftInPlace(fftBuffer);

            for (var i = 0; i < spectrum.Length; i++)
            {
                var mag = fftBuffer[i].Magnitude;
                spectrum[i] = (float)(mag * mag);
            }

            for (var mel = 0; mel < _melBins; mel++)
            {
                double energy = 0;
                var filter = _melFilterBank[mel];
                for (var bin = 0; bin < spectrum.Length; bin++)
                {
                    energy += spectrum[bin] * filter[bin];
                }

                var logEnergy = MathF.Log(MathF.Max((float)energy, 1e-10f));
                features[(frame * _melBins) + mel] = logEnergy;
            }
        }

        return features;
    }

    private static float[] BuildPoveyWindow(int length)
    {
        var window = new float[length];
        if (length == 1)
        {
            window[0] = 1f;
            return window;
        }

        for (var i = 0; i < length; i++)
        {
            var hann = 0.5f - 0.5f * MathF.Cos((2f * MathF.PI * i) / (length - 1));
            window[i] = MathF.Pow(hann, 0.85f);
        }

        return window;
    }

    private static float[][] BuildMelFilterBank(int melBins, int fftSize, int sampleRate)
    {
        var melMin = HzToMel(0);
        var melMax = HzToMel(sampleRate / 2f);
        var melPoints = new float[melBins + 2];

        for (var i = 0; i < melPoints.Length; i++)
        {
            melPoints[i] = melMin + (melMax - melMin) * i / (melBins + 1);
        }

        var hzPoints = melPoints.Select(MelToHz).ToArray();
        var binPoints = hzPoints.Select(hz => (int)Math.Floor((fftSize + 1) * hz / sampleRate)).ToArray();

        var filters = new float[melBins][];
        var fftBins = (fftSize / 2) + 1;

        for (var m = 0; m < melBins; m++)
        {
            filters[m] = new float[fftBins];
            var left = binPoints[m];
            var center = binPoints[m + 1];
            var right = binPoints[m + 2];

            if (center == left)
            {
                center++;
            }

            if (right == center)
            {
                right++;
            }

            for (var k = left; k < center && k < fftBins; k++)
            {
                filters[m][k] = (float)(k - left) / (center - left);
            }

            for (var k = center; k < right && k < fftBins; k++)
            {
                filters[m][k] = (float)(right - k) / (right - center);
            }
        }

        return filters;
    }

    private static float HzToMel(float hz)
    {
        return 2595f * MathF.Log10(1f + (hz / 700f));
    }

    private static float MelToHz(float mel)
    {
        return 700f * (MathF.Pow(10f, mel / 2595f) - 1f);
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
