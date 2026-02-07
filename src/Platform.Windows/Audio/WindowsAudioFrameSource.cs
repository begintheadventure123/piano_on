using NAudio.Wave;
using PianoActivityTracker.Core.Audio;

namespace PianoActivityTracker.Platform.Windows.Audio;

public sealed class WindowsAudioFrameSource : IAudioFrameSource, IDisposable
{
    private const int TargetSampleRate = 16_000;
    private const int FrameSamples = 16_000;
    private const int HopSamples = 4_000;

    private readonly object _sync = new();
    private readonly List<float> _sourceMonoBuffer = new();
    private readonly List<float> _targetBuffer = new();

    private WaveInEvent? _waveIn;
    private bool _running;
    private int _sourceSampleRate;
    private int _sourceChannels;
    private double _sourceReadPosition;
    private DateTime _captureStartUtc;
    private long _emittedHops;

    public event Action<AudioFrame>? OnFrameReady;

    public void Start()
    {
        lock (_sync)
        {
            if (_running)
            {
                return;
            }

            if (WaveInEvent.DeviceCount < 1)
            {
                throw new InvalidOperationException("No microphone input device is available.");
            }

            _waveIn = new WaveInEvent
            {
                BufferMilliseconds = 50,
                NumberOfBuffers = 3,
                WaveFormat = new WaveFormat(44_100, 16, 1)
            };

            _sourceSampleRate = _waveIn.WaveFormat.SampleRate;
            _sourceChannels = _waveIn.WaveFormat.Channels;
            _sourceReadPosition = 0;
            _sourceMonoBuffer.Clear();
            _targetBuffer.Clear();
            _captureStartUtc = DateTime.UtcNow;
            _emittedHops = 0;

            _waveIn.DataAvailable += OnDataAvailable;
            _waveIn.RecordingStopped += OnRecordingStopped;

            try
            {
                _waveIn.StartRecording();
                _running = true;
            }
            catch
            {
                CleanupWaveIn();
                throw;
            }
        }
    }

    public void Stop()
    {
        lock (_sync)
        {
            if (!_running)
            {
                return;
            }

            _running = false;
            _waveIn?.StopRecording();
            CleanupWaveIn();
            _sourceMonoBuffer.Clear();
            _targetBuffer.Clear();
            _sourceReadPosition = 0;
            _emittedHops = 0;
        }
    }

    public void Dispose()
    {
        Stop();
        GC.SuppressFinalize(this);
    }

    private void OnRecordingStopped(object? sender, StoppedEventArgs e)
    {
        if (e.Exception is not null)
        {
            // Surface as start/stop failures via UI flow, do not crash capture thread.
        }
    }

    private void OnDataAvailable(object? sender, WaveInEventArgs e)
    {
        List<AudioFrame>? pendingFrames = null;

        lock (_sync)
        {
            if (!_running || _waveIn is null)
            {
                return;
            }

            var sourceSamples = ConvertPcm16ToMono(e.Buffer, e.BytesRecorded, _sourceChannels);
            if (sourceSamples.Count == 0)
            {
                return;
            }

            _sourceMonoBuffer.AddRange(sourceSamples);

            if (_sourceSampleRate == TargetSampleRate)
            {
                _targetBuffer.AddRange(_sourceMonoBuffer);
                _sourceMonoBuffer.Clear();
            }
            else
            {
                ResampleSourceToTarget();
            }

            pendingFrames = ExtractFrames();
        }

        if (pendingFrames is null || pendingFrames.Count == 0)
        {
            return;
        }

        foreach (var frame in pendingFrames)
        {
            OnFrameReady?.Invoke(frame);
        }
    }

    private List<AudioFrame> ExtractFrames()
    {
        var frames = new List<AudioFrame>();

        while (_targetBuffer.Count >= FrameSamples)
        {
            var samples = _targetBuffer.GetRange(0, FrameSamples).ToArray();
            var startTime = _captureStartUtc + TimeSpan.FromSeconds((double)(_emittedHops * HopSamples) / TargetSampleRate);
            frames.Add(new AudioFrame(samples, TargetSampleRate, startTime));

            _targetBuffer.RemoveRange(0, HopSamples);
            _emittedHops++;
        }

        return frames;
    }

    private void ResampleSourceToTarget()
    {
        if (_sourceMonoBuffer.Count < 2)
        {
            return;
        }

        var step = (double)_sourceSampleRate / TargetSampleRate;

        while (_sourceReadPosition + 1 < _sourceMonoBuffer.Count)
        {
            var index = (int)_sourceReadPosition;
            var frac = _sourceReadPosition - index;

            var s0 = _sourceMonoBuffer[index];
            var s1 = _sourceMonoBuffer[index + 1];
            var sample = (float)(s0 + ((s1 - s0) * frac));
            _targetBuffer.Add(sample);

            _sourceReadPosition += step;
        }

        var consumed = Math.Max(0, (int)_sourceReadPosition - 1);
        if (consumed <= 0)
        {
            return;
        }

        _sourceMonoBuffer.RemoveRange(0, consumed);
        _sourceReadPosition -= consumed;
    }

    private static List<float> ConvertPcm16ToMono(byte[] buffer, int bytesRecorded, int channels)
    {
        var samples = new List<float>();
        if (bytesRecorded <= 0 || channels <= 0)
        {
            return samples;
        }

        var bytesPerSample = 2;
        var frameSize = bytesPerSample * channels;
        var frameCount = bytesRecorded / frameSize;

        for (var i = 0; i < frameCount; i++)
        {
            double sum = 0;
            for (var channel = 0; channel < channels; channel++)
            {
                var offset = (i * frameSize) + (channel * bytesPerSample);
                short value = (short)(buffer[offset] | (buffer[offset + 1] << 8));
                sum += value / 32768f;
            }

            samples.Add((float)(sum / channels));
        }

        return samples;
    }

    private void CleanupWaveIn()
    {
        if (_waveIn is null)
        {
            return;
        }

        _waveIn.DataAvailable -= OnDataAvailable;
        _waveIn.RecordingStopped -= OnRecordingStopped;
        _waveIn.Dispose();
        _waveIn = null;
    }
}
