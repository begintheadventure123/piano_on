using NAudio.CoreAudioApi;
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

    private IWaveIn? _waveIn;
    private bool _running;
    private int _sourceSampleRate;
    private int _sourceChannels;
    private WaveFormat? _sourceFormat;
    private string? _captureDeviceName;
    private double _sourceReadPosition;
    private DateTime _captureStartUtc;
    private long _emittedHops;
    private WaveFileWriter? _debugWriterProcessed;
    private WaveFileWriter? _debugWriterRaw;
    private TimeSpan _debugMaxDuration = TimeSpan.Zero;
    private long _debugSamplesWritten;
    private string? _debugRecordingPathProcessed;
    private string? _debugRecordingPathRaw;
    private int _lastRawBytes;
    private double _lastRawRms;
    private double _lastRawPeak;
    private DateTime? _debugStartUtc;
    private int _dataCallbackCount;

    public event Action<AudioFrame>? OnFrameReady;

    public string? CaptureInfo { get; private set; }

    public string? DebugRecordingStatus { get; private set; }

    public string? RawFormatInfo { get; private set; }

    public int LastRawBytes => _lastRawBytes;

    public double LastRawRms => _lastRawRms;

    public double LastRawPeak => _lastRawPeak;

    public int DataCallbackCount => _dataCallbackCount;

    public void Start()
    {
        lock (_sync)
        {
            if (_running)
            {
                return;
            }

            _waveIn = CreateCaptureDevice(out _captureDeviceName);
            _sourceFormat = _waveIn.WaveFormat;
            _sourceSampleRate = _sourceFormat.SampleRate;
            _sourceChannels = _sourceFormat.Channels;
            _sourceReadPosition = 0;
            _sourceMonoBuffer.Clear();
            _targetBuffer.Clear();
            _captureStartUtc = DateTime.UtcNow;
            _emittedHops = 0;
            _debugSamplesWritten = 0;
            _dataCallbackCount = 0;
            CaptureInfo = BuildCaptureInfo(_waveIn, _sourceFormat, _captureDeviceName);
            RawFormatInfo = BuildRawFormatInfo(_sourceFormat);
            UpdateDebugStatus("Debug recording disabled");

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
            StopDebugRecordingInternal("Debug recording stopped");
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

            if (_sourceFormat is null)
            {
                return;
            }

            _dataCallbackCount++;
            UpdateRawStats(e.Buffer, e.BytesRecorded, _sourceFormat);

            var sourceSamples = ConvertToMono(e.Buffer, e.BytesRecorded, _sourceFormat);
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
            AppendDebugRawBytes(e.Buffer, e.BytesRecorded);
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
            AppendDebugHopSamples(_targetBuffer, HopSamples);

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

    private static List<float> ConvertToMono(byte[] buffer, int bytesRecorded, WaveFormat format)
    {
        var samples = new List<float>();
        if (bytesRecorded <= 0 || format.Channels <= 0)
        {
            return samples;
        }

        var bytesPerSample = format.BitsPerSample / 8;
        var channels = format.Channels;
        var frameSize = bytesPerSample * channels;
        var frameCount = bytesRecorded / frameSize;

        if (bytesPerSample == 2 && format.Encoding == WaveFormatEncoding.Pcm)
        {
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

        if (bytesPerSample == 4 && format.Encoding == WaveFormatEncoding.IeeeFloat)
        {
            for (var i = 0; i < frameCount; i++)
            {
                double sum = 0;
                for (var channel = 0; channel < channels; channel++)
                {
                    var offset = (i * frameSize) + (channel * bytesPerSample);
                    var value = BitConverter.ToSingle(buffer, offset);
                    sum += value;
                }

                samples.Add((float)(sum / channels));
            }

            return samples;
        }

        return samples;
    }

    private static IWaveIn CreateCaptureDevice(out string? deviceName)
    {
        deviceName = null;
        try
        {
            var enumerator = new MMDeviceEnumerator();
            var device = enumerator.GetDefaultAudioEndpoint(DataFlow.Capture, Role.Multimedia);
            var capture = new WasapiCapture(device)
            {
                ShareMode = AudioClientShareMode.Shared
            };

            // Prefer the device mix format to avoid unsupported format errors.
            capture.WaveFormat = device.AudioClient.MixFormat;
            deviceName = device.FriendlyName;
            return capture;
        }
        catch
        {
            // Fall back to legacy WaveIn if WASAPI is unavailable.
        }

        try
        {
            var enumerator = new MMDeviceEnumerator();
            var device = enumerator.GetDefaultAudioEndpoint(DataFlow.Capture, Role.Communications);
            var capture = new WasapiCapture(device)
            {
                ShareMode = AudioClientShareMode.Shared,
                WaveFormat = device.AudioClient.MixFormat
            };

            deviceName = device.FriendlyName;
            return capture;
        }
        catch
        {
            // Continue to WaveIn fallback.
        }

        if (WaveInEvent.DeviceCount < 1)
        {
            throw new InvalidOperationException("No microphone input device is available.");
        }

        var waveIn = new WaveInEvent
        {
            BufferMilliseconds = 50,
            NumberOfBuffers = 3,
            WaveFormat = new WaveFormat(44_100, 16, 2)
        };

        deviceName = "WaveIn";
        return waveIn;
    }

    public void EnableDebugRecording(string processedPath, string rawPath, TimeSpan maxDuration)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(processedPath);
        ArgumentException.ThrowIfNullOrWhiteSpace(rawPath);

        lock (_sync)
        {
            _debugRecordingPathProcessed = processedPath;
            _debugRecordingPathRaw = rawPath;
            _debugMaxDuration = maxDuration <= TimeSpan.Zero ? TimeSpan.FromSeconds(10) : maxDuration;
            _debugSamplesWritten = 0;
            _debugStartUtc = DateTime.UtcNow;

            if (_debugWriterProcessed is not null)
            {
                _debugWriterProcessed.Dispose();
                _debugWriterProcessed = null;
            }

            if (_debugWriterRaw is not null)
            {
                _debugWriterRaw.Dispose();
                _debugWriterRaw = null;
            }

            UpdateDebugStatus($"Debug recording armed: {processedPath}");
        }
    }

    private void AppendDebugHopSamples(List<float> buffer, int hopSamples)
    {
        if (_debugRecordingPathProcessed is null || _debugMaxDuration <= TimeSpan.Zero)
        {
            return;
        }

        if (_debugWriterProcessed is null)
        {
            var format = WaveFormat.CreateIeeeFloatWaveFormat(TargetSampleRate, 1);
            _debugWriterProcessed = new WaveFileWriter(_debugRecordingPathProcessed, format);
            UpdateDebugStatus($"Debug recording started: {_debugRecordingPathProcessed}");
        }

        if (buffer.Count < hopSamples)
        {
            return;
        }

        var hop = buffer.GetRange(0, hopSamples).ToArray();
        _debugWriterProcessed.WriteSamples(hop, 0, hop.Length);
        _debugSamplesWritten += hop.Length;

        var maxSamples = (long)(_debugMaxDuration.TotalSeconds * TargetSampleRate);
        if (_debugSamplesWritten >= maxSamples)
        {
            StopDebugRecordingInternal($"Debug recording saved: {_debugRecordingPathProcessed}");
        }
    }

    private void AppendDebugRawBytes(byte[] buffer, int bytesRecorded)
    {
        if (_debugRecordingPathRaw is null || _debugMaxDuration <= TimeSpan.Zero)
        {
            return;
        }

        if (_sourceFormat is null)
        {
            return;
        }

        if (_debugWriterRaw is null)
        {
            _debugWriterRaw = new WaveFileWriter(_debugRecordingPathRaw, _sourceFormat);
        }

        _debugWriterRaw.Write(buffer, 0, bytesRecorded);

        if (_debugStartUtc is null)
        {
            _debugStartUtc = DateTime.UtcNow;
        }

        if (DateTime.UtcNow - _debugStartUtc >= _debugMaxDuration)
        {
            StopDebugRecordingInternal($"Debug recording saved: {_debugRecordingPathProcessed}");
        }
    }

    private void StopDebugRecordingInternal(string status)
    {
        if (_debugWriterProcessed is not null)
        {
            _debugWriterProcessed.Dispose();
            _debugWriterProcessed = null;
        }

        if (_debugWriterRaw is not null)
        {
            _debugWriterRaw.Dispose();
            _debugWriterRaw = null;
        }

        _debugRecordingPathProcessed = null;
        _debugRecordingPathRaw = null;
        _debugStartUtc = null;
        UpdateDebugStatus(status);
    }

    private void UpdateDebugStatus(string status)
    {
        DebugRecordingStatus = status;
    }

    private static string BuildCaptureInfo(IWaveIn capture, WaveFormat format, string? deviceName)
    {
        if (capture is WasapiCapture)
        {
            var name = string.IsNullOrWhiteSpace(deviceName) ? "Unknown device" : deviceName;
            return $"WASAPI: {name} | {format.Encoding} {format.SampleRate}Hz {format.BitsPerSample}bit {format.Channels}ch";
        }

        return $"WaveIn: {format.Encoding} {format.SampleRate}Hz {format.BitsPerSample}bit {format.Channels}ch";
    }

    private static string BuildRawFormatInfo(WaveFormat? format)
    {
        if (format is null)
        {
            return "Raw format: unknown";
        }

        return $"Raw format: {format.Encoding} {format.SampleRate}Hz {format.BitsPerSample}bit {format.Channels}ch";
    }

    private void UpdateRawStats(byte[] buffer, int bytesRecorded, WaveFormat format)
    {
        _lastRawBytes = bytesRecorded;
        if (bytesRecorded <= 0)
        {
            _lastRawRms = 0;
            _lastRawPeak = 0;
            return;
        }

        var bytesPerSample = format.BitsPerSample / 8;
        var channels = Math.Max(1, format.Channels);
        var frameSize = bytesPerSample * channels;
        var frameCount = bytesRecorded / frameSize;
        if (frameCount <= 0)
        {
            _lastRawRms = 0;
            _lastRawPeak = 0;
            return;
        }

        double sumSquares = 0;
        double peak = 0;
        if (bytesPerSample == 2 && format.Encoding == WaveFormatEncoding.Pcm)
        {
            for (var i = 0; i < frameCount; i++)
            {
                for (var channel = 0; channel < channels; channel++)
                {
                    var offset = (i * frameSize) + (channel * bytesPerSample);
                    short value = (short)(buffer[offset] | (buffer[offset + 1] << 8));
                    var sample = value / 32768.0;
                    sumSquares += sample * sample;
                    var abs = Math.Abs(sample);
                    if (abs > peak)
                    {
                        peak = abs;
                    }
                }
            }
        }
        else if (bytesPerSample == 4 && format.Encoding == WaveFormatEncoding.IeeeFloat)
        {
            for (var i = 0; i < frameCount; i++)
            {
                for (var channel = 0; channel < channels; channel++)
                {
                    var offset = (i * frameSize) + (channel * bytesPerSample);
                    var sample = BitConverter.ToSingle(buffer, offset);
                    sumSquares += sample * sample;
                    var abs = Math.Abs(sample);
                    if (abs > peak)
                    {
                        peak = abs;
                    }
                }
            }
        }

        var totalSamples = frameCount * channels;
        _lastRawRms = totalSamples > 0 ? Math.Sqrt(sumSquares / totalSamples) : 0;
        _lastRawPeak = peak;
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
        _sourceFormat = null;
        _captureDeviceName = null;
    }
}
