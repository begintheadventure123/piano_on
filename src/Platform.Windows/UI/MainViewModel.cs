using System.Collections.ObjectModel;
using System.ComponentModel;
using System.IO;
using System.Runtime.CompilerServices;
using System.Windows;
using System.Windows.Media;
using System.Windows.Threading;
using PianoActivityTracker.Core.Accumulation;
using PianoActivityTracker.Core.Audio;
using PianoActivityTracker.Core.Detection;
using PianoActivityTracker.Core.Storage;
using PianoActivityTracker.Platform.Windows.Audio;

namespace PianoActivityTracker.Platform.Windows.UI;

public sealed class MainViewModel : INotifyPropertyChanged, IDisposable
{
    private readonly IAudioFrameSource _audioSource;
    private readonly IPianoDetector _detector;
    private readonly ActivityAccumulator _accumulator;
    private readonly ISessionStore _sessionStore;
    private readonly DispatcherTimer _uiTimer;

    private bool _isRunning;
    private string _statusText = "Ready";
    private string _liveTimeText = "00:00:00";
    private string _summaryText = string.Empty;
    private string _errorText = string.Empty;
    private PointCollection _waveformPoints = new();
    private Visibility _waveformVisibility = Visibility.Collapsed;
    private double _waveformOpacity = 0.35;
    private double _waveformRms;
    private int _frameCount;
    private DateTime? _lastFrameUtc;
    private string _debugInfoText = "Debug: not started";
    private string _debugRecordingText = "Debug recording: off";
    private string _debugRawText = "Raw: n/a";
    private string _debugRawStatsText = "Raw stats: n/a";
    private string _debugCallbackText = "Callbacks: 0";

    private const int WaveformWidth = 240;
    private const int WaveformHeight = 70;
    private const int WaveformPointCount = 240;
    private const float SoundRmsThreshold = 0.003f;

    public MainViewModel()
        : this(
            new WindowsAudioFrameSource(),
            CreateDefaultDetector(out var detectorWarning),
            new ActivityAccumulator(),
            new JsonSessionStore())
    {
        if (!string.IsNullOrWhiteSpace(detectorWarning))
        {
            ErrorText = detectorWarning;
            StatusText = "Model missing";
        }
    }

    public MainViewModel(
        IAudioFrameSource audioSource,
        IPianoDetector detector,
        ActivityAccumulator accumulator,
        ISessionStore sessionStore)
    {
        _audioSource = audioSource;
        _detector = detector;
        _accumulator = accumulator;
        _sessionStore = sessionStore;

        StartCommand = new RelayCommand(StartSession, () => !IsRunning);
        StopCommand = new RelayCommand(StopSession, () => IsRunning);

        _audioSource.OnFrameReady += OnFrameReady;

        _uiTimer = new DispatcherTimer
        {
            Interval = TimeSpan.FromMilliseconds(200)
        };
        _uiTimer.Tick += (_, _) =>
        {
            LiveTimeText = FormatDuration(_accumulator.CurrentTotalPianoTime);
            UpdateStatusFromState();
            UpdateDebugFromSource();
        };

        History = new ObservableCollection<SessionHistoryItem>();
        ReloadHistory();
    }

    public event PropertyChangedEventHandler? PropertyChanged;

    public RelayCommand StartCommand { get; }

    public RelayCommand StopCommand { get; }

    public ObservableCollection<SessionHistoryItem> History { get; }

    public bool IsRunning
    {
        get => _isRunning;
        private set
        {
            if (_isRunning == value)
            {
                return;
            }

            _isRunning = value;
            OnPropertyChanged();
            StartCommand.RaiseCanExecuteChanged();
            StopCommand.RaiseCanExecuteChanged();
        }
    }

    public string StatusText
    {
        get => _statusText;
        private set
        {
            if (_statusText == value)
            {
                return;
            }

            _statusText = value;
            OnPropertyChanged();
        }
    }

    public string LiveTimeText
    {
        get => _liveTimeText;
        private set
        {
            if (_liveTimeText == value)
            {
                return;
            }

            _liveTimeText = value;
            OnPropertyChanged();
        }
    }

    public string SummaryText
    {
        get => _summaryText;
        private set
        {
            if (_summaryText == value)
            {
                return;
            }

            _summaryText = value;
            OnPropertyChanged();
        }
    }

    public string ErrorText
    {
        get => _errorText;
        private set
        {
            if (_errorText == value)
            {
                return;
            }

            _errorText = value;
            OnPropertyChanged();
        }
    }

    public PointCollection WaveformPoints
    {
        get => _waveformPoints;
        private set
        {
            if (ReferenceEquals(_waveformPoints, value))
            {
                return;
            }

            _waveformPoints = value;
            OnPropertyChanged();
        }
    }

    public Visibility WaveformVisibility
    {
        get => _waveformVisibility;
        private set
        {
            if (_waveformVisibility == value)
            {
                return;
            }

            _waveformVisibility = value;
            OnPropertyChanged();
        }
    }

    public double WaveformOpacity
    {
        get => _waveformOpacity;
        private set
        {
            if (Math.Abs(_waveformOpacity - value) < 0.001)
            {
                return;
            }

            _waveformOpacity = value;
            OnPropertyChanged();
        }
    }

    public double WaveformRms
    {
        get => _waveformRms;
        private set
        {
            if (Math.Abs(_waveformRms - value) < 0.0001)
            {
                return;
            }

            _waveformRms = value;
            OnPropertyChanged();
        }
    }

    public int FrameCount
    {
        get => _frameCount;
        private set
        {
            if (_frameCount == value)
            {
                return;
            }

            _frameCount = value;
            OnPropertyChanged();
        }
    }

    public string LastFrameText => _lastFrameUtc is null
        ? "No frames yet"
        : $"Last frame: {_lastFrameUtc.Value.ToLocalTime():HH:mm:ss}";

    public string DebugInfoText
    {
        get => _debugInfoText;
        private set
        {
            if (_debugInfoText == value)
            {
                return;
            }

            _debugInfoText = value;
            OnPropertyChanged();
        }
    }

    public string DebugRecordingText
    {
        get => _debugRecordingText;
        private set
        {
            if (_debugRecordingText == value)
            {
                return;
            }

            _debugRecordingText = value;
            OnPropertyChanged();
        }
    }

    public string DebugRawText
    {
        get => _debugRawText;
        private set
        {
            if (_debugRawText == value)
            {
                return;
            }

            _debugRawText = value;
            OnPropertyChanged();
        }
    }

    public string DebugRawStatsText
    {
        get => _debugRawStatsText;
        private set
        {
            if (_debugRawStatsText == value)
            {
                return;
            }

            _debugRawStatsText = value;
            OnPropertyChanged();
        }
    }

    public string DebugCallbackText
    {
        get => _debugCallbackText;
        private set
        {
            if (_debugCallbackText == value)
            {
                return;
            }

            _debugCallbackText = value;
            OnPropertyChanged();
        }
    }

    public void Dispose()
    {
        _uiTimer.Stop();
        _audioSource.OnFrameReady -= OnFrameReady;

        if (_audioSource is IDisposable disposable)
        {
            disposable.Dispose();
        }
    }

    private void StartSession()
    {
        ErrorText = string.Empty;
        SummaryText = string.Empty;
        LiveTimeText = "00:00:00";
        WaveformVisibility = Visibility.Collapsed;
        WaveformOpacity = 0.35;
        WaveformRms = 0;
        FrameCount = 0;
        _lastFrameUtc = null;
        OnPropertyChanged(nameof(LastFrameText));
        WaveformPoints = new PointCollection(BuildFlatWaveform());
        DebugInfoText = "Debug: starting";
        DebugRecordingText = "Debug recording: arming";

        var start = DateTime.UtcNow;
        _accumulator.StartSession(start);

        try
        {
            if (_audioSource is WindowsAudioFrameSource windowsSource)
            {
                var processedPath = Path.Combine(
                    Environment.GetFolderPath(Environment.SpecialFolder.DesktopDirectory),
                    $"piano_debug_processed_{DateTime.Now:yyyyMMdd_HHmmss}.wav");
                var rawPath = Path.Combine(
                    Environment.GetFolderPath(Environment.SpecialFolder.DesktopDirectory),
                    $"piano_debug_raw_{DateTime.Now:yyyyMMdd_HHmmss}.wav");
                windowsSource.EnableDebugRecording(processedPath, rawPath, TimeSpan.FromSeconds(10));
            }

            _audioSource.Start();
            IsRunning = true;
            StatusText = "Listening";
            WaveformVisibility = Visibility.Visible;
            _uiTimer.Start();
        }
        catch (Exception ex)
        {
            IsRunning = false;
            StatusText = "Error";
            ErrorText = $"Microphone unavailable: {ex.Message}";
        }
    }

    private void StopSession()
    {
        try
        {
            _audioSource.Stop();
        }
        catch (Exception ex)
        {
            ErrorText = $"Failed to stop capture cleanly: {ex.Message}";
        }

        _uiTimer.Stop();
        IsRunning = false;
        WaveformVisibility = Visibility.Collapsed;
        WaveformRms = 0;
        UpdateDebugFromSource();

        ActivitySummary? summary = null;
        try
        {
            summary = _accumulator.StopSession(DateTime.UtcNow);
            _sessionStore.Save(summary);
            ReloadHistory();
        }
        catch (Exception ex)
        {
            ErrorText = $"Failed to complete session: {ex.Message}";
        }

        StatusText = "Stopped";

        if (summary is not null)
        {
            SummaryText =
                $"Total Piano Time: {FormatDuration(summary.TotalPianoTime)}\n" +
                $"Segments: {summary.Segments.Count}";

            LiveTimeText = FormatDuration(summary.TotalPianoTime);
        }
    }

    private void OnFrameReady(AudioFrame frame)
    {
        var waveformPoints = BuildWaveformPoints(frame.Samples, out var hasSound, out var rms);

        try
        {
            var result = _detector.Process(frame);
            _accumulator.Process(result);
        }
        catch (Exception ex)
        {
            RunOnUi(() =>
            {
                ErrorText = $"Frame processing error: {ex.Message}";
                StatusText = "Error";
            });
            return;
        }

        RunOnUi(() =>
        {
            LiveTimeText = FormatDuration(_accumulator.CurrentTotalPianoTime);
            UpdateStatusFromState();
            WaveformPoints = new PointCollection(waveformPoints);
            WaveformVisibility = Visibility.Visible;
            WaveformOpacity = hasSound ? 1.0 : 0.35;
            WaveformRms = rms;
            FrameCount++;
            _lastFrameUtc = DateTime.UtcNow;
            OnPropertyChanged(nameof(LastFrameText));
        });
    }

    private void UpdateStatusFromState()
    {
        if (!IsRunning)
        {
            return;
        }

        StatusText = _accumulator.IsPianoDetected ? "Piano Detected" : "Listening";
    }

    private void UpdateDebugFromSource()
    {
        if (_audioSource is not WindowsAudioFrameSource windowsSource)
        {
            return;
        }

        DebugInfoText = windowsSource.CaptureInfo ?? "Debug: capture not started";
        DebugRecordingText = windowsSource.DebugRecordingStatus ?? "Debug recording: off";
        DebugRawText = windowsSource.RawFormatInfo ?? "Raw: unknown";
        DebugRawStatsText = $"Raw bytes: {windowsSource.LastRawBytes} | RMS {windowsSource.LastRawRms:0.0000} | Peak {windowsSource.LastRawPeak:0.0000}";
        DebugCallbackText = $"Callbacks: {windowsSource.DataCallbackCount}";
    }

    private void ReloadHistory()
    {
        History.Clear();

        foreach (var summary in _sessionStore.LoadAll())
        {
            History.Add(new SessionHistoryItem
            {
                SessionDateText = summary.SessionStart.ToLocalTime().ToString("yyyy-MM-dd HH:mm:ss"),
                TotalTimeText = FormatDuration(summary.TotalPianoTime),
                SegmentsCount = summary.Segments.Count
            });
        }
    }

    private static string FormatDuration(TimeSpan duration) =>
        duration < TimeSpan.Zero
            ? "00:00:00"
            : $"{(int)duration.TotalHours:00}:{duration.Minutes:00}:{duration.Seconds:00}";

    private static void RunOnUi(Action action)
    {
        var dispatcher = Application.Current?.Dispatcher;
        if (dispatcher is null || dispatcher.CheckAccess())
        {
            action();
            return;
        }

        dispatcher.Invoke(action);
    }

    private static Point[] BuildWaveformPoints(float[] samples, out bool hasSound, out double rms)
    {
        if (samples.Length == 0)
        {
            hasSound = false;
            rms = 0;
            return BuildFlatWaveform();
        }

        rms = 0.0;
        for (var i = 0; i < samples.Length; i++)
        {
            var sample = samples[i];
            rms += sample * sample;
        }

        rms = Math.Sqrt(rms / samples.Length);
        hasSound = rms >= SoundRmsThreshold;

        var step = Math.Max(1, samples.Length / WaveformPointCount);
        var xStep = (double)WaveformWidth / (WaveformPointCount - 1);

        var points = new Point[WaveformPointCount];
        for (var i = 0; i < WaveformPointCount; i++)
        {
            var index = Math.Min(samples.Length - 1, i * step);
            var sample = Math.Clamp(samples[index], -1f, 1f);
            var y = (1.0 - ((sample + 1.0) / 2.0)) * WaveformHeight;
            points[i] = new Point(i * xStep, y);
        }

        return points;
    }

    private static Point[] BuildFlatWaveform()
    {
        var mid = WaveformHeight / 2.0;
        return new[]
        {
            new Point(0, mid),
            new Point(WaveformWidth, mid)
        };
    }

    private static IPianoDetector CreateDefaultDetector(out string? warning)
    {
        warning = null;
        var modelDirCandidates = new[]
        {
            Path.Combine(AppContext.BaseDirectory, "assets", "models", "ast"),
            AppContext.BaseDirectory
        };

        foreach (var modelDir in modelDirCandidates)
        {
            try
            {
                if (!Directory.Exists(modelDir))
                {
                    continue;
                }

                return new AstOnnxPianoDetector(modelDir, new AstOnnxOptions
                {
                    Threshold = 0.18f,
                    TargetLabels = new[] { "Piano", "Electric piano", "Keyboard (musical)" }
                });
            }
            catch
            {
                // Keep trying candidate locations.
            }
        }

        warning = "AST model pack not found or failed to load. Run scripts/download-ast-model.ps1 and rebuild.";
        return new RuleBasedPianoDetector();
    }

    private void OnPropertyChanged([CallerMemberName] string? propertyName = null)
    {
        PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
    }
}
