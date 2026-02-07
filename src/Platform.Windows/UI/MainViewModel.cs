using System.Collections.ObjectModel;
using System.ComponentModel;
using System.Runtime.CompilerServices;
using System.Windows;
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

    public MainViewModel()
        : this(
            new WindowsAudioFrameSource(),
            new RuleBasedPianoDetector(),
            new ActivityAccumulator(),
            new JsonSessionStore())
    {
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

        var start = DateTime.UtcNow;
        _accumulator.StartSession(start);

        try
        {
            _audioSource.Start();
            IsRunning = true;
            StatusText = "Listening";
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

    private void OnPropertyChanged([CallerMemberName] string? propertyName = null)
    {
        PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
    }
}
