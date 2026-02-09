using System.Collections.ObjectModel;
using System.ComponentModel;
using System.IO;
using System.Runtime.CompilerServices;
using System.Windows;
using System.Windows.Media;
using System.Windows.Threading;
using Microsoft.Win32;
using NAudio.Wave;
using NAudio.Wave.SampleProviders;
using PianoActivityTracker.Core.Accumulation;
using PianoActivityTracker.Core.Audio;
using PianoActivityTracker.Core.Detection;
using PianoActivityTracker.Core.Storage;
using PianoActivityTracker.Platform.Windows.Audio;
using Forms = System.Windows.Forms;
using Point = System.Windows.Point;

namespace PianoActivityTracker.Platform.Windows.UI;

public sealed class MainViewModel : INotifyPropertyChanged, IDisposable
{
    private readonly IAudioFrameSource _audioSource;
    private readonly IPianoDetector _detector;
    private readonly ActivityAccumulator _accumulator;
    private readonly ISessionStore _sessionStore;
    private readonly DispatcherTimer _uiTimer;

    private bool _isRunning;
    private bool _isAnalyzingFile;
    private string _statusText = "Ready";
    private string _liveTimeText = "00:00:00";
    private string _summaryText = string.Empty;
    private string _errorText = string.Empty;
    private string _uploadFolderPath = GetDefaultUploadFolder();
    private string _selectedAudioFilePath = string.Empty;
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
    private const int AnalysisSampleRate = 16_000;
    private const int AnalysisFrameSamples = 16_000;
    private const int AnalysisHopSamples = 4_000;

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

        StartCommand = new RelayCommand(StartSession, () => !IsRunning && !IsAnalyzingFile);
        StopCommand = new RelayCommand(StopSession, () => IsRunning && !IsAnalyzingFile);
        BrowseFolderCommand = new RelayCommand(BrowseUploadFolder, () => !IsRunning && !IsAnalyzingFile);
        ReloadFilesCommand = new RelayCommand(ReloadAudioFiles, () => !IsRunning && !IsAnalyzingFile);
        BrowseFileCommand = new RelayCommand(BrowseAudioFile, () => !IsRunning && !IsAnalyzingFile);
        AnalyzeFileCommand = new RelayCommand(AnalyzeSelectedFile, () => !IsRunning && !IsAnalyzingFile && !string.IsNullOrWhiteSpace(SelectedAudioFilePath));

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

        AvailableAudioFiles = new ObservableCollection<string>();
        History = new ObservableCollection<SessionHistoryItem>();
        ReloadAudioFiles();
        ReloadHistory();
    }

    public event PropertyChangedEventHandler? PropertyChanged;

    public RelayCommand StartCommand { get; }

    public RelayCommand StopCommand { get; }

    public RelayCommand BrowseFolderCommand { get; }

    public RelayCommand ReloadFilesCommand { get; }

    public RelayCommand BrowseFileCommand { get; }

    public RelayCommand AnalyzeFileCommand { get; }

    public ObservableCollection<string> AvailableAudioFiles { get; }

    public ObservableCollection<SessionHistoryItem> History { get; }

    public bool IsAnalyzingFile
    {
        get => _isAnalyzingFile;
        private set
        {
            if (_isAnalyzingFile == value)
            {
                return;
            }

            _isAnalyzingFile = value;
            OnPropertyChanged();
            RaiseActionCommandsChanged();
        }
    }

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
            RaiseActionCommandsChanged();
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

    public string UploadFolderPath
    {
        get => _uploadFolderPath;
        set
        {
            if (_uploadFolderPath == value)
            {
                return;
            }

            _uploadFolderPath = value;
            OnPropertyChanged();
        }
    }

    public string SelectedAudioFilePath
    {
        get => _selectedAudioFilePath;
        set
        {
            if (_selectedAudioFilePath == value)
            {
                return;
            }

            _selectedAudioFilePath = value;
            OnPropertyChanged();
            AnalyzeFileCommand.RaiseCanExecuteChanged();
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

    private void RaiseActionCommandsChanged()
    {
        StartCommand.RaiseCanExecuteChanged();
        StopCommand.RaiseCanExecuteChanged();
        BrowseFolderCommand.RaiseCanExecuteChanged();
        ReloadFilesCommand.RaiseCanExecuteChanged();
        BrowseFileCommand.RaiseCanExecuteChanged();
        AnalyzeFileCommand.RaiseCanExecuteChanged();
    }

    private void BrowseUploadFolder()
    {
        using var dialog = new Forms.FolderBrowserDialog
        {
            Description = "Select folder that contains target audio files",
            InitialDirectory = Directory.Exists(UploadFolderPath) ? UploadFolderPath : AppContext.BaseDirectory,
            UseDescriptionForTitle = true,
            AutoUpgradeEnabled = true
        };

        if (dialog.ShowDialog() != Forms.DialogResult.OK || string.IsNullOrWhiteSpace(dialog.SelectedPath))
        {
            return;
        }

        UploadFolderPath = dialog.SelectedPath;
        ReloadAudioFiles();
    }

    private void ReloadAudioFiles()
    {
        AvailableAudioFiles.Clear();

        if (!Directory.Exists(UploadFolderPath))
        {
            return;
        }

        var extensions = new HashSet<string>(StringComparer.OrdinalIgnoreCase)
        {
            ".wav", ".m4a", ".mp3", ".wma", ".aac", ".flac"
        };

        var files = Directory.GetFiles(UploadFolderPath, "*.*", SearchOption.AllDirectories)
            .Where(path => extensions.Contains(Path.GetExtension(path)))
            .OrderBy(path => path, StringComparer.OrdinalIgnoreCase);

        foreach (var file in files)
        {
            AvailableAudioFiles.Add(file);
        }

        if (AvailableAudioFiles.Count > 0 && string.IsNullOrWhiteSpace(SelectedAudioFilePath))
        {
            SelectedAudioFilePath = AvailableAudioFiles[0];
        }
        else if (!string.IsNullOrWhiteSpace(SelectedAudioFilePath) && !AvailableAudioFiles.Contains(SelectedAudioFilePath))
        {
            SelectedAudioFilePath = AvailableAudioFiles.FirstOrDefault() ?? string.Empty;
        }
    }

    private void BrowseAudioFile()
    {
        var dialog = new Microsoft.Win32.OpenFileDialog
        {
            Title = "Select target audio file",
            Filter = "Audio Files (*.m4a;*.wav;*.mp3;*.wma;*.aac;*.flac)|*.m4a;*.wav;*.mp3;*.wma;*.aac;*.flac|All Files (*.*)|*.*",
            CheckFileExists = true,
            CheckPathExists = true,
            Multiselect = false,
            InitialDirectory = Directory.Exists(UploadFolderPath) ? UploadFolderPath : GetDefaultUploadFolder()
        };

        if (dialog.ShowDialog() != true || string.IsNullOrWhiteSpace(dialog.FileName))
        {
            return;
        }

        SelectedAudioFilePath = dialog.FileName;
        var selectedDir = Path.GetDirectoryName(dialog.FileName);
        if (!string.IsNullOrWhiteSpace(selectedDir))
        {
            UploadFolderPath = selectedDir;
            ReloadAudioFiles();
            SelectedAudioFilePath = dialog.FileName;
        }
    }

    private async void AnalyzeSelectedFile()
    {
        if (string.IsNullOrWhiteSpace(SelectedAudioFilePath) || !File.Exists(SelectedAudioFilePath))
        {
            ErrorText = "Please select an existing audio file first.";
            return;
        }

        ErrorText = string.Empty;
        SummaryText = string.Empty;
        StatusText = "Analyzing file...";
        IsAnalyzingFile = true;

        try
        {
            var analysis = await Task.Run(() => AnalyzeFileInternal(SelectedAudioFilePath));

            LiveTimeText = FormatDuration(analysis.Summary.TotalPianoTime);
            SummaryText =
                $"File: {Path.GetFileName(SelectedAudioFilePath)}\n" +
                $"Total Piano Time: {FormatDuration(analysis.Summary.TotalPianoTime)}\n" +
                $"Segments: {analysis.Summary.Segments.Count}\n" +
                $"Processed Frames: {analysis.ProcessedFrames}";
            StatusText = "Analysis complete";
            FrameCount = analysis.ProcessedFrames;
            _lastFrameUtc = DateTime.UtcNow;
            OnPropertyChanged(nameof(LastFrameText));
            _sessionStore.Save(analysis.Summary);
            ReloadHistory();

            if (!string.IsNullOrWhiteSpace(analysis.Warning))
            {
                ErrorText = analysis.Warning;
            }
        }
        catch (Exception ex)
        {
            StatusText = "Error";
            ErrorText = $"File analysis failed: {ex.Message}";
        }
        finally
        {
            IsAnalyzingFile = false;
        }
    }

    private static FileAnalysisResult AnalyzeFileInternal(string path)
    {
        var detector = CreateDefaultDetector(out var warning);
        try
        {
            var samples = LoadAsMono16k(path);
            var accumulator = new ActivityAccumulator();
            var sessionStart = DateTime.UtcNow;
            accumulator.StartSession(sessionStart);

            var processedFrames = 0;
            for (var offset = 0; offset + AnalysisFrameSamples <= samples.Length; offset += AnalysisHopSamples)
            {
                var frameSamples = new float[AnalysisFrameSamples];
                Array.Copy(samples, offset, frameSamples, 0, AnalysisFrameSamples);
                var timestamp = sessionStart + TimeSpan.FromSeconds((double)(processedFrames * AnalysisHopSamples) / AnalysisSampleRate);
                var frame = new AudioFrame(frameSamples, AnalysisSampleRate, timestamp);
                var detection = detector.Process(frame);
                accumulator.Process(detection);
                processedFrames++;
            }

            var sessionEnd = sessionStart + TimeSpan.FromSeconds((double)samples.Length / AnalysisSampleRate);
            var summary = accumulator.StopSession(sessionEnd);
            return new FileAnalysisResult(summary, processedFrames, warning);
        }
        finally
        {
            if (detector is IDisposable disposable)
            {
                disposable.Dispose();
            }
        }
    }

    private static float[] LoadAsMono16k(string path)
    {
        using var reader = new MediaFoundationReader(path);
        ISampleProvider sampleProvider = reader.ToSampleProvider();

        if (sampleProvider.WaveFormat.Channels > 1)
        {
            if (sampleProvider.WaveFormat.Channels != 2)
            {
                throw new NotSupportedException("Only mono/stereo input audio is supported for offline analysis.");
            }

            sampleProvider = new StereoToMonoSampleProvider(sampleProvider)
            {
                LeftVolume = 0.5f,
                RightVolume = 0.5f
            };
        }

        if (sampleProvider.WaveFormat.SampleRate != AnalysisSampleRate)
        {
            sampleProvider = new WdlResamplingSampleProvider(sampleProvider, AnalysisSampleRate);
        }

        var samples = new List<float>(AnalysisSampleRate * 60);
        var buffer = new float[AnalysisSampleRate];
        while (true)
        {
            var read = sampleProvider.Read(buffer, 0, buffer.Length);
            if (read <= 0)
            {
                break;
            }

            for (var i = 0; i < read; i++)
            {
                samples.Add(buffer[i]);
            }
        }

        return samples.ToArray();
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
        var dispatcher = System.Windows.Application.Current?.Dispatcher;
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

    private static string GetDefaultUploadFolder()
    {
        var current = AppContext.BaseDirectory;
        while (!string.IsNullOrWhiteSpace(current))
        {
            if (File.Exists(Path.Combine(current, "PianoActivityTracker.sln")) ||
                Directory.Exists(Path.Combine(current, ".git")))
            {
                return current;
            }

            current = Path.GetDirectoryName(current) ?? string.Empty;
        }

        return AppContext.BaseDirectory;
    }

    private readonly record struct FileAnalysisResult(ActivitySummary Summary, int ProcessedFrames, string? Warning);

    private void OnPropertyChanged([CallerMemberName] string? propertyName = null)
    {
        PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
    }
}
