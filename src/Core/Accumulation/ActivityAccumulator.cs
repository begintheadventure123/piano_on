using PianoActivityTracker.Core.Detection;

namespace PianoActivityTracker.Core.Accumulation;

public sealed class ActivityAccumulator
{
    public const int EnterDebounceFrames = 3;
    public const int ExitDebounceFrames = 4;

    private readonly List<ActivitySegment> _segments = new();

    private DateTime _sessionStart;
    private DateTime _lastTimestamp;
    private DateTime? _activeSegmentStart;
    private DateTime? _firstPositiveTimestamp;
    private DateTime? _firstNegativeTimestamp;
    private int _positiveStreak;
    private int _negativeStreak;
    private TimeSpan _closedTotal = TimeSpan.Zero;

    public ActivityState State { get; private set; } = ActivityState.Idle;

    public bool IsPianoDetected => State == ActivityState.PianoPlaying;

    public IReadOnlyList<ActivitySegment> Segments => _segments;

    public TimeSpan CurrentTotalPianoTime
    {
        get
        {
            if (State != ActivityState.PianoPlaying || _activeSegmentStart is null)
            {
                return _closedTotal;
            }

            var openTime = _lastTimestamp > _activeSegmentStart.Value
                ? _lastTimestamp - _activeSegmentStart.Value
                : TimeSpan.Zero;

            return _closedTotal + openTime;
        }
    }

    public void StartSession(DateTime sessionStart)
    {
        _segments.Clear();
        _closedTotal = TimeSpan.Zero;
        _sessionStart = sessionStart;
        _lastTimestamp = sessionStart;
        _activeSegmentStart = null;
        _firstPositiveTimestamp = null;
        _firstNegativeTimestamp = null;
        _positiveStreak = 0;
        _negativeStreak = 0;
        State = ActivityState.Listening;
    }

    public void Process(DetectionResult result)
    {
        if (State == ActivityState.Idle)
        {
            StartSession(result.Timestamp);
        }

        if (result.Timestamp < _lastTimestamp)
        {
            throw new ArgumentException("Detection timestamps must be monotonic.", nameof(result));
        }

        _lastTimestamp = result.Timestamp;

        if (result.IsPiano)
        {
            if (_positiveStreak == 0)
            {
                _firstPositiveTimestamp = result.Timestamp;
            }

            _positiveStreak++;
            _negativeStreak = 0;
            _firstNegativeTimestamp = null;

            if (State == ActivityState.Listening && _positiveStreak >= EnterDebounceFrames)
            {
                _activeSegmentStart = _firstPositiveTimestamp ?? result.Timestamp;
                State = ActivityState.PianoPlaying;
            }

            return;
        }

        if (_negativeStreak == 0)
        {
            _firstNegativeTimestamp = result.Timestamp;
        }

        _negativeStreak++;
        _positiveStreak = 0;
        _firstPositiveTimestamp = null;

        if (State == ActivityState.PianoPlaying && _negativeStreak >= ExitDebounceFrames)
        {
            var segmentEnd = _firstNegativeTimestamp ?? result.Timestamp;
            CloseSegment(segmentEnd);
            State = ActivityState.Listening;
        }
    }

    public ActivitySummary StopSession(DateTime sessionEnd)
    {
        if (State == ActivityState.Idle)
        {
            throw new InvalidOperationException("Cannot stop a session that never started.");
        }

        if (sessionEnd < _lastTimestamp)
        {
            throw new ArgumentException("Session end cannot be earlier than last detection timestamp.", nameof(sessionEnd));
        }

        _lastTimestamp = sessionEnd;

        if (State == ActivityState.PianoPlaying)
        {
            CloseSegment(sessionEnd);
            State = ActivityState.Listening;
        }

        var summary = new ActivitySummary
        {
            SessionStart = _sessionStart,
            SessionEnd = sessionEnd,
            TotalPianoTime = _closedTotal,
            Segments = _segments
                .Select(s => new ActivitySegment { Start = s.Start, End = s.End })
                .ToList()
        };

        State = ActivityState.Idle;
        return summary;
    }

    private void CloseSegment(DateTime segmentEnd)
    {
        if (_activeSegmentStart is null)
        {
            return;
        }

        var start = _activeSegmentStart.Value;
        _activeSegmentStart = null;
        _firstNegativeTimestamp = null;
        _negativeStreak = 0;

        if (segmentEnd <= start)
        {
            return;
        }

        if (_segments.Count > 0)
        {
            var last = _segments[^1];
            if (start <= last.End)
            {
                var merged = new ActivitySegment
                {
                    Start = last.Start,
                    End = segmentEnd > last.End ? segmentEnd : last.End
                };

                _closedTotal -= last.Duration;
                _closedTotal += merged.Duration;
                _segments[^1] = merged;
                return;
            }
        }

        var segment = new ActivitySegment
        {
            Start = start,
            End = segmentEnd
        };

        _segments.Add(segment);
        _closedTotal += segment.Duration;
    }
}
