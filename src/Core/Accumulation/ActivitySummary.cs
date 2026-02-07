namespace PianoActivityTracker.Core.Accumulation;

public sealed class ActivitySummary
{
    public DateTime SessionStart { get; init; }

    public DateTime SessionEnd { get; init; }

    public TimeSpan TotalPianoTime { get; init; }

    public List<ActivitySegment> Segments { get; init; } = new();
}
