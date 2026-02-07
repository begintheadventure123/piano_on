namespace PianoActivityTracker.Core.Accumulation;

public sealed class ActivitySegment
{
    public DateTime Start { get; init; }

    public DateTime End { get; init; }

    public TimeSpan Duration => End > Start ? End - Start : TimeSpan.Zero;
}
