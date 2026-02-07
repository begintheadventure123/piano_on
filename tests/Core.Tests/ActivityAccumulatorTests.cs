using PianoActivityTracker.Core.Accumulation;
using PianoActivityTracker.Core.Detection;

namespace Core.Tests;

public sealed class ActivityAccumulatorTests
{
    [Fact]
    public void EnterAndExitDebounce_AreAppliedCorrectly()
    {
        var t0 = new DateTime(2026, 2, 1, 10, 0, 0, DateTimeKind.Utc);
        var accumulator = new ActivityAccumulator();
        accumulator.StartSession(t0);

        Feed(accumulator, true, t0);
        Feed(accumulator, true, t0.AddMilliseconds(250));
        Feed(accumulator, true, t0.AddMilliseconds(500));

        Assert.Equal(ActivityState.PianoPlaying, accumulator.State);

        Feed(accumulator, false, t0.AddMilliseconds(750));
        Feed(accumulator, false, t0.AddMilliseconds(1000));
        Feed(accumulator, false, t0.AddMilliseconds(1250));

        Assert.Equal(ActivityState.PianoPlaying, accumulator.State);

        Feed(accumulator, false, t0.AddMilliseconds(1500));

        Assert.Equal(ActivityState.Listening, accumulator.State);
        Assert.Single(accumulator.Segments);
        Assert.Equal(TimeSpan.FromMilliseconds(750), accumulator.Segments[0].Duration);
    }

    [Fact]
    public void NegativeGapShorterThanExitDebounce_RemainsSingleSegment()
    {
        var t0 = new DateTime(2026, 2, 1, 10, 0, 0, DateTimeKind.Utc);
        var accumulator = new ActivityAccumulator();
        accumulator.StartSession(t0);

        Feed(accumulator, true, t0);
        Feed(accumulator, true, t0.AddMilliseconds(250));
        Feed(accumulator, true, t0.AddMilliseconds(500));
        Feed(accumulator, true, t0.AddMilliseconds(750));

        Feed(accumulator, false, t0.AddMilliseconds(1000));
        Feed(accumulator, false, t0.AddMilliseconds(1250));
        Feed(accumulator, false, t0.AddMilliseconds(1500));

        Feed(accumulator, true, t0.AddMilliseconds(1750));
        Feed(accumulator, true, t0.AddMilliseconds(2000));
        Feed(accumulator, true, t0.AddMilliseconds(2250));

        var summary = accumulator.StopSession(t0.AddMilliseconds(2500));

        Assert.Single(summary.Segments);
        Assert.Equal(TimeSpan.FromMilliseconds(2500), summary.TotalPianoTime);
    }

    [Fact]
    public void MultipleSegments_TotalTimeMatchesTimestamps()
    {
        var t0 = new DateTime(2026, 2, 1, 10, 0, 0, DateTimeKind.Utc);
        var accumulator = new ActivityAccumulator();
        accumulator.StartSession(t0);

        Feed(accumulator, true, t0);
        Feed(accumulator, true, t0.AddMilliseconds(250));
        Feed(accumulator, true, t0.AddMilliseconds(500));
        Feed(accumulator, false, t0.AddMilliseconds(750));
        Feed(accumulator, false, t0.AddMilliseconds(1000));
        Feed(accumulator, false, t0.AddMilliseconds(1250));
        Feed(accumulator, false, t0.AddMilliseconds(1500));

        Feed(accumulator, false, t0.AddMilliseconds(1750));
        Feed(accumulator, true, t0.AddMilliseconds(2000));
        Feed(accumulator, true, t0.AddMilliseconds(2250));
        Feed(accumulator, true, t0.AddMilliseconds(2500));
        Feed(accumulator, false, t0.AddMilliseconds(2750));
        Feed(accumulator, false, t0.AddMilliseconds(3000));
        Feed(accumulator, false, t0.AddMilliseconds(3250));
        Feed(accumulator, false, t0.AddMilliseconds(3500));

        var summary = accumulator.StopSession(t0.AddMilliseconds(4000));

        Assert.Equal(2, summary.Segments.Count);
        Assert.Equal(TimeSpan.FromMilliseconds(1500), summary.TotalPianoTime);
    }

    [Fact]
    public void NonMonotonicTimestamps_Throw()
    {
        var t0 = new DateTime(2026, 2, 1, 10, 0, 0, DateTimeKind.Utc);
        var accumulator = new ActivityAccumulator();
        accumulator.StartSession(t0);

        Feed(accumulator, true, t0.AddMilliseconds(500));

        Assert.Throws<ArgumentException>(() => Feed(accumulator, false, t0.AddMilliseconds(250)));
    }

    private static void Feed(ActivityAccumulator accumulator, bool isPiano, DateTime timestamp)
    {
        accumulator.Process(new DetectionResult(isPiano, isPiano ? 0.95f : 0.05f, timestamp));
    }
}
