namespace PianoActivityTracker.Platform.Windows.UI;

public sealed class SessionHistoryItem
{
    public required string SessionDateText { get; init; }

    public required string TotalTimeText { get; init; }

    public required int SegmentsCount { get; init; }
}
