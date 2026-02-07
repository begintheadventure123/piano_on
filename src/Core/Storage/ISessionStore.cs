using PianoActivityTracker.Core.Accumulation;

namespace PianoActivityTracker.Core.Storage;

public interface ISessionStore
{
    void Save(ActivitySummary summary);

    IReadOnlyList<ActivitySummary> LoadAll();
}
