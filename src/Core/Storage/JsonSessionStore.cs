using System.Text.Json;
using PianoActivityTracker.Core.Accumulation;

namespace PianoActivityTracker.Core.Storage;

public sealed class JsonSessionStore : ISessionStore
{
    private static readonly JsonSerializerOptions SerializerOptions = new()
    {
        WriteIndented = true,
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase
    };

    private readonly string _directoryPath;

    public JsonSessionStore(string? directoryPath = null)
    {
        _directoryPath = directoryPath
            ?? Path.Combine(
                Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
                "PianoActivityTracker",
                "sessions");

        Directory.CreateDirectory(_directoryPath);
    }

    public void Save(ActivitySummary summary)
    {
        ArgumentNullException.ThrowIfNull(summary);

        var safeTimestamp = summary.SessionStart.ToString("yyyyMMdd_HHmmss_fff");
        var fileName = $"session_{safeTimestamp}_{Guid.NewGuid():N}.json";
        var filePath = Path.Combine(_directoryPath, fileName);

        var payload = JsonSerializer.Serialize(summary, SerializerOptions);
        File.WriteAllText(filePath, payload);
    }

    public IReadOnlyList<ActivitySummary> LoadAll()
    {
        if (!Directory.Exists(_directoryPath))
        {
            return Array.Empty<ActivitySummary>();
        }

        var summaries = new List<ActivitySummary>();

        foreach (var file in Directory.EnumerateFiles(_directoryPath, "*.json", SearchOption.TopDirectoryOnly))
        {
            try
            {
                var json = File.ReadAllText(file);
                var summary = JsonSerializer.Deserialize<ActivitySummary>(json, SerializerOptions);
                if (summary is not null)
                {
                    summaries.Add(summary);
                }
            }
            catch
            {
                // Ignore malformed files to keep local history loading resilient.
            }
        }

        return summaries
            .OrderByDescending(s => s.SessionStart)
            .ToList();
    }
}
