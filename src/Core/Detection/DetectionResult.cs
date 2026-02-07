namespace PianoActivityTracker.Core.Detection;

public readonly struct DetectionResult
{
    public DetectionResult(bool isPiano, float confidence, DateTime timestamp)
    {
        IsPiano = isPiano;
        Confidence = confidence;
        Timestamp = timestamp;
    }

    public bool IsPiano { get; }

    public float Confidence { get; }

    public DateTime Timestamp { get; }
}
