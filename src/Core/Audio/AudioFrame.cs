namespace PianoActivityTracker.Core.Audio;

public readonly struct AudioFrame
{
    public AudioFrame(float[] samples, int sampleRate, DateTime startTime)
    {
        Samples = samples ?? throw new ArgumentNullException(nameof(samples));
        SampleRate = sampleRate;
        StartTime = startTime;
    }

    public float[] Samples { get; }

    public int SampleRate { get; }

    public DateTime StartTime { get; }
}
