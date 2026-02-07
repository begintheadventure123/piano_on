namespace PianoActivityTracker.Core.Audio;

public interface IAudioFrameSource
{
    event Action<AudioFrame>? OnFrameReady;

    void Start();

    void Stop();
}
