using PianoActivityTracker.Core.Audio;

namespace PianoActivityTracker.Core.Detection;

public interface IPianoDetector
{
    DetectionResult Process(AudioFrame frame);
}
