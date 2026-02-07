using PianoActivityTracker.Core.Audio;

namespace PianoActivityTracker.Core.Detection;

public sealed class RuleBasedPianoDetector : IPianoDetector
{
    private readonly FeatureExtractor _featureExtractor;
    private readonly float _threshold;

    public RuleBasedPianoDetector(FeatureExtractor? featureExtractor = null, float threshold = 0.60f)
    {
        _featureExtractor = featureExtractor ?? new FeatureExtractor();
        _threshold = threshold;
    }

    public DetectionResult Process(AudioFrame frame)
    {
        var feature = _featureExtractor.Extract(frame);

        var score =
            0.20f * ScoreRange(feature.RmsEnergy, 0.01f, 0.30f) +
            0.10f * ScoreRange(feature.ZeroCrossingRate, 0.01f, 0.22f) +
            0.20f * ScoreRange(feature.SpectralCentroid, 300f, 4200f) +
            0.15f * ScoreRange(feature.SpectralBandwidth, 200f, 3500f) +
            0.15f * ScoreRange(feature.SpectralRolloff85, 700f, 7000f) +
            0.20f * ScoreMfcc(feature);

        var confidence = Math.Clamp(score, 0f, 1f);
        return new DetectionResult(confidence >= _threshold, confidence, frame.StartTime);
    }

    private static float ScoreMfcc(FeatureVector feature)
    {
        var mean = (feature.Mfcc1 + feature.Mfcc2 + feature.Mfcc3 + feature.Mfcc4 + feature.Mfcc5) / 5f;
        return ScoreRange(mean, 0.4f, 2.8f);
    }

    private static float ScoreRange(float value, float min, float max)
    {
        if (value >= min && value <= max)
        {
            return 1f;
        }

        if (value < min)
        {
            var delta = min - value;
            return Math.Clamp(1f - (delta / Math.Max(min, 1e-3f)), 0f, 1f);
        }

        var overshoot = value - max;
        return Math.Clamp(1f - (overshoot / Math.Max(max, 1e-3f)), 0f, 1f);
    }
}
