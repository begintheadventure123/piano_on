using PianoActivityTracker.Core.Audio;

namespace PianoActivityTracker.Core.Detection;

public sealed class RuleBasedPianoDetector : IPianoDetector
{
    private readonly FeatureExtractor _featureExtractor;
    private readonly float _threshold;
    private double _noiseFloorRms = 0.01;
    private int _noiseCalibrationFrames;

    public RuleBasedPianoDetector(FeatureExtractor? featureExtractor = null, float threshold = 0.92f)
    {
        _featureExtractor = featureExtractor ?? new FeatureExtractor();
        _threshold = threshold;
    }

    public DetectionResult Process(AudioFrame frame)
    {
        var feature = _featureExtractor.Extract(frame);

        UpdateNoiseFloor(feature);

        if (IsClearlyNonPiano(feature))
        {
            return new DetectionResult(false, 0.05f, frame.StartTime);
        }

        if (IsSpeechLike(feature))
        {
            return new DetectionResult(false, 0.10f, frame.StartTime);
        }

        var score =
            0.20f * ScoreRange(feature.RmsEnergy, 0.01f, 0.30f) +
            0.10f * ScoreRange(feature.ZeroCrossingRate, 0.01f, 0.22f) +
            0.20f * ScoreRange(feature.SpectralCentroid, 300f, 4200f) +
            0.15f * ScoreRange(feature.SpectralBandwidth, 200f, 3500f) +
            0.15f * ScoreRange(feature.SpectralRolloff85, 700f, 7000f) +
            0.10f * ScoreMfcc(feature) +
            0.10f * ScoreBandBalance(feature) +
            0.10f * ScoreFlatness(feature);

        score -= SpeechGuardPenalty(feature);
        var confidence = Math.Clamp(score, 0f, 1f);
        return new DetectionResult(confidence >= _threshold, confidence, frame.StartTime);
    }

    private void UpdateNoiseFloor(FeatureVector feature)
    {
        if (_noiseCalibrationFrames >= 40)
        {
            return;
        }

        _noiseFloorRms = (_noiseFloorRms * _noiseCalibrationFrames + feature.RmsEnergy) / (_noiseCalibrationFrames + 1);
        _noiseCalibrationFrames++;
    }

    private static bool IsClearlyNonPiano(FeatureVector feature)
    {
        // Very quiet or low-frequency-heavy frames are almost never piano.
        if (feature.RmsEnergy < 0.03f)
        {
            return true;
        }

        if (feature.SpectralCentroid < 1100f && feature.SpectralRolloff85 < 2600f)
        {
            return true;
        }

        if (feature.HighBandEnergy < 0.25f)
        {
            return true;
        }

        return false;
    }

    private bool IsSpeechLike(FeatureVector feature)
    {
        var rmsGate = Math.Max(0.04f, (float)(_noiseFloorRms * 4.0));
        if (feature.RmsEnergy < rmsGate)
        {
            return true;
        }

        if (feature.SpectralCentroid < 1400f && feature.SpectralRolloff85 < 3200f && feature.HighBandEnergy < 0.45f)
        {
            return true;
        }

        if (feature.SpectralFlatness < 0.2f && feature.HighBandEnergy < 0.4f)
        {
            return true;
        }

        return false;
    }

    private static float ScoreMfcc(FeatureVector feature)
    {
        var mean = (feature.Mfcc1 + feature.Mfcc2 + feature.Mfcc3 + feature.Mfcc4 + feature.Mfcc5) / 5f;
        return ScoreRange(mean, 0.4f, 2.8f);
    }

    private static float ScoreBandBalance(FeatureVector feature)
    {
        var total = feature.LowBandEnergy + feature.MidBandEnergy + feature.HighBandEnergy + 1e-6f;
        var highRatio = feature.HighBandEnergy / total;
        var midRatio = feature.MidBandEnergy / total;

        var score = 0f;
        if (highRatio > 0.18f && midRatio > 0.35f)
        {
            score += 0.6f;
        }

        if (highRatio > 0.25f)
        {
            score += 0.4f;
        }

        return Math.Clamp(score, 0f, 1f);
    }

    private static float ScoreFlatness(FeatureVector feature)
    {
        // Piano tends to be moderately flat (not too tonal, not pure noise).
        return ScoreRange(feature.SpectralFlatness, 0.15f, 0.65f);
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

    private static float SpeechGuardPenalty(FeatureVector feature)
    {
        // Penalize low-centroid + low-rolloff frames (often speech or background).
        if (feature.SpectralCentroid < 1200f && feature.SpectralRolloff85 < 3000f)
        {
            return 0.15f;
        }

        if (feature.SpectralBandwidth < 500f)
        {
            return 0.10f;
        }

        return 0f;
    }
}
