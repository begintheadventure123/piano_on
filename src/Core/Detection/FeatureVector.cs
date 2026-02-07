namespace PianoActivityTracker.Core.Detection;

public readonly record struct FeatureVector(
    float RmsEnergy,
    float ZeroCrossingRate,
    float SpectralCentroid,
    float SpectralBandwidth,
    float SpectralRolloff85,
    float Mfcc1,
    float Mfcc2,
    float Mfcc3,
    float Mfcc4,
    float Mfcc5);
