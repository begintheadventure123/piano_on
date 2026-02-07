# Piano Activity Tracker — End-to-End Technical Specification (v2)

## 1. Project Overview

### 1.1 Goal
Create a local, privacy-first application that listens to microphone audio, determines whether **piano playing sound** is present, and accurately accumulates the **total duration of piano activity** during a user session.

This system is **not** intended to:
- Identify songs or melodies
- Perform pitch or note recognition
- Evaluate performance quality

### 1.2 Core Requirements
- Offline operation
- Deterministic timing accuracy
- Cross-platform core (Windows first, iOS later)
- No raw audio storage by default
- Robust against silence, short pauses, and background noise

---

## 2. System Architecture

```
┌─────────────────────────────┐
│        UI Host Layer        │
│  - Windows (WPF / WinUI)    │
│  - iOS (SwiftUI later)      │
└────────────▲────────────────┘
             │
┌────────────┴────────────────┐
│     Platform Audio Layer    │
│  - Windows: NAudio          │
│  - iOS: AVAudioEngine       │
└────────────▲────────────────┘
             │
┌────────────┴────────────────┐
│        Core Domain           │
│  (Pure logic, portable)     │
└─────────────────────────────┘
```

---

## 3. Core Domain Principles

- No platform-specific APIs
- No UI dependencies
- Time-based logic uses timestamps, never frame counts
- Replaceable detection engine (rule-based → ML)

---

## 4. Audio Normalization Contract

All platform layers must output audio in the following canonical format:

| Property      | Value            |
|---------------|------------------|
| Channels      | Mono             |
| Sample Rate   | 16000 Hz         |
| Encoding      | PCM float32      |
| Frame Size    | 1.0 second       |
| Hop Size      | 0.25 seconds     |

---

## 5. Core Data Models

### 5.1 AudioFrame

```csharp
struct AudioFrame
{
    float[] Samples;        // PCM mono samples
    int SampleRate;         // Must be 16000
    DateTime StartTime;     // Frame start timestamp
}
```

### 5.2 DetectionResult

```csharp
struct DetectionResult
{
    bool IsPiano;
    float Confidence;       // 0.0 – 1.0
    DateTime Timestamp;
}
```

---

## 6. Core Interfaces

### 6.1 IAudioFrameSource

```csharp
interface IAudioFrameSource
{
    event Action<AudioFrame> OnFrameReady;
    void Start();
    void Stop();
}
```

Responsibilities:
- Continuous audio capture
- Frame segmentation
- Format normalization

---

### 6.2 IPianoDetector

```csharp
interface IPianoDetector
{
    DetectionResult Process(AudioFrame frame);
}
```

---

## 7. Piano Detection — MVP Strategy

### 7.1 Feature Extraction
For each frame:
- RMS Energy
- Zero Crossing Rate
- Spectral Centroid
- Spectral Bandwidth
- Spectral Roll-off (85%)
- MFCC (coefficients 1–5)

### 7.2 Classification Logic
- Normalize features
- Weighted score aggregation
- Apply confidence threshold (default: 0.6)

> No ML dependency in MVP

---

## 8. Activity Accumulation Engine

### 8.1 States

```
Idle → Listening → PianoPlaying → Listening
```

### 8.2 Debounce Rules

| Transition | Condition |
|----------|-----------|
| Enter PianoPlaying | ≥3 consecutive piano frames |
| Exit PianoPlaying  | ≥4 consecutive non-piano frames |

### 8.3 Timing Rules
- Accumulate duration only in `PianoPlaying`
- Merge gaps shorter than exit debounce window
- Use frame timestamps for accuracy

---

## 9. Activity Models

### 9.1 ActivitySegment

```csharp
class ActivitySegment
{
    DateTime Start;
    DateTime End;
    TimeSpan Duration;
}
```

### 9.2 ActivitySummary

```csharp
class ActivitySummary
{
    DateTime SessionStart;
    DateTime SessionEnd;
    TimeSpan TotalPianoTime;
    List<ActivitySegment> Segments;
}
```

---

## 10. Persistence Layer

### 10.1 Interface

```csharp
interface ISessionStore
{
    void Save(ActivitySummary summary);
    List<ActivitySummary> LoadAll();
}
```

### 10.2 Default Implementation
- JSON files
- One file per session
- Stored locally only

---

## 11. Windows Platform Implementation

### 11.1 Technology Stack
- .NET 8
- WPF or WinUI 3
- NAudio

### 11.2 WindowsAudioFrameSource
Responsibilities:
- Microphone access
- Resampling
- Mono downmix
- Frame buffering

---

## 12. Windows UI (MVP)

### 12.1 Functional Requirements
- Start / Stop session
- Live status indicator (Listening / Piano Detected)
- Real-time accumulated time
- Session summary view

### 12.2 Explicit Non-Goals
- No waveform display
- No audio playback
- No background recording

---

## 13. Testing Strategy

### 13.1 Unit Tests
- Debounce correctness
- Segment merging
- Time accuracy

### 13.2 Manual Tests
- Piano only
- Piano + speech
- Silence
- TV background music

---

## 14. Acceptance Criteria

- Timing error ≤ ±5% over 30 minutes
- Zero accumulation during silence
- Stable for sessions ≥2 hours

---

## 15. iOS Migration Plan

### Reused
- Core Domain
- Detection Logic
- Accumulation Logic
- Persistence Models

### Replaced
- Audio capture layer
- UI layer

---

## 16. Repository Layout

```
/src
  /Core
    Audio
    Detection
    Accumulation
    Storage
  /Platform.Windows
    Audio
    UI
/tests
/docs
```

---

## END OF SPEC
