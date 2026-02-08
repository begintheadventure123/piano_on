# Piano Activity Tracker (Windows MVP)

Privacy-first local Windows MVP that listens to the microphone, detects piano activity, and saves per-session JSON summaries.

## Stack
- .NET 8
- WPF (`net8.0-windows`)
- NAudio (microphone capture)
- Core logic in platform-agnostic `src/Core`
- ONNX Runtime (AST AudioSet piano detector)

## Solution Layout
- `src/Core`: domain models, interfaces, detection, accumulation, storage
- `src/Platform.Windows`: Windows audio capture + WPF UI
- `tests/Core.Tests`: deterministic unit tests for `ActivityAccumulator`
- `docs/PianoActivityTracker.Spec.v2.md`: copied spec

## Build / Run (Windows)
```bash
dotnet restore

dotnet build PianoActivityTracker.sln

dotnet test tests/Core.Tests/Core.Tests.csproj

dotnet run --project src/Platform.Windows/Platform.Windows.csproj
```

## Model Setup (Required for High-Accuracy Piano Detection)
Download the AST AudioSet ONNX model pack (model + preprocessor config + labels):

```powershell
.\scripts\download-ast-model.ps1
```

This downloads `model.onnx` by default (recommended). Use `.\scripts\download-ast-model.ps1 -IncludeInt8` only if you explicitly want the quantized model too.

The app loads the model from `assets/models/ast` at runtime. If the model is missing or fails to load it will fall back to the rule-based detector and show an error message.

## Behavior
- Audio contract emitted into Core: mono, 16kHz, float32, 1.0s frames, 0.25s hop
- Debounce:
  - Enter piano state after `>= 3` consecutive positive detections
  - Exit piano state after `>= 4` consecutive negative detections
- Uses timestamps (never frame counts) for accumulation
- Default persistence: one JSON file per session under LocalAppData (`PianoActivityTracker/sessions`)
- Raw audio is not stored

## Assumptions
- System microphone provides PCM16 input through NAudio; app converts to mono float and resamples to 16kHz.
- High-accuracy detector uses the AST AudioSet ONNX model; rule-based detector remains as fallback.
- If microphone startup fails, UI shows an error and does not crash.

## Notes
- This environment could not reach NuGet (`https://api.nuget.org/v3/index.json`), so build/test execution here was blocked. Commands above should work on a normal Windows dev machine with NuGet access.
