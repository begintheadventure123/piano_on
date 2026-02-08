Param(
    [switch]$IncludeInt8
)

$ErrorActionPreference = "Stop"

$root = Resolve-Path (Join-Path $PSScriptRoot "..")
$dest = Join-Path $root "assets\\models\\ast"

if (-not (Test-Path $dest)) {
    New-Item -ItemType Directory -Path $dest | Out-Null
}

$files = @(
    @{
        Url = "https://huggingface.co/onnx-community/ast-finetuned-audioset-10-10-0.4593-ONNX/resolve/main/onnx/model.onnx"
        Name = "model.onnx"
    },
    @{
        Url = "https://huggingface.co/onnx-community/ast-finetuned-audioset-10-10-0.4593-ONNX/resolve/main/onnx/model_int8.onnx"
        Name = "model_int8.onnx"
    },
    @{
        Url = "https://huggingface.co/onnx-community/ast-finetuned-audioset-10-10-0.4593-ONNX/resolve/main/preprocessor_config.json"
        Name = "preprocessor_config.json"
    },
    @{
        Url = "https://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/class_labels_indices.csv"
        Name = "class_labels_indices.csv"
    }
)

if (-not $IncludeInt8) {
    $files = $files | Where-Object { $_.Name -ne "model_int8.onnx" }
}

foreach ($file in $files) {
    $target = Join-Path $dest $file.Name
    if (Test-Path $target) {
        Write-Host "Exists: $($file.Name)"
        continue
    }

    Write-Host "Downloading $($file.Name)..."
    Invoke-WebRequest -Uri $file.Url -OutFile $target
}

Write-Host "AST model pack ready at $dest"
