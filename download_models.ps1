$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$ModelDir = Join-Path $PSScriptRoot 'models'
New-Item -ItemType Directory -Force -Path $ModelDir | Out-Null
$Files = @(
    @{
        Name = 'kokoro-v1.0.fp16.onnx'
        Url = 'https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.1/kokoro-v1.0.fp16.onnx'
        Sha256 = 'f3a290d384fbb27966d462905c71a46cef9e5fd00516b40df32a0b4afe77ac96'
    },
    @{
        Name = 'kokoro-v1.0.int8.onnx'
        Url = 'https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.1/kokoro-v1.0.int8.onnx'
        Sha256 = 'ae315a79b623f244700e4afb9246c46a26066782e049ba174bf3ba433970ee9c'
    },
    @{
        Name = 'voices-v1.0.bin'
        Url = 'https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.1/voices-v1.0.bin'
        Sha256 = 'bca610b8308e8d99f32e6fe4197e7ec01679264efed0cac9140fe9c29f1fbf7d'
    }
)
foreach ($File in $Files) {
    $Destination = Join-Path $ModelDir $File.Name
    if (Test-Path $Destination) {
        $Existing = (Get-FileHash -Algorithm SHA256 -Path $Destination).Hash.ToLowerInvariant()
        if ($Existing -eq $File.Sha256) {
            Write-Host "Verified existing model: $($File.Name)"
            continue
        }
        Remove-Item -Force $Destination
    }
    Write-Host "Downloading $($File.Name)..."
    Invoke-WebRequest -Uri $File.Url -OutFile $Destination -MaximumRedirection 10
    $Actual = (Get-FileHash -Algorithm SHA256 -Path $Destination).Hash.ToLowerInvariant()
    if ($Actual -ne $File.Sha256) {
        Remove-Item -Force $Destination -ErrorAction SilentlyContinue
        throw "SHA256 check failed for $($File.Name). Expected $($File.Sha256), got $Actual."
    }
    Write-Host "Verified $($File.Name)"
}
Write-Host "All Kokoro model files verified in $ModelDir"