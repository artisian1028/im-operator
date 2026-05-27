# SoolinOperator CUDA Build Script
# Usage: powershell -ExecutionPolicy Bypass -File build_cuda.ps1 [-Clean] [-Config Release|Debug]
param([switch]$Clean, [string]$Config = "Release", [string]$BuildDir = "build_cuda")

$ErrorActionPreference = "Stop"
$srcDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$buildDir = Join-Path $srcDir $BuildDir
$cudaRoot = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6"

# Verify CUDA
if (-not (Test-Path "$cudaRoot\bin\nvcc.exe")) {
    Write-Host "ERROR: CUDA 12.6 not found at $cudaRoot" -ForegroundColor Red
    exit 1
}

if ($Clean -and (Test-Path $buildDir)) {
    Write-Host "Cleaning $buildDir..."
    Remove-Item -Recurse -Force $buildDir
}
if (-not (Test-Path $buildDir)) {
    New-Item -ItemType Directory -Path $buildDir | Out-Null
}

# Build cmake command line (no spaces in paths = safe for cmd)
$cmakeCmd = @"
cd /d "$buildDir" && cmake "$srcDir" -G Ninja `
  -DCMAKE_CUDA_COMPILER="$cudaRoot\bin\nvcc.exe" `
  -DCMAKE_CUDA_ARCHITECTURES=89 `
  -DCMAKE_CUDA_RUNTIME_LIBRARY=Shared `
  -DCMAKE_BUILD_TYPE=$Config `
  -DIM_OPERATOR_ENABLE_CUDA=ON `
  -DIM_OPERATOR_BUILD_TESTS=ON `
  -DIM_OPERATOR_BUILD_UNIT_TESTS=OFF `
  -DIM_OPERATOR_BUILD_BENCHMARKS=OFF
"@ -replace "`r`n", " " -replace "`n", " "

$buildCmd = "cd /d `"$buildDir`" && cmake --build . --config $Config --target jpeg_bench"

# Find VS and run build in its environment
$vsPath = & "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe" -latest -products * -property installationPath
$vcvars = "`"$vsPath\VC\Auxiliary\Build\vcvars64.bat`""

Write-Host "VS: $vsPath"
Write-Host "CUDA: $cudaRoot"
Write-Host "=== Configuring (Ninja + CUDA) ==="

# Run everything inside VS developer command prompt
# Use a single cmd invocation to preserve vcvars environment
$batch = @"
@echo off
call $vcvars > nul 2>&1
set PATH=$cudaRoot\bin;%PATH%
set CUDA_PATH=$cudaRoot
set CUDAToolkit_ROOT=$cudaRoot
echo VS+ CUDA environment ready
$cmakeCmd
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%
echo.
echo === Building jpeg_bench ===
$buildCmd
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%
echo.
echo === Running benchmark ===
jpeg_bench.exe
"@

$batchFile = Join-Path $buildDir "_build.bat"
$batch | Out-File -FilePath $batchFile -Encoding ASCII

$proc = Start-Process cmd.exe -ArgumentList "/c `"$batchFile`"" -Wait -NoNewWindow -PassThru
if ($proc.ExitCode -ne 0) {
    Write-Host "BUILD FAILED (exit code: $($proc.ExitCode))" -ForegroundColor Red
    # Show cmake error log if it exists
    $errLog = Join-Path $buildDir "CMakeFiles\CMakeConfigureLog.yaml"
    if (Test-Path $errLog) {
        Write-Host "--- CMake Error Log ---"
        Get-Content $errLog | Select-Object -Last 30
    }
    exit $proc.ExitCode
}
Write-Host "`n=== DONE ===" -ForegroundColor Green
