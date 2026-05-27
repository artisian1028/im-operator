param(
    [string]$Config = "Release",
    [string]$Target = "",
    [string]$Generator = "",
    [switch]$Clean = $false,
    [switch]$Rebuild = $false,
    [switch]$Run = $false,
    [switch]$Help = $false,
    [switch]$NoCuda = $false,
    [string]$BuildDirName = "build"
)

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$BuildDir = Join-Path $ProjectRoot $BuildDirName
$CacheFile = Join-Path $BuildDir "CMakeCache.txt"

function Show-Help {
    Write-Host "Usage: .\build.ps1 [options]" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Options:" -ForegroundColor Yellow
    Write-Host "  -Config <value>     Build configuration (Debug|Release), default: Release"
    Write-Host "  -Target <value>     Build specific target (e.g., folder_data_processor)"
    Write-Host "  -Generator <value>  CMake generator (e.g., 'Visual Studio 17 2022')"
    Write-Host "  -Clean              Clean build directory before building"
    Write-Host "  -Rebuild            Rebuild all targets (clean-first)"
    Write-Host "  -Run                Run the executable after successful build"
    Write-Host "  -Help               Show this help message"
    Write-Host "  -NoCuda             Disable CUDA GPU acceleration"
    Write-Host "  -BuildDirName <val> Build directory name, default: build (e.g., build_cuda)"
    Write-Host ""
    Write-Host "Examples:" -ForegroundColor Green
    Write-Host "  .\build.ps1                                     # Build all (Release, CUDA enabled)"
    Write-Host "  .\build.ps1 -Config Debug                       # Build all (Debug, CUDA enabled)"
    Write-Host "  .\build.ps1 -NoCuda                             # Build without CUDA acceleration"
    Write-Host "  .\build.ps1 -Target folder_data_processor         # Build specific target"
    Write-Host "  .\build.ps1 -Clean -Rebuild                     # Clean and rebuild all"
    Write-Host "  .\build.ps1 -Target folder_data_processor -Run    # Build and run"
}

if ($Help) {
    Show-Help
    exit 0
}

if ($Config -notin @("Debug", "Release")) {
    Write-Host "ERROR: Invalid Config '$Config'. Must be 'Debug' or 'Release'." -ForegroundColor Red
    Show-Help
    exit 1
}

Push-Location $ProjectRoot
try {
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "  SoolinOperator Build Script" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "  Configuration:   $Config" -ForegroundColor Gray
    Write-Host "  CUDA Acceleration: $(if ($NoCuda) { 'DISABLED' } else { 'ENABLED' })" -ForegroundColor $(if ($NoCuda) { 'Yellow' } else { 'Green' })
    Write-Host "  Build Directory: $BuildDirName" -ForegroundColor Gray
    if ($Generator) { Write-Host "  Generator:       $Generator" -ForegroundColor Gray }
    Write-Host ""

    if ($Clean -and (Test-Path $BuildDir)) {
        Write-Host "[1/3] Cleaning build directory..." -ForegroundColor Yellow
        Remove-Item -Path $BuildDir -Recurse -Force
        Write-Host "      Clean completed." -ForegroundColor Green
    }

    $NeedsConfig = -not (Test-Path $BuildDir)

    if (-not $NeedsConfig -and (Test-Path $CacheFile)) {
        $cachedCuda = Select-String -Path $CacheFile -Pattern "^IM_OPERATOR_ENABLE_CUDA:" -SimpleMatch
        if ($cachedCuda) {
            $cudaIsOn = $cachedCuda.Line -match "=ON$"
            if ($NoCuda -and $cudaIsOn) {
                Write-Host "      CUDA setting changed (ON -> OFF), reconfiguring..." -ForegroundColor Yellow
                $NeedsConfig = $true
            } elseif (-not $NoCuda -and -not $cudaIsOn) {
                Write-Host "      CUDA setting changed (OFF -> ON), reconfiguring..." -ForegroundColor Yellow
                $NeedsConfig = $true
            }
        }
    }

    if (-not $NeedsConfig -and (Test-Path $CacheFile)) {
        $cachedConfig = Select-String -Path $CacheFile -Pattern "^CMAKE_BUILD_TYPE:" -SimpleMatch
        if ($cachedConfig -and $cachedConfig.Line -notmatch "=${Config}$") {
            Write-Host "      Configuration changed ($($cachedConfig.Line.Split('=')[-1]) -> $Config), reconfiguring..." -ForegroundColor Yellow
            $NeedsConfig = $true
        }
    }

    if ($NeedsConfig) {
        Write-Host "[1/3] Configuring CMake..." -ForegroundColor Yellow
        New-Item -Path $BuildDir -ItemType Directory -Force | Out-Null

        $ConfigureArgs = @(
            "-B", $BuildDir,
            "-S", $ProjectRoot,
            "-DCMAKE_BUILD_TYPE=$Config"
        )

        if ($NoCuda) {
            $ConfigureArgs += "-DIM_OPERATOR_ENABLE_CUDA=OFF"
        }

        if ($Generator) {
            $ConfigureArgs += "-G"
            $ConfigureArgs += $Generator
        }

        & cmake $ConfigureArgs
        if ($LASTEXITCODE -ne 0) {
            Write-Host "ERROR: CMake configuration failed!" -ForegroundColor Red
            exit 1
        }
        Write-Host "      Configuration completed." -ForegroundColor Green
    } else {
        Write-Host "[1/3] CMake cache is up-to-date, skipping configuration." -ForegroundColor Gray
    }

    Write-Host "[2/3] Building project ($Config)..." -ForegroundColor Yellow

    $BuildArgs = @(
        "--build", $BuildDir,
        "--config", $Config
    )

    $effectiveRebuild = $Rebuild
    if ($Clean -and -not (Test-Path $BuildDir)) {
        $effectiveRebuild = $false
        Write-Host "      Skipping --clean-first (already cleaned via -Clean)." -ForegroundColor Gray
    }

    if ($effectiveRebuild) {
        $BuildArgs += "--clean-first"
    }

    if ($Target -ne "") {
        $BuildArgs += "--target"
        $BuildArgs += $Target
    }

    & cmake $BuildArgs

    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Build failed!" -ForegroundColor Red
        exit 1
    }

    Write-Host "      Build completed successfully." -ForegroundColor Green

    if ($Run -and $Target -ne "") {
        Write-Host "[3/3] Running executable..." -ForegroundColor Yellow

        $ExePath = Join-Path $BuildDir "$Config\$Target.exe"
        if (-not (Test-Path $ExePath)) {
            $ExePath = Join-Path $BuildDir "$Target.exe"
        }
        if (-not (Test-Path $ExePath)) {
            $found = Get-ChildItem -Path $BuildDir -Recurse -Filter "$Target.exe" | Select-Object -First 1
            if ($found) {
                $ExePath = $found.FullName
            }
        }

        if (Test-Path $ExePath) {
            Write-Host "      Running: $ExePath" -ForegroundColor Gray
            Write-Host ""
            & $ExePath
        } else {
            Write-Host "WARNING: Executable not found: $Target.exe in $BuildDir" -ForegroundColor Yellow
        }
    } else {
        Write-Host "[3/3] Build finished." -ForegroundColor Green
    }
}
finally {
    Pop-Location
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Build script completed!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan