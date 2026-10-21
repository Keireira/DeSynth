param([string]$Venv = '.venv')

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
Set-StrictMode -Version 3.0

$project = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$tools = Join-Path $project '.tools'
. (Join-Path $PSScriptRoot 'toolchain/windows.ps1')

# Discover existing tools before isolating rustup/Cargo and package caches.
$rustBin = Find-Rust
$node = Find-Executable 'node.exe' { param($version) $version -match '^v(24|2[6-9]|[3-9][0-9])\.\d+\.\d+$' }
$python = Find-Python $project $Venv
$manager = (Get-Content -Raw (Join-Path $project 'package.json') | ConvertFrom-Json).packageManager
if ($manager -notmatch '^pnpm@(\d+\.\d+\.\d+)(?:\+.+)?$') { throw 'package.json must pin a pnpm version.' }
$pnpmVersion = $Matches[1]
Assert-NativeCompiler

$environment = [ordered]@{
    CARGO_HOME = Join-Path $tools 'cargo'
    RUSTUP_HOME = Join-Path $tools 'rustup'
    MISE_CARGO_HOME = Join-Path $tools 'cargo'
    MISE_RUSTUP_HOME = Join-Path $tools 'rustup'
    MISE_DATA_DIR = Join-Path $tools 'mise/data'
    MISE_CACHE_DIR = Join-Path $tools 'mise/cache'
    MISE_STATE_DIR = Join-Path $tools 'mise/state'
    MISE_CONFIG_DIR = Join-Path $tools 'mise/config'
    MISE_SYSTEM_CONFIG_DIR = Join-Path $tools 'mise/system-config'
    MISE_GLOBAL_CONFIG_FILE = Join-Path $tools 'mise/config/config.toml'
    npm_config_cache = Join-Path $tools 'cache/npm'
    npm_config_store_dir = Join-Path $tools 'cache/pnpm'
    npm_config_state_dir = Join-Path $tools 'state/pnpm'
    PIP_CACHE_DIR = Join-Path $tools 'cache/pip'
    UV_CACHE_DIR = Join-Path $tools 'cache/uv'
    COREPACK_HOME = Join-Path $tools 'cache/corepack'
    PNPM_HOME = Join-Path $tools 'pnpm'
}
foreach ($entry in $environment.GetEnumerator()) {
    [Environment]::SetEnvironmentVariable($entry.Key, $entry.Value, 'Process')
    $directory = if ($entry.Key -eq 'MISE_GLOBAL_CONFIG_FILE') { Split-Path $entry.Value } else { $entry.Value }
    New-Item -ItemType Directory -Force -Path $directory | Out-Null
}
Write-Utf8 $environment.MISE_GLOBAL_CONFIG_FILE @'
[settings]
python.compile = false

[tools]
rust = { version = "1.94", profile = "minimal", components = ["rustfmt", "clippy"] }
'@
$pnpm = Find-Executable 'pnpm.cmd' { param($version) $version -eq $pnpmVersion }

if (-not $rustBin -or -not $node -or -not $python -or -not $pnpm) {
    $mise = Install-Mise $tools
    if (-not $rustBin) {
        $null = Install-Tool $mise 'rust@1.94' 'rustc'
        $sysroot = & $mise exec 'rust@1.94' -- rustc --print sysroot
        if ($LASTEXITCODE -ne 0) { throw 'Cannot resolve the local Rust toolchain.' }
        $rustBin = Join-Path ($sysroot -join "`n").Trim() 'bin'
    }
    if (-not $node) { $node = Install-Tool $mise 'node@24' 'node' }
    $env:PATH = (Split-Path $node) + ';' + $env:PATH
    if (-not $python) { $python = Install-Tool $mise 'python@3.13' 'python' }
    if (-not $pnpm) { $pnpm = Install-Tool $mise "npm:pnpm@$pnpmVersion" 'pnpm' }
}

$paths = @($rustBin, (Split-Path $node), (Split-Path $python), (Split-Path $pnpm)) + ($env:PATH -split ';')
$seen = [System.Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
$paths = @($paths | Where-Object { $_ -and $seen.Add($_.Replace('\', '/').TrimEnd('/')) })
$environment['PATH'] = $paths -join ';'
$environment['DESYNTH_BOOTSTRAP_PYTHON'] = $python
$lines = foreach ($entry in $environment.GetEnumerator()) {
    if ($entry.Value -match '[\r\n]') { throw "Invalid newline in $($entry.Key)" }
    $value = $entry.Value.Replace('\', '/').Replace('$', '$$').Replace('#', '\#')
    "export $($entry.Key) := $value"
}
Write-Utf8 (Join-Path $tools 'environment.mk') (($lines -join "`n") + "`n")
Write-Host "Tools ready: Rust $rustBin; Node $node; Python $python; pnpm $pnpmVersion"
