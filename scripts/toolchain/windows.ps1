function Write-Utf8([string]$Path, [string]$Text) {
    $temporary = "$Path.$PID.tmp"
    [IO.File]::WriteAllText($temporary, $Text, [Text.UTF8Encoding]::new($false))
    Move-Item -LiteralPath $temporary -Destination $Path -Force
}

function Read-Version([string]$Executable) {
    try {
        $version = & $Executable --version 2>$null
        if ($LASTEXITCODE -eq 0) { return ($version -join "`n").Trim() }
    } catch { }
    return ''
}

function Find-Executable([string]$Name, [scriptblock]$Accept) {
    foreach ($command in @(Get-Command $Name -All -CommandType Application -ErrorAction SilentlyContinue)) {
        if (& $Accept (Read-Version $command.Source)) { return $command.Source }
    }
    return $null
}

function Find-Rust {
    $rustc = Find-Executable 'rustc.exe' {
        param($version)
        $version -match '^rustc (\d+)\.(\d+)\.\d+' -and ([int]$Matches[1] -gt 1 -or [int]$Matches[2] -ge 94)
    }
    if ($rustc) {
        $sysroot = & $rustc --print sysroot
        $bin = Join-Path ($sysroot -join "`n").Trim() 'bin'
        if ($LASTEXITCODE -eq 0 -and
            (Test-Path -LiteralPath (Join-Path $bin 'cargo.exe')) -and
            (Test-Path -LiteralPath (Join-Path $bin 'rustfmt.exe')) -and
            (Test-Path -LiteralPath (Join-Path $bin 'cargo-clippy.exe'))) { return $bin }
    }
    return $null
}

function Find-Python([string]$Project, [string]$Venv) {
    $candidates = @()
    if ($env:DESYNTH_BOOTSTRAP_PYTHON) { $candidates += $env:DESYNTH_BOOTSTRAP_PYTHON }
    $directory = if ([IO.Path]::IsPathRooted($Venv)) { $Venv } else { Join-Path $Project $Venv }
    $config = Join-Path $directory 'pyvenv.cfg'
    if (Test-Path -LiteralPath $config) {
        foreach ($line in Get-Content -LiteralPath $config) {
            if ($line -match '^home\s*=\s*(.+)$') { $candidates += Join-Path $Matches[1].Trim() 'python.exe' }
        }
    }
    foreach ($name in @('python3.13.exe', 'python3.12.exe', 'python3.11.exe', 'python.exe', 'python3.exe')) {
        $candidates += @(Get-Command $name -All -CommandType Application -ErrorAction SilentlyContinue | ForEach-Object Source)
    }
    foreach ($candidate in $candidates) {
        if ((Test-Path -LiteralPath $candidate) -and (Read-Version $candidate) -match '^Python 3\.(11|12|13)\.') { return $candidate }
    }
    $launcher = Get-Command 'py.exe' -CommandType Application -ErrorAction SilentlyContinue
    if ($launcher) {
        # py -0p enumerates base interpreters without starting an unsupported one.
        foreach ($line in @(& $launcher.Source -0p 2>$null)) {
            if ($line -match '^\s*-(?:V:)?3\.(?:11|12|13)(?:-\d+)?\s+\*?\s*(.+python\.exe)\s*$') {
                $candidate = $Matches[1].Trim()
                if ((Read-Version $candidate) -match '^Python 3\.(11|12|13)\.') { return $candidate }
            }
        }
    }
    return $null
}

function Assert-NativeCompiler {
    $compiler = [bool](Get-Command 'cl.exe' -CommandType Application -ErrorAction SilentlyContinue)
    $locator = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio/Installer/vswhere.exe'
    if (-not $compiler -and (Test-Path -LiteralPath $locator)) {
        $installation = & $locator -latest -products '*' -requiresAny -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 Microsoft.VisualStudio.Component.VC.Tools.ARM64 -property installationPath
        $compiler = $LASTEXITCODE -eq 0 -and [bool]$installation
    }
    if (-not $compiler) {
        throw 'Install Visual Studio C++ Build Tools with MSVC and a Windows SDK, then rerun make setup. Bootstrap does not install system compilers.'
    }
    Assert-WindowsSdk
}

function Assert-WindowsSdk {
    $registry = Get-ItemProperty -LiteralPath 'HKLM:\SOFTWARE\Microsoft\Windows Kits\Installed Roots' -Name KitsRoot10 -ErrorAction SilentlyContinue
    $roots = @($env:WindowsSdkDir, (Join-Path ${env:ProgramFiles(x86)} 'Windows Kits/10'))
    if ($registry) { $roots += $registry.KitsRoot10 }
    $architecture = if ($env:PROCESSOR_ARCHITECTURE -eq 'ARM64' -or $env:PROCESSOR_ARCHITEW6432 -eq 'ARM64') { 'arm64' } else { 'x64' }
    foreach ($root in $roots | Where-Object { $_ }) {
        foreach ($version in @(Get-ChildItem -LiteralPath (Join-Path $root 'Lib') -Directory -ErrorAction SilentlyContinue)) {
            if ((Test-Path -LiteralPath (Join-Path $version.FullName "um/$architecture/kernel32.lib")) -and
                (Test-Path -LiteralPath (Join-Path $version.FullName "ucrt/$architecture/ucrt.lib")) -and
                (Test-Path -LiteralPath (Join-Path $root "Include/$($version.Name)/um/Windows.h")) -and
                (Test-Path -LiteralPath (Join-Path $root "Include/$($version.Name)/ucrt/stdio.h"))) { return }
        }
    }
    throw 'Install a Windows 10/11 SDK (headers, UCRT and libraries) through Visual Studio Installer, then rerun make setup.'
}

function Install-Mise([string]$Tools) {
    $executable = Join-Path $Tools 'bin/mise.exe'
    if (-not (Test-Path -LiteralPath $executable)) {
        $download = Join-Path $Tools 'downloads'
        New-Item -ItemType Directory -Force -Path $download | Out-Null
        $installer = Join-Path $download 'mise-install.ps1'
        [Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12
        Invoke-WebRequest -UseBasicParsing 'https://mise.run/powershell' -OutFile $installer
        $env:MISE_VERSION = '2026.10.4'
        $env:MISE_INSTALL_PATH = $executable
        $env:MISE_INSTALL_MODIFY_PATH = '0'
        $env:MISE_INSTALL_SHIMS_PATH = '0'
        & $installer
        if (-not (Test-Path -LiteralPath $executable)) { throw 'The local mise installation failed.' }
    }
    return $executable
}

function Install-Tool([string]$Mise, [string]$Tool, [string]$Executable) {
    & $Mise install --yes $Tool | Out-Host
    if ($LASTEXITCODE -ne 0) { throw "Cannot install $Tool into .tools." }
    $path = & $Mise which --tool $Tool $Executable
    if ($LASTEXITCODE -ne 0 -or -not $path) { throw "Cannot locate installed $Tool." }
    return ($path -join "`n").Trim()
}
