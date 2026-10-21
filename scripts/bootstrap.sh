#!/bin/sh
set -eu

project=$(CDPATH= cd -P "$(dirname "$0")/.." && pwd)
venv=.venv
if [ "${1-}" = '--venv' ] && [ "$#" -eq 2 ]; then
    venv=$2
elif [ "$#" -ne 0 ]; then
    printf '%s\n' 'Usage: sh scripts/bootstrap.sh [--venv PATH]' >&2
    exit 2
fi

case "$(uname -s)" in
    MINGW*|MSYS*|CYGWIN*)
        exec powershell.exe -NoProfile -ExecutionPolicy Bypass -File "$(cygpath -w "$project/scripts/bootstrap.ps1")" -Venv "$venv"
        ;;
esac

. "$project/scripts/toolchain/posix.sh"
tools=$project/.tools
rust_bin=$(find_rust)
node=$(find_node)
python=$(find_python "$venv")
pnpm_version=$(sed -n 's/.*"packageManager"[[:space:]]*:[[:space:]]*"pnpm@\([0-9][0-9.]*\).*".*/\1/p' "$project/package.json")
[ -n "$pnpm_version" ] || fail 'package.json must pin a pnpm version.'
command -v cc >/dev/null 2>&1 || fail 'Install the platform C/C++ compiler and SDK, then rerun make setup.'

export CARGO_HOME="$tools/cargo" RUSTUP_HOME="$tools/rustup"
export MISE_CARGO_HOME="$CARGO_HOME" MISE_RUSTUP_HOME="$RUSTUP_HOME"
export MISE_DATA_DIR="$tools/mise/data" MISE_CACHE_DIR="$tools/mise/cache"
export MISE_STATE_DIR="$tools/mise/state" MISE_CONFIG_DIR="$tools/mise/config"
export MISE_SYSTEM_CONFIG_DIR="$tools/mise/system-config"
export MISE_GLOBAL_CONFIG_FILE="$MISE_CONFIG_DIR/config.toml"
export npm_config_cache="$tools/cache/npm" npm_config_store_dir="$tools/cache/pnpm"
export npm_config_state_dir="$tools/state/pnpm" PNPM_HOME="$tools/pnpm"
export PIP_CACHE_DIR="$tools/cache/pip" UV_CACHE_DIR="$tools/cache/uv"
export COREPACK_HOME="$tools/cache/corepack"
mkdir -p "$CARGO_HOME" "$RUSTUP_HOME" "$MISE_CONFIG_DIR" "$tools/downloads"
cat > "$MISE_GLOBAL_CONFIG_FILE" <<'CONFIG'
[settings]
python.compile = false

[tools]
rust = { version = "1.94", profile = "minimal", components = ["rustfmt", "clippy"] }
CONFIG
pnpm=$(find_pnpm)

if [ -z "$rust_bin" ] || [ -z "$node" ] || [ -z "$python" ] || [ -z "$pnpm" ]; then
    mise=$(install_mise)
    if [ -z "$rust_bin" ]; then
        install_tool 'rust@1.94' rustc >/dev/null
        sysroot=$("$mise" exec 'rust@1.94' -- rustc --print sysroot)
        rust_bin=$sysroot/bin
    fi
    if [ -z "$node" ]; then node=$(install_tool 'node@24' node); fi
    PATH=$(dirname "$node"):$PATH
    export PATH
    if [ -z "$python" ]; then python=$(install_tool 'python@3.13' python); fi
    if [ -z "$pnpm" ]; then pnpm=$(install_tool "npm:pnpm@$pnpm_version" pnpm); fi
fi

PATH=$(printf '%s' "$rust_bin:$(dirname "$node"):$(dirname "$python"):$(dirname "$pnpm"):$PATH" |
    awk -v RS=: 'length && !seen[$0]++ { printf "%s%s", separator, $0; separator=":" }')
export PATH DESYNTH_BOOTSTRAP_PYTHON="$python"
temporary="$tools/environment.mk.$$.tmp"
trap 'rm -f "$temporary"' EXIT HUP INT TERM
{
    for name in PATH CARGO_HOME RUSTUP_HOME MISE_CARGO_HOME MISE_RUSTUP_HOME MISE_DATA_DIR MISE_CACHE_DIR MISE_STATE_DIR MISE_CONFIG_DIR MISE_SYSTEM_CONFIG_DIR MISE_GLOBAL_CONFIG_FILE npm_config_cache npm_config_store_dir npm_config_state_dir PNPM_HOME PIP_CACHE_DIR UV_CACHE_DIR COREPACK_HOME DESYNTH_BOOTSTRAP_PYTHON; do
        # The variable names above are fixed; never evaluate external text as code.
        eval 'value=${'"$name"'}'
        case "$value" in *'
'*) fail "Invalid newline in $name" ;; esac
        escaped=$(printf '%s' "$value" | sed -e 's/\$/$$/g' -e 's/#/\\#/g')
        printf 'export %s := %s\n' "$name" "$escaped"
    done
} > "$temporary"
mv "$temporary" "$tools/environment.mk"
printf 'Tools ready: Rust %s; Node %s; Python %s; pnpm %s\n' "$rust_bin" "$node" "$python" "$pnpm_version"
