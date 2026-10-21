fail() { printf '%s\n' "$*" >&2; exit 1; }

find_rust() {
    executable=$(command -v rustc || :)
    [ -n "$executable" ] || return 0
    version=$("$executable" --version 2>/dev/null || :)
    if printf '%s\n' "$version" | awk '/^rustc [0-9]+\.[0-9]+\./ { split($2,v,"."); if (v[1]>1 || (v[1]==1 && v[2]>=94)) exit 0; exit 1 } !/^rustc / { exit 1 }'; then
        sysroot=$("$executable" --print sysroot)
        if [ -x "$sysroot/bin/cargo" ] && [ -x "$sysroot/bin/rustfmt" ] && [ -x "$sysroot/bin/cargo-clippy" ]; then printf '%s\n' "$sysroot/bin"; fi
    fi
}

find_node() {
    executable=$(command -v node || :)
    [ -n "$executable" ] || return 0
    version=$("$executable" --version 2>/dev/null || :)
    if printf '%s\n' "$version" | grep -Eq '^v(24|2[6-9]|[3-9][0-9])\.[0-9]+\.[0-9]+$'; then printf '%s\n' "$executable"; fi
}

python_usable() { "$1" --version 2>/dev/null | grep -Eq '^Python 3\.(11|12|13)\.'; }

find_python() {
    if [ -n "${DESYNTH_BOOTSTRAP_PYTHON-}" ] && python_usable "$DESYNTH_BOOTSTRAP_PYTHON"; then
        printf '%s\n' "$DESYNTH_BOOTSTRAP_PYTHON"
        return
    fi
    case "$1" in /*) directory=$1 ;; *) directory=$project/$1 ;; esac
    if [ -f "$directory/pyvenv.cfg" ]; then
        base=$(sed -n 's/^home[[:space:]]*=[[:space:]]*//p' "$directory/pyvenv.cfg")
        for name in python3 python; do
            if [ -x "$base/$name" ] && python_usable "$base/$name"; then printf '%s\n' "$base/$name"; return; fi
        done
    fi
    for name in python3.13 python3.12 python3.11 python3 python; do
        executable=$(command -v "$name" || :)
        if [ -n "$executable" ] && python_usable "$executable"; then printf '%s\n' "$executable"; return; fi
    done
}

find_pnpm() {
    executable=$(command -v pnpm || :)
    if [ -n "$executable" ] && [ "$("$executable" --version 2>/dev/null || :)" = "$pnpm_version" ]; then printf '%s\n' "$executable"; fi
}

install_mise() {
    executable=$tools/bin/mise
    if [ ! -x "$executable" ]; then
        command -v curl >/dev/null 2>&1 || fail 'curl is required to bootstrap missing tools.'
        curl -fsSL --retry 3 https://mise.run -o "$tools/downloads/mise-install.sh"
        MISE_VERSION=2026.10.4 MISE_INSTALL_PATH="$executable" sh "$tools/downloads/mise-install.sh" >&2
        [ -x "$executable" ] || fail 'The local mise installation failed.'
    fi
    printf '%s\n' "$executable"
}

install_tool() {
    "$mise" install --yes "$1" >&2
    "$mise" which --tool "$1" "$2"
}
