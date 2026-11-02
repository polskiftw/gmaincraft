#!/usr/bin/env bash
# Run MCio Minecraft in a visible NVIDIA Gamescope window without locking the
# *host* mouse. Only this Gamescope process gets the SDL2 LD_PRELOAD override.
# The game and Python MCio process are launched with LD_PRELOAD removed.
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"
if [[ "${1:-}" == "--" ]]; then shift; fi
if [[ $# -eq 0 ]]; then
  printf 'Usage: %s <MCio Python command and arguments>\n' "$0" >&2
  exit 2
fi
if [[ ! -x "$root/mcio-data/xwayland-sandbox-wrapper.sh" ]]; then
  echo 'Missing local Xwayland sandbox wrapper; see docs/DEVELOPMENT.md' >&2
  exit 1
fi
if [[ -z "${WAYLAND_DISPLAY:-}" || -z "${XDG_RUNTIME_DIR:-}" ]]; then
  echo 'No usable Wayland environment; connect through the configured Gentoo Shell MCP sandbox.' >&2
  exit 1
fi
if ! command -v gamescope >/dev/null 2>&1; then
  echo 'Gamescope is not available in this environment.' >&2
  exit 1
fi
# Compiled output stays in the ignored /mcio-data/ tree, never in GitHub.
src="$root/tools/gamescope-no-grab.c"
lib="$root/mcio-data/gamescope-no-grab.so"
if [[ ! -f "$lib" || "$src" -nt "$lib" ]]; then
  tmp="$root/mcio-data/.gamescope-no-grab.$$.so"
  trap 'rm -f -- "$tmp"' EXIT
  gcc -shared -fPIC -O2 -Wall -Wextra -Werror -o "$tmp" "$src" -ldl
  mv -f -- "$tmp" "$lib"
  trap - EXIT
fi
# No --grab or --force-grab-cursor. The SDL2 shim also prevents automatic
# relative-pointer locking requested when Minecraft hides its cursor.
exec env SDL_VIDEODRIVER=wayland \
    WLR_XWAYLAND="$root/mcio-data/xwayland-sandbox-wrapper.sh" \
    ALSOFT_DRIVERS=null \
    LD_PRELOAD="$lib" \
    gamescope --backend sdl -W 800 -H 450 -w 800 -h 450 -- \
    env -u LD_PRELOAD "$@"
