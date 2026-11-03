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
# Optional test-only virtual refresh for the isolated nested compositor.
# When unfocused, Gamescope may throttle to a separate refresh; set BOTH.
# No host monitor, Wayland, system service or user graphics setting is changed.
backend="${GMAINCRAFT_GAMESCOPE_BACKEND:-sdl}"
if [[ "$backend" != 'sdl' && "$backend" != 'headless' ]]; then
  echo "Invalid GMAINCRAFT_GAMESCOPE_BACKEND: use sdl or headless" >&2
  exit 2
fi
if [[ "$backend" == 'headless' ]]; then
  echo 'Nested Gamescope backend: headless (game still renders RGB in Xwayland)' >&2
fi
refresh_args=()
if [[ -n "${GMAINCRAFT_GAMESCOPE_REFRESH_HZ:-}" ]]; then
  hz="$GMAINCRAFT_GAMESCOPE_REFRESH_HZ"
  if [[ ! "$hz" =~ ^[0-9]+$ ]] || (( hz < 30 || hz > 600 )); then
    echo "Invalid GMAINCRAFT_GAMESCOPE_REFRESH_HZ: expected integer 30..600" >&2
    exit 2
  fi
  refresh_args=(-r "$hz" -o "$hz")
  echo "Nested Gamescope refresh and unfocused refresh: $hz Hz" >&2
fi
# No --grab or --force-grab-cursor. The SDL2 shim also prevents automatic
# relative-pointer locking requested when Minecraft hides its cursor.
exec env SDL_VIDEODRIVER=wayland \
    WLR_XWAYLAND="$root/mcio-data/xwayland-sandbox-wrapper.sh" \
    ALSOFT_DRIVERS=null \
    LD_PRELOAD="$lib" \
    gamescope --backend "$backend" -W 800 -H 450 -w 800 -h 450 "${refresh_args[@]}" -- \
    env -u LD_PRELOAD "$@"
