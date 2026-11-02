#!/usr/bin/env bash
# One-shot, Minecraft-independent SDL2 event filter regression test.
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p -- "$root/mcio-data"
gcc -shared -fPIC -O2 -Wall -Wextra -Werror -o "$root/mcio-data/gamescope-no-grab.so" "$root/tools/gamescope-no-grab.c" -ldl
gcc -O2 -Wall -Wextra -Werror -o "$root/mcio-data/test-gamescope-mouse-filter" "$root/tests/test_gamescope_mouse_filter.c" -lSDL2
SDL_VIDEODRIVER=dummy LD_PRELOAD="$root/mcio-data/gamescope-no-grab.so" "$root/mcio-data/test-gamescope-mouse-filter"
