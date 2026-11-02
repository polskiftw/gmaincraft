#!/usr/bin/env bash
# Show GL_VENDOR / GL_RENDERER from the same nested GLX stack Minecraft uses.
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"
probe="$root/mcio-data/opengl-renderer-probe"
gcc -O2 -Wall -Wextra -Werror -o "$probe" tools/opengl-renderer-probe.c -lGL -lX11
exec tools/run-mcio-viewer.sh "$probe"
