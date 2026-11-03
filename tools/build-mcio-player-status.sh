#!/usr/bin/env bash
# Build an opt-in, read-only player-status/crosshair patch for the isolated
# DemoInstance. Original mod is ALWAYS backed up before --install; --restore
# switches back. No host-wide Gentoo/Prism/Minecraft settings are changed.
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
mode="${1:---build}"
case "$mode" in --build|--install|--restore) ;; *) echo 'Use --build, --install or --restore' >&2; exit 2;; esac
mod="$root/mcio-data/instances/DemoInstance/mods/mcio-1.6.1+1.21.3.jar"
backup="$root/mcio-data/mod_backups/mcio-1.6.1+1.21.3.pristine.jar"
if [[ "$mode" == '--restore' ]]; then
  [[ -f "$backup" ]] || { echo "No backed-up original mod at $backup" >&2; exit 1; }
  tmp="${mod}.gmaincraft.$$.tmp"
  trap 'rm -f -- "$tmp"' EXIT
  cp -- "$backup" "$tmp"
  mv -f -- "$tmp" "$mod"
  trap - EXIT
  echo 'RESTORED original MCio 1.6.1 mod to dedicated DemoInstance.'
  sha256sum "$mod"
  exit 0
fi
src="$root/mcio-data/mcio-upstream-inspect"
patch="$root/patches/mcio-1.6.1-player-status.patch"
[[ -f "$patch" ]] || { echo 'Missing tracked patch' >&2; exit 1; }
if [[ ! -d "$src/.git" ]]; then
  git clone --depth 1 --branch '1.6.1+1.21.3' https://github.com/twoturtles/MCio.git "$src"
fi
[[ "$(git -C "$src" rev-parse HEAD)" == 'e89bd1ea5187814a9a3b854dedf3573defac6f8a' ]] || {
  echo 'Unexpected upstream MCio commit; refusing to apply patch' >&2; exit 1;
}
if git -C "$src" apply --reverse --check "$patch" 2>/dev/null; then
  : # Already applied. Still verify only our exact two files changed.
elif git -C "$src" apply --check "$patch"; then
  git -C "$src" apply "$patch"
else
  echo 'MCio source modifications conflict with tracked patch' >&2
  exit 1
fi
actual="$root/mcio-data/mcio-current-diff.$$.patch"
trap 'rm -f -- "$actual"' EXIT
git -C "$src" diff -- src/client/java/net/twoturtles/MCioNetworkPacket.java \
  src/client/java/net/twoturtles/MCioObservationHandler.java > "$actual"
if ! cmp -s -- "$actual" "$patch"; then
  echo 'Unexpected source patch divergence; refusing to build an unreviewed mod' >&2
  exit 1
fi
[[ -z "$(git -C "$src" status --porcelain | grep -v '^ M src/client/java/net/twoturtles/MCioNetworkPacket.java$' | grep -v '^ M src/client/java/net/twoturtles/MCioObservationHandler.java$' || :)" ]] || {
  echo 'Unexpected tracked modifications in MCio upstream checkout' >&2; exit 1;
}
export JAVA_HOME="$root/mcio-data/instances/DemoInstance/runtime/java-runtime-delta/linux/java-runtime-delta"
export GRADLE_USER_HOME="$src/.gradle"
[[ -x "$JAVA_HOME/bin/java" ]] || { echo 'Missing local Java 21' >&2; exit 1; }
(cd "$src" && ./gradlew --no-daemon remapJar)
built="$src/build/libs/mcio-1.6.1+1.21.3.jar"
[[ -s "$built" ]] || { echo 'Expected remapped Fabric jar is missing' >&2; exit 1; }
echo 'BUILT patched MCio mod:'
sha256sum "$built"
if [[ "$mode" == '--install' ]]; then
  [[ -f "$mod" ]] || { echo 'Original MCio jar missing' >&2; exit 1; }
  mkdir -p -- "$(dirname -- "$backup")"
  if [[ -f "$backup" ]]; then
    if ! cmp -s -- "$mod" "$backup" && ! cmp -s -- "$mod" "$built"; then
      echo 'Unknown mod jar differs from BOTH original backup and patched build; aborting' >&2
      exit 1
    fi
  else
    cp -n -- "$mod" "$backup"
  fi
  # Never keep two mcio jars in the mods directory, Fabric would load both.
  tmp="${mod}.gmaincraft.$$.tmp"
  cp -- "$built" "$tmp"
  mv -f -- "$tmp" "$mod"
  echo 'INSTALLED ONLY in disposable DemoInstance. Original backed up at:'
  echo "$backup"
  echo 'GMAINCRAFT_STATUS_V1=1 is needed to emit the new packet option.'
fi
