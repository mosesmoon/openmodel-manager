#!/usr/bin/env bash
# Build the Linux packages on any x86_64 Linux machine with Docker:
#   AppImage + deb  (inside an Ubuntu 22.04 container)
#   Flatpak         (flatpak-builder from Flathub, installed per user; no sudo)
#
#   packaging/linux/build.sh            # everything
#   packaging/linux/build.sh appimage   # AppImage + deb only
#   packaging/linux/build.sh flatpak    # Flatpak only (needs the deb)
#
# Output: target/release/bundle/{appimage,deb,flatpak}/
set -euo pipefail
cd "$(dirname "$0")/../.."
ROOT=$PWD
WHAT=${1:-all}
VERSION=$(grep -m1 '"version"' app/src-tauri/tauri.conf.json | sed -E 's/.*"([0-9][^"]*)".*/\1/')

if [[ $WHAT == all || $WHAT == appimage ]]; then
  docker build -t openmodel-build:jammy packaging/linux
  mkdir -p "$ROOT/.cache/linux-home"
  docker run --rm -u "$(id -u):$(id -g)" \
    -e HOME=/home/build -e APPIMAGE_EXTRACT_AND_RUN=1 -e CARGO_TARGET_DIR=/src/target \
    -v "$ROOT":/src -v "$ROOT/.cache/linux-home":/home/build -w /src/app \
    openmodel-build:jammy bash -c "npm ci && npx tauri build --bundles appimage,deb"
fi

if [[ $WHAT == all || $WHAT == flatpak ]]; then
  DEB="$ROOT/target/release/bundle/deb/OpenModel-Manager_${VERSION}_amd64.deb"
  [[ -f $DEB ]] || { echo "build the deb first: $0 appimage" >&2; exit 1; }
  flatpak remote-add --user --if-not-exists flathub https://dl.flathub.org/repo/flathub.flatpakrepo
  flatpak install --user -y --noninteractive flathub org.flatpak.Builder
  STAGE="$ROOT/.cache/flatpak"
  rm -rf "$STAGE" && mkdir -p "$STAGE"
  cp packaging/flatpak/org.openmodel.manager.yml packaging/flatpak/org.openmodel.manager.desktop \
     packaging/flatpak/org.openmodel.manager.metainfo.xml app/src-tauri/icons/128x128@2x.png "$STAGE/"
  cp "$DEB" "$STAGE/openmodel-manager.deb"
  cd "$STAGE"
  flatpak run org.flatpak.Builder --user --install-deps-from=flathub --force-clean \
    --repo=repo build-dir org.openmodel.manager.yml
  mkdir -p "$ROOT/target/release/bundle/flatpak"
  flatpak build-bundle --runtime-repo=https://dl.flathub.org/repo/flathub.flatpakrepo \
    repo "$ROOT/target/release/bundle/flatpak/OpenModel-Manager_${VERSION}_x86_64.flatpak" org.openmodel.manager
fi

ls -la target/release/bundle/appimage target/release/bundle/deb target/release/bundle/flatpak 2>/dev/null || true
