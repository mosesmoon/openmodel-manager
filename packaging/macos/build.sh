#!/usr/bin/env bash
# Build the macOS disk image on a Mac (Apple's tools only run on macOS).
#
#   packaging/macos/build.sh
#
# Produces a universal (Apple Silicon + Intel) app in a DMG whose window shows
# the app next to an Applications shortcut: drag one onto the other to install.
# Output: target/universal-apple-darwin/release/bundle/dmg/
#
# Needs: Xcode command line tools (xcode-select --install), Rust (rustup) and
# Node.js 24.22.1 or newer (older releases are affected by CVE-2026-111497).
#
# The DMG is not signed or notarized. On first launch macOS refuses it as from
# an "unidentified developer": right-click the app → Open, or run
#   xattr -dr com.apple.quarantine /Applications/OpenModel-Manager.app
# Signing needs an Apple Developer ID: set APPLE_SIGNING_IDENTITY (and
# APPLE_ID / APPLE_PASSWORD / APPLE_TEAM_ID for notarization) before building.
set -euo pipefail
cd "$(dirname "$0")/../.."

node -e 'const [a,b,c]=process.versions.node.split(".").map(Number); if (a<24 || (a===24 && (b<22 || (b===22 && c<1)))) { console.error("Node.js " + process.versions.node + " is too old (need 24.22.1+)"); process.exit(1) }'
rustup target add aarch64-apple-darwin x86_64-apple-darwin

cd app
npm ci
npx tauri build --target universal-apple-darwin --bundles dmg
ls -la ../target/universal-apple-darwin/release/bundle/dmg/
