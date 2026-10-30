#!/bin/sh
# Сборка установщика для macOS: dist/ArxNova-<версия>-macOS-<арх>.dmg
set -eu
cd "$(dirname "$0")"
VER=$(grep -m1 '^version' ../pyproject.toml | cut -d'"' -f2)
ARCH=$(uname -m)
uv run --project .. --with pyinstaller --with pillow python make_icon.py icon.png
mkdir -p icon.iconset
for s in 16 32 64 128 256 512; do
  sips -z $s $s icon.png --out icon.iconset/icon_${s}x${s}.png >/dev/null
  sips -z $((s*2)) $((s*2)) icon.png --out icon.iconset/icon_${s}x${s}@2x.png >/dev/null
done
iconutil -c icns icon.iconset -o icon.icns
rm -rf build dist icon.iconset
uv run --project .. --with pyinstaller python third_party.py
uv run --project .. --with pyinstaller --with pillow pyinstaller --noconfirm --clean arxnova.spec
STAGE=dist/dmg; rm -rf "$STAGE"; mkdir -p "$STAGE"
cp -R "dist/Arx Nova.app" "$STAGE/"
ln -s /Applications "$STAGE/Applications"
cp ../LICENSE "$STAGE/LICENSE.txt"; cp THIRD_PARTY_LICENSES.txt "$STAGE/"
hdiutil create -quiet -volname "Arx Nova" -srcfolder "$STAGE" -ov -format UDZO "dist/ArxNova-$VER-macOS-$ARCH.dmg"
rm -rf "$STAGE"
ls -la dist/*.dmg
