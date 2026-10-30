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
# на машинах GitHub hdiutil порой отвечает «Resource busy» — повторить (выпуск 0.2.45)
for try in 1 2 3 4 5; do
  hdiutil create -quiet -volname "Arx Nova" -srcfolder "$STAGE" -ov -format UDZO "dist/ArxNova-$VER-macOS-$ARCH.dmg" && break
  [ "$try" = 5 ] && exit 1
  echo "hdiutil: попытка $try не удалась, повтор"; sleep 10
done
rm -rf "$STAGE"
ls -la dist/*.dmg
