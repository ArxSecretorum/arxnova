#!/bin/sh
# Пакет .deb для Linux (x86_64 / aarch64 — Raspberry Pi): dist/arxnova_<версия>_<арх>.deb
# Ставит /opt/arxnova, команду /usr/bin/arxnova и ярлык «Arx Nova» в меню.
set -eu
cd "$(dirname "$0")/.."
VER=$(grep -m1 '^version' ../pyproject.toml | cut -d'"' -f2)
case "$(uname -m)" in x86_64) ARCH=amd64;; aarch64|arm64) ARCH=arm64;; *) ARCH=$(uname -m);; esac
uv run --project .. --with pillow python make_icon.py icon.png
uv run --project .. --with pyinstaller --with pillow pyinstaller --noconfirm --clean arxnova.spec
ROOT=dist/deb; rm -rf "$ROOT"
mkdir -p "$ROOT/DEBIAN" "$ROOT/opt" "$ROOT/usr/bin" "$ROOT/usr/share/applications" "$ROOT/usr/share/icons/hicolor/512x512/apps"
cp -R dist/arxnova "$ROOT/opt/arxnova"
ln -s /opt/arxnova/arxnova "$ROOT/usr/bin/arxnova"
cp icon.png "$ROOT/usr/share/icons/hicolor/512x512/apps/arxnova.png"
cat > "$ROOT/usr/share/applications/arxnova.desktop" <<DESK
[Desktop Entry]
Type=Application
Name=Arx Nova
Comment=Сервер медиатеки для Arx Player
Exec=/opt/arxnova/arxnova
Icon=arxnova
Categories=AudioVideo;Audio;
Terminal=false
DESK
cat > "$ROOT/DEBIAN/control" <<CTRL
Package: arxnova
Version: $VER
Architecture: $ARCH
Maintainer: Arx <noreply@github.com>
Depends: libc6
Section: sound
Priority: optional
Description: Arx Nova — сервер домашней FLAC-медиатеки для Arx Player
 Слушать свою музыку из любой точки, поток по вкусу. Запустите «Arx Nova» из меню
 или «arxnova» в терминале; на сервере без экрана — «arxnova setup» и «arxnova service install».
CTRL
dpkg-deb --build --root-owner-group "$ROOT" "dist/arxnova_${VER}_${ARCH}.deb"
rm -rf "$ROOT"
ls -la dist/*.deb
