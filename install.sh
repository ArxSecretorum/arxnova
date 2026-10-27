#!/bin/sh
# Arx Nova — сервер домашней медиатеки. Установка одной командой (macOS, Linux):
#   curl -LsSf https://raw.githubusercontent.com/ArxSecretorum/arxnova/main/install.sh | sh
# Ставит uv (менеджер Python от Astral) в домашнюю папку, сам сервер — последней версией
# с GitHub и включает автозапуск. Ничего не спрашивает: папку с музыкой и согласие на модель
# звука владелец даёт в Arx Player на телефоне (у сервера на Pi и NAS часто нет ни экрана,
# ни клавиатуры). Кто хочет по-старому, с вопросами в терминале: ARXNOVA_ASK=1.
# Права администратора не нужны.
set -eu

REPO="ArxSecretorum/arxnova"
say() { printf '%s\n' "$*"; }

if command -v uv >/dev/null 2>&1; then
  UV="$(command -v uv)"
elif [ -x "$HOME/.local/bin/uv" ]; then
  UV="$HOME/.local/bin/uv"
else
  say "Ставлю uv — менеджер Python (в домашнюю папку)…"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  UV="$HOME/.local/bin/uv"
fi

# Последняя версия — готовый пакет из релизов GitHub (git не нужен).
WHEEL="${ARXNOVA_WHEEL:-}"
if [ -z "$WHEEL" ]; then
  WHEEL="$(curl -fsSL "https://api.github.com/repos/$REPO/releases/latest" \
    | grep -o '"browser_download_url": *"[^"]*\.whl"' | head -1 | sed 's/.*"\(http[^"]*\)"/\1/')"
fi
if [ -z "$WHEEL" ]; then
  say "Не нашёл релиз Arx Nova на GitHub. Проверьте связь и попробуйте ещё раз."
  exit 1
fi

say "Ставлю сервер Arx Nova…"
"$UV" tool install --force --python 3.12 "$WHEEL"
BIN="$("$UV" tool dir --bin)/arxnova"

if [ -n "${ARXNOVA_ASK:-}" ] && [ -r /dev/tty ]; then
  "$BIN" setup </dev/tty
else
  "$BIN" setup --no-models </dev/null
fi

if [ -z "${ARXNOVA_NO_SERVICE:-}" ]; then
  "$BIN" service install
fi

say ""
say "Готово. Сервер работает и запускается сам."
say "Дальше — с телефона, в этой же домашней сети: Arx Player → Настройки → Arx Nova."
say "Сервер появится в списке сам — нажмите «Настроить этот сервер», выберите папку с музыкой"
say "и согласитесь на модель разбора. Терминал больше не нужен."
say "Проверка установки:  $BIN doctor"
