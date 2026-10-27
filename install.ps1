# Arx Nova — сервер домашней медиатеки. Установка одной командой (Windows, PowerShell):
#   irm https://raw.githubusercontent.com/ArxSecretorum/arxnova/main/install.ps1 | iex
# Ставит uv (менеджер Python от Astral) в профиль пользователя, сервер — последней версией
# с GitHub и включает автозапуск. Ничего не спрашивает: папку с музыкой и согласие на модель
# звука владелец даёт в Arx Player на телефоне. С вопросами в консоли: $env:ARXNOVA_ASK=1.
# Права администратора не нужны.
$ErrorActionPreference = "Stop"
$Repo = "ArxSecretorum/arxnova"

$uv = (Get-Command uv -ErrorAction SilentlyContinue).Source
if (-not $uv) {
    $cand = Join-Path $env:USERPROFILE ".local\bin\uv.exe"
    if (Test-Path $cand) { $uv = $cand } else {
        Write-Host "Ставлю uv — менеджер Python (в профиль пользователя)…"
        powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
        $uv = $cand
    }
}

$wheel = $env:ARXNOVA_WHEEL
if (-not $wheel) {
    $rel = Invoke-RestMethod "https://api.github.com/repos/$Repo/releases/latest"
    $wheel = ($rel.assets | Where-Object { $_.name -like "*.whl" } | Select-Object -First 1).browser_download_url
}
if (-not $wheel) { throw "Не нашёл релиз Arx Nova на GitHub. Проверьте связь и попробуйте ещё раз." }

Write-Host "Ставлю сервер Arx Nova…"
& $uv tool install --force --python 3.12 $wheel
$bin = Join-Path (& $uv tool dir --bin) "arxnova.exe"

if ($env:ARXNOVA_ASK) { & $bin setup } else { & $bin setup --no-models }
if (-not $env:ARXNOVA_NO_SERVICE) { & $bin service install }

Write-Host ""
Write-Host "Готово. Сервер работает и запускается сам при входе в Windows."
Write-Host "Дальше — с телефона, в этой же домашней сети: Arx Player → Настройки → Arx Nova."
Write-Host "Сервер появится в списке сам — нажмите «Настроить этот сервер», выберите папку с музыкой"
Write-Host "и согласитесь на модель разбора. Консоль больше не нужна."
Write-Host "Проверка установки:  $bin doctor"
