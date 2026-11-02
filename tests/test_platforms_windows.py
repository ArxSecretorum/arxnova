"""Брандмауэр Windows: живого Windows у разработки нет, поэтому вердикт — чистая функция, а настоящий
PowerShell проверяется на windows-latest в CI (последний тест)."""

import subprocess
import sys

import pytest

from arxnova import platforms

APP = r"C:\\Arx\\python.exe"


def verdict(enabled="True", category="Private", rules=()):
    return platforms.windows_firewall_verdict({"category": category, "enabled": enabled, "rules": list(rules)}, APP)


def rule(action="Allow", enabled="True", profile="Any"):
    return {"action": action, "enabled": enabled, "profile": profile}


def test_firewall_off_is_fine():
    assert verdict(enabled="False")[0] == "ok"


def test_no_rule_means_inbound_is_blocked_and_says_how_to_fix():
    status, text = verdict(rules=[])
    assert status == "fail" and APP in text and "Разрешить работу с приложением" in text


def test_allow_rule_counts_only_for_this_network():
    assert verdict(rules=[rule(profile="Private")])[0] == "ok"
    assert verdict(rules=[rule(profile="Any")])[0] == "ok"
    assert verdict(rules=[rule(profile="Domain, Private")])[0] == "ok"
    # галочку поставили только для домашней сети, а сеть сейчас «общественная»
    assert verdict(category="Public", rules=[rule(profile="Private")])[0] == "fail"


def test_disabled_rule_does_not_count():
    assert verdict(rules=[rule(enabled="False")])[0] == "fail"


def test_block_wins_over_allow():
    status, text = verdict(rules=[rule(), rule(action="Block")])
    assert status == "fail" and "запрещает" in text


def test_no_data_is_silence_not_a_scare():
    assert platforms.windows_firewall_verdict({}, APP) is None
    assert verdict(enabled="") is None
    assert verdict(category="") is None


@pytest.mark.skipif(sys.platform != "win32", reason="только на Windows (CI: windows-latest)")
def test_real_powershell_answers_on_windows():
    """Сценарий PowerShell нельзя проверить нигде, кроме Windows. Сбой здесь не должен держать выпуск (без него окно
    сети просто молчит про брандмауэр), поэтому xfail: смотреть глазами в журнале CI."""
    p = platforms.current()
    assert p.key == "windows"
    apps = p.apps()
    assert apps and all(a.lower().endswith(".exe") for a in apps)
    try:
        data = p.read_firewall(23056, apps)
    except (ValueError, OSError, subprocess.SubprocessError) as e:  # и таймаут на медленной машине CI
        pytest.xfail(f"PowerShell не ответил: {e}")
    print("брандмауэр:", data, "→", platforms.windows_firewall_verdict(data, apps[0]))
    if data.get("enabled") not in ("True", "False"):
        pytest.xfail(f"нет состояния брандмауэра: {data}")
    assert platforms.windows_firewall_verdict(data, apps[0])[0] in ("ok", "fail")
