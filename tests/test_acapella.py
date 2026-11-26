"""Акапеллы — инструмент диджея: стиль «Acapella», в поток не идут (замечание владельца 26.10.2026)."""

from arxnova.nova import is_acapella


def test_acapella_detected_by_title_or_tag():
    assert is_acapella("It's Not Over (Accapella)", "Acappella")
    assert is_acapella("Space Between (Acapella)", None)
    assert is_acapella("Breathless (Keyapella)", "")
    assert is_acapella("Panda Style (Pandapella)", None)
    assert is_acapella("I Am Somebody (Acapella)", "Drum & Bass")
    assert is_acapella("Shine In", "Instrumentals/Acapellas")
    assert is_acapella("Song", "A Cappella")


def test_music_is_not_acapella():
    # красное: без исключений сведение из диджейского микса и обычные треки тоже ушли бы из потока
    assert not is_acapella("It's Not Over (Influx UK's Dig Deeper Mix)", "Drum & Bass")
    assert not is_acapella("Music Is Everything [Danny Byrd Instrumental]' + Klute 'Part Of Me [Acapella]", "Drum & Bass")
    assert not is_acapella("Capella Grey", None)
    assert not is_acapella("Shine In (Instrumental)", "Instrumentals/Acapellas")
