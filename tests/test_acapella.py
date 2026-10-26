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


def test_dnb_tempo_fix_only_for_dnb_tempo_and_tag():
    import numpy as np

    from arxnova.nova import _dnb_tempo_fix

    labels = ["Electronic---Drum n Bass", "Electronic---Chiptune", "Electronic---House", "Electronic---Ambient"]
    ids = np.array([1, 2, 3, 4])
    styles = np.array([
        [0.3, 0.6, 0.05, 0.05],  # 175 BPM, тег DnB, модель: Chiptune — на самом деле DnB
        [0.1, 0.1, 0.7, 0.1],    # 128 BPM: хаус-ремикс на альбоме DnB — не трогаем
        [0.3, 0.6, 0.05, 0.05],  # 175 BPM, но тег — не DnB — не трогаем
        [0.01, 0.4, 0.3, 0.29],  # 175 BPM, тег DnB, но DnB у модели последний — не трогаем
    ], np.float32)
    top = styles.argmax(axis=1)
    got = _dnb_tempo_fix(ids, styles, top, labels, {1: 175, 2: 128, 3: 175, 4: 175},
                         {1: "Drum & Bass", 2: "Drum & Bass", 3: "Electronic", 4: "Drum & Bass"})
    assert list(got) == [0, 2, 1, 1]
