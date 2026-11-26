"""Стиль трека по совокупности свидетельств (решение 0023): звук трека, альбом, исполнитель, похожие треки исполнителя,
тег, MusicBrainz. Веса подобраны по эталону Discogs на медиатеке владельца и проверены на исполнителях вне обучения."""

import numpy as np

from arxnova import style_evidence as SE

LABELS = ["Electronic---Drum n Bass", "Electronic---Downtempo", "Electronic---House", "Electronic---Ambient"]
DNB, DOWN, HOUSE, AMB = range(4)


def _lib(tracks):
    """tracks: (исполнитель, альбом, название, активации, тег, сборник, вектор микса) → входы decide."""
    ids = np.arange(1, len(tracks) + 1)
    meta = {int(i): {"artist": t[0], "albumartist": None, "album": t[1], "title": t[2], "genre_raw": t[4], "compilation": t[5]}
            for i, t in zip(ids, tracks)}
    styles = np.array([t[3] for t in tracks], np.float32)
    mix = np.array([t[6] for t in tracks], np.float32)
    return ids, styles, meta, mix


def _t(artist, album, title, acts, tag=None, comp=0, mix=(1.0, 0.0)):
    return (artist, album, title, acts, tag, comp, mix)


SURE_DNB, SURE_HOUSE, SURE_AMB = [0.9, 0.02, 0.02, 0.02], [0.02, 0.02, 0.9, 0.02], [0.02, 0.02, 0.02, 0.8]


def test_album_and_artist_pull_unsure_track():
    """Трек, который модель слышит неуверенно (Downtempo 0,35 / DnB 0,30), на альбоме из DnB — DnB: так у 34 треков с
    эталоном DnB сервер говорил Downtempo. Тот же звук без альбома и исполнителя остаётся Downtempo."""
    unsure = [0.30, 0.35, 0.02, 0.02]
    ids, styles, meta, mix = _lib([_t("Logistics", "Fear Not", f"t{i}", SURE_DNB) for i in range(3)]
                                  + [_t("Logistics", "Fear Not", "Running Late", unsure), _t("Someone", None, "Alone", unsure)])
    top = SE.decide(ids, styles, LABELS, meta, mix)
    assert list(top) == [DNB, DNB, DNB, DNB, DOWN]


def test_compilation_is_not_album_context():
    """Сборник — не альбом: соседние треки других исполнителей о треке ничего не говорят."""
    unsure = [0.30, 0.35, 0.02, 0.02]
    ids, styles, meta, mix = _lib([_t(f"Artist {i}", "Hospitality 2015", f"t{i}", SURE_DNB, comp=1) for i in range(3)]
                                  + [_t("Other", "Hospitality 2015", "Slow One", unsure, comp=1)])
    assert SE.decide(ids, styles, LABELS, meta, mix)[3] == DOWN
    # «Various Artists» в поле исполнителя — не исполнитель: такие треки друг другу не свидетели
    ids, styles, meta, mix = _lib([_t("Various Artists", None, f"t{i}", SURE_DNB) for i in range(4)]
                                  + [_t("Various Artists", None, "Slow One", unsure)])
    assert SE.decide(ids, styles, LABELS, meta, mix)[4] == DOWN


def test_remix_follows_the_remixer():
    """Ремикс звучит как автор ремикса, а не как исполнитель оригинала: «Ready for Your Love (Etherwood Remix)» у хаусных
    Gorgon City — DnB (на Discogs у сингла стоит House — стиль оригинала). Альбом оригинала ремиксу не свидетель."""
    unsure = [0.35, 0.02, 0.40, 0.02]
    ids, styles, meta, mix = _lib([_t("Etherwood", None, f"e{i}", SURE_DNB) for i in range(3)]
                                  + [_t("Gorgon City", "Sirens", f"g{i}", SURE_HOUSE) for i in range(3)]
                                  + [_t("Gorgon City", "Sirens", "Ready for Your Love (Etherwood Remix)", unsure),
                                     _t("Gorgon City", "Sirens", "Imagination", unsure)])
    top = SE.decide(ids, styles, LABELS, meta, mix)
    assert top[6] == DNB and top[7] == HOUSE
    assert SE.remixers("A (Calibre & High Contrast Remix)") == {"calibre", "high contrast"}
    assert SE.remixers("Song (Radio Edit)") == set() and SE.remixers("X (Original Mix)") == set()


def test_remix_by_unknown_remixer_keeps_context_of_the_original():
    """Автора ремикса в медиатеке нет (Banco de Gaia «Kuos (Bluetech Remix)») — свидетели те же, что у оригинала: его
    альбом и исполнитель. Без них ремикс оставался Deep House по одному звуку; у 125 таких ремиксов с эталоном совпадение
    было 82 %, с контекстом оригинала — 94 %."""
    unsure = [0.02, 0.05, 0.32, 0.12]
    ids, styles, meta, mix = _lib([_t("Banco de Gaia", "Last Train To Lhasa", f"t{i}", SURE_AMB) for i in range(3)]
                                  + [_t("Banco de Gaia", "Last Train To Lhasa", "Kuos (Bluetech Remix)", unsure)])
    assert SE.decide(ids, styles, LABELS, meta, mix)[3] == AMB


def test_tag_supports_but_does_not_override_sure_sound():
    """Тег файла — свидетельство, а не приказ: неуверенному звуку помогает, уверенный не перебивает."""
    ids, styles, meta, mix = _lib([_t("A", None, "a", [0.02, 0.30, 0.40, 0.02], tag="Downtempo"),
                                   _t("B", None, "b", [0.02, 0.10, 0.90, 0.02], tag="Downtempo"),
                                   _t("C", None, "c", [0.30, 0.02, 0.40, 0.02], tag="Drum & Bass")])
    assert list(SE.decide(ids, styles, LABELS, meta, mix)) == [DOWN, HOUSE, DNB]


def test_genre_word_in_tag_supports_styles_of_that_genre():
    """В теге часто жанр, а не стиль: «Jazz» у Anouar Brahem. Он поддерживает все стили этого жанра — Folk 0,27 против
    Contemporary Jazz 0,22 решается в пользу джаза (на эталоне Discogs у него было 19 ошибок из 42, стало 8)."""
    labels = ["Folk, World, & Country---Folk", "Jazz---Contemporary Jazz", "Electronic---Ambient"]
    ids, styles, meta, mix = _lib([_t("Anouar Brahem", None, "L'aube", [0.27, 0.22, 0.05], tag="Jazz"),
                                   _t("Someone", None, "x", [0.27, 0.22, 0.05], tag=None),
                                   _t("Other", None, "y", [0.60, 0.05, 0.05], tag="Jazz")])
    assert list(SE.decide(ids, styles, labels, meta, mix)) == [1, 0, 0]
    assert SE.tag_genres("ECM Style Jazz") == {"Jazz"} and SE.tag_genres("エレクトロニック") == {"Electronic"}
    assert SE.tag_genres("Drum & Bass") == set() and SE.tag_genres(None) == set()


def test_similar_tracks_of_a_many_sided_artist():
    """У исполнителя две стороны (Calibre: DnB и эмбиент): неуверенный трек сверяется с теми его треками, что звучат
    похоже (вектор микса), а не со всем каталогом сразу."""
    unsure = [0.32, 0.02, 0.02, 0.30]
    tracks = [_t("Calibre", None, f"a{i}", SURE_AMB, mix=(1.0, 0.02 * i)) for i in range(12)]
    tracks += [_t("Calibre", None, f"d{i}", SURE_DNB, mix=(0.02 * i, 1.0)) for i in range(12)]
    tracks += [_t("Calibre", None, "Hills", unsure, mix=(1.0, 0.1)), _t("Calibre", None, "Roller", unsure, mix=(0.1, 1.0))]
    ids, styles, meta, mix = _lib(tracks)
    top = SE.decide(ids, styles, LABELS, meta, mix)
    assert top[24] == AMB and top[25] == DNB


def test_musicbrainz_genres_are_evidence_when_present():
    """Жанры MusicBrainz, если сервер их спрашивал, — ещё одно свидетельство (на эталоне Discogs +1 % совпадений)."""
    ids, styles, meta, mix = _lib([_t("A", None, "a", [0.30, 0.02, 0.40, 0.02])])
    assert SE.decide(ids, styles, LABELS, meta, mix)[0] == HOUSE
    assert SE.decide(ids, styles, LABELS, meta, mix, mb={1: {"drum and bass": 3}})[0] == DNB


def test_musicbrainz_only_when_plausible_for_the_model():
    """London Elektricity Big Band «Remember The Future (Live)»: оценки 0,08–0,12, первым Neo Soul, в MusicBrainz у
    исполнителя «drum and bass» — DnB (владелец 23.10.2026). Жанр, который модель почти исключает («doom metal» у
    соула: ложное совпадение альбома), не проходит; уверенный звук MusicBrainz не перебивает."""
    labels = ["Electronic---Drum n Bass", "Funk / Soul---Neo Soul", "Jazz---Big Band", "Rock---Doom Metal"]
    ids, styles, meta, mix = _lib([_t("Big Band", None, "a", [0.08, 0.12, 0.03, 0.0]),
                                   _t("Sure", None, "b", [0.90, 0.01, 0.01, 0.0]),
                                   _t("Wrong", None, "c", [0.01, 0.15, 0.01, 0.001])])
    mb = {1: {"big band": 1, "drum and bass": 1}, 2: {"rock": 5, "neo soul": 1}, 3: {"doom metal": 5}}
    assert list(SE.decide(ids, styles, labels, meta, mix, mb=mb)) == [0, 0, 1]


FAM_LABELS = ["Electronic---Ambient", "Electronic---Downtempo", "Electronic---House", "Electronic---Trance",
              "Electronic---Drum n Bass", "Electronic---Dubstep"]
FAMILY = ["chill", "chill", "house", "house", "dnb", "bass"]
F_AMB, F_DOWN, F_HOUSE, F_TRANCE, F_DNB, F_DUB = range(6)


def _album(tracks_acts, bpms, mixes=None):
    """Один альбом одного исполнителя: активации и темп по трекам."""
    tracks = [_t("Banco de Gaia", "Last Train To Lhasa", f"t{i}", a) for i, a in enumerate(tracks_acts)]
    ids, styles, meta, mix = _lib(tracks)
    return ids, styles, meta, mix, {int(i): b for i, b in zip(ids, bpms)}


def test_track_that_stands_out_of_its_album_keeps_its_own_family():
    """Banco de Gaia «Kuos (Gnomes Mix)»: альбом — эмбиент, трек — танцевальный, 129 BPM, модель слышит House 0,42 и Trance
    0,25, а Ambient — 0,10. Раньше альбом перевешивал, и трек становился Ambient (владелец 25.11.2026: «Ambient — это
    правильно?»). Звук уверен, темп подходит его семейству и не подходит семейству альбома — верим треку. Так же
    Logistics «Raygun» (126 BPM, House 0,95) на альбоме DnB."""
    amb = [0.8, 0.5, 0.02, 0.02, 0.02, 0.02]
    kuos = [0.10, 0.08, 0.42, 0.25, 0.02, 0.02]
    ids, styles, meta, mix, bpm = _album([amb] * 5 + [kuos], [95, 90, 100, 85, 92, 129])
    assert SE.decide(ids, styles, FAM_LABELS, meta, mix)[5] == F_AMB                       # без семейств и темпа — как альбом
    top = SE.decide(ids, styles, FAM_LABELS, meta, mix, family=FAMILY, bpm=bpm)
    assert list(top[:5]) == [F_AMB] * 5 and top[5] == F_HOUSE
    # темп не подходит семейству звука (трек на 95 BPM не хаус) или подходит и альбому — остаётся с альбомом
    for slow in (95, None):
        b = dict(bpm)
        b[6] = slow
        assert SE.decide(ids, styles, FAM_LABELS, meta, mix, family=FAMILY, bpm=b)[5] == F_AMB
    # звук не уверен: своё семейство не вдвое сильнее семейства альбома
    weak = [0.30, 0.08, 0.42, 0.25, 0.02, 0.02]
    ids, styles, meta, mix, bpm = _album([amb] * 5 + [weak], [95, 90, 100, 85, 92, 129])
    assert SE.decide(ids, styles, FAM_LABELS, meta, mix, family=FAMILY, bpm=bpm)[5] == F_AMB


def test_dnb_heard_as_chill_stays_dnb_because_of_tempo():
    """Обратный случай: атмосферный DnB, который модель слышит как Downtempo, на альбоме DnB. Темп 174 подходит DnB и не
    подходит спокойной музыке — контекст прав (на эталоне синглов в таких спорах он прав 10 раз, звук — 2)."""
    dnb = [0.02, 0.02, 0.02, 0.02, 0.9, 0.05]
    soft = [0.2, 0.6, 0.02, 0.02, 0.15, 0.02]
    for tempo in (174, 87):  # и половинный темп: 87 подходит и DnB, и спокойной музыке — спор не решён, остаётся контекст
        ids, styles, meta, mix, bpm = _album([dnb] * 5 + [soft], [174, 172, 175, 174, 173, tempo])
        assert SE.decide(ids, styles, FAM_LABELS, meta, mix, family=FAMILY, bpm=bpm)[5] == F_DNB
    # а дабстеп на 140 на том же альбоме — выбивается: темп подходит басовой музыке и не подходит DnB
    dub = [0.02, 0.02, 0.02, 0.02, 0.2, 0.8]
    ids, styles, meta, mix, bpm = _album([dnb] * 5 + [dub], [174, 172, 175, 174, 173, 140])
    assert SE.decide(ids, styles, FAM_LABELS, meta, mix, family=FAMILY, bpm=bpm)[5] == F_DUB


def _clap_sign(z: float) -> np.ndarray:
    """Единичный вектор CLAP с заданной приметой «звук транса против звука хауса» (в долях разброса)."""
    v = SE.cue_vectors()
    d = v[:len(SE.TRANCE_TEXTS)].mean(axis=0) - v[len(SE.TRANCE_TEXTS):].mean(axis=0)
    base = np.random.default_rng(0).normal(size=d.shape).astype(np.float32)
    lo, hi = -20.0, 20.0
    for _ in range(60):
        mid = (lo + hi) / 2
        x = base + mid * d
        x = (x / np.linalg.norm(x)).astype(np.float32)
        if float(SE.trance_sign(x[None])[0]) < z:
            lo = mid
        else:
            hi = mid
    return x


def test_acid_sound_picks_trance_over_house_inside_the_dance_family():
    """Внутри танцевального семейства «хаус или транс» решают вместе звук модели и описательная примета CLAP: кислотные
    линии 303, скрип фильтра, психоделический лид, суперсо против соул-вокала, диско и хаус-пиано (владелец 25.11.2026:
    «там же кислотные эти хлюпанья трансовые»). У «Kuos (Gnomes Mix)» примета +1,4…+1,5 разброса — Trance; у трека без
    неё — House."""
    amb = [0.8, 0.5, 0.02, 0.02, 0.02, 0.02]
    kuos = [0.10, 0.08, 0.42, 0.25, 0.02, 0.02]
    ids, styles, meta, mix, bpm = _album([amb] * 5 + [kuos], [95, 90, 100, 85, 92, 129])
    acid = np.stack([_clap_sign(0.0)] * 5 + [_clap_sign(1.5)])
    plain = np.stack([_clap_sign(0.0)] * 6)
    assert SE.decide(ids, styles, FAM_LABELS, meta, mix, family=FAMILY, bpm=bpm, clap=acid)[5] == F_TRANCE
    assert SE.decide(ids, styles, FAM_LABELS, meta, mix, family=FAMILY, bpm=bpm, clap=plain)[5] == F_HOUSE
    assert abs(float(SE.trance_sign(acid)[5]) - 1.5) < 0.05 and abs(float(SE.trance_sign(plain)[0])) < 0.05
