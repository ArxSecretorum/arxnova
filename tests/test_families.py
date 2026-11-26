"""Семейства стилей: порядок карусели и «родственные» стили потока. Таблица сверена с Discogs и MusicBrainz 25.11.2026
(решение 0023): семейство нашего стиля совпадает с семейством стилей эталона у 98,3 % синглов и 96,6 % треков альбомов
(было 96,0 и 92,7 %); в «прочее» на медиатеке владельца попадало 183 трека, стало 3."""

from arxnova.nova import style_family


def test_new_classical_sits_with_ambient():
    """Ólafur Arnalds, Max Richter, Nils Frahm: одни и те же треки в базах то Ambient, то Modern Classical, то
    Contemporary, то Neo-Classical (на Discogs у 35 % изданий Modern Classical стоит и Ambient) — одно семейство."""
    fam = style_family("Electronic", "Ambient")
    assert fam == "chill"
    assert style_family("Electronic", "Modern Classical") == fam
    assert style_family("Classical", "Contemporary Classical") == fam
    assert style_family("Classical", "Neo-Classical") == fam
    assert style_family("Classical", "Baroque") == "classical"  # остальная классика — своим семейством
    for style in ("Drone", "Dark Ambient", "New Age", "Tribal", "Abstract", "Glitch", "Berlin-School"):
        assert style_family("Electronic", style) == "chill", style


def test_house_techno_trance_by_name():
    """Goa Trance, Neo Trance, Euro House, Tribal House раньше были «прочим»: в семействе хауса стояли только перечисленные."""
    for style in ("Goa Trance", "Neo Trance", "Euro House", "Tribal House", "Acid House", "Dub Techno", "Psy-Trance", "Techno"):
        assert style_family("Electronic", style) == "house", style
    assert style_family("Electronic", "Ambient House") == "chill"  # названо явно — правило по имени не перебивает


def test_neighbours_across_discogs_genres():
    """Соседи, которых Discogs разносит по разным жанрам: Folk Rock (Rock) рядом с Folk, Darkwave (Electronic) — с
    Ethereal и Goth Rock (Rock), Synth-pop (Electronic) — с поп-музыкой."""
    assert style_family("Rock", "Folk Rock") == style_family("Folk, World, & Country", "Folk") == "folk"
    assert style_family("Electronic", "Neofolk") == "folk"
    assert style_family("Electronic", "Darkwave") == style_family("Rock", "Ethereal") == style_family("Rock", "Goth Rock") == "rock"
    assert style_family("Electronic", "Synth-pop") == style_family("Pop", "Indie Pop") == "pop"
    assert style_family("Electronic", "Hip Hop") == "hiphop"
    assert style_family("Electronic", "Drum n Bass") == "dnb" and style_family("Electronic", "Dubstep") == "bass"
