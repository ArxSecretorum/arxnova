"""Значок Arx Nova в семейном стиле Arx (как у Arx Player и Arx Notes): щит-крепость с
зубцами, сиреневый #6C7CE0 с тёмно-синей тенью #3A4593 на тёмном фоне #14161F; внутри —
четырёхлучевая звезда «нова». Геометрия щита — из ic_launcher_foreground.xml плеера."""
import math
import sys

from PIL import Image, ImageDraw

S = 1024
SC = 4 * 0.9 * 1.12  # вьюпорт 256 → 1024, масштаб группы 0.9; крупнее — щит читается в Доке
OX = S / 2 - 128 * SC
OY = S / 2 - 128 * SC + 8 * 4 * 0.9

SHIELD = [(73, 56), (95, 56), (95, 72), (117, 72), (117, 56), (139, 56), (139, 72), (161, 72),
          (161, 56), (183, 56), (183, 142), (128, 198), (73, 142)]


def pt(x, y, dx=0, dy=0):
    return (OX + (x + dx) * SC, OY + (y + dy) * SC)


img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
d = ImageDraw.Draw(img)
d.rounded_rectangle([60, 60, S - 60, S - 60], radius=200, fill=(20, 22, 31, 255))
d.polygon([pt(x, y, 3, 4) for x, y in SHIELD], fill=(58, 69, 147, 255))   # тень
d.polygon([pt(x, y) for x, y in SHIELD], fill=(108, 124, 224, 255))       # щит

# звезда «нова»: 4 луча, центр чуть выше середины щита
cx, cy = pt(128, 120)
R, r = 40 * SC, 9 * SC
star = []
for i in range(8):
    a = math.pi / 2 * (i // 2) + (math.pi / 4 if i % 2 else 0) - math.pi / 2
    rad = R if i % 2 == 0 else r
    star.append((cx + rad * math.cos(a), cy + rad * math.sin(a)))
d.polygon(star, fill=(231, 234, 247, 255))
out = sys.argv[1] if len(sys.argv) > 1 else "icon.png"
if out.endswith(".ico"):
    img.save(out, sizes=[(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
else:
    img.save(out)
