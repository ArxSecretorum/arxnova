"""Значок Arx Nova в семейном стиле Arx. Основа — icon_1024.png: плитка, тень и башня взяты один
в один из значка Arx Player (те же размеры, цвета, «сквиркл» macOS 824/1024 и мягкая тень), внутри
вместо знака плеера — четырёхлучевая звезда «нова». Скрипт только выдаёт нужный формат."""
import sys

from PIL import Image

img = Image.open(__file__.rsplit("/", 1)[0] + "/icon_1024.png" if "/" in __file__ else "icon_1024.png").convert("RGBA")
out = sys.argv[1] if len(sys.argv) > 1 else "icon.png"
if out.endswith(".ico"):
    img.save(out, sizes=[(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
else:
    img.save(out)
