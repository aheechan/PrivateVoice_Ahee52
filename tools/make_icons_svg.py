"""Render the chat-wave SVG into the full visual set for 1.0.4."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(r"C:\Local Repositories\PrivateVoice")
SVG = Path(r"E:\微信缓存\xwechat_files\wxid_npr3b54oq4g922_c6a0\msg\file\2026-11\chat-wave.svg")
OUT = ROOT / "results" / "logo"
DIST = ROOT / "dist"
ASSETS = ROOT / "assets"
OUT.mkdir(parents=True, exist_ok=True)
ASSETS.mkdir(exist_ok=True)


def render_svg(size: int) -> Image.Image:
    import sys
    sys.path.insert(0, str(ROOT))
    from PySide6.QtCore import QByteArray, Qt
    from PySide6.QtGui import QImage, QPainter
    from PySide6.QtSvg import QSvgRenderer

    data = QByteArray(SVG.read_bytes())
    renderer = QSvgRenderer(data)
    img = QImage(size, size, QImage.Format_ARGB32)
    img.fill(Qt.transparent)
    p = QPainter(img)
    renderer.render(p)
    p.end()
    ptr = img.bits()
    arr = np.frombuffer(ptr, np.uint8, count=img.sizeInBytes()).reshape((size, size, 4))
    im = Image.fromarray(arr[:, :, [2, 1, 0, 3]].copy(), "RGBA")

    # SVG 自带浅色方形底：裁到紫色圆角块，再用圆角 alpha 去掉白边
    rgb = np.asarray(im.convert("RGB"), np.int16)
    mx, mn = rgb.max(axis=2), rgb.min(axis=2)
    strong = ((mx - mn) > 45) & (mx > 110)
    ys, xs = np.where(strong)
    if xs.size:
        x0, x1, y0, y1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
        side = max(x1 - x0 + 1, y1 - y0 + 1)
        cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
        box = (cx - side // 2, cy - side // 2, cx + side // 2, cy + side // 2)
        im = im.crop(box).resize((size, size), Image.LANCZOS)
        mask = Image.new("L", (size, size), 0)
        ImageDraw.Draw(mask).rounded_rectangle(
            [1, 1, size - 2, size - 2], radius=int(size * 0.225), fill=255)
        im.putalpha(mask)
    return im


def font(name: str, size: int):
    for p in (rf"C:\Windows\Fonts\{name}", r"C:\Windows\Fonts\segoeuib.ttf",
              r"C:\Windows\Fonts\msyh.ttc"):
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            continue
    return ImageFont.load_default()


def gradient(size, c1=(20, 16, 38), c2=(54, 32, 90), angle=120) -> Image.Image:
    w, h = size
    y, x = (a.astype("float32") / 255 for a in np.mgrid[0:h, 0:w])
    a = math.radians(angle)
    t = np.clip(x * math.cos(a) + y * math.sin(a), 0, 1)
    arr = np.zeros((h, w, 3), "float32")
    for i in range(3):
        arr[..., i] = c1[i] + (c2[i] - c1[i]) * t
    return Image.fromarray(arr.astype("uint8"), "RGB")


def export_icons(master: Image.Image) -> None:
    sizes = {"16": 16, "32": 32, "48": 48, "64": 64, "128": 128, "256": 256}
    for name, px in sizes.items():
        master.resize((px, px), Image.LANCZOS).save(DIST / f"app_{name}.png")
        master.resize((px, px), Image.LANCZOS).save(ASSETS / f"app_{name}.png")
    ico = [(256, 256), (128, 128), (64, 64), (48, 48), (32, 32), (16, 16)]
    master.resize((256, 256), Image.LANCZOS).save(DIST / "app.ico", sizes=ico)
    master.resize((256, 256), Image.LANCZOS).save(ASSETS / "app.ico", sizes=ico)
    master.save(OUT / "icon_chatwave_1024.png")
    print("icons exported")


def wizard(master: Image.Image) -> None:
    W, H = 328, 628
    im = gradient((W, H), (18, 14, 34), (48, 28, 84), angle=105)
    glow = Image.new("RGB", (W, H), (0, 0, 0))
    gd = ImageDraw.Draw(glow)
    for i in range(30, 0, -1):
        f = i / 30
        r = 250 * f
        gd.ellipse([W / 2 - r, 150 - r, W / 2 + r, 150 + r],
                   fill=(int(72 * (1 - f) ** 2), int(42 * (1 - f) ** 2),
                         int(122 * (1 - f) ** 2)))
    glow = glow.filter(ImageFilter.GaussianBlur(38))
    im = Image.fromarray(np.clip(np.asarray(im, np.int16) + np.asarray(glow, np.int16),
                                 0, 255).astype("uint8")).convert("RGBA")
    mark = master.resize((188, 188), Image.LANCZOS)
    im.alpha_composite(mark, ((W - 188) // 2, 58))
    d = ImageDraw.Draw(im)
    d.text((W / 2, 272), "PrivateVoice_Ahee52", font=font("segoeuib.ttf", 27),
           fill=(240, 235, 255), anchor="ma")
    d.line([(W * 0.32, 320), (W * 0.68, 320)], fill=(118, 102, 168), width=2)
    d.text((W / 2, 344), "Private voice stays here.", font=font("segoeui.ttf", 17),
           fill=(150, 138, 184), anchor="ma")
    d.text((W / 2, H - 46), "v1.0.4", font=font("segoeui.ttf", 16),
           fill=(118, 108, 148), anchor="ma")
    im.convert("RGB").resize((164, 314), Image.LANCZOS).save(OUT / "installer_wizard.png")
    im.convert("RGB").save(ASSETS / "wizard_image.bmp")
    print("wizard art exported")


def wizard_small(master: Image.Image) -> None:
    im = Image.new("RGBA", (256, 256), (0, 0, 0, 0))
    im.alpha_composite(master.resize((220, 220), Image.LANCZOS), (18, 18))
    im.convert("RGB").save(ASSETS / "wizard_small.bmp")
    print("wizard small exported")


def social(master: Image.Image) -> None:
    W, H = 1280, 640
    im = gradient((W, H), (16, 12, 30), (58, 34, 96), angle=125).convert("RGBA")
    mark = master.resize((196, 196), Image.LANCZOS)
    im.alpha_composite(mark, (84, 88))
    d = ImageDraw.Draw(im)
    d.text((88, 306), "PrivateVoice_Ahee52", font=font("segoeuib.ttf", 60),
           fill=(244, 240, 255))
    d.text((90, 392), "Local. Private. Yours.", font=font("segoeui.ttf", 33),
           fill=(188, 174, 218))
    d.text((90, 450), "本地运行的中文情感陪伴器 · 语音克隆 · 全离线",
           font=font("msyh.ttc", 25), fill=(154, 142, 188))
    im.convert("RGB").save(OUT / "social_preview.png")
    print("social exported")


def header(master: Image.Image) -> None:
    W, H = 1280, 400
    im = gradient((W, H), (17, 13, 30), (52, 32, 88), angle=120).convert("RGBA")
    mark = master.resize((168, 168), Image.LANCZOS)
    im.alpha_composite(mark, (96, 116))
    d = ImageDraw.Draw(im)
    d.text((300, 128), "PrivateVoice_Ahee52", font=font("segoeuib.ttf", 66),
           fill=(244, 240, 255))
    d.text((304, 222), "本地运行的中文情感陪伴器", font=font("msyh.ttc", 30),
           fill=(186, 172, 216))
    d.text((304, 268), "Local · Private · Companion", font=font("segoeui.ttf", 24),
           fill=(150, 138, 184))
    im.convert("RGB").save(OUT / "readme_header.png")
    print("header exported")


if __name__ == "__main__":
    master = render_svg(1024)
    export_icons(master)
    wizard(master)
    wizard_small(master)
    social(master)
    header(master)
