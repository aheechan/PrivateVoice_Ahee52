"""方案 B 视觉系统：透明底标志 + 图标全家桶 + 安装器图形 + GitHub 社交卡。"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(r"C:\Local Repositories\PrivateVoice")
OUT = ROOT / "results" / "logo"
DIST = ROOT / "dist"
ASSETS = ROOT / "assets"

BG1, BG2 = (20, 16, 38), (54, 32, 90)
PURPLE, VIOLET, PINK = (139, 92, 246), (167, 139, 250), (236, 122, 214)
AMBER, WHITE = (255, 184, 107), (255, 255, 255)


def font(name: str, size: int):
    for path in (rf"C:\Windows\Fonts\{name}", r"C:\Windows\Fonts\segoeuib.ttf",
                 r"C:\Windows\Fonts\msyhbd.ttc"):
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    return ImageFont.load_default()


def gradient(size: tuple[int, int], c1=BG1, c2=BG2, angle=120) -> Image.Image:
    w, h = size
    y, x = (a.astype("float32") / 255 for a in np.mgrid[0:h, 0:w])
    a = math.radians(angle)
    t = np.clip(x * math.cos(a) + y * math.sin(a), 0, 1)
    arr = np.zeros((h, w, 3), "float32")
    for i in range(3):
        arr[..., i] = c1[i] + (c2[i] - c1[i]) * t
    return Image.fromarray(arr.astype("uint8"), "RGB")


def draw_mark(size: int = 1024) -> Image.Image:
    """透明底的气泡 + 声波标志（方案 B）。"""
    im = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    cx, cy = size * 0.5, size * 0.47
    w, h = size * 0.52, size * 0.40
    d.rounded_rectangle([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2],
                        radius=size * 0.11, fill=(*PURPLE, 255))
    d.polygon([(cx - w * 0.12, cy + h / 2 - size * 0.012),
               (cx - w * 0.02, cy + h / 2 + size * 0.115),
               (cx + w * 0.16, cy + h / 2 - size * 0.02)], fill=(*PURPLE, 255))

    gloss = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    gd = ImageDraw.Draw(gloss)
    gd.ellipse([cx - w * 0.72, cy - h * 1.02, cx + w * 0.72, cy + h * 0.50],
               fill=(*PINK, 72))
    gloss = gloss.filter(ImageFilter.GaussianBlur(size * 0.055))
    im = Image.alpha_composite(im, gloss)

    d = ImageDraw.Draw(im)
    heights = [0.10, 0.19, 0.26, 0.19, 0.10]
    for i, hh in enumerate(heights):
        bx = cx + (i - 2) * size * 0.062
        bw = size * 0.045
        d.rounded_rectangle([bx - bw / 2, cy - size * 0.005 - size * hh / 2,
                             bx + bw / 2, cy - size * 0.005 + size * hh / 2],
                            radius=bw / 2, fill=(255, 255, 255, 245))
    for i, alpha in enumerate((235, 140)):
        rr = size * 0.018
        px = cx + (i - 0.5) * size * 0.062
        py = cy + h * 0.34
        d.ellipse([px - rr, py - rr, px + rr, py + rr], fill=(255, 255, 255, alpha))
    return im


def rounded_bg(size: int, radius_ratio: float = 0.235) -> Image.Image:
    bg = gradient((size, size))
    mask = Image.new("L", (size, size), 0)
    d = ImageDraw.Draw(mask)
    d.rounded_rectangle([0, 0, size - 1, size - 1], radius=int(size * radius_ratio),
                        fill=255)
    out = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    out.paste(bg, (0, 0), mask)
    return out


def app_icon(size: int = 1024) -> Image.Image:
    canvas = rounded_bg(size)
    mark = draw_mark(int(size * 0.92)).resize((size, size), Image.LANCZOS)
    return Image.alpha_composite(canvas, mark).convert("RGB")


def export_icons() -> None:
    src = app_icon()
    for name, px in {"16": 16, "32": 32, "48": 48, "64": 64,
                     "128": 128, "256": 256}.items():
        src.resize((px, px), Image.LANCZOS).save(DIST / f"app_{name}.png")
    ico = [(256, 256), (128, 128), (64, 64), (48, 48), (32, 32), (16, 16)]
    src.resize((256, 256), Image.LANCZOS).save(DIST / "app.ico", sizes=ico)
    src.resize((256, 256), Image.LANCZOS).save(ASSETS / "app.ico", sizes=ico)
    src.resize((512, 512), Image.LANCZOS).save(OUT / "icon_b_512.png")
    print("icons exported")


def wizard_image() -> None:
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
                                 0, 255).astype("uint8"))
    mark = draw_mark(360).resize((176, 176), Image.LANCZOS)
    im = im.convert("RGBA")
    im.alpha_composite(mark, ((W - 176) // 2, 62))
    d = ImageDraw.Draw(im)
    d.text((W / 2, 266), "PrivateVoice_Ahee52", font=font("segoeuib.ttf", 27),
           fill=(240, 235, 255), anchor="ma")
    d.text((W / 2, 306), "本地 · 私有 · 陪伴", font=font("msyh.ttc", 21),
           fill=(178, 166, 210), anchor="ma")
    d.line([(W * 0.32, 348), (W * 0.68, 348)], fill=(118, 102, 168), width=2)
    d.text((W / 2, 372), "Your voice stays here.", font=font("segoeui.ttf", 17),
           fill=(150, 138, 184), anchor="ma")
    d.text((W / 2, H - 46), "v1.0.5", font=font("segoeui.ttf", 16),
           fill=(118, 108, 148), anchor="ma")
    im.convert("RGB").resize((164, 314), Image.LANCZOS).save(OUT / "installer_wizard.png")
    im.convert("RGB").save(ASSETS / "wizard_image.bmp")
    print("wizard image exported")


def wizard_small() -> None:
    im = Image.new("RGB", (256, 256), BG1)
    mark = draw_mark(256).resize((200, 200), Image.LANCZOS)
    im = im.convert("RGBA")
    im.alpha_composite(mark, (28, 28))
    im.convert("RGB").save(ASSETS / "wizard_small.bmp")
    im.convert("RGB").save(OUT / "installer_small.png")
    print("wizard small exported")


def social_card() -> None:
    W, H = 1280, 640
    im = gradient((W, H), (16, 12, 30), (58, 34, 96), angle=125)
    im = im.convert("RGBA")

    # 右侧极淡气泡水印
    wm = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    wd = ImageDraw.Draw(wm)
    wd.rounded_rectangle([820, 120, 1200, 380], radius=70, fill=(*VIOLET, 16))
    wd.polygon([(880, 380), (925, 480), (995, 380)], fill=(*VIOLET, 16))
    im = Image.alpha_composite(im, wm.filter(ImageFilter.GaussianBlur(3)))

    mark = draw_mark(420).resize((190, 190), Image.LANCZOS)
    im.alpha_composite(mark, (86, 92))
    d = ImageDraw.Draw(im)
    d.text((92, 306), "PrivateVoice_Ahee52", font=font("segoeuib.ttf", 60),
           fill=(244, 240, 255))
    d.text((94, 392), "Local. Private. Yours.", font=font("segoeui.ttf", 33),
           fill=(188, 174, 218))
    d.text((94, 450), "本地运行的中文情感陪伴器 · 语音克隆 · 全离线",
           font=font("msyh.ttc", 25), fill=(154, 142, 188))
    d.line([(96, 508), (152, 508)], fill=(*AMBER, 220), width=4)
    im.convert("RGB").save(OUT / "social_preview.png")
    print("social card exported")


def readme_header() -> None:
    W, H = 1280, 400
    im = gradient((W, H), (17, 13, 30), (52, 32, 88), angle=120).convert("RGBA")
    mark = draw_mark(420).resize((168, 168), Image.LANCZOS)
    im.alpha_composite(mark, (96, 116))
    d = ImageDraw.Draw(im)
    d.text((300, 128), "PrivateVoice_Ahee52", font=font("segoeuib.ttf", 66),
           fill=(244, 240, 255))
    d.text((304, 222), "本地运行的中文情感陪伴器", font=font("msyh.ttc", 30),
           fill=(186, 172, 216))
    d.text((304, 268), "Local · Private · Companion", font=font("segoeui.ttf", 24),
           fill=(150, 138, 184))
    im.convert("RGB").save(OUT / "readme_header.png")
    print("readme header exported")


if __name__ == "__main__":
    export_icons()
    wizard_image()
    wizard_small()
    social_card()
    readme_header()
