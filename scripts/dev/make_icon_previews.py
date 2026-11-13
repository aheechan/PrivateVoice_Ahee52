"""生成桌面/任务栏图标预览（用于验收视觉效果，不依赖真实安装）。"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(r"C:\Local Repositories\PrivateVoice")
OUT = ROOT / "results" / "logo"
ICON = ROOT / "dist" / "app_256.png"


def font(name: str, size: int):
    for p in (rf"C:\Windows\Fonts\{name}", r"C:\Windows\Fonts\segoeuib.ttf",
              r"C:\Windows\Fonts\msyh.ttc"):
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            continue
    return ImageFont.load_default()


def wallpaper(w: int, h: int) -> Image.Image:
    y, x = (a.astype("float32") / 255 for a in np.mgrid[0:h, 0:w])
    arr = np.zeros((h, w, 3), "float32")
    for i, (c1, c2) in enumerate(zip((26, 22, 44), (74, 52, 120))):
        arr[..., i] = c1 + (c2 - c1) * np.clip(x * 0.7 + y * 0.5, 0, 1)
    im = Image.fromarray(arr.astype("uint8"))
    glow = Image.new("RGB", (w, h), (0, 0, 0))
    d = ImageDraw.Draw(glow)
    for i in range(40, 0, -1):
        f = i / 40
        r = w * 0.55 * f
        d.ellipse([w * 0.72 - r, h * 0.28 - r, w * 0.72 + r, h * 0.28 + r],
                  fill=(int(90 * (1 - f) ** 2), int(50 * (1 - f) ** 2),
                        int(140 * (1 - f) ** 2)))
    glow = glow.filter(ImageFilter.GaussianBlur(w * 0.03))
    return Image.fromarray(np.clip(np.asarray(im, np.int16) + np.asarray(glow, np.int16),
                                   0, 255).astype("uint8"))


def desktop_preview() -> None:
    W, H = 1280, 800
    im = wallpaper(W, H).convert("RGBA")
    d = ImageDraw.Draw(im, "RGBA")

    icon = Image.open(ICON).convert("RGBA")
    # 桌面图标 64px（真实桌面是 48/64）
    small = icon.resize((64, 64), Image.LANCZOS)
    im.alpha_composite(small, (64, 64))
    d.text((64, 140), "PrivateVoice_Ahee52", font=font("msyh.ttc", 16),
           fill=(240, 236, 255))
    # 选中框（只画描边，不覆盖图标）
    d.rounded_rectangle([58, 58, 136, 166], radius=6, outline=(120, 170, 255, 170),
                        width=2)

    # 任务栏
    bar_h = 56
    d.rectangle([0, H - bar_h, W, H], fill=(18, 15, 30, 235))
    d.line([(0, H - bar_h), (W, H - bar_h)], fill=(255, 255, 255, 26), width=1)
    start = Image.new("RGBA", (26, 26), (0, 0, 0, 0))
    sd = ImageDraw.Draw(start)
    for i in range(4):
        for j in range(4):
            sd.rectangle([2 + i * 6, 2 + j * 6, 5 + i * 6, 5 + j * 6],
                         fill=(150, 170, 210, 220))
    im.alpha_composite(start, (18, H - bar_h + 15))

    task_icon = icon.resize((36, 36), Image.LANCZOS)
    im.alpha_composite(task_icon, (72, H - bar_h + 10))
    # 任务栏激活指示条
    d.rounded_rectangle([84, H - 6, 112, H - 2], radius=2, fill=(139, 92, 246, 235))
    im.convert("RGB").save(OUT / "preview_desktop.png")
    print("desktop preview saved")


def sizes_preview() -> None:
    W, H = 900, 260
    im = Image.new("RGB", (W, H), (17, 14, 30))
    d = ImageDraw.Draw(im)
    icon = Image.open(ICON).convert("RGB")
    x = 40
    for px in (256, 128, 64, 48, 32, 16):
        ic = icon.resize((px, px), Image.LANCZOS)
        im.paste(ic, (x, 60))
        d.text((x, 26), f"{px}px", font=font("segoeui.ttf", 18), fill=(190, 178, 220))
        x += px + 44
    im.save(OUT / "preview_icon_sizes.png")
    print("sizes preview saved")


if __name__ == "__main__":
    desktop_preview()
    sizes_preview()
