"""生成三个方案的水平 lockup（图标 + 字标），用于 README / 网站。"""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

OUT = Path(r"C:\Local Repositories\PrivateVoice\results\logo")
CONCEPTS = {
    "a": "concept_a_wave_smile.png",
    "b": "concept_b_bubble.png",
    "c": "concept_c_embrace.png",
}
W, H = 1600, 500
TEXT_MAIN = (243, 240, 255)
TEXT_SUB = (168, 156, 200)


def font(name: str, size: int) -> ImageFont.FreeTypeFont:
    for path in (rf"C:\Windows\Fonts\{name}", r"C:\Windows\Fonts\segoeuib.ttf",
                 r"C:\Windows\Fonts\msyhbd.ttc"):
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    return ImageFont.load_default()


def lockup(key: str, icon_file: str) -> None:
    canvas = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    icon = Image.open(OUT / icon_file).convert("RGBA").resize((300, 300), Image.LANCZOS)
    canvas.alpha_composite(icon, (40, (H - 300) // 2))

    d = ImageDraw.Draw(canvas)
    f_main = font("segoeuib.ttf", 104)
    f_sub = font("segoeui.ttf", 38)
    x = 400
    d.text((x, 150), "PrivateVoice", font=f_main, fill=TEXT_MAIN)
    # 字距拉开的副标
    sub = "A H E E 5 2"
    d.text((x + 6, 285), sub, font=f_sub, fill=TEXT_SUB)
    canvas.save(OUT / f"lockup_{key}.png")

    # 深底版本，方便直接放进 README
    dark = Image.new("RGBA", (W, H), (17, 13, 30, 255))
    dark.alpha_composite(canvas)
    dark.convert("RGB").save(OUT / f"lockup_{key}_dark.png")
    print("lockup", key)


if __name__ == "__main__":
    for k, f in CONCEPTS.items():
        lockup(k, f)
