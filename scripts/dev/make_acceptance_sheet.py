"""把所有视觉验收图拼成一张总览。"""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

OUT = Path(r"C:\Local Repositories\PrivateVoice\results\logo")


def font(name: str, size: int):
    for p in (rf"C:\Windows\Fonts\{name}", r"C:\Windows\Fonts\segoeuib.ttf",
              r"C:\Windows\Fonts\msyh.ttc"):
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            continue
    return ImageFont.load_default()


def main() -> None:
    items = [
        ("安装向导（合成预览）", "preview_installer.png", 640),
        ("主界面 + 标题栏图标", "preview_app_window.png", 560),
        ("桌面 / 任务栏", "preview_desktop.png", 640),
        ("GitHub 社交预览 1280×640", "social_preview.png", 640),
        ("README 头图", "readme_header.png", 640),
        ("图标尺寸 256→16", "preview_icon_sizes.png", 640),
        ("设置 · 语音识别（中/英 ASR）", "preview_settings_asr.png", 470),
    ]
    pad, label_h, cols = 34, 58, 2
    rows = (len(items) + 1) // cols
    cell_w = 660
    cell_h = 520
    W = pad + cols * (cell_w + pad)
    H = pad + rows * (cell_h + label_h + pad)
    sheet = Image.new("RGB", (W, H), (12, 10, 22))
    d = ImageDraw.Draw(sheet)
    f = font("msyh.ttc", 22)

    for idx, (title, fname, target_w) in enumerate(items):
        r, c = divmod(idx, cols)
        x = pad + c * (cell_w + pad)
        y = pad + r * (cell_h + label_h + pad)
        p = OUT / fname
        if not p.exists():
            continue
        im = Image.open(p).convert("RGB")
        scale = min(target_w / im.width, cell_h / im.height)
        im = im.resize((int(im.width * scale), int(im.height * scale)), Image.LANCZOS)
        bg = Image.new("RGB", (cell_w, cell_h), (20, 16, 34))
        bg.paste(im, ((cell_w - im.width) // 2, (cell_h - im.height) // 2))
        sheet.paste(bg, (x, y))
        d.text((x + 2, y + cell_h + 14), title, font=f, fill=(226, 219, 245))

    sheet.save(OUT / "acceptance_sheet.png")
    print("sheet saved", sheet.size)


if __name__ == "__main__":
    main()
