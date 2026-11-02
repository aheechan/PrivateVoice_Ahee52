"""角色库：CRUD、状态机、.vcp 导入导出。

目录约定：
    characters/index.json
    characters/<id>/profile.json / avatar.png / dataset/ / checkpoints/
                    reference.wav / reference.txt
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import time
import uuid
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

VCP_VERSION = 1
VCP_MAGIC = "voicelab.character.vcp"
STATUSES = ("draft", "training", "ready", "failed")

_REQUIRED_VCP = ("manifest.json", "profile.json", "reference.wav")


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _slug(name: str) -> str:
    s = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]+", "-", name.strip()).strip("-").lower()
    return s[:24] or "char"


def _sha256_dir(path: Path) -> str:
    h = hashlib.sha256()
    if not path.exists():
        return ""
    for f in sorted(p for p in path.rglob("*") if p.is_file()):
        h.update(f.name.encode("utf-8"))
        with f.open("rb") as fh:
            for block in iter(lambda: fh.read(1 << 20), b""):
                h.update(block)
    return h.hexdigest()


@dataclass
class Character:
    id: str
    name: str
    avatar: str = ""
    persona: str = ""
    system_prompt_ref: str = "prompts/companion_system.txt"
    greeting: str = ""
    llm: dict[str, Any] = field(default_factory=lambda: {"temperature": 0.8, "max_tokens": 256})
    voice: dict[str, Any] = field(default_factory=lambda: {
        "mode": "finetune", "engine": "gpt-sovits-v3",
        "gpt_ckpt": "", "sovits_ckpt": "", "reference": "", "reference_text": "",
    })
    dataset: dict[str, Any] = field(default_factory=lambda: {
        "duration_sec": 0.0, "clips": 0, "sha256": "",
    })
    status: str = "draft"
    created_at: str = field(default_factory=_now)
    last_used: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Character":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in known})


class CharacterStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.index_path = self.root / "index.json"
        self._index: dict[str, Any] = self._load_index()

    # -- index ---------------------------------------------------------
    def _load_index(self) -> dict[str, Any]:
        if self.index_path.exists():
            try:
                data = json.loads(self.index_path.read_text(encoding="utf-8"))
                data.setdefault("characters", [])
                return data
            except Exception:
                pass
        return {"version": 1, "characters": [], "active_id": ""}

    def _save_index(self) -> None:
        self.index_path.write_text(
            json.dumps(self._index, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    @property
    def active_id(self) -> str:
        return self._index.get("active_id") or ""

    def set_active(self, cid: str) -> None:
        if cid and not (self.root / cid).exists():
            raise KeyError(cid)
        self._index["active_id"] = cid
        if cid:
            self.touch(cid)
        self._save_index()

    # -- 查询 ----------------------------------------------------------
    def dir_of(self, cid: str) -> Path:
        return self.root / cid

    def exists(self, cid: str) -> bool:
        return (self.dir_of(cid) / "profile.json").exists()

    def list(self, status: Iterable[str] | None = None) -> list[Character]:
        out: list[Character] = []
        for entry in self._index.get("characters", []):
            p = self.dir_of(entry["id"]) / "profile.json"
            if not p.exists():
                continue
            try:
                c = Character.from_dict(json.loads(p.read_text(encoding="utf-8")))
            except Exception:
                continue
            if status is None or c.status in status:
                out.append(c)
        out.sort(key=lambda c: c.last_used or c.created_at, reverse=True)
        return out

    def get(self, cid: str) -> Character:
        p = self.dir_of(cid) / "profile.json"
        if not p.exists():
            raise KeyError(f"角色不存在: {cid}")
        return Character.from_dict(json.loads(p.read_text(encoding="utf-8")))

    def name_taken(self, name: str, exclude: str | None = None) -> bool:
        return any(c.name == name.strip() and c.id != exclude for c in self.list())

    # -- 写入 ----------------------------------------------------------
    def create(self, name: str, **kwargs: Any) -> Character:
        name = name.strip()
        if not name:
            raise ValueError("角色名不能为空")
        if self.name_taken(name):
            raise ValueError(f"角色名「{name}」已存在")
        cid = f"{_slug(name)}-{uuid.uuid4().hex[:6]}"
        c = Character(id=cid, name=name, **kwargs)
        self.dir_of(cid).mkdir(parents=True, exist_ok=True)
        for sub in ("dataset", "checkpoints"):
            (self.dir_of(cid) / sub).mkdir(exist_ok=True)
        self.save(c)
        self._index.setdefault("characters", []).append({
            "id": cid, "name": name, "status": c.status,
            "avatar": c.avatar, "created_at": c.created_at, "last_used": "",
        })
        self._save_index()
        return c

    def save(self, c: Character) -> Path:
        d = self.dir_of(c.id)
        d.mkdir(parents=True, exist_ok=True)
        p = d / "profile.json"
        p.write_text(json.dumps(c.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        for entry in self._index.get("characters", []):
            if entry["id"] == c.id:
                entry.update(name=c.name, status=c.status, avatar=c.avatar,
                             last_used=c.last_used)
                break
        self._save_index()
        return p

    def set_status(self, cid: str, status: str) -> Character:
        if status not in STATUSES:
            raise ValueError(f"未知状态: {status}")
        c = self.get(cid)
        c.status = status
        self.save(c)
        return c

    def touch(self, cid: str) -> None:
        try:
            c = self.get(cid)
        except KeyError:
            return
        c.last_used = _now()
        self.save(c)

    def delete(self, cid: str) -> None:
        d = self.dir_of(cid)
        if d.exists():
            shutil.rmtree(d, ignore_errors=True)
        self._index["characters"] = [
            e for e in self._index.get("characters", []) if e["id"] != cid
        ]
        if self._index.get("active_id") == cid:
            self._index["active_id"] = ""
        self._save_index()

    # -- 数据集 --------------------------------------------------------
    def set_dataset(self, cid: str, duration_sec: float, clips: int) -> Character:
        c = self.get(cid)
        c.dataset = {
            "duration_sec": round(float(duration_sec), 2),
            "clips": int(clips),
            "sha256": _sha256_dir(self.dir_of(cid) / "dataset"),
        }
        self.save(c)
        return c

    def set_voice(self, cid: str, gpt_ckpt: str, sovits_ckpt: str,
                  reference: str, reference_text: str) -> Character:
        c = self.get(cid)
        c.voice.update({
            "mode": "finetune", "engine": "gpt-sovits-v3",
            "gpt_ckpt": str(gpt_ckpt), "sovits_ckpt": str(sovits_ckpt),
            "reference": str(reference), "reference_text": reference_text,
        })
        self.save(c)
        return c

    def avatar_path(self, cid: str) -> Path:
        return self.dir_of(cid) / "avatar.png"

    def set_avatar(self, cid: str, src: str | Path) -> Path:
        dst = self.avatar_path(cid)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        c = self.get(cid)
        c.avatar = "avatar.png"
        self.save(c)
        return dst

    # -- .vcp ----------------------------------------------------------
    def export_vcp(self, cid: str, dest: str | Path | None = None) -> Path:
        c = self.get(cid)
        if c.status != "ready":
            raise RuntimeError("只有 ready 状态的角色才能导出")
        d = self.dir_of(cid)
        dest = Path(dest) if dest else (self.root / f"{cid}.vcp")
        dest.parent.mkdir(parents=True, exist_ok=True)
        manifest = {
            "magic": VCP_MAGIC,
            "vcp_version": VCP_VERSION,
            "app": "VoiceLab Stage2",
            "id": c.id,
            "name": c.name,
            "created_at": c.created_at,
            "exported_at": _now(),
        }
        with zipfile.ZipFile(dest, "w", compression=zipfile.ZIP_DEFLATED) as z:
            z.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
            z.writestr("profile.json", json.dumps(c.to_dict(), ensure_ascii=False, indent=2))
            for name in ("avatar.png", "reference.wav", "reference.txt"):
                p = d / name
                if p.exists():
                    z.write(p, name)
            for p in sorted((d / "checkpoints").rglob("*")) if (d / "checkpoints").exists() else []:
                if p.is_file():
                    z.write(p, f"checkpoints/{p.relative_to(d / 'checkpoints')}")
            ds = d / "dataset"
            if ds.exists():
                for p in sorted(ds.rglob("*")):
                    if p.is_file() and p.suffix in (".wav", ".json", ".list", ".txt"):
                        z.write(p, f"dataset/{p.relative_to(ds)}")
        return dest

    def import_vcp(self, archive: str | Path, overwrite: bool = False) -> Character:
        archive = Path(archive)
        if not archive.exists():
            raise FileNotFoundError(archive)
        if not zipfile.is_zipfile(archive):
            raise ValueError("不是有效的 .vcp 文件（不是 zip 容器）")
        with zipfile.ZipFile(archive) as z:
            names = set(z.namelist())
            missing = [n for n in _REQUIRED_VCP if n not in names]
            if missing:
                raise ValueError(f".vcp 缺少必要文件：{', '.join(missing)}")
            manifest = json.loads(z.read("manifest.json").decode("utf-8"))
            if manifest.get("magic") != VCP_MAGIC:
                raise ValueError("不是 VoiceLab 角色包（magic 不匹配）")
            ver = int(manifest.get("vcp_version", 0))
            if ver > VCP_VERSION:
                raise ValueError(
                    f"角色包版本 v{ver} 高于当前应用支持的 v{VCP_VERSION}，请升级应用"
                )
            profile = json.loads(z.read("profile.json").decode("utf-8"))
            c = Character.from_dict(profile)
            if not c.id or not c.name:
                raise ValueError("profile.json 缺少 id 或 name")
            if not any(n.startswith("checkpoints/") for n in names):
                raise ValueError("角色包里没有 checkpoints/，无法用于推理")

            cid = c.id
            if self.exists(cid):
                if not overwrite:
                    raise ValueError(f"角色 {cid} 已存在，请先删除或选择覆盖")
                self.delete(cid)
            target = self.dir_of(cid)
            target.mkdir(parents=True, exist_ok=True)
            for name in names:
                if name in ("manifest.json", "profile.json"):
                    continue
                if name.endswith("/"):
                    continue
                z.extract(name, target)
        c.status = "ready"
        self.save(c)
        self._index.setdefault("characters", []).append({
            "id": c.id, "name": c.name, "status": c.status,
            "avatar": c.avatar, "created_at": c.created_at, "last_used": "",
        })
        self._save_index()
        return c

    # -- 参考音频 ------------------------------------------------------
    def set_reference(self, cid: str, wav: str | Path, text: str) -> Character:
        d = self.dir_of(cid)
        dst = d / "reference.wav"
        src = Path(wav)
        if src.resolve() != dst.resolve():
            shutil.copyfile(src, dst)
        (d / "reference.txt").write_text(text.strip() + "\n", encoding="utf-8")
        c = self.get(cid)
        c.voice["reference"] = "reference.wav"
        c.voice["reference_text"] = text.strip()
        self.save(c)
        return c


__all__ = ["Character", "CharacterStore", "STATUSES", "VCP_VERSION"]
