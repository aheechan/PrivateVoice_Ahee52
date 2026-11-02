""".vcp 导入导出往返一致 + 角色库基本 CRUD。"""
from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.core.characters import CharacterStore, VCP_VERSION  # noqa: E402


@pytest.fixture()
def store(tmp_path) -> CharacterStore:
    return CharacterStore(tmp_path / "characters")


def _make_ready(store: CharacterStore, name: str = "小满"):
    ch = store.create(name, persona="测试人设", greeting="你好")
    d = store.dir_of(ch.id)
    (d / "checkpoints").mkdir(exist_ok=True)
    (d / "checkpoints" / "gpt.ckpt").write_bytes(b"fake-gpt")
    (d / "checkpoints" / "sovits.pth").write_bytes(b"fake-sovits")
    audio = (0.1 * np.random.randn(24000)).astype(np.float32)
    sf.write(str(d / "reference.wav"), audio, 24000, subtype="PCM_16")
    (d / "reference.txt").write_text("测试参考文本\n", encoding="utf-8")
    ch = store.set_reference(ch.id, d / "reference.wav", "测试参考文本")
    store.set_voice(ch.id, str(d / "checkpoints" / "gpt.ckpt"),
                    str(d / "checkpoints" / "sovits.pth"), "reference.wav", "测试参考文本")
    store.set_status(ch.id, "ready")
    return store.get(ch.id)


def test_create_and_list(store):
    ch = store.create("阿离")
    assert store.get(ch.id).name == "阿离"
    assert [c.id for c in store.list()] == [ch.id]
    with pytest.raises(ValueError):
        store.create("阿离")          # 重名
    with pytest.raises(ValueError):
        store.create("   ")           # 空名


def test_export_import_roundtrip(store, tmp_path):
    ch = _make_ready(store)
    pkg = store.export_vcp(ch.id, tmp_path / "x.vcp")
    assert pkg.exists()
    with zipfile.ZipFile(pkg) as z:
        names = set(z.namelist())
        assert "manifest.json" in names
        assert any(n.startswith("checkpoints/") for n in names)
        manifest = json.loads(z.read("manifest.json"))
        assert manifest["vcp_version"] == VCP_VERSION

    store.delete(ch.id)
    assert not store.exists(ch.id)

    back = store.import_vcp(pkg)
    assert back.id == ch.id
    assert back.name == ch.name
    assert back.status == "ready"
    assert back.voice["reference_text"] == "测试参考文本"
    assert (store.dir_of(back.id) / "checkpoints" / "gpt.ckpt").read_bytes() == b"fake-gpt"


def test_import_rejects_broken_package(store, tmp_path):
    bad = tmp_path / "bad.vcp"
    with zipfile.ZipFile(bad, "w") as z:
        z.writestr("profile.json", "{}")
    with pytest.raises(ValueError):
        store.import_vcp(bad)

    future = tmp_path / "future.vcp"
    with zipfile.ZipFile(future, "w") as z:
        z.writestr("manifest.json", json.dumps(
            {"magic": "voicelab.character.vcp", "vcp_version": VCP_VERSION + 5}))
        z.writestr("profile.json", json.dumps({"id": "a", "name": "b"}))
        z.writestr("reference.wav", b"x")
        z.writestr("checkpoints/gpt.ckpt", b"x")
    with pytest.raises(ValueError):
        store.import_vcp(future)


def test_status_and_touch(store):
    ch = _make_ready(store)
    store.touch(ch.id)
    assert store.get(ch.id).last_used
    store.set_status(ch.id, "failed")
    assert store.get(ch.id).status == "failed"
    with pytest.raises(ValueError):
        store.set_status(ch.id, "unknown")
