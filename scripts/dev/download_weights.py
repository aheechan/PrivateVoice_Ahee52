"""Resumable weight downloader for Stage 1 (hf-mirror, no huggingface_hub needed).

Writes logs/weights.sha256 with: sha256  size_bytes  local_path  url
"""
import hashlib
import os
import sys
import time
import urllib.request
import urllib.error

MIRROR = "https://hf-mirror.com"
ROOT = r"C:\Local Repositories\PrivateVoice"
PM = os.path.join(ROOT, "GPT-SoVITS", "GPT_SoVITS", "pretrained_models")
LOG = os.path.join(ROOT, "logs", "weights.sha256")

# (repo, remote_path, local_path, expected_bytes or None)
MANIFEST = [
    ("lj1995/GPT-SoVITS", "s1v3.ckpt", "s1v3.ckpt", 155300000),
    ("lj1995/GPT-SoVITS", "s2Gv3.pth", "s2Gv3.pth", 769000000),
    ("lj1995/GPT-SoVITS", "chinese-hubert-base/config.json", "chinese-hubert-base/config.json", None),
    ("lj1995/GPT-SoVITS", "chinese-hubert-base/preprocessor_config.json", "chinese-hubert-base/preprocessor_config.json", None),
    ("lj1995/GPT-SoVITS", "chinese-hubert-base/pytorch_model.bin", "chinese-hubert-base/pytorch_model.bin", 188800000),
    ("lj1995/GPT-SoVITS", "chinese-roberta-wwm-ext-large/config.json", "chinese-roberta-wwm-ext-large/config.json", None),
    ("lj1995/GPT-SoVITS", "chinese-roberta-wwm-ext-large/pytorch_model.bin", "chinese-roberta-wwm-ext-large/pytorch_model.bin", 651200000),
    ("lj1995/GPT-SoVITS", "chinese-roberta-wwm-ext-large/tokenizer.json", "chinese-roberta-wwm-ext-large/tokenizer.json", None),
]

BV = "models--nvidia--bigvgan_v2_24khz_100band_256x"
for rel in ["config.json", "bigvgan_generator.pt", "bigvgan.py", "activations.py",
            "env.py", "utils.py", "meldataset.py",
            "alias_free_activation/cuda/__init__.py",
            "alias_free_activation/cuda/activation1d.py",
            "alias_free_activation/cuda/anti_alias_activation.cpp",
            "alias_free_activation/cuda/anti_alias_activation_cuda.cu",
            "alias_free_activation/cuda/compat.h",
            "alias_free_activation/cuda/load.py",
            "alias_free_activation/cuda/type_shim.h",
            "alias_free_activation/torch/__init__.py",
            "alias_free_activation/torch/act.py",
            "alias_free_activation/torch/filter.py",
            "alias_free_activation/torch/resample.py"]:
    exp = 450100000 if rel == "bigvgan_generator.pt" else None
    MANIFEST.append(("nvidia/bigvgan_v2_24khz_100band_256x", rel, BV + "/" + rel, exp))


def sha256_of(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def download(url, dest, expected=None):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    part = dest + ".part"
    if os.path.exists(dest) and (expected is None or abs(os.path.getsize(dest) - expected) < 200000):
        return "exists"
    have = os.path.getsize(part) if os.path.exists(part) else 0
    req = urllib.request.Request(url, headers={"User-Agent": "voicelab/1.0"})
    if have:
        req.add_header("Range", "bytes=%d-" % have)
    try:
        resp = urllib.request.urlopen(req, timeout=60)
    except urllib.error.HTTPError as e:
        if e.code == 416 and os.path.exists(part):
            os.replace(part, dest)
            return "resumed-complete"
        raise
    total = resp.headers.get("Content-Length")
    total = int(total) + have if total else None
    mode = "ab" if have and resp.status == 206 else "wb"
    if mode == "wb":
        have = 0
    t0 = time.time()
    done = have
    with open(part, mode) as f:
        while True:
            b = resp.read(1 << 20)
            if not b:
                break
            f.write(b)
            done += len(b)
            if total and done % (64 << 20) < (1 << 20):
                mb = done / 1e6
                sp = done / max(1e-6, time.time() - t0) / 1e6
                print("   %.0f/%.0f MB  %.1f MB/s" % (mb, total / 1e6, sp), flush=True)
    os.replace(part, dest)
    return "downloaded"


def main():
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as lg:
        lg.write("\n# === run %s ; mirror=%s ===\n" % (time.strftime("%Y-%m-%dT%H:%M:%S"), MIRROR))
    fail = 0
    for repo, remote, local, expected in MANIFEST:
        url = "%s/%s/resolve/main/%s" % (MIRROR, repo, remote)
        dest = os.path.join(PM, local.replace("/", os.sep))
        print("[..] %s" % local, flush=True)
        try:
            st = download(url, dest, expected)
        except Exception as e:
            print("[FAIL] %s -> %s: %s" % (local, type(e).__name__, e), flush=True)
            with open(LOG, "a", encoding="utf-8") as lg:
                lg.write("FAILED\t%s\t%s\t%s\n" % (local, url, e))
            fail += 1
            continue
        size = os.path.getsize(dest)
        digest = sha256_of(dest)
        print("[ok] %s  %s  %.1f MB  (%s)" % (local, digest[:16], size / 1e6, st), flush=True)
        with open(LOG, "a", encoding="utf-8") as lg:
            lg.write("%s\t%d\t%s\t%s\n" % (digest, size, local, url))
    print("DONE failures=%d" % fail)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
