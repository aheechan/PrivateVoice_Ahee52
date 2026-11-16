"""Download torch/torchaudio cu128 wheels directly (resumable, sha256-verified)."""
import hashlib
import os
import sys
import time
import urllib.parse
import urllib.request

OUT = r"C:\Local Repositories\PrivateVoice\wheels"
WHEELS = [
    ("https://download-r2.pytorch.org/whl/cu128/torch-2.11.0%2Bcu128-cp312-cp312-win_amd64.whl",
     "7c78215c3af4f62e63f2b2e360f1722fc719b0853c7ac22666483d9810613a4c"),
    ("https://download-r2.pytorch.org/whl/cu128/torchaudio-2.11.0%2Bcu128-cp312-cp312-win_amd64.whl",
     "e0203d44b6dcf4c59f2ce38f997616e663b2a23a9e0b20ebb90ea0c787b5e86a"),
]


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def fetch(url, dest):
    if os.path.exists(dest):
        return "exists"
    part = dest + ".part"
    have = os.path.getsize(part) if os.path.exists(part) else 0
    req = urllib.request.Request(url, headers={"User-Agent": "voicelab/1.0"})
    if have:
        req.add_header("Range", "bytes=%d-" % have)
    resp = urllib.request.urlopen(req, timeout=60)
    cl = resp.headers.get("Content-Length")
    total = (int(cl) + have) if (cl and resp.status == 206) else (int(cl) if cl else None)
    mode = "ab" if (have and resp.status == 206) else "wb"
    if mode == "wb":
        have = 0
    t0, done = time.time(), have
    with open(part, mode) as f:
        while True:
            b = resp.read(1 << 20)
            if not b:
                break
            f.write(b)
            done += len(b)
            if total and done % (128 << 20) < (1 << 20):
                print("   %5.0f / %5.0f MB   %.1f MB/s" % (done / 1e6, total / 1e6,
                                                           done / 1e6 / max(1e-6, time.time() - t0)),
                      flush=True)
    os.replace(part, dest)
    return "downloaded"


def main():
    os.makedirs(OUT, exist_ok=True)
    ok = True
    for url, want in WHEELS:
        name = os.path.basename(urllib.parse.unquote(url.split("#")[0]))
        dest = os.path.join(OUT, name)
        print("[..] %s" % name, flush=True)
        try:
            st = fetch(url, dest)
        except Exception as e:
            print("[FAIL] %s: %s" % (name, e), flush=True)
            ok = False
            continue
        got = sha256_of(dest)
        good = (got == want)
        ok &= good
        print("[%s] %s  %d bytes  (%s) sha256_match=%s"
              % ("ok" if good else "BAD", name, os.path.getsize(dest), st, good), flush=True)
        if not good:
            print("   want=%s\n   got =%s" % (want, got), flush=True)
    print("ALL_OK" if ok else "ERRORS")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
