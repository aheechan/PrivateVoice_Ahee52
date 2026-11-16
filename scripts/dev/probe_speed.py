import time, urllib.request, urllib.error, sys

URL = ("https://download-r2.pytorch.org/whl/cu128/"
       "torch-2.11.0%2Bcu128-cp312-cp312-win_amd64.whl")


def probe(url, label):
    req = urllib.request.Request(url, headers={"User-Agent": "probe/1.0",
                                               "Range": "bytes=0-31457279"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            total = r.headers.get("Content-Range", "?")
            n = 0
            while True:
                b = r.read(1 << 16)
                if not b:
                    break
                n += len(b)
                if n >= 16 << 20:
                    break
            dt = time.time() - t0
            print("%s: status=%s got=%.1fMB in %.1fs -> %.1f MB/s ; %s"
                  % (label, r.status, n / 1e6, dt, n / 1e6 / max(dt, 1e-6), total), flush=True)
    except Exception as e:
        print("%s: FAIL %s: %s" % (label, type(e).__name__, e), flush=True)


if __name__ == "__main__":
    probe(URL, "direct")
