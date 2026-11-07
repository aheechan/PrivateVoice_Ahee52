"""Try to build the BigVGAN CUDA extension with a given CUDA_HOME (no model load)."""
import os
import sys
import time

os.environ["MSVC_VCVARS64_PATH"] = r"C:\Local Repositories\PrivateVoice\tools\vs\VC\Auxiliary\Build\vcvars64.bat"
if len(sys.argv) > 1:
    os.environ["CUDA_HOME"] = sys.argv[1]
    os.environ["CUDA_PATH"] = sys.argv[1]

import torch
from torch.utils import cpp_extension

print("torch", torch.__version__, "| CUDA_HOME seen by torch:", cpp_extension.CUDA_HOME, flush=True)
from aquatts.bigvgan.cuda import load as kload

t0 = time.time()
try:
    mod = kload.load()
    print("BUILD_OK in %.1fs -> %r" % (time.time() - t0, mod), flush=True)
    print("has fwd:", hasattr(mod, "forward"))
except Exception as exc:
    print("BUILD_FAIL after %.1fs: %s: %s" % (time.time() - t0, type(exc).__name__, exc), flush=True)
