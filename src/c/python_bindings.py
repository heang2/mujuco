"""
Python ctypes bindings for the C fast_gae and running_stats modules.

Falls back to pure-Python implementations if the shared library is not found,
so the rest of the project works without compilation.

Compile the shared library first:
    # Linux / macOS
    cd src/c
    gcc -O3 -march=native -ffast-math -shared -fPIC \\
        -o fast_gae.so fast_gae.c running_stats.c -lm

    # Windows (MSVC)
    cl /O2 /LD fast_gae.c running_stats.c /Fe:fast_gae.dll

    # Windows (MinGW/Clang)
    gcc -O3 -shared -fPIC -o fast_gae.dll fast_gae.c running_stats.c -lm

Then verify the speedup:
    python src/c/python_bindings.py --benchmark
"""

import ctypes
import os
import platform
import time
from pathlib import Path
from typing import Optional

import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Library loading
# ──────────────────────────────────────────────────────────────────────────────

_LIB_DIR  = Path(__file__).parent
_EXT      = ".dll" if platform.system() == "Windows" else ".so"
_LIB_PATH = _LIB_DIR / f"fast_gae{_EXT}"

_lib: Optional[ctypes.CDLL] = None


def _load_lib() -> Optional[ctypes.CDLL]:
    global _lib
    if _lib is not None:
        return _lib
    if not _LIB_PATH.exists():
        return None
    try:
        lib = ctypes.CDLL(str(_LIB_PATH))

        # ── compute_gae ───────────────────────────────────────────────
        lib.compute_gae.argtypes = [
            ctypes.POINTER(ctypes.c_float),   # rewards
            ctypes.POINTER(ctypes.c_float),   # values
            ctypes.POINTER(ctypes.c_float),   # dones
            ctypes.POINTER(ctypes.c_float),   # advantages (out)
            ctypes.POINTER(ctypes.c_float),   # returns (out)
            ctypes.c_int32,                   # n
            ctypes.c_float,                   # gamma
            ctypes.c_float,                   # gae_lambda
            ctypes.c_float,                   # last_value
        ]
        lib.compute_gae.restype = None

        # ── compute_gae_batched ────────────────────────────────────────
        lib.compute_gae_batched.argtypes = [
            ctypes.POINTER(ctypes.c_float),   # rewards [B,n]
            ctypes.POINTER(ctypes.c_float),   # values  [B,n]
            ctypes.POINTER(ctypes.c_float),   # dones   [B,n]
            ctypes.POINTER(ctypes.c_float),   # advantages (out)
            ctypes.POINTER(ctypes.c_float),   # returns (out)
            ctypes.c_int32,                   # B
            ctypes.c_int32,                   # n
            ctypes.c_float,                   # gamma
            ctypes.c_float,                   # gae_lambda
            ctypes.POINTER(ctypes.c_float),   # last_values [B]
        ]
        lib.compute_gae_batched.restype = None

        # ── normalize_advantages ──────────────────────────────────────
        lib.normalize_advantages.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int32,
            ctypes.c_float,
        ]
        lib.normalize_advantages.restype = None

        # ── RunningStats ──────────────────────────────────────────────
        lib.running_stats_create.argtypes  = [ctypes.c_int32]
        lib.running_stats_create.restype   = ctypes.c_void_p
        lib.running_stats_destroy.argtypes = [ctypes.c_void_p]
        lib.running_stats_destroy.restype  = None
        lib.running_stats_update.argtypes  = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int32
        ]
        lib.running_stats_update.restype   = None
        lib.running_stats_normalize.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_float),
            ctypes.c_int32, ctypes.c_float, ctypes.c_float,
        ]
        lib.running_stats_normalize.restype  = None
        lib.running_stats_get_mean.argtypes  = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_float)
        ]
        lib.running_stats_get_mean.restype   = None
        lib.running_stats_get_count.argtypes = [ctypes.c_void_p]
        lib.running_stats_get_count.restype  = ctypes.c_double

        _lib = lib
        return lib
    except OSError:
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Pure-Python fallbacks (used when .so not compiled)
# ──────────────────────────────────────────────────────────────────────────────

def _py_compute_gae(
    rewards: np.ndarray,
    values:  np.ndarray,
    dones:   np.ndarray,
    gamma:   float,
    gae_lambda: float,
    last_value: float,
) -> tuple[np.ndarray, np.ndarray]:
    n          = len(rewards)
    advantages = np.zeros(n, dtype=np.float32)
    returns    = np.zeros(n, dtype=np.float32)
    gae        = 0.0
    for t in reversed(range(n)):
        next_val  = last_value if t == n - 1 else values[t + 1]
        next_done = 0.0        if t == n - 1 else dones[t + 1]
        delta     = rewards[t] + gamma * next_val * (1.0 - dones[t]) - values[t]
        gae       = delta + gamma * gae_lambda * (1.0 - dones[t]) * gae
        advantages[t] = gae
        returns[t]    = gae + values[t]
    return advantages, returns


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def compute_gae(
    rewards:    np.ndarray,
    values:     np.ndarray,
    dones:      np.ndarray,
    gamma:      float = 0.99,
    gae_lambda: float = 0.95,
    last_value: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute GAE advantages and discounted returns.

    Uses the compiled C library if available, otherwise falls back to Python.

    Returns:
        advantages: np.ndarray (n,) float32
        returns:    np.ndarray (n,) float32
    """
    rewards = np.ascontiguousarray(rewards, dtype=np.float32)
    values  = np.ascontiguousarray(values,  dtype=np.float32)
    dones   = np.ascontiguousarray(dones,   dtype=np.float32)
    n       = len(rewards)

    lib = _load_lib()
    if lib is not None:
        advantages = np.zeros(n, dtype=np.float32)
        returns    = np.zeros(n, dtype=np.float32)
        lib.compute_gae(
            rewards.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            values.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            dones.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            advantages.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            returns.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_int32(n),
            ctypes.c_float(gamma),
            ctypes.c_float(gae_lambda),
            ctypes.c_float(last_value),
        )
        return advantages, returns
    else:
        return _py_compute_gae(rewards, values, dones, gamma, gae_lambda, last_value)


def normalize_advantages(advantages: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Normalise advantages in-place. Uses C library if available."""
    adv = np.ascontiguousarray(advantages, dtype=np.float32)
    lib = _load_lib()
    if lib is not None:
        lib.normalize_advantages(
            adv.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_int32(len(adv)),
            ctypes.c_float(eps),
        )
        return adv
    else:
        return (adv - adv.mean()) / (adv.std() + eps)


class CRunningStats:
    """
    Fast observation normaliser backed by the C running_stats module.

    Falls back to a pure-Python implementation if the library isn't compiled.

    Usage:
        rs = CRunningStats(obs_dim=11)
        rs.update(obs_batch)      # shape (B, 11)
        normed = rs.normalize(obs)  # shape (11,)
    """

    def __init__(self, dim: int, clip: float = 10.0, eps: float = 1e-8):
        self.dim   = dim
        self.clip  = clip
        self.eps   = eps
        lib        = _load_lib()

        if lib is not None:
            self._handle = lib.running_stats_create(ctypes.c_int32(dim))
            self._lib    = lib
            self._native = True
        else:
            # Pure-Python fallback
            self._mean   = np.zeros(dim, dtype=np.float64)
            self._var    = np.ones(dim,  dtype=np.float64)
            self._count  = 1e-4
            self._native = False

    def update(self, obs: np.ndarray) -> None:
        obs = np.ascontiguousarray(obs, dtype=np.float32)
        if obs.ndim == 1:
            obs = obs[np.newaxis]
        B = obs.shape[0]

        if self._native:
            self._lib.running_stats_update(
                ctypes.c_void_p(self._handle),
                obs.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                ctypes.c_int32(B),
            )
        else:
            batch_mean = obs.mean(axis=0)
            batch_var  = obs.var(axis=0)
            delta      = batch_mean - self._mean
            total      = self._count + B
            self._mean += delta * B / total
            self._var   = (self._var * self._count + batch_var * B
                          + delta ** 2 * self._count * B / total) / total
            self._count = total

    def normalize(self, obs: np.ndarray) -> np.ndarray:
        obs = np.ascontiguousarray(obs, dtype=np.float32)
        was_1d = obs.ndim == 1
        if was_1d:
            obs = obs[np.newaxis]
        B   = obs.shape[0]
        out = obs.copy()

        if self._native:
            self._lib.running_stats_normalize(
                ctypes.c_void_p(self._handle),
                out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                ctypes.c_int32(B),
                ctypes.c_float(self.eps),
                ctypes.c_float(self.clip),
            )
        else:
            out = np.clip(
                (out - self._mean) / np.sqrt(self._var + self.eps),
                -self.clip, self.clip,
            ).astype(np.float32)

        return out[0] if was_1d else out

    def __del__(self):
        if self._native and hasattr(self, "_handle") and self._handle:
            self._lib.running_stats_destroy(ctypes.c_void_p(self._handle))

    @property
    def using_c_library(self) -> bool:
        return self._native


# ──────────────────────────────────────────────────────────────────────────────
# Benchmark
# ──────────────────────────────────────────────────────────────────────────────

def _run_benchmark(n: int = 8192, reps: int = 200) -> None:
    print(f"\nGAE Benchmark  (n={n}, reps={reps})")
    print("─" * 45)

    rewards = np.random.randn(n).astype(np.float32)
    values  = np.random.randn(n).astype(np.float32)
    dones   = (np.random.rand(n) < 0.02).astype(np.float32)

    # Python baseline
    t0 = time.perf_counter()
    for _ in range(reps):
        _py_compute_gae(rewards, values, dones, 0.99, 0.95, 0.0)
    py_ms = (time.perf_counter() - t0) / reps * 1000

    # C library (if available)
    lib = _load_lib()
    if lib is not None:
        advantages = np.zeros(n, dtype=np.float32)
        returns    = np.zeros(n, dtype=np.float32)
        t0 = time.perf_counter()
        for _ in range(reps):
            lib.compute_gae(
                rewards.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                values.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                dones.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                advantages.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                returns.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                ctypes.c_int32(n),
                ctypes.c_float(0.99),
                ctypes.c_float(0.95),
                ctypes.c_float(0.0),
            )
        c_ms = (time.perf_counter() - t0) / reps * 1000
        print(f"  Python  : {py_ms:.3f} ms / call")
        print(f"  C       : {c_ms:.3f} ms / call")
        print(f"  Speedup : {py_ms / c_ms:.1f}×")
    else:
        print(f"  Python  : {py_ms:.3f} ms / call")
        print(f"  C       : library not compiled — run build instructions above")

    # CRunningStats benchmark
    print(f"\nRunningStats Normalise Benchmark  (dim=111, batch=2048)")
    print("─" * 45)
    obs_batch = np.random.randn(2048, 111).astype(np.float32)

    rs_c  = CRunningStats(111)
    rs_c.update(obs_batch)
    t0 = time.perf_counter()
    for _ in range(reps):
        rs_c.normalize(obs_batch)
    c_ms2 = (time.perf_counter() - t0) / reps * 1000
    mode = "C library" if rs_c.using_c_library else "Python fallback"
    print(f"  {mode}: {c_ms2:.3f} ms / call")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark", action="store_true")
    args = p.parse_args()
    if args.benchmark:
        _run_benchmark()
    else:
        # Quick sanity check
        r  = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        v  = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)
        d  = np.zeros(4, dtype=np.float32)
        adv, ret = compute_gae(r, v, d, gamma=0.99, gae_lambda=0.95, last_value=0.5)
        print(f"advantages : {adv}")
        print(f"returns    : {ret}")
        print(f"C library  : {'loaded' if _load_lib() else 'not compiled (using Python)'}")
