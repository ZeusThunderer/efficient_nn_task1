"""Measure SmallCNN forward-pass latency, memory, and energy on one GPU."""

from __future__ import annotations

import csv
import math
import threading
import time
from pathlib import Path

import numpy as np
import pynvml
import torch

from models import SmallCNN

BASE_S = (32, 64, 128, 224, 256, 384, 512)
BASE_B = (1, 2, 4, 8, 16, 32, 64, 128, 256)
# Beyond the homework grid, to reach the 8 GB OOM wall. Not calibration points.
OOM_S = (576, 640)
OOM_B = (330, 400)
SEED = 0
WARMUP = 10
LATENCY_REPEATS = 30
ENERGY_MIN_SECONDS = 1.0
POWER_PERIOD_S = 0.01  # 100 Hz
CSV_PATH = Path(__file__).resolve().parent / "results" / "measurements.csv"
COLUMNS = ("S", "B", "latency", "memory", "energy", "status", "is_validation")


def _is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def build_grid(seed: int = SEED) -> tuple[list[int], list[int], list[tuple[int, int, bool]]]:
    """Cartesian grid. is_validation is True when S or B is not from the base lists.

    Homework core is (7+4) x (9+3) = 132. OOM_S and OOM_B add the large
    points needed to hit the 8 GB limit; those rows are validation, not fit data.
    """
    rng = np.random.default_rng(seed)
    s_candidates = [s for s in range(32, 513, 16) if s not in BASE_S]
    extra_s = sorted(int(x) for x in rng.choice(s_candidates, size=4, replace=False))
    b_candidates = [b for b in range(1, 257) if not _is_power_of_two(b)]
    extra_b = sorted(int(x) for x in rng.choice(b_candidates, size=3, replace=False))

    base_s = set(BASE_S)
    base_b = set(BASE_B)
    sizes = (*BASE_S, *extra_s, *OOM_S)
    batches = (*BASE_B, *extra_b, *OOM_B)
    configs: list[tuple[int, int, bool]] = []
    for s in sizes:
        for b in batches:
            is_validation = s not in base_s or b not in base_b
            configs.append((s, b, is_validation))
    configs.sort(key=lambda item: (item[0] * item[0] * item[1], item[0], item[1]))
    return extra_s, extra_b, configs


def _trapezoid(y: np.ndarray, x: np.ndarray) -> float:
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))


class PowerSampler:
    """Background NVML power poller. Samples are (perf_counter seconds, watts)."""

    def __init__(self, handle: object, period_s: float = POWER_PERIOD_S) -> None:
        self._handle = handle
        self._period_s = period_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.samples: list[tuple[float, float]] = []

    def start(self) -> None:
        self.samples = []
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="nvml-power", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None

    def _loop(self) -> None:
        while not self._stop.is_set():
            milliwatts = pynvml.nvmlDeviceGetPowerUsage(self._handle)
            self.samples.append((time.perf_counter(), milliwatts / 1000.0))
            self._stop.wait(self._period_s)


def _energy_joules(samples: list[tuple[float, float]], t0: float, t1: float, repeats: int) -> float:
    window = [(t, w) for t, w in samples if t0 <= t <= t1]
    duration = t1 - t0
    if len(window) >= 2:
        ts = np.array([t for t, _ in window], dtype=np.float64)
        ws = np.array([w for _, w in window], dtype=np.float64)
        joules = _trapezoid(ws, ts)
    else:
        watts = [w for _, w in (window or samples)] or [0.0]
        joules = (sum(watts) / len(watts)) * duration
    return joules / repeats


def measure_one(
    model: torch.nn.Module,
    image_size: int,
    batch: int,
    sampler: PowerSampler,
) -> tuple[float, int, float]:
    x = torch.randn(batch, 3, image_size, image_size, device="cuda")
    try:
        with torch.inference_mode():
            for _ in range(WARMUP):
                model(x)
            torch.cuda.synchronize()

            times: list[float] = []
            for _ in range(LATENCY_REPEATS):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                model(x)
                torch.cuda.synchronize()
                times.append(time.perf_counter() - t0)
            latency = float(np.median(times))

            torch.cuda.reset_peak_memory_stats()
            model(x)
            torch.cuda.synchronize()
            memory = int(torch.cuda.max_memory_allocated())

            repeats = max(LATENCY_REPEATS, math.ceil(ENERGY_MIN_SECONDS / latency))
            torch.cuda.synchronize()
            sampler.start()
            try:
                t_start = time.perf_counter()
                for _ in range(repeats):
                    model(x)
                torch.cuda.synchronize()
                t_end = time.perf_counter()
            finally:
                sampler.stop()
            energy = _energy_joules(sampler.samples, t_start, t_end, repeats)
        return latency, memory, energy
    finally:
        del x


def _load_done(path: Path) -> set[tuple[int, int]]:
    done: set[tuple[int, int]] = set()
    if not path.exists() or path.stat().st_size == 0:
        return done
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            done.add((int(row["S"]), int(row["B"])))
    return done


def _format_ok(latency: float, memory: int, energy: float) -> str:
    return f"ok latency={latency:.6g} s memory={memory} B energy={energy:.6g} J"


def _device_name(name: str | bytes) -> str:
    if isinstance(name, bytes):
        return name.decode()
    return name


def select_rtx_2070() -> tuple[object, int, int, str]:
    """Pick the physical RTX 2070. NVML order and torch order differ on this machine."""
    nvml_index = None
    handle = None
    nvml_name = ""
    for index in range(pynvml.nvmlDeviceGetCount()):
        candidate = pynvml.nvmlDeviceGetHandleByIndex(index)
        name = _device_name(pynvml.nvmlDeviceGetName(candidate))
        if "2070" in name and "5070" not in name:
            nvml_index = index
            handle = candidate
            nvml_name = name
            break
    if handle is None or nvml_index is None:
        raise SystemExit("RTX 2070 not found via NVML")

    torch_index = None
    for index in range(torch.cuda.device_count()):
        name = torch.cuda.get_device_name(index)
        if "2070" in name and "5070" not in name:
            torch_index = index
            break
    if torch_index is None:
        raise SystemExit("RTX 2070 not found via torch")

    torch.cuda.set_device(torch_index)
    return handle, nvml_index, torch_index, nvml_name


def main() -> None:
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available")

    extra_s, extra_b, configs = build_grid()
    expected = (len(BASE_S) + 4 + len(OOM_S)) * (len(BASE_B) + 3 + len(OOM_B))
    if len(configs) != expected:
        raise SystemExit(f"expected {expected} configs, got {len(configs)}")
    print(f"extra S (seed={SEED}): {extra_s}", flush=True)
    print(f"extra B (seed={SEED}): {extra_b}", flush=True)
    print(f"OOM S: {list(OOM_S)}", flush=True)
    print(f"OOM B: {list(OOM_B)}", flush=True)
    n_val = sum(1 for _, _, is_val in configs if is_val)
    print(f"grid: {len(configs)} configs, validation={n_val}, calibration={len(configs) - n_val}", flush=True)

    pynvml.nvmlInit()
    try:
        handle, nvml_index, torch_index, name = select_rtx_2070()
        print(f"GPU: {name} (nvml={nvml_index}, cuda={torch_index})", flush=True)
        sampler = PowerSampler(handle)

        model = SmallCNN().cuda().eval()

        CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
        done = _load_done(CSV_PATH)
        new_file = not CSV_PATH.exists() or CSV_PATH.stat().st_size == 0
        total = len(configs)

        with CSV_PATH.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=COLUMNS)
            if new_file:
                writer.writeheader()
                f.flush()

            for index, (image_size, batch, is_validation) in enumerate(configs, start=1):
                left = total - index
                prefix = (
                    f"[{index}/{total}] left={left} S={image_size} B={batch} "
                    f"validation={is_validation}"
                )
                if (image_size, batch) in done:
                    print(f"{prefix} -> skip (already in csv)", flush=True)
                    continue

                try:
                    latency, memory, energy = measure_one(model, image_size, batch, sampler)
                    row = {
                        "S": image_size,
                        "B": batch,
                        "latency": latency,
                        "memory": memory,
                        "energy": energy,
                        "status": "ok",
                        "is_validation": is_validation,
                    }
                    print(f"{prefix} -> {_format_ok(latency, memory, energy)}", flush=True)
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                    row = {
                        "S": image_size,
                        "B": batch,
                        "latency": "",
                        "memory": "",
                        "energy": "",
                        "status": "OOM",
                        "is_validation": is_validation,
                    }
                    print(f"{prefix} -> OOM", flush=True)

                writer.writerow(row)
                f.flush()
    finally:
        pynvml.nvmlShutdown()


if __name__ == "__main__":
    main()
