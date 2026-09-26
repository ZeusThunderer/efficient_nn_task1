"""Predicted-vs-measured plots for FLOPs, memory, latency, and energy."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from equations import energy, flops, latency, memory

ROOT = Path(__file__).resolve().parent
CSV_PATH = ROOT / "results" / "measurements.csv"
THETA_PATH = ROOT / "results" / "theta.json"
FIG_DIR = ROOT / "results" / "figures"

BATCHES = (1, 8, 64, 400)
S_LINE = np.arange(32, 769, 16, dtype=np.float64)
# Free memory reported by mem_get_info on the 8 GB RTX 2070, not the nameplate size.
USABLE_VRAM_BYTES = 6.98 * 1024**3
DPI = 160


def _load_measurements(path: Path) -> dict[str, np.ndarray]:
    s_vals: list[float] = []
    b_vals: list[float] = []
    latency_vals: list[float] = []
    memory_vals: list[float] = []
    energy_vals: list[float] = []
    status: list[str] = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            s_vals.append(float(row["S"]))
            b_vals.append(float(row["B"]))
            status.append(row["status"].strip())
            ok = row["status"].strip().lower() == "ok"
            latency_vals.append(float(row["latency"]) if ok else np.nan)
            memory_vals.append(float(row["memory"]) if ok else np.nan)
            energy_vals.append(float(row["energy"]) if ok else np.nan)
    return {
        "S": np.asarray(s_vals, dtype=np.float64),
        "B": np.asarray(b_vals, dtype=np.float64),
        "latency": np.asarray(latency_vals, dtype=np.float64),
        "memory": np.asarray(memory_vals, dtype=np.float64),
        "energy": np.asarray(energy_vals, dtype=np.float64),
        "status": np.asarray(status),
    }


def _load_theta(path: Path) -> tuple[tuple[float, float, float], dict]:
    payload = json.loads(path.read_text())
    lat = payload["latency"]
    theta_l = (lat["theta_0_Bps"], lat["theta_1_FLOPs"], lat["theta_2_s"])
    eng = payload["energy"]
    theta_e = {
        "P_idle": eng["P_idle_W"],
        "e_mem": eng["e_mem_J_per_byte"],
        "e_compute": eng["e_compute_J_per_FLOP"],
        "theta_L": theta_l,
    }
    return theta_l, theta_e


def _s_at_vram(batch: float) -> float:
    """Image size where memory(S, B) equals usable VRAM (6.98 GiB)."""
    base = float(memory(0.0, batch))
    per_pixel = float(memory(1.0, batch)) - base
    return float(np.sqrt((USABLE_VRAM_BYTES - base) / per_pixel))


def _ok_mask(data: dict[str, np.ndarray], batch: int) -> np.ndarray:
    return (data["B"] == batch) & (data["status"] == "ok")


def _oom_mask(data: dict[str, np.ndarray], batch: int) -> np.ndarray:
    return (data["B"] == batch) & (data["status"] == "OOM")


def _style_axes(ax: plt.Axes, ylabel: str, title: str) -> None:
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("image size S (pixels)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both", linestyle=":", alpha=0.5)
    ax.legend()


def _plot_curves(
    ax: plt.Axes,
    predict,
    data: dict[str, np.ndarray],
    column: str | None,
) -> None:
    for batch, color in zip(BATCHES, plt.rcParams["axes.prop_cycle"].by_key()["color"]):
        y_line = np.asarray(predict(S_LINE, batch), dtype=np.float64)
        ax.plot(S_LINE, y_line, color=color, label=f"B={batch} predicted")
        if column is None:
            continue
        mask = _ok_mask(data, batch)
        if np.any(mask):
            order = np.argsort(data["S"][mask])
            ax.scatter(
                data["S"][mask][order],
                data[column][mask][order],
                color=color,
                marker="o",
                zorder=3,
                label=f"B={batch} measured",
            )


def _mark_oom(ax: plt.Axes, data: dict[str, np.ndarray], predict) -> None:
    labeled_cross = False
    labeled_line = False
    for batch in BATCHES:
        mask = _oom_mask(data, batch)
        if np.any(mask):
            s_oom = data["S"][mask]
            y_oom = np.asarray(predict(s_oom, batch), dtype=np.float64)
            ax.scatter(
                s_oom,
                y_oom,
                marker="x",
                color="red",
                s=80,
                zorder=4,
                label="OOM measured" if not labeled_cross else None,
            )
            labeled_cross = True
        s_limit = _s_at_vram(batch)
        if S_LINE[0] <= s_limit <= S_LINE[-1]:
            ax.axvline(
                s_limit,
                color="black",
                linestyle="--",
                linewidth=1,
                label="usable VRAM" if not labeled_line else None,
            )
            labeled_line = True


def plot_flops(fig_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5.5))
    _plot_curves(ax, flops, {}, None)
    _style_axes(ax, "FLOPs", "FLOPs vs image size (model only; not measured)")
    fig.tight_layout()
    fig.savefig(fig_dir / "flops.png", dpi=DPI)
    plt.close(fig)


def plot_memory(data: dict[str, np.ndarray], fig_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5.5))
    _plot_curves(ax, memory, data, "memory")
    ax.axhline(USABLE_VRAM_BYTES, color="black", linestyle="--", linewidth=1, label="usable VRAM")
    _style_axes(ax, "bytes", "Peak memory vs image size")
    fig.tight_layout()
    fig.savefig(fig_dir / "memory.png", dpi=DPI)
    plt.close(fig)


def plot_latency(
    data: dict[str, np.ndarray],
    theta_l: tuple[float, float, float],
    fig_dir: Path,
) -> None:
    def predict(image_size, batch):
        return latency(image_size, batch, theta_l)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    _plot_curves(ax, predict, data, "latency")
    _mark_oom(ax, data, predict)
    _style_axes(ax, "seconds", "Latency vs image size")
    fig.tight_layout()
    fig.savefig(fig_dir / "latency.png", dpi=DPI)
    plt.close(fig)


def plot_energy(data: dict[str, np.ndarray], theta_e: dict, fig_dir: Path) -> None:
    def predict(image_size, batch):
        return energy(image_size, batch, theta_e)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    _plot_curves(ax, predict, data, "energy")
    _mark_oom(ax, data, predict)
    _style_axes(ax, "joules", "Energy vs image size")
    fig.tight_layout()
    fig.savefig(fig_dir / "energy.png", dpi=DPI)
    plt.close(fig)


def main() -> None:
    data = _load_measurements(CSV_PATH)
    theta_l, theta_e = _load_theta(THETA_PATH)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plot_flops(FIG_DIR)
    plot_memory(data, FIG_DIR)
    plot_latency(data, theta_l, FIG_DIR)
    plot_energy(data, theta_e, FIG_DIR)
    print(f"wrote {FIG_DIR}/{{flops,memory,latency,energy}}.png", flush=True)


if __name__ == "__main__":
    main()
