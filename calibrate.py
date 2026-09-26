"""Fit latency and energy parameters on non-OOM calibration rows."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit, least_squares

from equations import energy, latency

CSV_PATH = Path(__file__).resolve().parent / "results" / "measurements.csv"
THETA_PATH = Path(__file__).resolve().parent / "results" / "theta.json"


def _load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _split(
    rows: list[dict[str, str]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return train and validation arrays (S, B, latency_s, energy_J). OOM rows dropped."""
    train: list[tuple[float, float, float, float]] = []
    valid: list[tuple[float, float, float, float]] = []
    for row in rows:
        if row["status"].strip().upper() == "OOM":
            continue
        sample = (
            float(row["S"]),
            float(row["B"]),
            float(row["latency"]),
            float(row["energy"]),
        )
        if row["is_validation"].strip().lower() == "true":
            valid.append(sample)
        else:
            train.append(sample)
    if not train:
        raise SystemExit("no calibration rows (is_validation=False, status!=OOM)")
    train_arr = np.asarray(train, dtype=np.float64)
    valid_arr = np.asarray(valid, dtype=np.float64) if valid else np.empty((0, 4))
    return train_arr[:, 0], train_arr[:, 1], train_arr[:, 2], train_arr[:, 3], valid_arr


def _latency_p0(image_size: np.ndarray, batch: np.ndarray, measured: np.ndarray) -> np.ndarray:
    mem = 196.0 * batch * image_size**2 + 6544.0 * batch + 4159872.0
    comp = 17712.0 * batch * image_size**2 + 313344.0 * batch
    theta_2 = float(np.percentile(measured, 10))
    busy = measured > 2.0 * theta_2
    if not np.any(busy):
        busy = np.ones(measured.shape, dtype=bool)
    theta_0 = float(np.median(mem[busy] / measured[busy]))
    theta_1 = float(np.median(comp[busy] / measured[busy]))
    return np.array([max(theta_0, 1.0), max(theta_1, 1.0), max(theta_2, 1e-9)], dtype=np.float64)


def fit_latency(image_size: np.ndarray, batch: np.ndarray, measured: np.ndarray) -> np.ndarray:
    p0 = _latency_p0(image_size, batch, measured)

    def residual(theta: np.ndarray) -> np.ndarray:
        pred = np.asarray(latency(image_size, batch, theta), dtype=np.float64)
        return pred - measured

    result = least_squares(residual, p0, bounds=(0.0, np.inf), x_scale="jac")
    if not result.success:
        raise SystemExit(f"latency fit failed: {result.message}")
    return result.x


def fit_energy(
    image_size: np.ndarray,
    batch: np.ndarray,
    measured: np.ndarray,
    theta_l: np.ndarray,
) -> np.ndarray:
    """Fit P_idle and e_compute. e_mem stays 0.

    bytes_moved and FLOPs are both proportional to B*S^2, so e_mem and
    e_compute are not separately identifiable. energy() still takes e_mem.
    """

    def model(sb: tuple[np.ndarray, np.ndarray], p_idle: float, e_compute: float) -> np.ndarray:
        s, b = sb
        pred = energy(
            s,
            b,
            {
                "P_idle": p_idle,
                "e_mem": 0.0,
                "e_compute": e_compute,
                "theta_L": theta_l,
            },
        )
        return np.asarray(pred, dtype=np.float64)

    popt, _ = curve_fit(
        model,
        (image_size, batch),
        measured,
        p0=(40.0, 1e-11),
        bounds=(0.0, np.inf),
        maxfev=20000,
    )
    return np.array([popt[0], 0.0, popt[1]], dtype=np.float64)


def _metrics(measured: np.ndarray, predicted: np.ndarray) -> tuple[float, float]:
    resid = predicted - measured
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((measured - np.mean(measured)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else float("nan")
    rel = float(np.mean(np.abs(resid) / np.abs(measured)) * 100.0)
    return r2, rel


def _report(name: str, split: str, measured: np.ndarray, predicted: np.ndarray) -> None:
    r2, rel = _metrics(measured, predicted)
    print(f"{name:8s} {split:10s}  R2={r2:.4f}  relative_error={rel:.2f}%", flush=True)


def main() -> None:
    rows = _load_rows(CSV_PATH)
    image_size, batch, lat_meas, energy_meas, valid = _split(rows)

    theta_l = fit_latency(image_size, batch, lat_meas)
    theta_e = fit_energy(image_size, batch, energy_meas, theta_l)

    lat_hat = np.asarray(latency(image_size, batch, theta_l), dtype=np.float64)
    energy_theta = {
        "P_idle": float(theta_e[0]),
        "e_mem": float(theta_e[1]),
        "e_compute": float(theta_e[2]),
        "theta_L": theta_l,
    }
    energy_hat = np.asarray(energy(image_size, batch, energy_theta), dtype=np.float64)

    _report("latency", "train", lat_meas, lat_hat)
    _report("energy", "train", energy_meas, energy_hat)

    if len(valid) > 0:
        vs, vb, vlat, venergy = valid[:, 0], valid[:, 1], valid[:, 2], valid[:, 3]
        vlat_hat = np.asarray(latency(vs, vb, theta_l), dtype=np.float64)
        venergy_hat = np.asarray(energy(vs, vb, energy_theta), dtype=np.float64)
        _report("latency", "validation", vlat, vlat_hat)
        _report("energy", "validation", venergy, venergy_hat)
    else:
        print("validation: no rows", flush=True)

    payload = {
        "latency": {
            "theta_0_Bps": float(theta_l[0]),
            "theta_1_FLOPs": float(theta_l[1]),
            "theta_2_s": float(theta_l[2]),
        },
        "energy": {
            "P_idle_W": float(theta_e[0]),
            "e_mem_J_per_byte": float(theta_e[1]),
            "e_compute_J_per_FLOP": float(theta_e[2]),
            "theta_L": {
                "theta_0_Bps": float(theta_l[0]),
                "theta_1_FLOPs": float(theta_l[1]),
                "theta_2_s": float(theta_l[2]),
            },
        },
    }
    THETA_PATH.parent.mkdir(parents=True, exist_ok=True)
    THETA_PATH.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {THETA_PATH}", flush=True)


if __name__ == "__main__":
    main()
