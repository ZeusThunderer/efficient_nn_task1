"""Analytical cost model for SmallCNN (numpy only, broadcast over S and B)."""

from __future__ import annotations

from typing import Any, Mapping, Sequence, Union

import numpy as np

Theta = Union[Sequence[float], Mapping[Any, float]]
ArrayLike = Union[float, int, np.ndarray]

_WEIGHTS_BYTES = 4 * (
    4704 + 51200 + 73728 + 32768 + 589824 + 131072 + 131072 + 25600
)


def _as_float64(x: ArrayLike) -> np.ndarray:
    return np.asarray(x, dtype=np.float64)


def _as_result(x: np.ndarray) -> Union[float, np.ndarray]:
    if x.ndim == 0:
        return float(x)
    return x


def _theta_value(theta: Theta, index: int) -> float:
    if isinstance(theta, Mapping):
        for key in (index, f"theta_{index}"):
            if key in theta:
                return float(theta[key])
        raise KeyError(f"theta missing key for index {index} (tried {index}, 'theta_{index}')")
    return float(theta[index])


def flops(image_size: ArrayLike, batch: ArrayLike) -> Union[float, np.ndarray]:
    """Floating-point operations of one forward pass.

    Parameters
    ----------
    image_size : scalar or ndarray
        Side length S of the square input (pixels).
    batch : scalar or ndarray
        Batch size B.

    Returns
    -------
    float or ndarray
        FLOPs (dimensionless count of floating-point operations).
    """
    s = _as_float64(image_size)
    b = _as_float64(batch)
    out = 2.0 * b * (8856.0 * s * s + 156672.0)
    return _as_result(out)


def memory(image_size: ArrayLike, batch: ArrayLike) -> Union[float, np.ndarray]:
    """Peak GPU memory of one forward pass (analytical model).

    Parameters
    ----------
    image_size : scalar or ndarray
        Side length S of the square input (pixels).
    batch : scalar or ndarray
        Batch size B.

    Returns
    -------
    float or ndarray
        Memory in bytes.
    """
    s = _as_float64(image_size)
    b = _as_float64(batch)
    out = _WEIGHTS_BYTES + 44.0 * b * s * s
    return _as_result(out)


def latency(
    image_size: ArrayLike,
    batch: ArrayLike,
    theta: Theta,
) -> Union[float, np.ndarray]:
    """Wall-clock time of one forward pass (bottleneck = max of memory vs compute).

    Parameters
    ----------
    image_size : scalar or ndarray
        Side length S of the square input (pixels).
    batch : scalar or ndarray
        Batch size B.
    theta : sequence or mapping
        Calibrated throughput parameters:
        - theta[0] (or ``theta_0``): effective memory bandwidth, bytes/s
        - theta[1] (or ``theta_1``): effective compute throughput, FLOP/s

    Returns
    -------
    float or ndarray
        Latency in seconds.
    """
    s = _as_float64(image_size)
    b = _as_float64(batch)
    theta_0 = _theta_value(theta, 0)
    theta_1 = _theta_value(theta, 1)
    mem_max = 196.0 * b * s * s + 6544.0 * b + 4159872.0
    comp_max = 17712.0 * b * s * s + 313344.0 * b
    out = np.maximum(mem_max / theta_0, comp_max / theta_1)
    return _as_result(out)


def energy(
    image_size: ArrayLike,
    batch: ArrayLike,
    theta_energy: Mapping[str, Any],
) -> Union[float, np.ndarray]:
    """Energy of one forward pass (additive model, not max of terms).

    E = P_idle * L(B, S, theta_L) + e_mem * bytes_moved + e_compute * FLOPs

    Parameters
    ----------
    image_size : scalar or ndarray
        Side length S of the square input (pixels).
    batch : scalar or ndarray
        Batch size B.
    theta_energy : dict
        - ``P_idle``: idle/static power draw during the forward pass, watts (W)
        - ``e_mem``: energy per byte moved, joules per byte (J/byte)
        - ``e_compute``: energy per FLOP, joules per FLOP (J/FLOP)
        - ``theta_L``: same object as ``theta`` in :func:`latency` (bytes/s and FLOP/s)

    Returns
    -------
    float or ndarray
        Energy in joules (J).
    """
    p_idle = float(theta_energy["P_idle"])
    e_mem = float(theta_energy["e_mem"])
    e_compute = float(theta_energy["e_compute"])
    theta_l = theta_energy["theta_L"]

    s = _as_float64(image_size)
    b = _as_float64(batch)
    l_sec = np.asarray(latency(image_size, batch, theta_l), dtype=np.float64)
    flops_val = np.asarray(flops(image_size, batch), dtype=np.float64)
    bytes_moved = 196.0 * b * s * s + 6544.0 * b + 4159872.0

    out = p_idle * l_sec + e_mem * bytes_moved + e_compute * flops_val
    return _as_result(out)


if __name__ == "__main__":
    s = np.array([32, 64, 128])
    b = np.array([[1], [2], [4]])
    f = flops(s, b)
    assert f.shape == (3, 3)
    m = memory(s, b)
    assert m.shape == (3, 3)
    th = (1e12, 1e12)
    lat = latency(s, b, th)
    assert lat.shape == (3, 3)
    te = {
        "P_idle": 10.0,
        "e_mem": 1e-12,
        "e_compute": 1e-15,
        "theta_L": th,
    }
    e = energy(s, b, te)
    assert e.shape == (3, 3)
    print("broadcast OK:", f.shape, m.shape, lat.shape, e.shape)
