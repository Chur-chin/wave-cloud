"""
test_functions.py  —  run with: python -m pytest tests/ -v
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from wave_cloud.registry import registry
from wave_cloud import functions   # register all functions


def invoke(name, payload={}):
    result = registry.invoke(name, payload)
    assert result["status"] == "ok", f"{name} failed: {result.get('error')}"
    return result["data"]


# ── lyapunov ────────────────────────────────────────────────────────
def test_lyapunov_returns_float():
    data = invoke("lyapunov", {"w0": 1.0, "g1": 0.8, "T": 80})
    assert isinstance(data["lambda_max"], float)
    assert isinstance(data["is_chaotic"], bool)

def test_lyapunov_chaotic_regime():
    data = invoke("lyapunov", {"w0": 1.0, "g1": 1.2, "T": 100})
    assert data["lambda_max"] != 0.0

def test_lyapunov_has_params():
    data = invoke("lyapunov", {"w0": 1.5, "g1": 0.5, "T": 60})
    assert "params" in data
    assert data["params"]["w0"] == 1.5


# ── bifurcation ─────────────────────────────────────────────────────
def test_bifurcation_returns_lists():
    data = invoke("bifurcation", {"w0": 1.0, "n_points": 10, "T": 120})
    assert isinstance(data["g1_values"], list)
    assert isinstance(data["x_peaks"],   list)
    assert len(data["g1_values"]) == len(data["x_peaks"])

def test_bifurcation_n_peaks_positive():
    data = invoke("bifurcation", {"w0": 1.0, "n_points": 15, "T": 130})
    assert data["n_peaks"] > 0


# ── synapse ─────────────────────────────────────────────────────────
def test_synapse_ltp_at_zero():
    data = invoke("synapse", {"theta": 0.0, "intensity": 1e6})
    assert data["plasticity"] == "LTP"
    assert data["delta_EF_meV"] > 0

def test_synapse_ltd_at_ninety():
    data = invoke("synapse", {"theta": 90.0, "intensity": 1e6})
    assert data["plasticity"] == "LTD"
    assert data["delta_EF_meV"] < 0

def test_synapse_weight_sum():
    data = invoke("synapse", {"theta": 45.0})
    assert abs(data["wI"] + data["wII"] - 1.0) < 1e-6

def test_synapse_neutral():
    data = invoke("synapse", {"theta": 45.0, "intensity": 1.0})
    assert data["plasticity"] == "neutral"


# ── polariton ───────────────────────────────────────────────────────
def test_polariton_band_I():
    data = invoke("polariton", {"band": "I", "d_nm": 10})
    assert len(data["omega_cm"]) > 0
    assert all(v >= 0 for v in data["q_re"])

def test_polariton_band_II():
    data = invoke("polariton", {"band": "II", "d_nm": 10})
    assert data["band"] == "II"

def test_polariton_length_consistency():
    data = invoke("polariton", {"n_points": 50})
    assert len(data["omega_cm"]) == len(data["q_re"]) == 50


# ── stdp ────────────────────────────────────────────────────────────
def test_stdp_ltp_positive():
    data = invoke("stdp", {"dt_min": 0, "dt_max": 80, "n_points": 50})
    assert all(v >= 0 for v in data["dW"])

def test_stdp_ltd_negative():
    data = invoke("stdp", {"dt_min": -80, "dt_max": 0, "n_points": 50})
    assert all(v <= 0 for v in data["dW"])

def test_stdp_length():
    data = invoke("stdp", {"n_points": 100})
    assert len(data["delta_t"]) == 100 == len(data["dW"])


# ── hbn_epsilon ─────────────────────────────────────────────────────
def test_hbn_epsilon_band_I_negative():
    data = invoke("hbn_epsilon", {"omega_min": 760, "omega_max": 825, "n_points": 10})
    assert any(v < 0 for v in data["eps_perp_re"])

def test_hbn_epsilon_lengths():
    data = invoke("hbn_epsilon", {"n_points": 50})
    n = len(data["omega_cm"])
    assert n == len(data["eps_perp_re"]) == len(data["eps_par_re"]) == 50


# ── registry ────────────────────────────────────────────────────────
def test_registry_lists_all():
    fns = registry.list_functions()
    expected = {"lyapunov", "bifurcation", "scaling_law",
                "synapse", "polariton", "stdp", "hbn_epsilon"}
    assert expected.issubset(set(fns))

def test_registry_unknown_function():
    result = registry.invoke("nonexistent", {})
    assert result["status"] == "error"
    assert "not found" in result["error"]
