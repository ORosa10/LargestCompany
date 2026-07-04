"""Persist and restore the latest modeling artifacts between Streamlit phases."""

from __future__ import annotations

from pathlib import Path
import pickle
from typing import Any


STORE_DIR = Path.home() / ".largestcompany"
SNAPSHOT_PATH = STORE_DIR / "latest_simulation.pkl"


def _atomic_pickle(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(".tmp")
    with temporary_path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    temporary_path.replace(path)
    return path


def save_simulation_snapshot(
    *,
    result: Any,
    simulation_inputs: Any,
    run_metadata: dict | None,
    source: str,
) -> Path:
    payload = {
        "result": result,
        "simulation_inputs": simulation_inputs,
        "run_metadata": run_metadata or {},
        "source": source,
    }
    return _atomic_pickle(SNAPSHOT_PATH, payload)


def load_simulation_snapshot() -> dict | None:
    if not SNAPSHOT_PATH.exists():
        return None
    try:
        with SNAPSHOT_PATH.open("rb") as handle:
            payload = pickle.load(handle)
    except (OSError, EOFError, pickle.UnpicklingError):
        return None
    required = {"result", "simulation_inputs", "run_metadata", "source"}
    return payload if isinstance(payload, dict) and required.issubset(payload) else None


def phase_artifact_path(phase: str) -> Path:
    safe_name = "".join(character for character in str(phase).lower() if character.isalnum() or character in {"_", "-"})
    if not safe_name:
        raise ValueError("Phase artifact name cannot be empty.")
    return STORE_DIR / f"{safe_name}.pkl"


def save_phase_artifact(phase: str, payload: dict) -> Path:
    if not isinstance(payload, dict):
        raise ValueError("Phase artifact payload must be a dictionary.")
    return _atomic_pickle(phase_artifact_path(phase), payload)


def load_phase_artifact(phase: str) -> dict | None:
    path = phase_artifact_path(phase)
    if not path.exists():
        return None
    try:
        with path.open("rb") as handle:
            payload = pickle.load(handle)
    except (OSError, EOFError, pickle.UnpicklingError):
        return None
    return payload if isinstance(payload, dict) else None
