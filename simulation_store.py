"""Persist and restore the latest modeling artifacts between Streamlit phases."""

from __future__ import annotations

from pathlib import Path
import pickle
import shutil
import subprocess
from typing import Any


STORE_DIR = Path.home() / ".largestcompany"
SNAPSHOT_PATH = STORE_DIR / "latest_simulation.pkl"

# Repo-backed persistence: a "Save session" commits the local artifacts here so a
# fresh clone (a new Codespace) restores the work automatically on load.
REPO_STATE_DIR = Path(__file__).resolve().parent / "saved_state"


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
    _restore_from_repo("latest_simulation.pkl")
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
    _restore_from_repo(path.name)
    if not path.exists():
        return None
    try:
        with path.open("rb") as handle:
            payload = pickle.load(handle)
    except (OSError, EOFError, pickle.UnpicklingError):
        return None
    return payload if isinstance(payload, dict) else None



def _restore_from_repo(filename: str) -> None:
    """If a local artifact is missing but a saved copy exists in the repo, restore it.

    This makes a freshly cloned checkout (for example a brand-new Codespace) pick
    up the last saved session transparently the first time a page loads it.
    """

    local_path = STORE_DIR / filename
    repo_path = REPO_STATE_DIR / filename
    if local_path.exists() or not repo_path.exists():
        return
    try:
        STORE_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy2(repo_path, local_path)
    except OSError:
        pass


def save_session_to_repo() -> str:
    """Copy the current session artifacts into the repo and commit/push them.

    Returns a human-readable status message for the UI. Git operations are best
    effort: in Codespaces the provided token can push; elsewhere the commit still
    lands locally and the message says so.
    """

    if not STORE_DIR.exists():
        return "No session data to save yet - run Phase 1 first."
    REPO_STATE_DIR.mkdir(parents=True, exist_ok=True)
    copied = 0
    for pickle_path in STORE_DIR.glob("*.pkl"):
        try:
            shutil.copy2(pickle_path, REPO_STATE_DIR / pickle_path.name)
            copied += 1
        except OSError:
            continue
    if copied == 0:
        return "No session data to save yet - run Phase 1 first."

    repo = REPO_STATE_DIR.parent
    git = ["git", "-C", str(repo)]
    try:
        subprocess.run(git + ["add", "saved_state"], check=True, capture_output=True, text=True)
        commit = subprocess.run(
            git + ["-c", "user.email=app@largestcompany", "-c", "user.name=LargestCompany app",
                   "commit", "-m", "Save Streamlit session state"],
            capture_output=True, text=True,
        )
        blob = (commit.stdout + commit.stderr).lower()
        if "nothing to commit" in blob:
            return f"Session already up to date ({copied} files saved)."
        push = subprocess.run(git + ["push"], capture_output=True, text=True)
        if push.returncode != 0:
            return (
                f"Saved and committed {copied} files, but the push failed "
                f"({push.stderr.strip()[:160]}). Run `git push` in the terminal."
            )
        return f"Saved {copied} files and pushed to GitHub. A new Codespace will restore them."
    except Exception as exc:  # noqa: BLE001 - surface any git failure to the UI
        return f"Copied {copied} files into the repo, but git failed: {exc}"


def saved_session_files() -> list[str]:
    """Names of session artifacts currently committed in the repo."""

    if not REPO_STATE_DIR.exists():
        return []
    return sorted(path.name for path in REPO_STATE_DIR.glob("*.pkl"))
