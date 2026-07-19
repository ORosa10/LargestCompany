import pickle

import pandas as pd

import simulation_store


def test_load_phase_artifact_restores_from_repo(tmp_path, monkeypatch):
    home = tmp_path / "home_store"
    repo = tmp_path / "repo_state"
    repo.mkdir(parents=True)
    monkeypatch.setattr(simulation_store, "STORE_DIR", home)
    monkeypatch.setattr(simulation_store, "REPO_STATE_DIR", repo)

    payload = {"selected_ticker": "NVDA", "note": pd.DataFrame({"a": [1, 2]})}
    with (repo / "phase4.pkl").open("wb") as handle:
        pickle.dump(payload, handle)

    assert not (home / "phase4.pkl").exists()
    restored = simulation_store.load_phase_artifact("phase4")
    assert restored is not None
    assert restored["selected_ticker"] == "NVDA"
    # the file was materialized into the local store
    assert (home / "phase4.pkl").exists()


def test_local_artifact_takes_precedence_over_repo(tmp_path, monkeypatch):
    home = tmp_path / "home_store"
    repo = tmp_path / "repo_state"
    home.mkdir(parents=True)
    repo.mkdir(parents=True)
    monkeypatch.setattr(simulation_store, "STORE_DIR", home)
    monkeypatch.setattr(simulation_store, "REPO_STATE_DIR", repo)

    with (home / "phase2.pkl").open("wb") as handle:
        pickle.dump({"source": "local"}, handle)
    with (repo / "phase2.pkl").open("wb") as handle:
        pickle.dump({"source": "repo"}, handle)

    restored = simulation_store.load_phase_artifact("phase2")
    assert restored["source"] == "local"


def test_saved_session_files_lists_repo_pickles(tmp_path, monkeypatch):
    repo = tmp_path / "repo_state"
    repo.mkdir(parents=True)
    monkeypatch.setattr(simulation_store, "REPO_STATE_DIR", repo)
    (repo / "phase4.pkl").write_bytes(b"x")
    (repo / "latest_simulation.pkl").write_bytes(b"y")
    (repo / "notes.txt").write_text("ignore me")
    assert simulation_store.saved_session_files() == ["latest_simulation.pkl", "phase4.pkl"]
