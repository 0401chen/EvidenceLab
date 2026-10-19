"""Offline regressions for two browser tabs editing the same saved experiment."""
from __future__ import annotations

import json


def seeded(client):
    papers = client.post("/api/demo/seed").json()["papers"]
    paper_id = papers[0]["id"]
    return papers, client.get(f"/api/papers/{paper_id}").json()


def database_snapshot(store):
    with store.connect() as db:
        return {table: [tuple(row) for row in db.execute(f"SELECT * FROM {table} ORDER BY rowid")]
                for table in ("papers", "evidence", "experiments", "experiment_history", "reports", "report_states")}


def test_two_tabs_reject_stale_edit_without_changing_history_or_reports(client):
    papers, tab_a = seeded(client)
    paper_id = tab_a["id"]
    tab_b = client.get(f"/api/papers/{paper_id}").json()
    experiment_id = tab_a["experiments"][0]["id"]
    token = tab_a["experiment_revisions"][experiment_id]
    assert token == tab_b["experiment_revisions"][experiment_id]
    assert len(token) == 64
    first = client.patch(f"/api/experiments/{experiment_id}", json={"name": "Tab A saved correction", "expected_revision": token})
    assert first.status_code == 200, first.text
    current = client.get(f"/api/papers/{paper_id}").json()
    assert current["experiment_revisions"][experiment_id] != token
    # A fresh comparison must not be invalidated by the rejected Tab B request.
    report = client.post("/api/compare", json={"paper_ids": [p["id"] for p in papers]}).json()
    before = database_snapshot(client.app.state.store)
    rejected = client.patch(f"/api/experiments/{experiment_id}", json={"name": "Tab B stale correction", "expected_revision": token})
    assert rejected.status_code == 409
    assert "重新载入" in rejected.json()["detail"]
    assert database_snapshot(client.app.state.store) == before
    assert client.get(f"/api/reports/{report['id']}/export").status_code == 200
    assert client.get(f"/api/papers/{paper_id}").json() == current


def test_valid_revision_updates_successfully_without_persisting_tokens(client):
    _, before = seeded(client)
    experiment_id = before["experiments"][0]["id"]
    response = client.patch(f"/api/experiments/{experiment_id}", json={
        "name": "Reviewed with a current revision", "expected_revision": before["experiment_revisions"][experiment_id],
    })
    assert response.status_code == 200, response.text
    assert "expected_revision" not in response.json()
    assert "_revision" not in response.json()
    assert all("expected_revision" not in note for note in response.json()["notes"])
    with client.app.state.store.connect() as db:
        payload = json.loads(db.execute("SELECT payload FROM experiments WHERE id=?", (experiment_id,)).fetchone()[0])
        paper = json.loads(db.execute("SELECT payload FROM papers WHERE id=?", (before["id"],)).fetchone()[0])
    assert payload == response.json()
    assert "_revision" not in payload and "expected_revision" not in payload
    assert "experiment_revisions" not in paper
    history = client.get(f"/api/experiments/{experiment_id}/history").json()
    assert history[0]["experiment"] == before["experiments"][0]


def test_legacy_client_without_revision_remains_compatible(client):
    _, paper = seeded(client)
    experiment_id = paper["experiments"][0]["id"]
    response = client.patch(f"/api/experiments/{experiment_id}", json={"name": "Legacy edit without token"})
    assert response.status_code == 200, response.text
    assert response.json()["name"] == "Legacy edit without token"
    current = client.get(f"/api/papers/{paper['id']}").json()
    assert current["experiment_revisions"][experiment_id] != paper["experiment_revisions"][experiment_id]


def test_revision_is_compared_again_in_the_actual_write_transaction(client, monkeypatch):
    _, paper = seeded(client)
    experiment_id = paper["experiments"][0]["id"]
    store = client.app.state.store
    update = store.update_experiment
    winner = {**paper["experiments"][0], "name": "Concurrent winner after API read"}

    def write_between_read_and_update(experiment, expected=None, expected_revision=None):
        assert update(winner)
        return update(experiment, expected=expected, expected_revision=expected_revision)

    monkeypatch.setattr(store, "update_experiment", write_between_read_and_update)
    response = client.patch(f"/api/experiments/{experiment_id}", json={
        "name": "Losing edit", "expected_revision": paper["experiment_revisions"][experiment_id],
    })
    assert response.status_code == 409
    assert store.get_experiment(experiment_id) == winner
    assert len(store.experiment_history(experiment_id)) == 1  # Only the winner archives a prior record.


def test_store_revision_check_is_independent_of_the_api_read_snapshot(client):
    _, paper = seeded(client)
    store = client.app.state.store
    experiment = paper["experiments"][0]
    token = paper["experiment_revisions"][experiment["id"]]
    reordered = dict(reversed(list(experiment.items())))
    assert store.experiment_payload_revision(reordered) == token
    assert store.update_experiment({**experiment, "name": "First writer"}, expected_revision=token)
    before = database_snapshot(store)
    assert not store.update_experiment({**experiment, "name": "Stale direct writer"}, expected_revision=token)
    assert database_snapshot(store) == before
