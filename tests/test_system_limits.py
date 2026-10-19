"""Reject expensive research operations before constructing the comparison graph."""


def test_excessive_cross_paper_pairs_are_explained_without_saving_a_report(client, monkeypatch):
    papers = client.post("/api/demo/seed").json()["papers"][:2]
    store = client.app.state.store
    before = client.get("/api/reports").json()
    rows = [{"paper_id": paper["id"]} for paper in papers for _ in range(50)]
    monkeypatch.setattr(store, "experiments_for", lambda ids: rows)
    response = client.post("/api/compare", json={"paper_ids": [paper["id"] for paper in papers]})
    assert response.status_code == 422
    assert "2000" in response.json()["detail"]
    assert "减少" in response.json()["detail"]
    assert client.get("/api/reports").json() == before
