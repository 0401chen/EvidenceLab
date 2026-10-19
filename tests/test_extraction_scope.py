"""Explicit source scope is preserved and never silently becomes whole-document."""
import pymupdf
import pytest


def upload_pages(client):
    document = pymupdf.open()
    for index in (1, 2):
        page = document.new_page()
        page.insert_text((70, 70), f"Results {index}\nExperiment: Model-{index}\nMethod: Model-{index}\n"
                         f"Metric: accuracy\nValue: {80 + index}%\nDataset: SyntheticSet\nSplit: test-A")
    content = document.tobytes()
    document.close()
    response = client.post('/api/papers/upload', files={'files': ('synthetic.pdf', content, 'application/pdf')})
    assert response.status_code == 200
    return response.json()['papers'][0]['id']


def test_selected_page_candidate_has_only_selected_sources_and_keeps_current_records(client):
    identifier = upload_pages(client)
    original = client.get(f'/api/papers/{identifier}').json()
    response = client.post(f'/api/papers/{identifier}/extract', json={
        'mode': 'rule', 'reparse': True, 'pages': [2, 2], 'max_results': 1})
    assert response.status_code == 200
    detail = response.json()
    candidate = detail['extraction_candidates'][0]
    assert candidate['extraction_scope'] == {'pages': [2], 'total_pages': 2,
                                              'max_results': 1, 'truncated': False, 'completeness': 'not_assessed'}
    assert len(candidate['experiments']) == 1 and candidate['experiments'][0]['value'] == 82
    assert {source['page'] for source in candidate['source_evidence']} == {2}
    assert detail['experiments'] == original['experiments']
    assert detail['evidence'] == original['evidence']


@pytest.mark.parametrize('pages', [[], [0], [3], [True], ['2']])
def test_invalid_page_request_leaves_candidates_and_current_data_unchanged(client, pages):
    identifier = upload_pages(client)
    original = client.get(f'/api/papers/{identifier}').json()
    response = client.post(f'/api/papers/{identifier}/extract', json={'mode': 'rule', 'pages': pages})
    assert response.status_code == 422
    assert client.get(f'/api/papers/{identifier}').json() == original
