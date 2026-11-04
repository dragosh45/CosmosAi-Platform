"""The release check rejects missing assets and stale successful responses."""
from pathlib import Path
from urllib.parse import urlsplit

import pytest
import requests

ROOT = Path(__file__).resolve().parents[1]


class ReleaseFiles:
    def __init__(self, failure=None):
        self.failure = failure
        self.visited = []

    def get(self, url, **kwargs):
        path = urlsplit(url).path
        self.visited.append(path)
        if path.startswith('/assets/'):
            file = ROOT / 'apps/api-gateway/static' / path.removeprefix('/assets/')
        elif path in ('/learn/', '/learn/viewer.js'):
            file = ROOT / 'apps/learning-replay' / (path.removeprefix('/learn/') or 'index.html')
        else:
            file = ROOT / 'apps/learning-content/site' / path.removeprefix('/learn/')
            if file.is_dir():
                file = file / 'index.html'
        response = requests.Response()
        response.status_code = 200
        response._content = file.read_bytes()
        response.url = url
        if self.failure and path.endswith('/cnn-fundamentals/example.js'):
            response.status_code = 404 if self.failure == 'missing' else 200
            response._content = b'old build'
        return response


def test_release_learning_covers_courses_examples_and_replay(load_service_module):
    checker = load_service_module('release_checker', 'scripts/check_release_demo.py')
    session = ReleaseFiles()
    assert checker.check_learning(session, 'https://example.test') >= 20
    for path in ['/learn/courses/MAL/', '/learn/examples/cnn-fundamentals/math.js',
                 '/learn/examples/lab1-preprocessing/trace.json', '/learn/viewer.js']:
        assert path in session.visited


@pytest.mark.parametrize('failure', ['missing', 'stale'])
def test_release_learning_rejects_missing_or_stale_examples(load_service_module, failure):
    checker = load_service_module('release_checker_bad', 'scripts/check_release_demo.py')
    with pytest.raises((AssertionError, requests.HTTPError)):
        checker.check_learning(ReleaseFiles(failure), 'https://example.test')
