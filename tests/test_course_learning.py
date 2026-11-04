"""Course navigation resolves to real concepts/examples and excludes private sources."""

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "apps/learning-content/site"


def load(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_course_pages_link_to_existing_anchors_and_worked_examples():
    from test_learning_content import Links

    concepts = Links((SITE / "concepts/index.html").read_text())
    courses = json.loads((SITE / "learning_map.json").read_text())["courses"]
    assert {course["id"] for course in courses} == {"AIA", "AIMLA", "MAL"}
    mal = next(course for course in courses if course["id"] == "MAL")
    assert len(mal["lectures"][0]["concepts"]) == 9
    assert all(concept["id"].startswith("mal-cnn-") for concept in mal["lectures"][0]["concepts"])
    for course in courses:
        page = SITE / "courses" / course["id"] / "index.html"
        assert page.is_file()
        assert f'href="/learn/courses/{course["id"]}/"' in (SITE / "courses/index.html").read_text()
        for lecture in course["lectures"]:
            for concept in lecture["concepts"]:
                assert concept["id"] in concepts.ids
            for example in lecture["examples"]:
                if example["url"].startswith("/learn/examples/"):
                    assert (SITE / example["url"].removeprefix("/learn/") / "index.html").is_file()
                else:
                    replay = Links((ROOT / "apps/learning-replay/index.html").read_text())
                    assert example["url"].partition("#")[2] in replay.ids


def test_course_export_ignores_private_metadata_and_cleans_retired_pages(tmp_path):
    module = load(ROOT / "scripts/build_learning_site.py")
    mapping = tmp_path / "map.yaml"
    mapping.write_text(json.dumps({"courses": [{"id": "AIA", "title": "AI",
        "private_notes": "secret text", "pdf": "private.pdf", "lectures": [
            {"id": "L1", "title": "Lesson 1", "concepts": ["weights"], "examples": ["weighted-sum"],
             "notes": "never publish this"}]}]}))
    courses = module.load_courses({"course_map": "map.yaml"}, tmp_path,
                                  [{"id": "weights", "title": "Weights", "topic": "weights"}],
                                  {"weights": "weights"})
    from jinja2 import Environment, FileSystemLoader, select_autoescape
    env = Environment(loader=FileSystemLoader(ROOT / "apps/learning-content/templates"),
                      autoescape=select_autoescape(["html"]))
    module.write_courses(tmp_path / "site", courses, env)
    assert not any(secret in (tmp_path / "site/learning_map.json").read_text()
                   for secret in ["secret text", "private.pdf", "never publish"])
    module.write_courses(tmp_path / "site", [], env)
    assert not (tmp_path / "site/courses/AIA").exists()


@pytest.mark.parametrize("field,value,message", [("concepts", ["absent"], "Unknown course concept"),
                                                ("examples", ["absent"], "Unknown worked example")])
def test_course_map_rejects_broken_learning_links(tmp_path, field, value, message):
    module = load(ROOT / "scripts/build_learning_site.py")
    lesson = {"id": "L1", "title": "Lesson", field: value}
    (tmp_path / "map.yaml").write_text(json.dumps({"courses": [{"id": "AIA", "title": "AI", "lectures": [lesson]}]}))
    with pytest.raises(ValueError, match=message):
        module.load_courses({"course_map": "map.yaml"}, tmp_path, [], {})


def test_default_neuron_trace_is_reproducible_and_relu_handles_negative_sum():
    example = load(ROOT / "scripts/examples/weighted_sum.py")
    trace = json.loads((ROOT / "apps/learning-examples/weighted-sum/trace.json").read_text())
    actual = example.calculate(trace["inputs"], trace["weights"], trace["bias"])
    # Python versions may sum these binary floats with different last bits.
    assert actual.keys() == trace.keys()
    for key in trace:
        assert actual[key] == pytest.approx(trace[key], abs=1e-12)
    assert trace["products"] == pytest.approx([.8, -.2, -.3])
    assert trace["weighted_sum"] == pytest.approx(.4)
    assert trace["relu"] == pytest.approx(.4)
    assert example.calculate([2], [-1], .1)["relu"] == 0
    with pytest.raises(ValueError):
        example.calculate([1, 2], [1], 0)


def test_public_bundle_excludes_local_study_materials():
    for path in SITE.rglob("*"):
        if path.is_file():
            assert path.suffix not in {".pdf", ".zip"}
            assert "private-img" not in path.parts
            if path.suffix in {".html", ".json"}:
                text = path.read_text()
                assert "LOCAL STUDY" not in text
                assert "course-private" not in text
                assert "Curs1_Notite" not in text
