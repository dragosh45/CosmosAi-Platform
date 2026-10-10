"""Keep the low-cost Cloud Run release contract reviewable without cloud access."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read(relative_path: str) -> str:
    return (ROOT / relative_path).read_text()


def test_release_images_accept_cloud_run_port_overrides():
    for dockerfile in (
        "apps/api-gateway/Dockerfile",
        "apps/inference-router/Dockerfile",
        "apps/galaxy-classifier-service/Dockerfile.release",
    ):
        assert "${PORT:-8000}" in read(dockerfile)


def test_cloud_run_sidecars_use_distinct_localhost_ports_and_startup_order():
    service = read("deploy/cloud-run/service.template.yaml")
    for port in ("8080", "8001", "8002"):
        assert port in service
    assert "http://127.0.0.1:8001" in service
    assert "http://127.0.0.1:8002" in service
    assert '"api-gateway":["inference-router"]' in service
    assert '"inference-router":["galaxy-classifier"]' in service
    assert 'run.googleapis.com/maxScale: "1"' in service
    assert 'autoscaling.knative.dev/minScale: "0"' in service
    assert "containerConcurrency: 1" in service


def test_cloud_build_publishes_all_three_release_images():
    build = read("deploy/cloud-run/cloudbuild.yaml")
    assert "apps/api-gateway/Dockerfile" in build
    assert "apps/inference-router/Dockerfile" in build
    assert "apps/galaxy-classifier-service/Dockerfile.release" in build
    for image in ("gateway:${_TAG}", "router:${_TAG}", "classifier:${_TAG}"):
        assert image in build


def test_public_access_requires_a_separate_confirmed_action():
    deploy = read("deploy/cloud-run/deploy.sh")
    publish = read("deploy/cloud-run/make-public.sh")
    assert "add-iam-policy-binding" not in deploy
    assert "roles/run.invoker" not in deploy
    assert 'confirmation" == "PUBLIC"' in publish
    assert "roles/run.invoker" in publish
