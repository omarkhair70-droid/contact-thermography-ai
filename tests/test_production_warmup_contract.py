from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_production_container_pays_dino_cold_start_before_serving_exams():
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    asgi = (ROOT / "app" / "asgi.py").read_text(encoding="utf-8")

    assert "MUMGUARD_DINOV2_WARMUP=1" in dockerfile
    assert "MUMGUARD_DINOV2_WARMUP_MODE=blocking" in dockerfile
    assert "--start-period=240s" in dockerfile
    assert 'warmup_mode == "blocking"' in asgi
    assert "encode_patches_batch" in asgi
    assert "np.zeros((224, 224, 3)" in asgi
    assert "_warm_dinov2_runtime(fail_hard=True)" in asgi
    assert "DINOV2_WARMUP_READY" in asgi
