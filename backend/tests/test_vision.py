from pathlib import Path
import numpy as np
import pytest
from backend.app.ingest.ffmpeg_utils import generate_synthetic_video, create_proxy_video
from backend.app.vision.scenes import detect_scene_cuts, compute_cut_density
from backend.app.vision.clipzs import classify_proxy_frames
from backend.app.vision.faces import extract_faces_from_proxy
from backend.app.vision.ocr import compute_visual_score


@pytest.fixture(scope="session")
def sample_proxy(tmp_path_factory) -> Path:
    temp_dir = tmp_path_factory.mktemp("vision_test")
    src = temp_dir / "src.mp4"
    proxy = temp_dir / "proxy.mp4"
    generate_synthetic_video(src, duration=10.0, with_audio=False)
    create_proxy_video(src, proxy)
    return proxy


def test_scene_cut_density():
    cuts = [2.0, 5.0, 8.0]
    density = compute_cut_density(cuts, duration=10.0, window_sec=4.0)
    assert len(density) == 10
    assert not np.isnan(density).any()
    assert density[5] > 0.0


def test_clipzs_classification(sample_proxy: Path):
    action_prob, trailer_prob, available = classify_proxy_frames(sample_proxy, duration=10.0)
    assert len(action_prob) == 10
    assert len(trailer_prob) == 10
    assert not np.isnan(action_prob).any()


def test_face_extraction_fallback_no_faces(sample_proxy: Path):
    # Synthetic color pattern video contains no human faces
    tracks, face_score, available = extract_faces_from_proxy(sample_proxy, duration=10.0)
    assert len(face_score) == 10
    assert available is False
    assert len(tracks) == 0
    assert (face_score == 0.0).all()


def test_compute_visual_score(sample_proxy: Path):
    v_score, act, cuts, available = compute_visual_score(
        sample_proxy,
        duration=10.0,
        keywords=["GTA", "Vice City"],
    )
    assert len(v_score) == 10
    assert not np.isnan(v_score).any()
    assert (v_score >= -3.0).all() and (v_score <= 3.0).all()
    assert available is True
