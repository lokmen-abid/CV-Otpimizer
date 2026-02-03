import pytest
import numpy as np
from analyzers import embeddings


def test_chunk_text_small():
    text = "Short paragraph"
    chunks = embeddings._chunk_text(text, max_chars=100)
    assert isinstance(chunks, list)
    assert len(chunks) == 1


def test_chunk_text_large():
    text = ("A" * 3500) + "\n\n" + ("B" * 4000)
    chunks = embeddings._chunk_text(text, max_chars=3000, overlap=100)
    assert all(len(c) <= 3100 for c in chunks)


def test_get_embeddings_no_model(monkeypatch):
    # Simulate absence of sentence-transformers by monkeypatching _load_model
    monkeypatch.setattr(embeddings, "_load_model", lambda name: None)
    res = embeddings.get_embeddings(["hello world"], model_name="nonexistent-model")
    assert res is None


def test_compute_embedding_similarity_with_stub(monkeypatch, tmp_path):
    # Create a stub model with deterministic encode
    class StubModel:
        def get_sentence_embedding_dimension(self):
            return 3
        def encode(self, batch, convert_to_numpy=True):
            import numpy as np
            out = []
            for i, text in enumerate(batch):
                arr = np.array([1.0, 0.0, 0.0]) if text == "same" else np.array([0.0, 1.0, 0.0])
                out.append(arr)
            return np.vstack(out)

    monkeypatch.setattr(embeddings, "_load_model", lambda name: StubModel())
    sim = embeddings.compute_embedding_similarity("same", "same", model_name="stub")
    assert isinstance(sim, dict)
    assert sim.get('method') == 'sentence-transformers'
    assert sim.get('score') == pytest.approx(1.0, rel=1e-3)

    sim2 = embeddings.compute_embedding_similarity("same", "different", model_name="stub")
    assert sim2.get('score') == pytest.approx(0.0, abs=0.001)


def test_cache_write_and_read(monkeypatch, tmp_path):
    # Use stub model and test that cache files are written to EMBEDDING_CACHE_DIR
    d = tmp_path / "embcache"
    d.mkdir()
    monkeypatch.setenv('EMBEDDING_CACHE_DIR', str(d))
    # override module constants
    monkeypatch.setattr(embeddings, 'EMBEDDING_CACHE_DIR', str(d))
    monkeypatch.setattr(embeddings, '_CACHE', None)

    class StubModel2:
        def get_sentence_embedding_dimension(self):
            return 2
        def encode(self, batch, convert_to_numpy=True):
            import numpy as np
            return np.array([[1.0, 0.0]])

    monkeypatch.setattr(embeddings, '_load_model', lambda name: StubModel2())
    res = embeddings.get_embeddings(["hello world"], model_name="stub", use_cache=True)
    assert res is not None
    # check that cache files were created
    files = list(d.iterdir())
    assert any(str(f).endswith('.npy') for f in files)
