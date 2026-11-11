"""Unit tests for config/config.yaml and config/settings.py."""

import os
from pathlib import Path
from config.settings import load_settings, get_settings, CONFIG_YAML_PATH, BASE_DIR


def test_yaml_file_exists():
    assert CONFIG_YAML_PATH.exists(), f"File {CONFIG_YAML_PATH} tidak ditemukan!"


def test_settings_load_defaults():
    s = load_settings()
    assert s.app.name == "rag_langchain"
    assert s.app.air_gapped is True
    assert s.vector_store.port == 8001
    assert s.vector_store.collection_name == "rag_langchain_kb"
    assert s.chunking.chunk_size == 256
    assert s.chunking.chunk_overlap == 26
    assert s.embeddings.model_name == "nomic-ai/nomic-embed-text-v1.5"
    assert s.pre_retrieval.hybrid_search_enabled is True
    assert s.pre_retrieval.dense_weight == 0.7
    assert s.pre_retrieval.sparse_weight == 0.3
    assert s.pre_retrieval.fetch_k == 20
    assert s.post_retrieval.reranker.enabled is True
    assert s.post_retrieval.reranker.model_name == "cross-encoder/ms-marco-MiniLM-L-6-v2"
    assert s.llm.model == "qwen2.5-7b-int4"
    assert s.llm.temperature == 0.1
    assert s.hardware.shared_vram_limit_gb == 4.5


def test_directories_created():
    s = get_settings()
    assert s.paths.data_raw_dir.exists()
    assert s.paths.data_processed_dir.exists()
    assert s.paths.data_dlq_dir.exists()
    assert s.paths.logs_dir.exists()
    assert s.paths.conversion_log_path.name == "conversion.log"
    assert s.paths.conversion_history_path.name == "conversion_history.jsonl"


def test_env_override():
    os.environ["CHROMA_PORT"] = "9999"
    os.environ["CHUNK_SIZE"] = "512"
    os.environ["EMBEDDING_DEVICE"] = "cuda"

    try:
        s = load_settings()
        assert s.vector_store.port == 9999
        assert s.chunking.chunk_size == 512
        assert s.embeddings.device == "cuda"
    finally:
        # Reset env
        os.environ.pop("CHROMA_PORT", None)
        os.environ.pop("CHUNK_SIZE", None)
        os.environ.pop("EMBEDDING_DEVICE", None)


if __name__ == "__main__":
    test_yaml_file_exists()
    test_settings_load_defaults()
    test_directories_created()
    test_env_override()
    print("✅ All config tests passed successfully!")
