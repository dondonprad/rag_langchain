"""Configuration loader combining config.yaml and .env variables.

Menyediakan akses konfigurasi terpusat, type-safe, dan otomatis menyinkronkan
variabel lingkungan (.env) untuk isolasi air-gapped serta optimasi hardware.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

try:
    import yaml
except ImportError:
    yaml = None


# Path root project
BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_YAML_PATH = BASE_DIR / "config" / "config.yaml"
ENV_PATH = BASE_DIR / ".env"

# Muat berkas .env jika tersedia
if ENV_PATH.exists() and load_dotenv is not None:
    load_dotenv(dotenv_path=ENV_PATH)
elif load_dotenv is not None:
    load_dotenv()


def _get_env(key: str, default: Any, cast_type: type = str) -> Any:
    """Mengambil variabel environment dengan fallback default dan casting tipe."""
    val = os.getenv(key)
    if val is None:
        return default
    if cast_type == bool:
        return str(val).strip().lower() in ("true", "1", "yes", "y", "on")
    if cast_type in (int, float):
        try:
            return cast_type(val)
        except ValueError:
            return default
    return cast_type(val)


@dataclass
class AppConfig:
    name: str = "rag_langchain"
    version: str = "0.1.0"
    description: str = ""
    environment: str = "local"
    air_gapped: bool = True
    telemetry_enabled: bool = False


@dataclass
class PathsConfig:
    data_raw_dir: Path = BASE_DIR / "data" / "raw"
    data_processed_dir: Path = BASE_DIR / "data" / "processed"
    data_dlq_dir: Path = BASE_DIR / "data" / "dlq"
    dlq_error_log: Path = BASE_DIR / "data" / "dlq" / "error.log"
    bm25_index_path: Path = BASE_DIR / "data" / "processed" / "bm25_index.pkl"
    lineage_cache_path: Path = BASE_DIR / "data" / "processed" / "lineage_cache.json"
    logs_dir: Path = BASE_DIR / "logs"
    conversion_log_path: Path = BASE_DIR / "logs" / "conversion.log"
    conversion_history_path: Path = BASE_DIR / "logs" / "conversion_history.jsonl"

    def ensure_directories(self) -> None:
        """Membuat direktori kerja jika belum ada."""
        self.data_raw_dir.mkdir(parents=True, exist_ok=True)
        self.data_processed_dir.mkdir(parents=True, exist_ok=True)
        self.data_dlq_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class IngestionConfig:
    supported_extensions: List[str] = field(default_factory=lambda: [".md"])
    encoding: str = "utf-8"
    hashing_algorithm: str = "sha256"
    min_file_size_chars: int = 50
    page_tag_patterns: List[str] = field(
        default_factory=lambda: [
            r"\[Hal:\s*(\d+)\]",
            r"<!--\s*page:\s*(\d+)\s*-->",
        ]
    )


@dataclass
class ChunkingConfig:
    chunk_size: int = 256
    chunk_overlap: int = 26
    headers_to_split_on: List[List[str]] = field(
        default_factory=lambda: [
            ["#", "Header 1"],
            ["##", "Header 2"],
            ["###", "Header 3"],
        ]
    )


@dataclass
class EmbeddingsConfig:
    model_name: str = "nomic-ai/nomic-embed-text-v1.5"
    device: str = "cpu"
    batch_size: int = 16
    normalize_embeddings: bool = True


@dataclass
class VectorStoreConfig:
    provider: str = "chroma"
    host: str = "localhost"
    port: int = 8001
    collection_name: str = "rag_langchain_kb"
    persistence: bool = True
    distance_metric: str = "cosine"


@dataclass
class SparseIndexConfig:
    provider: str = "bm25"
    k1: float = 1.5
    b: float = 0.75


@dataclass
class PreRetrievalConfig:
    multi_query_enabled: bool = True
    multi_query_num_queries: int = 3
    hybrid_search_enabled: bool = True
    dense_weight: float = 0.7
    sparse_weight: float = 0.3
    fetch_k: int = 20


@dataclass
class RerankerConfig:
    enabled: bool = True
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    device: str = "cpu"
    batch_size: int = 16
    top_k: int = 5
    score_threshold: float = 0.0


@dataclass
class PostRetrievalConfig:
    reranker: RerankerConfig = field(default_factory=RerankerConfig)
    compression_enabled: bool = True
    compression_similarity_threshold: float = 0.7
    reordering_enabled: bool = True
    citation_format_template: str = (
        "[Sumber: {document_name} | Bagian: {section} | Hal: {page_number}]"
    )


@dataclass
class LLMConfig:
    provider: str = "ollama"
    base_url: str = "http://localhost:11434"
    base_url_docker: str = "http://host.docker.internal:11434"
    model: str = "qwen2.5-7b-int4"
    temperature: float = 0.1
    max_tokens: int = 256
    num_ctx: int = 2048


@dataclass
class HardwareConfig:
    shared_vram_limit_gb: float = 4.5
    ollama_num_parallel: int = 1
    sycl_immediate_cmd_lists: int = 1
    ze_pci_id_device_order: int = 1


@dataclass
class Settings:
    app: AppConfig = field(default_factory=AppConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    ingestion: IngestionConfig = field(default_factory=IngestionConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    embeddings: EmbeddingsConfig = field(default_factory=EmbeddingsConfig)
    vector_store: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    sparse_index: SparseIndexConfig = field(default_factory=SparseIndexConfig)
    pre_retrieval: PreRetrievalConfig = field(default_factory=PreRetrievalConfig)
    post_retrieval: PostRetrievalConfig = field(default_factory=PostRetrievalConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)

    def enforce_airgap_environment(self) -> None:
        """Menyetel environment flags untuk isolasi offline / air-gapped total."""
        if self.app.air_gapped:
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
        if not self.app.telemetry_enabled:
            os.environ["ANONYMIZED_TELEMETRY"] = "FALSE"
        # Flags hardware Intel Level Zero / SYCL
        os.environ["SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS"] = str(
            self.hardware.sycl_immediate_cmd_lists
        )
        os.environ["ZE_ENABLE_PCI_ID_DEVICE_ORDER"] = str(
            self.hardware.ze_pci_id_device_order
        )


def _load_yaml_dict(yaml_file: Path) -> Dict[str, Any]:
    """Membaca file YAML secara aman jika PyYAML tersedia."""
    if not yaml_file.exists():
        return {}
    if yaml is None:
        return {}
    with open(yaml_file, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def load_settings(config_file: Optional[Path] = None) -> Settings:
    """Membuat objek Settings dengan memadukan config.yaml dan .env overrides."""
    target_config = config_file or CONFIG_YAML_PATH
    y = _load_yaml_dict(target_config)

    app_data = y.get("app", {})
    paths_data = y.get("paths", {})
    ingestion_data = y.get("ingestion", {})
    chunking_data = y.get("chunking", {})
    emb_data = y.get("embeddings", {})
    vs_data = y.get("vector_store", {})
    si_data = y.get("sparse_index", {})
    pr_data = y.get("pre_retrieval", {})
    post_data = y.get("post_retrieval", {})
    llm_data = y.get("llm", {})
    hw_data = y.get("hardware", {})

    # 1. App
    app = AppConfig(
        name=app_data.get("name", "rag_langchain"),
        version=app_data.get("version", "0.1.0"),
        description=app_data.get("description", ""),
        environment=_get_env("RUNTIME_ENV", app_data.get("environment", "local")),
        air_gapped=app_data.get("air_gapped", True),
        telemetry_enabled=app_data.get("telemetry_enabled", False),
    )

    # 2. Paths
    raw_dir = _get_env("DATA_RAW_DIR", paths_data.get("data_raw_dir", "./data/raw"))
    proc_dir = _get_env(
        "DATA_PROCESSED_DIR", paths_data.get("data_processed_dir", "./data/processed")
    )
    dlq_dir = _get_env("DATA_DLQ_DIR", paths_data.get("data_dlq_dir", "./data/dlq"))
    logs_dir = _get_env("LOGS_DIR", paths_data.get("logs_dir", "./logs"))

    paths = PathsConfig(
        data_raw_dir=(BASE_DIR / raw_dir).resolve()
        if not Path(raw_dir).is_absolute()
        else Path(raw_dir),
        data_processed_dir=(BASE_DIR / proc_dir).resolve()
        if not Path(proc_dir).is_absolute()
        else Path(proc_dir),
        data_dlq_dir=(BASE_DIR / dlq_dir).resolve()
        if not Path(dlq_dir).is_absolute()
        else Path(dlq_dir),
        dlq_error_log=(
            BASE_DIR / paths_data.get("dlq_error_log", "./data/dlq/error.log")
        ).resolve(),
        bm25_index_path=(
            BASE_DIR
            / paths_data.get("bm25_index_path", "./data/processed/bm25_index.pkl")
        ).resolve(),
        lineage_cache_path=(
            BASE_DIR
            / paths_data.get(
                "lineage_cache_path", "./data/processed/lineage_cache.json"
            )
        ).resolve(),
        logs_dir=(BASE_DIR / logs_dir).resolve()
        if not Path(logs_dir).is_absolute()
        else Path(logs_dir),
        conversion_log_path=(
            BASE_DIR
            / paths_data.get("conversion_log_path", "./logs/conversion.log")
        ).resolve(),
        conversion_history_path=(
            BASE_DIR
            / paths_data.get("conversion_history_path", "./logs/conversion_history.jsonl")
        ).resolve(),
    )
    paths.ensure_directories()

    # 3. Ingestion
    ingestion = IngestionConfig(
        supported_extensions=ingestion_data.get("supported_extensions", [".md"]),
        encoding=ingestion_data.get("encoding", "utf-8"),
        hashing_algorithm=ingestion_data.get("hashing_algorithm", "sha256"),
        min_file_size_chars=ingestion_data.get("min_file_size_chars", 50),
        page_tag_patterns=ingestion_data.get(
            "page_tag_patterns",
            [r"\[Hal:\s*(\d+)\]", r"<!--\s*page:\s*(\d+)\s*-->"],
        ),
    )

    # 4. Chunking
    chunking = ChunkingConfig(
        chunk_size=_get_env(
            "CHUNK_SIZE", chunking_data.get("chunk_size", 256), int
        ),
        chunk_overlap=_get_env(
            "CHUNK_OVERLAP", chunking_data.get("chunk_overlap", 26), int
        ),
        headers_to_split_on=chunking_data.get(
            "headers_to_split_on",
            [["#", "Header 1"], ["##", "Header 2"], ["###", "Header 3"]],
        ),
    )

    # 5. Embeddings
    embeddings = EmbeddingsConfig(
        model_name=_get_env(
            "EMBEDDING_MODEL_NAME",
            emb_data.get("model_name", "nomic-ai/nomic-embed-text-v1.5"),
        ),
        device=_get_env("EMBEDDING_DEVICE", emb_data.get("device", "cpu")),
        batch_size=_get_env(
            "EMBEDDING_BATCH_SIZE", emb_data.get("batch_size", 16), int
        ),
        normalize_embeddings=emb_data.get("normalize_embeddings", True),
    )

    # 6. Vector Store
    vector_store = VectorStoreConfig(
        provider=vs_data.get("provider", "chroma"),
        host=_get_env("CHROMA_HOST", vs_data.get("host", "localhost")),
        port=_get_env("CHROMA_PORT", vs_data.get("port", 8001), int),
        collection_name=_get_env(
            "CHROMA_COLLECTION_NAME",
            vs_data.get("collection_name", "rag_langchain_kb"),
        ),
        persistence=_get_env(
            "CHROMA_PERSISTENCE", vs_data.get("persistence", True), bool
        ),
        distance_metric=vs_data.get("distance_metric", "cosine"),
    )

    # 7. Sparse Index
    sparse_index = SparseIndexConfig(
        provider=si_data.get("provider", "bm25"),
        k1=float(si_data.get("k1", 1.5)),
        b=float(si_data.get("b", 0.75)),
    )

    # 8. Pre-Retrieval
    pr_mq = pr_data.get("multi_query", {})
    pr_hy = pr_data.get("hybrid_search", {})
    pre_retrieval = PreRetrievalConfig(
        multi_query_enabled=pr_mq.get("enabled", True),
        multi_query_num_queries=int(pr_mq.get("num_queries", 3)),
        hybrid_search_enabled=pr_hy.get("enabled", True),
        dense_weight=float(pr_hy.get("dense_weight", 0.7)),
        sparse_weight=float(pr_hy.get("sparse_weight", 0.3)),
        fetch_k=int(pr_hy.get("fetch_k", 20)),
    )

    # 9. Post-Retrieval
    post_rerank = post_data.get("reranker", {})
    post_comp = post_data.get("compression", {})
    reranker = RerankerConfig(
        enabled=post_rerank.get("enabled", True),
        model_name=post_rerank.get(
            "model_name", "cross-encoder/ms-marco-MiniLM-L-6-v2"
        ),
        device=post_rerank.get("device", "cpu"),
        batch_size=int(post_rerank.get("batch_size", 16)),
        top_k=_get_env("RAG_TOP_K", post_rerank.get("top_k", 5), int),
        score_threshold=float(post_rerank.get("score_threshold", 0.0)),
    )

    post_retrieval = PostRetrievalConfig(
        reranker=reranker,
        compression_enabled=post_comp.get("enabled", True),
        compression_similarity_threshold=float(
            post_comp.get("similarity_threshold", 0.7)
        ),
        reordering_enabled=post_data.get("reordering", {}).get("enabled", True),
        citation_format_template=post_data.get("citation", {}).get(
            "format_template",
            "[Sumber: {document_name} | Bagian: {section} | Hal: {page_number}]",
        ),
    )

    # 10. LLM
    llm = LLMConfig(
        provider=llm_data.get("provider", "ollama"),
        base_url=_get_env("OLLAMA_URL", llm_data.get("base_url", "http://localhost:11434")),
        base_url_docker=_get_env(
            "OLLAMA_URL_DOCKER",
            llm_data.get("base_url_docker", "http://host.docker.internal:11434"),
        ),
        model=_get_env("OLLAMA_MODEL", llm_data.get("model", "qwen2.5-7b-int4")),
        temperature=_get_env(
            "LLM_TEMPERATURE", llm_data.get("temperature", 0.1), float
        ),
        max_tokens=_get_env(
            "LLM_MAX_TOKENS", llm_data.get("max_tokens", 256), int
        ),
        num_ctx=_get_env("LLM_NUM_CTX", llm_data.get("num_ctx", 2048), int),
    )

    # 11. Hardware
    hardware = HardwareConfig(
        shared_vram_limit_gb=_get_env(
            "SHARED_VRAM_LIMIT_GB", hw_data.get("shared_vram_limit_gb", 4.5), float
        ),
        ollama_num_parallel=_get_env(
            "OLLAMA_NUM_PARALLEL", hw_data.get("ollama_num_parallel", 1), int
        ),
        sycl_immediate_cmd_lists=_get_env(
            "SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS",
            hw_data.get("sycl_immediate_cmd_lists", 1),
            int,
        ),
        ze_pci_id_device_order=_get_env(
            "ZE_ENABLE_PCI_ID_DEVICE_ORDER",
            hw_data.get("ze_pci_id_device_order", 1),
            int,
        ),
    )

    settings = Settings(
        app=app,
        paths=paths,
        ingestion=ingestion,
        chunking=chunking,
        embeddings=embeddings,
        vector_store=vector_store,
        sparse_index=sparse_index,
        pre_retrieval=pre_retrieval,
        post_retrieval=post_retrieval,
        llm=llm,
        hardware=hardware,
    )

    settings.enforce_airgap_environment()
    return settings


# Singleton instance
settings: Settings = load_settings()


def get_settings() -> Settings:
    """Mengembalikan singleton settings aktif."""
    return settings
