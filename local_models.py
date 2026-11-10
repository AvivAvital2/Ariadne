"""Pinned Mac model candidate; hardware and output quality remain unqualified."""

from dataclasses import dataclass
from types import MappingProxyType

MAC_PROFILE_ID = 'qwen-mac-q4-v1'
MAC_CHAT_MODEL = 'bartowski/Qwen_Qwen3.5-9B-GGUF'
MAC_EMBEDDING_MODEL = 'Qwen/Qwen3-Embedding-8B-GGUF'
MAC_WEIGHT_FORMAT = 'gguf-q4_k_m'
MAC_RECIPE = 'qwen3-embedding-8b-gguf-q4-k-m-3072-v1'
MAC_ACCURACY_PROFILE_ID = 'qwen-mac-27b-q4-thinking-v1'
MAC_ACCURACY_CHAT_MODEL = 'bartowski/Qwen3.8-27B-GGUF'
GEMMA_PROFILE_ID = 'gemma4-12b-qwen8b-mac-q4-v1'
GEMMA_CHAT_MODEL = 'unsloth/gemma-4-12b-it-GGUF'
MAC_PROFILE_IDS = (MAC_PROFILE_ID, MAC_ACCURACY_PROFILE_ID, GEMMA_PROFILE_ID)


@dataclass(frozen=True)
class ModelArtifact:
    repository: str
    revision: str
    filename: str
    sha256: str
    size: int

    @property
    def url(self) -> str:
        return f'https://huggingface.co/{self.repository}/resolve/{self.revision}/{self.filename}'


# SHA-256 and byte sizes from the publishers' LFS metadata, checked 2026-10-22.
MAC_ARTIFACTS = MappingProxyType({
    'llm': ModelArtifact(
        MAC_CHAT_MODEL, '182be2fd6c7bc44887d88a91cb03ff009cc9f549',
        'Qwen_Qwen3.5-9B-Q4_K_M.gguf',
        'd784ce9eda1a5a7b51e8f705a9e6310844bf4f173654d115823c775fdea56d43',
        6169341984,
    ),
    'embeddings': ModelArtifact(
        MAC_EMBEDDING_MODEL, '69d0e58a13e463cd99a9b83e3f5fee7c10265fab',
        'Qwen3-Embedding-8B-Q4_K_M.gguf',
        '3fcd3febec8b3fd64435204db75bf0dd73b91e8d0661e0331acfe7e7c3120b85',
        4676804928,
    ),
})

# Separate profile: existing 9B libraries retain their original artifact and mode.
# Publisher LFS metadata checked 2026-10-22; these are file bytes, not runtime RAM.
MAC_ACCURACY_ARTIFACTS = MappingProxyType({
    'llm': ModelArtifact(
        MAC_ACCURACY_CHAT_MODEL, '87410e7b9a5c21e6ca70c0c9cc5237de909fe621',
        'Qwen3.8-27B-Q4_K_M.gguf',
        '880a52af8317974f7b4b064a6bc29ff49e188b91aa218e08eea3078b6b1cc673',
        17442399968,
    ),
    'embeddings': MAC_ARTIFACTS['embeddings'],
})

# Publisher LFS metadata checked 2026-11-10. Text generation only; no projector.
GEMMA_ARTIFACTS = MappingProxyType({
    'llm': ModelArtifact(
        GEMMA_CHAT_MODEL, 'fc034cfff751157913579611efad8462ac1be606',
        'gemma-4-12b-it-Q4_K_M.gguf',
        '0a270ec9fe6b34f4a0d33992b6135117b484ebc4766ab76b51d4ae8c457e4c42',
        7121861440,
    ),
    'embeddings': MAC_ARTIFACTS['embeddings'],
})


def mac_model_profile() -> dict:
    """Return fresh configuration, binding to GGUF weights and embedded tokenizers."""
    return {
        'llm': {
            'provider': 'local',
            'model': MAC_CHAT_MODEL,
            'served_model': 'Qwen/Qwen3.5-9B',
            'model_revision': MAC_ARTIFACTS['llm'].revision,
            'weight_format': MAC_WEIGHT_FORMAT,
            'base_url': 'http://127.0.0.1:18080/v1',
            'context_window': 32768,
            'max_output_tokens': 8192,
            'max_concurrency': 1,
        },
        'embeddings': {
            'provider': 'local',
            'model': MAC_EMBEDDING_MODEL,
            'served_model': 'Qwen/Qwen3-Embedding-8B',
            'model_revision': MAC_ARTIFACTS['embeddings'].revision,
            'weight_format': MAC_WEIGHT_FORMAT,
            'base_url': 'http://127.0.0.1:18081/v1',
            'dimensions': 3072,
            'recipe': MAC_RECIPE,
            'max_concurrency': 1,
        },
    }


def mac_accuracy_profile() -> dict:
    """32 GB qualification candidate; hardware fit and factual quality are unproven."""
    profile = mac_model_profile()
    profile['llm'].update(
        model=MAC_ACCURACY_CHAT_MODEL,
        served_model='Qwen/Qwen3.8-27B',
        model_revision=MAC_ACCURACY_ARTIFACTS['llm'].revision,
        timeout_seconds=600.0,
        # max_output_tokens reserves total reasoning plus final tokens, not two budgets.
        request_options={
            'chat_template_kwargs': {'enable_thinking': True},
            'reasoning_effort': 'xhigh', 'reasoning_format': 'deepseek',
            'temperature': 1.0, 'top_p': 0.95, 'top_k': 20,
            'min_p': 0.0, 'presence_penalty': 0.0, 'repeat_penalty': 1.0,
        },
    )
    return profile


def managed_model_profile(profile_id: str) -> dict:
    """Return fresh configuration; never reinterpret an unknown persisted marker."""
    if profile_id == MAC_PROFILE_ID:
        return mac_model_profile()
    if profile_id == MAC_ACCURACY_PROFILE_ID:
        return mac_accuracy_profile()
    if profile_id == GEMMA_PROFILE_ID:
        return gemma_model_profile()
    raise ValueError(f'Unknown managed local model profile: {profile_id!r}')


def gemma_model_profile() -> dict:
    """Gemma generation with the exact existing Qwen Q4 embedding space."""
    profile = mac_model_profile()
    profile['llm'].update(
        model=GEMMA_CHAT_MODEL,
        served_model='google/gemma-4-12b-it',
        model_revision=GEMMA_ARTIFACTS['llm'].revision,
        context_window=16384,
        timeout_seconds=600.0,
        request_options={
            'chat_template_kwargs': {'enable_thinking': False},
            'reasoning_format': 'auto',
            'temperature': 1.0, 'top_p': 0.95, 'top_k': 64,
            'min_p': 0.0, 'presence_penalty': 0.0, 'repeat_penalty': 1.0,
        },
    )
    return profile
