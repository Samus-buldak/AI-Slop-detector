"""Настройки. Переменные окружения имеют приоритет над локальным .env."""
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for raw_line in (ROOT / '.env').read_text(encoding='utf-8-sig').splitlines() if (ROOT / '.env').is_file() else []:
    line = raw_line.strip()
    if line and not line.startswith('#') and '=' in line:
        key, value = line.split('=', 1)
        key = key.strip()
        if key and key not in os.environ:
            os.environ[key] = value.strip().strip('"\'')

API_KEY = os.getenv('API_KEY', '').strip()
if API_KEY.lower() in {'', 'sk-...', 'sk-xxx', 'your_api_key_here',
                       'your_qwen_api_key_here', 'paste_your_qwen_api_key_here',
                       'paste_your_api_key_here', 'changeme'}:
    API_KEY = ''
API_URL = (os.getenv('API_URL', 'https://deepcode.ci.nsu.ru/api/v1').strip()
           or 'https://deepcode.ci.nsu.ru/api/v1').rstrip('/')
MODEL = os.getenv('MODEL', 'Qwen3.8-27B').strip() or 'Qwen3.8-27B'
EMBED_MODEL = os.getenv('EMBED_MODEL', 'qwen3-embedding-local').strip() or 'qwen3-embedding-local'
# Qwen endpoints: local_qwen.py redirects embeddings to LM Studio when selected.
# The standard run uses the Qwen API configured by API_URL/API_KEY/MODEL.
SEMANTIC_API_URL = (os.getenv('SEMANTIC_API_URL', API_URL).strip() or API_URL).rstrip('/')
SEMANTIC_API_KEY = os.getenv('SEMANTIC_API_KEY', API_KEY).strip()
if SEMANTIC_API_KEY.lower() in {'', 'sk-...', 'sk-xxx', 'your_api_key_here',
                                'your_qwen_api_key_here', 'paste_your_qwen_api_key_here',
                                'paste_your_api_key_here', 'changeme'}:
    SEMANTIC_API_KEY = ''
SEMANTIC_MODEL = os.getenv('SEMANTIC_MODEL', MODEL).strip() or MODEL
VISION_MODEL = os.getenv('VISION_MODEL', SEMANTIC_MODEL).strip() or SEMANTIC_MODEL
VISION_API_URL = (os.getenv('VISION_API_URL', SEMANTIC_API_URL).strip() or SEMANTIC_API_URL).rstrip('/')
VISION_API_KEY = os.getenv('VISION_API_KEY', SEMANTIC_API_KEY).strip()
if VISION_API_KEY.lower() in {'', 'sk-...', 'sk-xxx', 'your_api_key_here',
                              'your_qwen_api_key_here', 'paste_your_qwen_api_key_here',
                              'paste_your_api_key_here', 'changeme'}:
    VISION_API_KEY = ''
def _int_setting(name: str, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        value = default
    return max(minimum, min(maximum, value))


def _float_setting(name: str, default: float, minimum: float, maximum: float) -> float:
    try:
        value = float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        value = default
    if not math.isfinite(value):
        value = default
    return max(minimum, min(maximum, value))


SEM_BATCH = _int_setting('SEM_BATCH', 8, 1, 64)
SEM_CONC = _int_setting('SEM_CONC', 8, 1, 32)
EMB_BATCH = _int_setting('EMB_BATCH', 32, 1, 256)
EMB_CONC = _int_setting('EMB_CONC', 6, 1, 32)
API_TIMEOUT = _float_setting('API_TIMEOUT', 120., 5., 600.)
API_RETRIES = _int_setting('API_RETRIES', 1, 0, 5)
MODEL_DIR = ROOT / os.getenv('DETECTOR_MODEL_DIR', 'model')
DATA_DIR = ROOT / 'data'
