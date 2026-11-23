# src/features/semantic_features.py
import json
import hashlib
from typing import Dict
from src.utils.cache import SemanticCache
from src.api.local_qwen import get_client
from src.utils.logger import get_logger

logger = get_logger(__name__)
cache = SemanticCache()

def analyze_semantics(text: str, model_name: str = "qwen2.5-1.5b-instruct") -> Dict:
    neutral = {
        "emotional_volatility": 0.5,
        "formality_score": 0.5,
        "perplexity_estimate": 0.5
    }
    
    if not text or len(text.strip()) < 10:
        return neutral
        
    cache_key = hashlib.sha256(f"{model_name}:{text[:500]}".encode()).hexdigest()
    cached_result = cache.get(cache_key)
    if cached_result:
        return cached_result
        
    client = get_client()
    prompt = f"""
    Оцени текст по шкале 0.0-1.0. Верни ТОЛЬКО JSON.
    1. emotional_volatility: 0 (монотонно) - 1 (эмоциональные качели).
    2. formality_score: 0 (сленг) - 1 (бюрократия).
    3. perplexity_estimate: 0 (шаблонно/AI) - 1 (неожиданно/человек).
    
    Текст: {text[:1000]}
    """
    
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            response_format={"type": "json_object"},
            timeout=15
        )
        result = json.loads(response.choices[0].message.content)
        
        # Валидация
        for k in neutral:
            if k not in result: result[k] = 0.5
            else: result[k] = max(0.0, min(1.0, float(result[k])))
            
        cache.set(cache_key, result)
        return result
        
    except Exception as e:
        logger.error(f"Ошибка семантического анализа: {e}")
        return neutral