# semantic_features.py
import json
import hashlib
import sqlite3
from typing import Dict
from config import get_llm_client

# Кэш для повторных запросов
CACHE_DB = "semantic_cache.db"

def _get_cache_key(text: str, model: str) -> str:
    return hashlib.sha256(f"{model}:{text}".encode()).hexdigest()

def _get_from_cache(key: str) -> Dict:
    try:
        conn = sqlite3.connect(CACHE_DB)
        cursor = conn.execute("SELECT result FROM cache WHERE key=?", (key,))
        row = cursor.fetchone()
        conn.close()
        if row:
            return json.loads(row[0])
    except Exception:
        pass
    return None

def _save_to_cache(key: str, result: Dict):
    try:
        conn = sqlite3.connect(CACHE_DB)
        conn.execute("CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, result TEXT)")
        conn.execute("INSERT OR REPLACE INTO cache VALUES (?, ?)", (key, json.dumps(result)))
        conn.commit()
        conn.close()
    except Exception:
        pass

def analyze_segment_semantics(text: str, model_name: str = "qwen2.5-7b-instruct") -> Dict:
    """
    Анализирует текст и возвращает семантические метрики.
    При сбое возвращает нейтральные значения, чтобы не ломать пайплайн.
    """
    neutral = {
        "emotional_volatility": 0.5,
        "formality_score": 0.5,
        "perplexity_estimate": 0.5
    }
    
    if not text or len(text.strip()) < 10:
        return neutral
    
    # Проверяем кэш
    cache_key = _get_cache_key(text[:1000], model_name)
    cached = _get_from_cache(cache_key)
    if cached:
        return cached
    
    client = get_llm_client()
    
    prompt = f"""
    Ты — лингвистический анализатор. Оцени текст по трем шкалам от 0.0 до 1.0.
    Верни ТОЛЬКО валидный JSON.

    1. emotional_volatility: Насколько резко меняются эмоции? (0 - монотонно, 1 - эмоциональные качели)
    2. formality_score: Насколько язык официальный? (0 - сленг/чат, 1 - бюрократический отчет)
    3. perplexity_estimate: Насколько текст шаблонный? (0 - предсказуемо/AI, 1 - неожиданно/человек)

    Текст:
    ---
    {text[:1000]}
    ---
    """
    
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            response_format={"type": "json_object"},
            timeout=30.0
        )
        
        content = response.choices[0].message.content
        result = json.loads(content)
        
        # Валидация значений
        for key in neutral:
            if key not in result or not isinstance(result[key], (int, float)):
                result[key] = neutral[key]
            else:
                result[key] = max(0.0, min(1.0, float(result[key])))
        
        _save_to_cache(cache_key, result)
        return result
        
    except Exception as e:
        print(f"[semantic_features] Ошибка: {e}. Использую нейтральные значения.")
        return neutral