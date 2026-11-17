# src/core/decision_engine.py
import json
from typing import Dict, List, Optional
from src.utils.logger import get_logger

logger = get_logger(__name__)

class DecisionEngine:
    def __init__(self, thresholds_path: str = "artifacts/model/thresholds.json"):
        self.thresholds = self._load_thresholds(thresholds_path)
        
    def _load_thresholds(self, path: str) -> Dict:
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(f"Файл порогов {path} не найден. Используются дефолтные значения.")
            return {"human_cut": 0.3, "ai_cut": 0.7}

    def make_verdict(self, final_score: float, zone_verdicts: List[Dict]) -> Dict:
        """
        Принимает финальный скор и вердикты по зонам.
        Возвращает итоговый вердикт (HUMAN/AI/UNCERTAIN).
        """
        human_cut = self.thresholds.get("human_cut", 0.3)
        ai_cut = self.thresholds.get("ai_cut", 0.7)
        
        # Проверка на конфликт зон (гибридное авторство)
        has_ai = any(z.get("verdict") == "AI" for z in zone_verdicts)
        has_human = any(z.get("verdict") == "HUMAN" for z in zone_verdicts)
        
        if has_ai and has_human:
            verdict = "UNCERTAIN"
            reason = "Обнаружен конфликт авторства в разных частях письма."
        elif final_score < human_cut:
            verdict = "HUMAN"
            reason = "Скор ниже порога человека."
        elif final_score >= ai_cut:
            verdict = "AI"
            reason = "Скор выше порога AI."
        else:
            verdict = "UNCERTAIN"
            reason = "Скор находится в зоне неопределенности."
            
        return {
            "verdict": verdict,
            "final_score": round(final_score, 4),
            "reason": reason,
            "thresholds_used": {"human": human_cut, "ai": ai_cut}
        }