# src/models/thresholds.py
import numpy as np
import json

def find_optimal_thresholds(scores: np.ndarray, labels: np.ndarray):
    """
    Находит пороги по метрике Юдена (Youden's J).
    """
    best_j = -1
    best_threshold = 0.5
    
    # Упрощенный перебор для примера
    thresholds = np.linspace(0, 1, 100)
    for t in thresholds:
        preds = (scores >= t).astype(int)
        tp = np.sum((preds == 1) & (labels == 1))
        fp = np.sum((preds == 1) & (labels == 0))
        fn = np.sum((preds == 0) & (labels == 1))
        tn = np.sum((preds == 0) & (labels == 0))
        
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        j = tpr - fpr
        if j > best_j:
            best_j = j
            best_threshold = t
            
    return {
        "human_cut": best_threshold - 0.1, # Зона неопределенности
        "ai_cut": best_threshold + 0.1,
        "optimal_j": best_j
    }