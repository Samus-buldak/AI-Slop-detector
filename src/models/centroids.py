# src/models/centroids.py
import numpy as np
import json

class CentroidCalculator:
    @staticmethod
    def calculate_centroids(embeddings: np.ndarray, labels: list) -> dict:
        """
        embeddings: массив векторов [N, D]
        labels: список меток ('AI' или 'HUMAN')
        """
        ai_mask = np.array([l == 'AI' for l in labels])
        human_mask = ~ai_mask
        
        centroid_ai = np.mean(embeddings[ai_mask], axis=0) if np.any(ai_mask) else None
        centroid_human = np.mean(embeddings[human_mask], axis=0) if np.any(human_mask) else None
        
        return {
            "ai": centroid_ai.tolist() if centroid_ai is not None else None,
            "human": centroid_human.tolist() if centroid_human is not None else None
        }

    @staticmethod
    def save(centroids: dict, path: str):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(centroids, f)