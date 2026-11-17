# src/models/decision_tree.py
import pickle
from sklearn.tree import DecisionTreeClassifier
from src.utils.logger import get_logger

logger = get_logger(__name__)

class ModelManager:
    @staticmethod
    def train(X_train, y_train, max_depth=3):
        clf = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
        clf.fit(X_train, y_train)
        logger.info("Дерево решений обучено.")
        return clf

    @staticmethod
    def save(model, path: str):
        with open(path, 'wb') as f:
            pickle.dump(model, f)
        logger.info(f"Модель сохранена в {path}")

    @staticmethod
    def load(path: str):
        with open(path, 'rb') as f:
            return pickle.load(f)