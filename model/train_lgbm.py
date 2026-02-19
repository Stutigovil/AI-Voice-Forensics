"""
LightGBM Training Module
========================
Trains a LightGBM classifier for detecting AI-generated voice.

Features:
- Binary classification (AI vs Human)
- SHAP explainability support
- Cross-validation with hyperparameter tuning
- Model persistence and versioning
"""

import os
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import joblib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AIVoiceClassifier:
    """
    LightGBM classifier for AI voice detection.
    
    This classifier uses a combination of audio and text features
    to distinguish between human and AI-generated voice samples.
    
    Labels:
        0 = Human voice
        1 = AI-generated voice
    """
    
    def __init__(
        self,
        num_leaves: int = 31,
        max_depth: int = -1,
        learning_rate: float = 0.05,
        n_estimators: int = 200,
        min_child_samples: int = 20,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42,
        class_weight: str = "balanced"
    ):
        """
        Initialize the classifier.
        
        Args:
            num_leaves: Maximum number of leaves in each tree
            max_depth: Maximum depth of trees (-1 for no limit)
            learning_rate: Learning rate for boosting
            n_estimators: Number of boosting rounds
            min_child_samples: Minimum samples in a leaf
            subsample: Subsample ratio for training
            colsample_bytree: Feature subsample ratio
            random_state: Random seed for reproducibility
            class_weight: Class weight strategy
        """
        self.params = {
            'num_leaves': num_leaves,
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'n_estimators': n_estimators,
            'min_child_samples': min_child_samples,
            'subsample': subsample,
            'colsample_bytree': colsample_bytree,
            'random_state': random_state,
            'class_weight': class_weight,
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'verbose': -1
        }
        
        self.model = None
        self.feature_names = None
        self.training_metrics = {}
        self.is_fitted = False
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
        validation_split: float = 0.2,
        early_stopping_rounds: int = 50
    ) -> Dict:
        """
        Train the classifier.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Labels (0=Human, 1=AI)
            feature_names: Optional names for features
            validation_split: Fraction of data for validation
            early_stopping_rounds: Early stopping patience
            
        Returns:
            Dictionary with training metrics
        """
        logger.info(f"Training with {X.shape[0]} samples, {X.shape[1]} features")
        
        # Store feature names
        self.feature_names = feature_names or [f"feature_{i}" for i in range(X.shape[1])]
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, 
            test_size=validation_split, 
            stratify=y,
            random_state=self.params['random_state']
        )
        
        logger.info(f"Train: {X_train.shape[0]} samples, Val: {X_val.shape[0]} samples")
        
        # Create LightGBM datasets
        train_data = lgb.Dataset(X_train, label=y_train, feature_name=self.feature_names)
        val_data = lgb.Dataset(X_val, label=y_val, feature_name=self.feature_names)
        
        # Train model
        callbacks = [
            lgb.early_stopping(stopping_rounds=early_stopping_rounds),
            lgb.log_evaluation(period=50)
        ]
        
        self.model = lgb.train(
            self.params,
            train_data,
            valid_sets=[train_data, val_data],
            valid_names=['train', 'val'],
            num_boost_round=self.params['n_estimators'],
            callbacks=callbacks
        )
        
        self.is_fitted = True
        
        # Evaluate on validation set
        y_pred_proba = self.model.predict(X_val)
        y_pred = (y_pred_proba >= 0.5).astype(int)
        
        # Calculate metrics
        self.training_metrics = {
            'accuracy': accuracy_score(y_val, y_pred),
            'precision': precision_score(y_val, y_pred),
            'recall': recall_score(y_val, y_pred),
            'f1': f1_score(y_val, y_pred),
            'roc_auc': roc_auc_score(y_val, y_pred_proba),
            'best_iteration': self.model.best_iteration,
            'train_samples': X_train.shape[0],
            'val_samples': X_val.shape[0],
            'n_features': X.shape[1]
        }
        
        logger.info(f"Training complete. Validation metrics:")
        for metric, value in self.training_metrics.items():
            if isinstance(value, float):
                logger.info(f"  {metric}: {value:.4f}")
        
        return self.training_metrics
    
    def cross_validate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_folds: int = 5
    ) -> Dict:
        """
        Perform cross-validation.
        
        Args:
            X: Feature matrix
            y: Labels
            n_folds: Number of CV folds
            
        Returns:
            Dictionary with CV scores
        """
        logger.info(f"Running {n_folds}-fold cross-validation...")
        
        # Create sklearn-compatible estimator
        clf = lgb.LGBMClassifier(**self.params)
        
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        
        # Compute CV scores
        cv_scores = {
            'accuracy': cross_val_score(clf, X, y, cv=cv, scoring='accuracy'),
            'f1': cross_val_score(clf, X, y, cv=cv, scoring='f1'),
            'roc_auc': cross_val_score(clf, X, y, cv=cv, scoring='roc_auc')
        }
        
        results = {}
        for metric, scores in cv_scores.items():
            results[f'{metric}_mean'] = scores.mean()
            results[f'{metric}_std'] = scores.std()
            logger.info(f"{metric}: {scores.mean():.4f} (+/- {scores.std():.4f})")
        
        return results
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make binary predictions.
        
        Args:
            X: Feature matrix
            
        Returns:
            Binary predictions (0=Human, 1=AI)
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        proba = self.model.predict(X)
        return (proba >= 0.5).astype(int)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probabilities.
        
        Args:
            X: Feature matrix
            
        Returns:
            Probability of AI-generated class
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        return self.model.predict(X)
    
    def get_feature_importance(
        self,
        importance_type: str = "gain",
        top_k: int = 20
    ) -> Dict[str, float]:
        """
        Get feature importance scores.
        
        Args:
            importance_type: Type of importance ('gain', 'split')
            top_k: Number of top features to return
            
        Returns:
            Dictionary of feature names and importance scores
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        importance = self.model.feature_importance(importance_type=importance_type)
        
        # Create feature-importance pairs
        feature_importance = dict(zip(self.feature_names, importance))
        
        # Sort by importance
        sorted_importance = dict(
            sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:top_k]
        )
        
        return sorted_importance
    
    def explain_prediction(
        self,
        X: np.ndarray,
        use_shap: bool = True
    ) -> Dict:
        """
        Explain a single prediction using SHAP or feature contributions.
        
        Args:
            X: Single sample feature vector (1, n_features)
            use_shap: Whether to use SHAP values
            
        Returns:
            Dictionary with prediction explanation
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        # Ensure 2D input
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        prediction = self.predict(X)[0]
        probability = self.predict_proba(X)[0]
        
        explanation = {
            'prediction': int(prediction),
            'prediction_label': 'AI-Generated' if prediction == 1 else 'Human',
            'confidence': float(probability) if prediction == 1 else float(1 - probability),
            'probability_ai': float(probability)
        }
        
        if use_shap:
            try:
                import shap
                
                explainer = shap.TreeExplainer(self.model)
                shap_values = explainer.shap_values(X)
                
                # Get top contributing features
                feature_contributions = dict(zip(self.feature_names, shap_values[0]))
                sorted_contributions = dict(
                    sorted(feature_contributions.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
                )
                
                explanation['shap_values'] = sorted_contributions
                explanation['expected_value'] = float(explainer.expected_value)
                
            except ImportError:
                logger.warning("SHAP not available. Falling back to simple feature importance.")
                explanation['top_features'] = self.get_feature_importance(top_k=10)
        else:
            explanation['top_features'] = self.get_feature_importance(top_k=10)
        
        return explanation
    
    def save(self, filepath: Union[str, Path]) -> None:
        """
        Save the model to disk.
        
        Args:
            filepath: Path to save the model
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Save model and metadata
        model_data = {
            'model': self.model,
            'params': self.params,
            'feature_names': self.feature_names,
            'training_metrics': self.training_metrics,
            'save_time': datetime.now().isoformat()
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to: {filepath}")
    
    def load(self, filepath: Union[str, Path]) -> None:
        """
        Load a saved model.
        
        Args:
            filepath: Path to the saved model
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.params = model_data['params']
        self.feature_names = model_data['feature_names']
        self.training_metrics = model_data['training_metrics']
        self.is_fitted = True
        
        logger.info(f"Model loaded from: {filepath}")
        logger.info(f"Training accuracy: {self.training_metrics.get('accuracy', 'N/A')}")


def train_model(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    **kwargs
) -> AIVoiceClassifier:
    """
    Convenience function to train and optionally save a model.
    
    Args:
        X: Feature matrix
        y: Labels
        feature_names: Optional feature names
        save_path: Optional path to save trained model
        **kwargs: Additional parameters for AIVoiceClassifier
        
    Returns:
        Trained AIVoiceClassifier
    """
    classifier = AIVoiceClassifier(**kwargs)
    classifier.fit(X, y, feature_names=feature_names)
    
    if save_path:
        classifier.save(save_path)
    
    return classifier


def create_training_data_placeholder() -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Create placeholder training data for testing.
    
    In production, replace this with actual feature extraction
    from ASVspoof or custom datasets.
    
    Returns:
        X, y, feature_names
    """
    np.random.seed(42)
    
    n_samples = 1000
    n_features = 80
    
    # Simulate features for human (class 0) and AI (class 1)
    # Human samples: higher disfluency, more pitch variation
    X_human = np.random.randn(n_samples // 2, n_features)
    X_human[:, 0] += 2  # Higher disfluency rate
    X_human[:, 1] += 1  # More pitch variation
    
    # AI samples: lower perplexity, more uniform features
    X_ai = np.random.randn(n_samples // 2, n_features)
    X_ai[:, 0] -= 1  # Lower disfluency
    X_ai[:, 2] += 1  # Lower perplexity (log scale)
    
    X = np.vstack([X_human, X_ai])
    y = np.array([0] * (n_samples // 2) + [1] * (n_samples // 2))
    
    # Shuffle
    indices = np.random.permutation(n_samples)
    X = X[indices]
    y = y[indices]
    
    feature_names = [f"feature_{i}" for i in range(n_features)]
    
    return X, y, feature_names


if __name__ == "__main__":
    # Example training run
    print("AI Voice Detection - Model Training")
    print("=" * 50)
    
    # Create placeholder data
    X, y, feature_names = create_training_data_placeholder()
    
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Class distribution: Human={sum(y==0)}, AI={sum(y==1)}")
    
    # Train model
    classifier = AIVoiceClassifier()
    metrics = classifier.fit(X, y, feature_names=feature_names)
    
    print("\nTraining Results:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value}")
    
    # Get feature importance
    print("\nTop 10 Important Features:")
    importance = classifier.get_feature_importance(top_k=10)
    for feature, score in importance.items():
        print(f"  {feature}: {score:.4f}")
    
    # Save model
    classifier.save("model/saved/lgbm_model.pkl")
    print("\nModel saved successfully!")
