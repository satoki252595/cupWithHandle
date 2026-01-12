"""
Machine Learning Module for Cup with Handle Pattern Detection
=============================================================

This module provides ML-enhanced pattern classification:
- CNN image classification (chart patterns)
- GBM feature-based classification
- Hybrid ensemble combining both approaches
"""

from .feature_extractor import FeatureExtractor
from .data_generator import MLDataGenerator, TrainingExample, SyntheticDataGenerator
from .ensemble import SimpleEnsemble, HybridEnsemble, create_ensemble_from_checkpoint

__all__ = [
    'FeatureExtractor',
    'MLDataGenerator',
    'TrainingExample',
    'SyntheticDataGenerator',
    'SimpleEnsemble',
    'HybridEnsemble',
    'create_ensemble_from_checkpoint',
]
