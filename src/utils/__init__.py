# Import utility functions for easy access
from .data_processing import clean_text, preprocess
from .metrics import compute_metrics
from .visualization import plot_confusion_matrix

__all__ = [
    'clean_text',
    'preprocess',
    'compute_metrics',
    'plot_confusion_matrix'
]
