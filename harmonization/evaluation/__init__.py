"""
Module related to the Human Alignment benchmark
"""

from .click_me import load_clickme_val, evaluate_tf_model
from .metrics import dice, spearman_correlation, intersection_over_union
