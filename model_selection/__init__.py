"""
The :mod:`sklearn.model_selection` module.
"""

from ._split import (
    StratifiedGroupKFold,
    RepeatedStratifiedGroupKFold,
    StratifiedSampleFromGroupKFold,
    RepeatedStratifiedSampleFromGroupKFold,
    StratifiedGroupShuffleSplit,
    StratifiedSampleFromGroupShuffleSplit,
)

from ._search import ExtendedGridSearchCV
from ._search_successive_halving import ExtendedHalvingGridSearchCV
from ._search import ExtendedRandomizedSearchCV
from ._search_successive_halving import ExtendedHalvingRandomSearchCV

from ._validation import cross_val_score
from ._validation import cross_val_predict
from ._validation import cross_validate
from ._validation import learning_curve
from ._validation import permutation_test_score, shuffle_y
from ._validation import validation_curve


__all__ = [
    "ExtendedGridSearchCV",
    "ExtendedHalvingGridSearchCV",
    "ExtendedRandomizedSearchCV",
    "ExtendedHalvingRandomSearchCV",
    "StratifiedGroupKFold",
    "StratifiedSampleFromGroupKFold",
    "StratifiedGroupShuffleSplit",
    "StratifiedSampleFromGroupShuffleSplit",
    "RepeatedStratifiedGroupKFold",
    "RepeatedStratifiedSampleFromGroupKFold",
    "cross_val_predict",
    "cross_val_score",
    "cross_validate",
    "learning_curve",
    "permutation_test_score",
    "shuffle_y",
    "validation_curve",
]


# TODO: remove this check once the estimator is no longer experimental.
def __getattr__(name):
    if name in {"ExtendedHalvingGridSearchCV", "ExtendedHalvingRandomizedSearchCV"}:
        raise ImportError(
            f"{name} is experimental and the API might change without any "
            "deprecation cycle. To use it, you need to explicitly import "
            "enable_halving_search_cv:\n"
            "from sklearn.experimental import enable_halving_search_cv"
        )
    raise AttributeError(f"module {__name__} has no attribute {name}")
