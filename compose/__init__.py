"""Meta-estimators for building composite models with transformers

In addition to its current contents, this module will eventually be home to
refurbished versions of Pipeline and FeatureUnion.

"""

from ._column_transformer import (ExtendedColumnTransformer,
                                  make_extended_column_transformer,
                                  make_column_selector)


__all__ = [
    'ExtendedColumnTransformer',
    'make_extended_column_transformer',
    'make_column_selector',
]
