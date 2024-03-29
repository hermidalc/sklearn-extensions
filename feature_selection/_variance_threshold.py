# Author: Lars Buitinck
# License: 3-clause BSD

import numpy as np
from sklearn.feature_selection import VarianceThreshold as SklearnVarianceThreshold

from ..feature_selection import ExtendedSelectorMixin


class VarianceThreshold(ExtendedSelectorMixin, SklearnVarianceThreshold):
    def _get_support_mask(self):
        return super()._get_support_mask()
