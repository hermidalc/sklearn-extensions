# Author: Lars Buitinck
# License: 3-clause BSD

from sklearn.feature_selection import VarianceThreshold as BaseVarianceThreshold
from sklearn.utils.validation import check_is_fitted

from ..feature_selection import ExtendedSelectorMixin


class VarianceThreshold(ExtendedSelectorMixin, BaseVarianceThreshold):
    def _get_support_mask(self):
        check_is_fitted(self)
        return self.variances_ > self.threshold
