from sklearn.utils.validation import check_memory


class CachedFitMixin:
    """Mixin for caching pipeline nested estimator fits"""

    def fit(self, *args, **kwargs):
        memory = check_memory(self.memory)
        fit = memory.cache(super().fit)
        cached_self = fit(*args, **kwargs)
        vars(self).update(vars(cached_self))
        return self
