import numpy as np

from sklearn.preprocessing import FunctionTransformer


def shifted_log(X, base, shift):
    return np.log(X + shift) / np.log(base)


class LogTransformer(FunctionTransformer):
    """Log transformer

    Parameters
    ----------
    base : int (default = 2)
        Base to use when taking logarithm

    shift : float (default = 1.0)
        Value to shift data by before taking logarithm
    """

    def __init__(self, base=2, shift=1):
        self.base = base
        self.shift = shift
        super().__init__(
            func=shifted_log,
            check_inverse=False,
            kw_args={"base": base, "shift": shift},
            validate=True,
        )
