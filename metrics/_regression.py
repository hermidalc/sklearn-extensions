"""Metrics to assess performance on regression task

Functions named as ``*_score`` return a scalar value to maximize: the higher
the better

Function named as ``*_error`` or ``*_loss`` return a scalar value to minimize:
the lower the better
"""

# Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Mathieu Blondel <mathieu@mblondel.org>
#          Olivier Grisel <olivier.grisel@ensta.org>
#          Arnaud Joly <a.joly@ulg.ac.be>
#          Jochen Wersdorfer <jochen@wersdoerfer.de>
#          Lars Buitinck
#          Joel Nothman <joel.nothman@gmail.com>
#          Karan Desai <karandesai281196@gmail.com>
#          Noel Dawe <noel@dawe.me>
#          Manoj Kumar <manojkumarsivaraj334@gmail.com>
#          Michael Eickenberg <michael.eickenberg@gmail.com>
#          Konstantin Shmelkov <konstantin.shmelkov@polytechnique.edu>
#          Christian Lorentzen <lorentzen.ch@googlemail.com>
# License: BSD 3 clause


import numpy as np
from scipy.special import xlogy

from sklearn.utils.validation import check_array, check_consistent_length
from sklearn.utils.validation import column_or_1d


__ALL__ = [
    "mean_tweedie_deviance",
    "mean_poisson_deviance",
    "mean_gamma_deviance",
]


def _check_reg_targets(y_true, y_pred, multioutput, dtype="numeric"):
    """Check that y_true and y_pred belong to the same regression task

    Parameters
    ----------
    y_true : array-like

    y_pred : array-like

    multioutput : array-like or string in ['raw_values', uniform_average',
        'variance_weighted'] or None
        None is accepted due to backward compatibility of r2_score().

    Returns
    -------
    type_true : one of {'continuous', continuous-multioutput'}
        The type of the true target data, as output by
        'utils.multiclass.type_of_target'

    y_true : array-like of shape (n_samples, n_outputs)
        Ground truth (correct) target values.

    y_pred : array-like of shape (n_samples, n_outputs)
        Estimated target values.

    multioutput : array-like of shape (n_outputs) or string in ['raw_values',
        uniform_average', 'variance_weighted'] or None
        Custom output weights if ``multioutput`` is array-like or
        just the corresponding argument if ``multioutput`` is a
        correct keyword.
    dtype: str or list, default="numeric"
        the dtype argument passed to check_array

    """
    check_consistent_length(y_true, y_pred)
    y_true = check_array(y_true, ensure_2d=False, dtype=dtype)
    y_pred = check_array(y_pred, ensure_2d=False, dtype=dtype)

    if y_true.ndim == 1:
        y_true = y_true.reshape((-1, 1))

    if y_pred.ndim == 1:
        y_pred = y_pred.reshape((-1, 1))

    if y_true.shape[1] != y_pred.shape[1]:
        raise ValueError("y_true and y_pred have different number of output "
                         "({0}!={1})".format(y_true.shape[1], y_pred.shape[1]))

    n_outputs = y_true.shape[1]
    allowed_multioutput_str = ('raw_values', 'uniform_average',
                               'variance_weighted')
    if isinstance(multioutput, str):
        if multioutput not in allowed_multioutput_str:
            raise ValueError("Allowed 'multioutput' string values are {}. "
                             "You provided multioutput={!r}".format(
                                 allowed_multioutput_str,
                                 multioutput))
    elif multioutput is not None:
        multioutput = check_array(multioutput, ensure_2d=False)
        if n_outputs == 1:
            raise ValueError("Custom weights are useful only in "
                             "multi-output cases.")
        elif n_outputs != len(multioutput):
            raise ValueError(("There must be equally many custom weights "
                              "(%d) as outputs (%d).") %
                             (len(multioutput), n_outputs))
    y_type = 'continuous' if n_outputs == 1 else 'continuous-multioutput'

    return y_type, y_true, y_pred, multioutput


def mean_tweedie_deviance(y_true, y_pred, sample_weight=None, power=0):
    """Mean Tweedie deviance regression loss.

    Read more in the :ref:`User Guide <mean_tweedie_deviance>`.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth (correct) target values.

    y_pred : array-like of shape (n_samples,)
        Estimated target values.

    sample_weight : array-like, shape (n_samples,), optional
        Sample weights.

    power : float, default=0
        Tweedie power parameter. Either power <= 0 or power >= 1.

        The higher `p` the less weight is given to extreme
        deviations between true and predicted targets.

        - power < 0: Extreme stable distribution. Requires: y_pred > 0.
        - power = 0 : Normal distribution, output corresponds to
          mean_squared_error. y_true and y_pred can be any real numbers.
        - power = 1 : Poisson distribution. Requires: y_true >= 0 and
          y_pred > 0.
        - 1 < p < 2 : Compound Poisson distribution. Requires: y_true >= 0
          and y_pred > 0.
        - power = 2 : Gamma distribution. Requires: y_true > 0 and y_pred > 0.
        - power = 3 : Inverse Gaussian distribution. Requires: y_true > 0
          and y_pred > 0.
        - otherwise : Positive stable distribution. Requires: y_true > 0
          and y_pred > 0.

    Returns
    -------
    loss : float
        A non-negative floating point value (the best value is 0.0).

    Examples
    --------
    >>> from sklearn.metrics import mean_tweedie_deviance
    >>> y_true = [2, 0, 1, 4]
    >>> y_pred = [0.5, 0.5, 2., 2.]
    >>> mean_tweedie_deviance(y_true, y_pred, power=1)
    1.4260...
    """
    y_type, y_true, y_pred, _ = _check_reg_targets(
        y_true, y_pred, None, dtype=[np.float64, np.float32])
    if y_type == 'continuous-multioutput':
        raise ValueError("Multioutput not supported in mean_tweedie_deviance")
    check_consistent_length(y_true, y_pred, sample_weight)

    if sample_weight is not None:
        sample_weight = column_or_1d(sample_weight)
        sample_weight = sample_weight[:, np.newaxis]

    message = ("Mean Tweedie deviance error with power={} can only be used on "
               .format(power))
    if power < 0:
        # 'Extreme stable', y_true any real number, y_pred > 0
        if (y_pred <= 0).any():
            raise ValueError(message + "strictly positive y_pred.")
        dev = 2 * (np.power(np.maximum(y_true, 0), 2 - power)
                   / ((1 - power) * (2 - power))
                   - y_true * np.power(y_pred, 1 - power)/(1 - power)
                   + np.power(y_pred, 2 - power)/(2 - power))
    elif power == 0:
        # Normal distribution, y_true and y_pred any real number
        dev = (y_true - y_pred)**2
    elif power < 1:
        raise ValueError("Tweedie deviance is only defined for power<=0 and "
                         "power>=1.")
    elif power == 1:
        # Poisson distribution, y_true >= 0, y_pred > 0
        if (y_true < 0).any() or (y_pred <= 0).any():
            raise ValueError(message + "non-negative y_true and strictly "
                             "positive y_pred.")
        dev = 2 * (xlogy(y_true, y_true/y_pred) - y_true + y_pred)
    elif power == 2:
        # Gamma distribution, y_true and y_pred > 0
        if (y_true <= 0).any() or (y_pred <= 0).any():
            raise ValueError(message + "strictly positive y_true and y_pred.")
        dev = 2 * (np.log(y_pred/y_true) + y_true/y_pred - 1)
    else:
        if power < 2:
            # 1 < p < 2 is Compound Poisson, y_true >= 0, y_pred > 0
            if (y_true < 0).any() or (y_pred <= 0).any():
                raise ValueError(message + "non-negative y_true and strictly "
                                           "positive y_pred.")
        else:
            if (y_true <= 0).any() or (y_pred <= 0).any():
                raise ValueError(message + "strictly positive y_true and "
                                           "y_pred.")

        dev = 2 * (np.power(y_true, 2 - power)/((1 - power) * (2 - power))
                   - y_true * np.power(y_pred, 1 - power)/(1 - power)
                   + np.power(y_pred, 2 - power)/(2 - power))

    return np.average(dev, weights=sample_weight)


def mean_poisson_deviance(y_true, y_pred, sample_weight=None):
    """Mean Poisson deviance regression loss.

    Poisson deviance is equivalent to the Tweedie deviance with
    the power parameter `p=1`.

    Read more in the :ref:`User Guide <mean_tweedie_deviance>`.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth (correct) target values. Requires y_true >= 0.

    y_pred : array-like of shape (n_samples,)
        Estimated target values. Requires y_pred > 0.

    sample_weight : array-like, shape (n_samples,), optional
        Sample weights.

    Returns
    -------
    loss : float
        A non-negative floating point value (the best value is 0.0).

    Examples
    --------
    >>> from sklearn.metrics import mean_poisson_deviance
    >>> y_true = [2, 0, 1, 4]
    >>> y_pred = [0.5, 0.5, 2., 2.]
    >>> mean_poisson_deviance(y_true, y_pred)
    1.4260...
    """
    return mean_tweedie_deviance(
        y_true, y_pred, sample_weight=sample_weight, power=1
    )


def mean_gamma_deviance(y_true, y_pred, sample_weight=None):
    """Mean Gamma deviance regression loss.

    Gamma deviance is equivalent to the Tweedie deviance with
    the power parameter `p=2`. It is invariant to scaling of
    the target variable, and measures relative errors.

    Read more in the :ref:`User Guide <mean_tweedie_deviance>`.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth (correct) target values. Requires y_true > 0.

    y_pred : array-like of shape (n_samples,)
        Estimated target values. Requires y_pred > 0.

    sample_weight : array-like, shape (n_samples,), optional
        Sample weights.

    Returns
    -------
    loss : float
        A non-negative floating point value (the best value is 0.0).

    Examples
    --------
    >>> from sklearn.metrics import mean_gamma_deviance
    >>> y_true = [2, 0.5, 1, 4]
    >>> y_pred = [0.5, 0.5, 2., 2.]
    >>> mean_gamma_deviance(y_true, y_pred)
    1.0568...
    """
    return mean_tweedie_deviance(
        y_true, y_pred, sample_weight=sample_weight, power=2
    )
