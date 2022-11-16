"""
Compute calibration curve from predictions and probabilities.
"""

import numpy as np


def get_calibration_curve(y_true, y_pred, n_bins=10):

    assert (
        np.shape(y_true)[-1] == 1
    ), "input data should be binary. use 'get_multiclass_calibration_curve' instead!"

    weights, prob_pred, _, _ = get_bin_stats_uniform(
        y_true, y_pred, n_bins, cal_type="all"
    )

    return prob_pred, weights


def get_multiclass_calibration_curve(
    y_true, y_pred, n_bins=10, cal_type="all"
):

    if cal_type == "all":
        prob_pred = np.reshape(y_pred, [-1, 1])
        prob_true = np.reshape(y_true, [-1, 1])

    elif cal_type == "max":
        prob_pred = np.amax(y_pred, axis=-1, keepdims=True)
        prob_true = np.array([[y_true[j, i]] for i, j in enumerate(np.argmax(y_pred))])

    else:
        raise Exception()

    return get_calibration_curve(prob_true, prob_pred, n_bins=n_bins)


def get_bin_stats_uniform(y_true, y_pred, n_bins=10, cal_type="all"):
    weights = []
    prob_pred = []
    prob_true = []

    if cal_type == "all":
        for j in range(n_bins):
            bin_mask = np.logical_and(j / n_bins <= y_pred, y_pred < (j + 1) / n_bins)
            correct_size = np.sum(bin_mask.astype(int) * y_true)
            false_size = np.sum(bin_mask.astype(int) * (1 - y_true))
            bin_weight = correct_size + false_size
            weights.append(bin_weight)
            prob_pred.append(np.mean(y_pred[bin_mask]))
            prob_true.append(correct_size / bin_weight)

    elif cal_type == "max":
        for j in range(n_bins):
            bin_mask = np.logical_and(
                j / n_bins <= np.amax(y_pred, axis=-1, keepdims=True),
                np.amax(y_pred, axis=-1, keepdims=True) < (j + 1) / n_bins,
            )
            correct_size = np.sum(
                bin_mask.astype(int) * np.argmax(y_true, axis=-1)
                == np.argmax(y_pred, axis=-1)
            )
            false_size = np.sum(
                bin_mask.astype(int)
                * (1 - np.argmax(y_true, axis=-1) == np.argmax(y_pred, axis=-1))
            )
            bin_weight = correct_size + false_size
            weights.append(bin_weight)
            prob_pred.append(np.mean(np.amax(y_pred, axis=-1, keepdims=True)[bin_mask]))
            prob_true.append(correct_size / bin_weight)

    else:
        raise Exception()

    calibration_error = get_calibration_error(prob_true, prob_pred, weights)

    return weights, prob_pred, prob_true, calibration_error


def get_calibration_error(prob_true, prob_pred, weights):

    weights = np.array(weights)
    prob_pred = np.array(prob_pred)
    n = np.sum(weights)

    ece = np.sum(weights * np.abs(prob_true - prob_pred)) / n

    return ece
