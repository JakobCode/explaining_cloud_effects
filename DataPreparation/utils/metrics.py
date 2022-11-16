"""
Different metrics for prediction evluation.
"""

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from scipy.special import psi, gammaln

def precision(outputs):
    """
    Compute precision value of a Dirichlet Distribution from logits.

    Arguments
    outputs (nparray):      Network output / logits
    """
    outputs = outputs.astype("float64")
    outputs = np.clip(outputs, -25, 25)
    return np.sum(np.exp(outputs), axis=-1)


# Returns
# Input:    logits outputs of a neural network
def max_probability(outputs, from_logits=True):
    """
    Computes and returns largest probability value based on received logits.

    Arguments
    outputs (nparray):      Network output / logits
    from_logits (boolean):  compute from logits or probabilities (default: False)
    """
    if from_logits:
        return np.max(_get_prob(outputs), axis=-1)
    else:
        return np.max(outputs, axis=-1)


# Returns mutual information based on received logits
# Input:    logits outputs of a neural network
def mutual_information(outputs):
    """
    Computes mutual information based on received logits.

    Arguments
    outputs (nparray):      Network output / logits
    """
    outputs = outputs.astype("float64")
    outputs = np.clip(outputs, -25, 25)
    alpha_c = np.exp(outputs)
    alpha_0 = np.sum(alpha_c, axis=-1)

    gammaln_alpha_c = gammaln(alpha_c)
    gammaln_alpha_0 = gammaln(alpha_0)

    psi_alpha_c = psi(alpha_c)
    psi_alpha_0 = psi(alpha_0)
    psi_alpha_0 = np.expand_dims(psi_alpha_0, axis=1)

    temp_mat = np.sum((alpha_c - 1) * (psi_alpha_c - psi_alpha_0), axis=1)

    metric = np.sum(gammaln_alpha_c, axis=-1) - gammaln_alpha_0 - temp_mat
    return metric


def _get_prob(outputs):
    """
    Computes and returns the probability from network logits.

    Arguments
    outputs (nparray):      Network output / logits
    """
    logits = outputs.astype("float64")
    logits = np.clip(logits, -25, 25)
    alpha_c = np.exp(logits)
    # alpha_c = np.clip(alpha_c, 10e-25, 10e25)
    alpha_0 = np.sum(alpha_c, axis=-1)
    alpha_0 = np.expand_dims(alpha_0, axis=-1)

    return alpha_c / alpha_0


# Returns entropy based on received logits
# Input:    logits outputs of a neural network
def entropy(outputs, from_logits=True):
    """
    Computes the entropy from a network prediction.

    Arguments
    outputs (nparray):      Network output / logits
    from_logits (boolean):  compute from logits or probabilities (default: False)
    """
    prob = _get_prob(outputs) if from_logits else outputs
    exp_prob = np.log(np.clip(prob, 1e-20, 1))

    ent = -np.sum(prob * exp_prob, axis=-1)
    return ent


def logsum(outputs):
    """
    Computes the sum over the logits of a network prediction.

    Arguments
    outputs (nparray):      Network output / logits
    """
    return np.sum(outputs, axis=-1)



def auroc_dict(outputs, binary_label):
    """
    Computes AUPR and AUROC scores for different metrics for out-of-distribution performance.

    Arguments:
    outputs (nparray):          Output of a neural network.
    binary_label (nparray):     Labels for in- (0) and out-of-distribution (1).
    """
    res_dict = {}

    # precision metric
    print(np.shape(outputs))
    print(np.shape(binary_label))
    print(np.shape(precision(outputs=outputs)))

    res_dict["roc_precision"] = 100 * roc_auc_score(
        binary_label, precision(outputs=outputs)
    )
    res_dict["pr_precision"] = 100 * average_precision_score(
        binary_label, precision(outputs=outputs)
    )

    res_dict["roc_Log-Sum"] = 100 * roc_auc_score(binary_label, logsum(outputs=outputs))
    res_dict["pr_Log-Sum"] = 100 * average_precision_score(
        binary_label, logsum(outputs=outputs)
    )

    # max probability metric
    res_dict["roc_max_probability"] = 100 * roc_auc_score(
        binary_label, max_probability(outputs=outputs)
    )
    res_dict["pr_max_probability"] = 100 * average_precision_score(
        binary_label, max_probability(outputs=outputs)
    )

    # mutual information metric
    res_dict["roc_mutual_information"] = 100 * roc_auc_score(
        binary_label, mutual_information(outputs=outputs)
    )
    res_dict["pr_mutual_information"] = 100 * average_precision_score(
        binary_label, mutual_information(outputs=outputs)
    )

    # entropy metric
    res_dict["roc_entropy"] = 100 * roc_auc_score(
        binary_label, entropy(outputs=outputs)
    )
    res_dict["pr_entropy"] = 100 * average_precision_score(
        binary_label, entropy(outputs=outputs)
    )

    return res_dict


def get_dictionary_from_logits(pred_in, pred_out):
    """
    Computes AUPR and AUROC scores for different metrics for out-of-distribution performance.

    Arguments:
    pred_in (nparray):      Network predictions for in-distribution examples.
    pred_out (nparray):     Network predictions for ood examples.
    """
    labels = np.concatenate([np.zeros(len(pred_in)), np.ones(len(pred_out))], axis=0)
    preds = np.concatenate([pred_in, pred_out], axis=0)

    return auroc_dict(preds, labels)
