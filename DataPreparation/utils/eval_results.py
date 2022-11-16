"""
Evaluate the performance for given predictions and ground truths and save it to a pickle file.
"""
import numpy as np

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    balanced_accuracy_score,
    fbeta_score,
)
from DataPreparation.utils.metrics import max_probability, mutual_information, entropy
from DataPreparation.utils.calibration_curve import get_bin_stats_uniform

import pickle as pkl


def evaluate_results(
    ids, predicted_logits, y_true, save_name="results.pkl", from_logits=True
):
    ids = np.array(ids)
    predicted_logits = np.array(predicted_logits)
    y_true = np.array(y_true)

    total_dict = get_result_dict(
        ids=ids,
        predicted_logits=predicted_logits,
        y_true=y_true,
        from_logits=from_logits,
    )

    info = {"All classes": total_dict}

    for class_id in range(len(predicted_logits[0])):
        class_filter = [y_true[i][class_id] == 1 for i in range(len(y_true))]

        info[f"Class {class_id}"] = get_result_dict(
            ids=ids[class_filter],
            predicted_logits=predicted_logits[class_filter,],
            y_true=y_true[class_filter,],
            from_logits=from_logits,
        )

    print("saving metrics...")
    with open(save_name, "wb") as f:
        pkl.dump(info, f)


def get_result_dict(ids, predicted_logits, y_true, from_logits):
    if len(ids) == 0:
        return {"num samples": 0}

    logits = np.clip(np.array(predicted_logits).astype("float64"), -20, 20)

    log_sum = np.sum(logits, axis=-1)
    if not from_logits:
        alpha = logits
        alpha0 = log_sum
        predicted_probs = logits
    else:
        from_logits = True
        alpha = np.exp(logits)
        alpha0 = np.sum(alpha, axis=-1)
        predicted_probs = alpha / np.sum(alpha, axis=-1, keepdims=True)

    # convert predicted probabilities into one/multi-hot labels
    loc = np.argmax(predicted_probs, axis=-1)
    y_predicted = np.zeros_like(predicted_probs).astype(np.float32)
    for i in range(len(loc)):
        y_predicted[i, loc[i]] = 1

    y_true = np.asarray(y_true)

    # --------------------------- evaluation with metrics
    # general
    macro_f1 = f1_score(y_true, y_predicted, average="macro")
    micro_f1 = f1_score(y_true, y_predicted, average="micro")

    macro_f2 = fbeta_score(y_true, y_predicted, beta=2, average="macro")
    micro_f2 = fbeta_score(y_true, y_predicted, beta=2, average="micro")

    macro_prec = precision_score(y_true, y_predicted, average="macro")
    micro_prec = precision_score(y_true, y_predicted, average="micro")

    macro_rec = recall_score(y_true, y_predicted, average="macro")
    micro_rec = recall_score(y_true, y_predicted, average="micro")

    accuracy = accuracy_score(y_true=y_true, y_pred=y_predicted)
    print(y_true[0])
    print(y_predicted[0])
    balanced_accuracy = balanced_accuracy_score(
        y_true=np.argmax(y_true, -1), y_pred=np.argmax(y_predicted, -1)
    )

    # average accuracy, \
    # zero-sample classes are not excluded

    # uncertainty measures
    prob_weights, prob_pred, prob_true, calibration_error = get_bin_stats_uniform(
        y_true, predicted_probs, n_bins=10, cal_type="all"
    )

    (
        prob_weights_max,
        prob_pred_max,
        prob_true_max,
        calibration_error_max,
    ) = get_bin_stats_uniform(y_true, predicted_probs, n_bins=10, cal_type="max")

    max_prob = (max_probability(logits, from_logits=from_logits),)
    mutual_info = mutual_information(logits)
    ent = entropy(logits, from_logits=from_logits)

    info = {
        "num samples": len(max_prob),
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "max_prob": max_prob,
        "mutual_info": mutual_info,
        "ent": ent,
        "logsum": log_sum,
        "alpha0": alpha0,
        "macroPrec": macro_prec,
        "microPrec": micro_prec,
        "macroRec": macro_rec,
        "microRec": micro_rec,
        "macroF1": macro_f1,
        "microF1": micro_f1,
        "macroF2": macro_f2,
        "microF2": micro_f2,
        "calibration_curve_all": [prob_pred, prob_true],
        "calibration_curve_max": [prob_pred_max, prob_true_max],
        "calibration_error_all": calibration_error,
        "calibration_error_max": calibration_error_max,
        "calibration_bin_cont_all": prob_weights,
        "calibration_bin_cont_max": prob_weights_max,
    }

    k_list = list(info.keys())
    for k in k_list:
        print(k)
        print(np.shape(info[k]))
        info[k + "_mean_std"] = [np.mean(info[k]), np.std(info[k])]

    info["ids"] = ids

    return info
