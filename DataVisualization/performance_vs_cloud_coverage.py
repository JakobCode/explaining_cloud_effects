"""
Generates a barplot based on the result pickle file
showing the relation between cloud coverage and predictive
performance.
"""

import sys
from pathlib import Path
import pickle as pkl
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score
import os
import argparse
import matplotlib.pyplot as plt

from matplotlib import rcParams

# Setup Matplotlib Parameters
rcParams["axes.titlepad"] = 14
rcParams["xtick.labelsize"] = 11
rcParams["ytick.labelsize"] = 11
rcParams["xtick.major.pad"] = "8"
rcParams["ytick.major.pad"] = "8"
rcParams.update({"font.size": 11})
plt.yticks(fontsize=11)
plt.xticks(fontsize=11)

plt.rc("legend", fontsize=11)  # using a size in points
title_size = 14


def plot_performance_cloud_clouster(
    result_pkl_path: str,
    cloud_coverage_pkl_path: str, 
    target_folder: str="./ResulPlots", 
    num_bins: int=10
):
    """
    Method to plot the predictive performance of samples cloustered into different
    intensities of cloud coverages.

    Arguments:
    result_pkl_path (str):      Path to pickle file with prediction on test samples
    target_folder (str):        Path where the plots are saved into (defaul: "./ResulPlots").
    num_bins (int):             Number of clusters (default: 10).
    """
    os.makedirs(target_folder, exist_ok=True)

    with open(result_pkl_path, "rb") as f:
        res_dict = pkl.load(f)
        
    y_pred_cloudy = np.array(res_dict["y_pred_cloudy"])
    y_pred_clear = np.array(res_dict["y_pred_clear"])
    y_true = np.array(res_dict["y_true"])

    with open(cloud_coverage_pkl_path, "rb") as f:
        cloud_coverage_dict = pkl.load(f)

    cloud_coverage = [cloud_coverage_dict[sample_id.replace("_s2_", "_s2_cloudy_")] for sample_id in res_dict["sample_id"]]

    bins = np.stack(
        [
            np.arange(num_bins) * 100 / num_bins,
            (np.arange(num_bins) + 1) * 100 / num_bins,
        ],
        axis=-1,
    )

    bin_cont = []
    performance = {
        "cloudy": {"confidence": [], "accuracy": [], "avg. accuracy": []},
        "clear": {"confidence": [], "accuracy": [], "avg. accuracy": []},
    }

    for single_bin in bins:

        mask = np.logical_and(
            cloud_coverage <= single_bin[1], cloud_coverage >= single_bin[0]
        )

        bin_cont.append(
            {
                "cloudy": y_pred_cloudy[mask],
                "clear": y_pred_clear[mask],
                "true": y_true[mask],
            }
        )

        performance["cloudy"]["confidence"].append(
            np.mean(np.max(bin_cont[-1]["cloudy"], axis=-1))
        )
        performance["cloudy"]["accuracy"].append(
            accuracy_score(
                y_true=np.argmax(bin_cont[-1]["true"], axis=-1),
                y_pred=np.argmax(bin_cont[-1]["cloudy"], axis=-1),
            )
        )
        performance["cloudy"]["avg. accuracy"].append(
            balanced_accuracy_score(
                y_true=np.argmax(bin_cont[-1]["true"], axis=-1),
                y_pred=np.argmax(bin_cont[-1]["cloudy"], axis=-1),
            )
        )

    performance["clear"]["accuracy"].append(
        accuracy_score(
            y_true=np.argmax(y_true, axis=-1), y_pred=np.argmax(y_pred_clear, axis=-1)
        )
    )
    performance["clear"]["confidence"].append(np.mean(np.max(y_pred_clear, axis=-1)))
    performance["clear"]["avg. accuracy"].append(
        balanced_accuracy_score(
            y_true=np.argmax(y_true, axis=-1), y_pred=np.argmax(y_pred_clear, axis=-1)
        )
    )

    x_val = np.arange(1, num_bins + 1, step=1)
    x_val = np.append(x_val, x_val[-1] + 2)
    width = 0.3

    plt.bar(
        x=x_val - 0.30,
        height=np.concatenate(
            [performance["cloudy"]["confidence"], performance["clear"]["confidence"]]
        ),
        label="Avg. Confidence",
        width=width,
    )
    plt.bar(
        x=x_val,
        height=np.concatenate(
            [performance["cloudy"]["accuracy"], performance["clear"]["accuracy"]]
        ),
        label="Accuracy",
        width=width,
    )
    plt.bar(
        x=x_val + 0.3,
        height=np.concatenate(
            [
                performance["cloudy"]["avg. accuracy"],
                performance["clear"]["avg. accuracy"],
            ]
        ),
        label="Avg. Accuracy",
        width=width,
    )

    plt.ylim([0, 1])
    plt.axvline(x=0.5 * (x_val[-1] + x_val[-2]), color="black", linestyle="--")
    plt.xticks(
        x_val,
        [f"{int(b[0])}% to {int(b[1])}%" for b in bins] + ["Clear Test Set"],
        rotation=-90,
    )
    plt.title(
        "Classification Performance on Clear Test Set \nand Different Ranges of Cloud Coverage",
        fontsize=title_size,
    )
    plt.legend(loc="upper left")
    plt.tight_layout()

    plt.savefig(os.path.join(target_folder, "cloudy_performance.pdf"))
    plt.savefig(os.path.join(target_folder, "cloudy_performance.png"))


if __name__ == "__main__":
    sys.path.append(str(Path().resolve()))

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--predictions_pkl_path",
        type=str,
        help="Path to pickle file with prediction on test samples.",
    )

    parser.add_argument(
        "--cloud_coverage_pkl_path",
        type=str,
        help="Path to pickle file with cloud coverage.",
        default="./pkl_files/cloud_coverage.pkl"
    )


    parser.add_argument(
        "--target_folder",
        type=str,
        default="./ResultPlots/CloudyPerformance",
        help="Path where the plots are saved into.",
    )

    parser.add_argument("--num_bins", type=int, help="Number of clusters.", default=10)

    args = parser.parse_args()

    plot_performance_cloud_clouster(
        result_pkl_path=args.predictions_pkl_path,
        cloud_coverage_pkl_path=args.cloud_coverage_pkl_path,
        target_folder=args.target_folder,
        num_bins=args.num_bins,
    )
