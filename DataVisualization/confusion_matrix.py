"""
Generating confusion matrix from cloudy and clear predictions
on SEN12MSCR saved in result pickle files.
"""

import sys
from pathlib import Path
sys.path.append(str(Path().resolve()))
import pickle as pkl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import os
import argparse
from DataHandling.data_loader_sen12ms import class_names

# Setup matplotlib parameters for plots
rcParams["axes.titlepad"] = 14
rcParams["xtick.labelsize"] = 18
rcParams["ytick.labelsize"] = 18
rcParams["xtick.major.pad"] = "8"
rcParams["ytick.major.pad"] = "8"
rcParams.update({"font.size": 18})

plt.rc("legend", fontsize=18)  # using a size in points
title_size = 28
fig_size = 10

filenames_list = []


def print_confusion_matrix(
    pkl_file: str,
    cloud_cover_file: str,
    target_folder: str,
    target_name: str,
    cloud_cover_filter:int
):
    """
    Function to create confusion matrices to visualize the performance on cloudy and clear samples.

    Args:
    filename (str):         Path to pickle file with predictions.
    cloud_cover_file (str):     Path to pickle file with cloud coverage.
    target_folder (str):        Target folder for plots.
    target_name (str):          Target name for plots.
    cloud_cover_filter (int):   Percentage of threshold cloud coverage for cloudy evaluation.
    """
    os.makedirs(target_folder, exist_ok=True)

    with open(cloud_cover_file, "rb") as f:
        cloud_file = pkl.load(f)

    with open(pkl_file, "rb") as f:
        p = pkl.load(f)

    y_true = np.argmax(p["y_true"], axis=-1)
    y_pred_clear = np.argmax(p["y_pred_clear"], axis=-1)
    y_pred_cloudy = np.argmax(p["y_pred_cloudy"], axis=-1)

    id_filter = [
        s >= cloud_cover_filter
        for s in np.array(
            [cloud_file[idx.replace("_s2_", "_s2_cloudy_")] for idx in p["sample_id"]]
        )
    ]

    y_true = y_true[id_filter]
    y_pred_clear = y_pred_clear[id_filter]
    y_pred_cloudy = y_pred_cloudy[id_filter]

    conf_matrix_clear = confusion_matrix(
        y_true, y_pred_clear, labels=np.arange(10), normalize=None
    )

    _, ax = plt.subplots(figsize=(fig_size, fig_size))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=np.around(conf_matrix_clear, 2), display_labels=class_names
    )

    disp.plot(ax=ax)
    plt.xticks(rotation=90)
    plt.title(f"Confusion Matrix {target_name}", fontsize=title_size)
    plt.tight_layout()
    plt.savefig(
        os.path.join(target_folder, "confusion_matrix_" + target_name + "_clear.pdf")
    )

    conf_matrix_cloudy = confusion_matrix(
        y_true, y_pred_cloudy, labels=np.arange(10), normalize=None
    )
    _, ax = plt.subplots(figsize=(fig_size, fig_size))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=np.around(conf_matrix_cloudy, 2), display_labels=class_names
    )

    disp.plot(ax=ax)
    plt.xticks(rotation=90)
    plt.title(f"Confusion Matrix {target_name}", fontsize=title_size)
    plt.tight_layout()
    plt.savefig(
        os.path.join(target_folder, "confusion_matrix_" + target_name + "_cloudy.pdf")
    )

    conf_matrix_clear = confusion_matrix(
        y_true, y_pred_clear, labels=np.arange(10), normalize="true"
    )
    _, ax = plt.subplots(figsize=(fig_size, fig_size))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=np.around(conf_matrix_clear, 2), display_labels=class_names
    )

    disp.plot(ax=ax)
    disp.im_.set_clim(0, 1)
    plt.xticks(rotation=90)
    plt.title(f"Confusion Matrix {target_name}", fontsize=title_size)
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            target_folder, "confusion_matrix_" + target_name + "_clear_norm.pdf"
        )
    )

    conf_matrix_cloudy = confusion_matrix(
        y_true, y_pred_cloudy, labels=np.arange(10), normalize="true"
    )
    _, ax = plt.subplots(figsize=(fig_size, fig_size))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=np.around(conf_matrix_cloudy, 2), display_labels=class_names
    )

    disp.plot(ax=ax)
    disp.im_.set_clim(0, 1)
    plt.xticks(rotation=90)
    plt.title(f"Confusion Matrix {target_name}", fontsize=title_size)
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            target_folder, "confusion_matrix_" + target_name + "_cloudy_norm.pdf"
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--pkl_file", type=str, help="Path to pickle file with predictions.",
        default="./pkl_files/predictions_ResNet50_s2.pkl"
    )

    parser.add_argument(
        "--cloud_cover_file", type=str, help="Path to pickle file with cloud coverage.",
        default="./pkl_files/cloud_coverage.pkl"
    )

    parser.add_argument(
        "--target_folder",
        type=str,
        help="Target folder for plots.",
        default="./ResultPlots/ConfusionMatrix/",
    )

    parser.add_argument(
        "--target_name",
        type=str,
        help="target name for plots.",
        default="",
    )

    parser.add_argument(
        "--cloud_cover_filter",
        type=int,
        help="Percentage of threshold cloud coverage for cloudy evaluation.",
        default=0
    )

    args = parser.parse_args()

    pkl_file_arg = args.pkl_file
    cloud_cover_file_arg = args.cloud_cover_file
    target_folder_arg = args.target_folder
    target_name_arg = args.target_name

    cloud_cover_filter_arg = args.cloud_cover_filter

    print_confusion_matrix(
        pkl_file=pkl_file_arg,
        cloud_cover_file=cloud_cover_file_arg,
        target_folder=target_folder_arg,
        target_name=target_name_arg,
        cloud_cover_filter=cloud_cover_filter_arg,
    )
