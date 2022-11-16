"""
Visualizes the saliency maps statistics computed with
run_and_plot_grad_cam.py as violin plots.
"""

import sys
from pathlib import Path
sys.path.append(str(Path().resolve()))
import pickle as pkl
import matplotlib.pyplot as plt
import os
import numpy as np
import argparse
from DataHandling.data_loader_sen12ms import class_names

# Setup matplotlib parameters for plots
from matplotlib import rcParams

rcParams["axes.titlepad"] = 14
rcParams["xtick.labelsize"] = 12
rcParams["ytick.labelsize"] = 12
plt.rc("legend", fontsize=11)
title_size = 13


def plot_violin_plots(
    save_path: str, stat_dict_clear_path: str = None, stat_dict_cloudy_path: str = None
):
    """
    Function to visualize the saliency maps based violine plots from the paper.

    Args:
        save_path (str):                 Path where the violin plots are saved to.
        stat_dict_clear_path (str):     Path to the pickle file with the GradCam.
                                        statistics of the clear samples (default: None).
        stat_dict_cloudy_path (str):    Path to the pickle file with the GradCam.
                                        statistics of the cloudy samples (default: None).
    """

    os.makedirs(save_path, exist_ok=True)
    num_points = 100000

    if stat_dict_clear_path is None:
        print("Skip stats on clear images (argument is None) ...")
    else:
        print("Process stats on clear images ...")
        with open(stat_dict_clear_path, "rb") as f:
            statdict = pkl.load(f)

        d_clear = []
        class_sub_names = []

        for class_id in range(10):
            if class_id in [
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                9,
            ]:  # not all classes in test data set!
                d_clear.append(np.array(statdict[class_id]).flatten())
                class_sub_names.append(class_names[class_id])

        r = plt.violinplot(
            dataset=d_clear,
            vert=True,
            widths=0.5,
            showmeans=True,
            showextrema=True,
            showmedians=True,
            points=num_points,
            bw_method=None,
            data=None,
        )
        r["cmeans"].set_color("r")
        r["cmedians"].set_color("g")
        plt.title("Saliency distribution over the non-cloudy test set")
        plt.xticks(np.arange(1, len(class_sub_names) + 1), class_sub_names)
        plt.xticks(rotation=45)
        plt.ylim(0, 0.8)
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, "GradCamStatistics_clear.pdf"))
        plt.savefig(os.path.join(save_path, "GradCamStatistics_clear.png"))
        plt.clf()

    if stat_dict_clear_path is None:
        print("Skip stats on cloudy images (argument is None) ...")
    else:
        print("Process stas on cloudy images ...")
        with open(stat_dict_cloudy_path, "rb") as f:
            statdict_cloudy = pkl.load(f)

        d_cloudy = []
        class_sub_names = []

        class_sub_names = []
        for class_id in range(10):
            if class_id in [
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                9,
            ]:  # not all classes in test data set!
                d_cloudy.append(np.array(statdict_cloudy[class_id]).flatten())
                class_sub_names.append(class_names[class_id])

        r = plt.violinplot(
            dataset=d_cloudy,
            vert=True,
            widths=0.5,
            showmeans=True,
            showextrema=True,
            showmedians=True,
            points=num_points,
            bw_method=None,
            data=None,
        )
        r["cmeans"].set_color("r")
        r["cmedians"].set_color("g")
        plt.title("Saliency distribution over the cloudy test samples")
        plt.xticks(np.arange(1, len(class_sub_names) + 1), class_sub_names)
        plt.xticks(rotation=45)
        plt.ylim(0, 0.8)
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, "GradCamStatistics_cloudy.pdf"))
        plt.savefig(os.path.join(save_path, "GradCamStatistics_cloudy.png"))
        plt.clf()

    if stat_dict_cloudy_path is not None and stat_dict_clear_path is not None:
        print("Plot clear and cloudy statistics together ...")
        d_clear.reverse()
        d_cloudy.reverse()
        class_sub_names.reverse()

        r = plt.violinplot(
            dataset=d_cloudy,
            vert=False,
            widths=1.0,
            showmeans=True,
            showextrema=True,
            showmedians=True,
            points=num_points,
            bw_method=None,
            data=None,
        )
        r["cmeans"].set_color("r")
        r["cmedians"].set_color("g")
        r = plt.violinplot(
            dataset=[-x for x in d_clear],
            vert=False,
            widths=1.0,
            showmeans=True,
            showextrema=True,
            showmedians=True,
            points=num_points,
            bw_method=None,
            data=None,
        )
        r["cmeans"].set_color("r")
        r["cmedians"].set_color("g")
        plt.vlines(x=0, ymin=0.5, ymax=9.5, colors="black", linestyles="-")
        plt.title("Saliency Distributions", y=1.06)
        plt.yticks(np.arange(1, len(class_sub_names) + 1), class_sub_names)
        x_max = 0.6
        plt.xlim(-x_max, x_max)
        plt.xticks(
            np.arange(-x_max, x_max + 0.1, 0.1),
            [
                str(np.round(x, decimals=1))
                for x in np.abs(np.arange(-x_max, x_max + 0.1, 0.1))
            ],
        )
        plt.ylim(0.5, 9.5)
        plt.text(y=8.9, x=-0.425, s="Clear Samples", fontdict={"size": 11})
        plt.text(y=8.9, x=0.15, s="Cloudy Samples", fontdict={"size": 11})
        plt.xlabel("Saliency")
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, "GradCamStatistics_both.pdf"))
        plt.savefig(os.path.join(save_path, "GradCamStatistics_both.png"))
        plt.clf()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--stats_clear_path",
        type=str,
        help="Path to pickle file with clear gradcam statistics.",
    )

    parser.add_argument(
        "--stats_cloudy_path",
        type=str,
        help="Path to pickle file with cloudy gradcam statistics.",
    )

    parser.add_argument(
        "--save_path",
        type=str,
        help="Target folder for plots.",
        default="./ResultPlots/grad_cam/Statistics",
    )

    args = parser.parse_args()

    save_path_arg = args.save_path
    stat_dict_clear_path_arg = args.stats_clear_path
    stat_dict_cloudy_path_arg = args.stats_cloudy_path

    plot_violin_plots(
        save_path=save_path_arg,
        stat_dict_clear_path=stat_dict_clear_path_arg,
        stat_dict_cloudy_path=stat_dict_cloudy_path_arg,
    )
