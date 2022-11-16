"""
Generates bar plots visualizing the class distributions of the original original
SEN12MS data set and the SEN12MSCR subset.

Generates bar plots visualizing the number of samples for different level of
cloud coverage in SEN12MSCR.
"""

import sys
from pathlib import Path
sys.path.append(str(Path().resolve()))
import os
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
from matplotlib import rcParams

import argparse

from DataHandling.data_loader_sen12ms import class_names, SEN12MS

# Setup Matplotlib
rcParams["axes.titlepad"] = 14
rcParams["xtick.labelsize"] = 11
rcParams["ytick.labelsize"] = 11
rcParams["xtick.major.pad"] = "8"
rcParams["ytick.major.pad"] = "8"
rcParams.update({"font.size": 11})
plt.yticks(fontsize=11)
plt.xticks(fontsize=11)

plt.rc("legend", fontsize=11)  # using a size in points


def create_bar_plot_cloud_coverage(
    samples_per_bin, bins_str, target_folder="./ResultPlots/ClassDistributions"
    ):

    assert len(samples_per_bin) == len(bins_str)

    os.makedirs(target_folder, exist_ok=True)
    fig, ax = plt.subplots()
    fig.subplots_adjust(left=0.2)

    y_pos = np.arange(len(bins_str))
    ax.barh(y_pos, samples_per_bin, align="center")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(bins_str)
    ax.set_ylabel("Cloud coverage", rotation=90)

    ax.set_title("Cloud Coverage in the Cloudy Test Split")
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel("Number of Samples")
    ax.grid(color="gray", linestyle="-", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(target_folder, "CloudCoverDistribution.png"))
    plt.savefig(os.path.join(target_folder, "CloudCoverDistribution.pdf"))

    #print(f"Plots saved to '{target_folder}':")
    print("  (1)  CloudCoverDistribution.png")
    print("  (2)  CloudCoverDistribution.pdf")
    print("")


def create_bar_plot_classes(
    classes_set_1, classes_set_2, target_folder="./ResultPlots/ClassDistributions"
):

    os.makedirs(target_folder, exist_ok=True)

    fig, ax = plt.subplots()
    fig.subplots_adjust(left=0.2)

    y_pos = np.arange(len(class_names))

    ax.barh(
        y_pos - 0.2,
        classes_set_1,
        align="center",
        height=0.4,
        label="Original Test Data",
    )
    ax.barh(
        y_pos + 0.2, classes_set_2, align="center", height=0.4, label="Cloudy Test Data"
    )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(class_names)

    ax.set_title("Class Distribution for the Test Splits")
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel("Number of Samples")
    ax.grid(color="gray", linestyle="-", linewidth=0.5)
    plt.legend()
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    plt.savefig(os.path.join(target_folder, "ClassDistributions.pdf"))
    plt.savefig(os.path.join(target_folder, "ClassDistributions.png"))

    print(f"Plots saved to '{target_folder}':")
    print("  (1)  ClassDistributions.png")
    print("  (2)  ClassDistributions.pdf")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_root_path", type=str, help="path to SEN12MSCR dataset",
    )

    parser.add_argument(
        "--label_split_dir",
        type=str,
        default="./DataHandling/DataSplits",
        help="path to label data and split list",
    )

    parser.add_argument(
        "--cloud_cover_pkl", type=str, help="Cloud coverage pickle-file.",
        default="./pkl_files/cloud_coverage.pkl"
    )

    parser.add_argument(
        "--target_folder",
        type=str,
        default="./ResultPlots/ClassDistributions",
        help="Target folder where the results are saved to.",
    )

    args = parser.parse_args()

    data_dir = args.data_root_path
    target_folder_arg = args.target_folder
    label_split_dir = args.label_split_dir
    cloud_cover_pkl = args.cloud_cover_pkl

    with open(cloud_cover_pkl, "rb") as f:
        cloud_coverage = pkl.load(f)

    # Loading cloudy subset of test set
    data_gen_cloudy = SEN12MS(
        path=data_dir,
        ls_dir=label_split_dir,
        label_type="single_label",
        subset="test_cloudy",
        igbp_s=True,
    )

    a_cloudy = np.array(
        [
            cloud_coverage[sample["id"].replace("_s2_", "_s2_cloudy_")]
            for sample in data_gen_cloudy.samples
            if sample["id"].replace("_s2_", "_s2_cloudy_") in cloud_coverage
        ]
    )

    # Loading original test set
    data_gen_clear = SEN12MS(
        path=data_dir,
        ls_dir=label_split_dir,
        label_type="single_label",
        subset="test",
        igbp_s=True,
    )

    a_labels_clear = data_gen_clear.get_labels()
    a_labels_cloudy = data_gen_cloudy.get_labels()

    # Creating bins of size 5
    bins = np.arange(0, 100, 5)

    # Storing all the bin names in one array
    bins_str_s = [
        "0%",
        "0% to 5%",
        "5% to 10%",
        "10% to 15%",
        "15% to 20%",
        "20% to 25%",
        "25% to 30%",
        "30% to 35%",
        "35% to 40%",
        "40% to 45%",
        "45% to 50%",
        "50% to 55%",
        "55% to 60%",
        "60% to 65%",
        "65% to 70%",
        "70% to 75%",
        "75% to 80%",
        "80% to 85%",
        "85% to 90%",
        "90% to 95%",
        "95% to 100%",
        "100%",
    ]

    samples_per_bin_s = np.zeros(22)

    # Segregating into bins based on a condition
    for i in bins:
        b = list(filter(lambda x: np.logical_and(x > i, x <= i + 5), a_cloudy))

        samples_per_bin_s[int(i / 5)+1] = len(b)
    
    samples_per_bin_s[0] = len(list(filter(lambda x: x==0, a_cloudy)))
    samples_per_bin_s[-1] = len(list(filter(lambda x: x==100, a_cloudy)))

    print("Plot distribution of cloud coverage in cloudy test set.")
    create_bar_plot_cloud_coverage(
        samples_per_bin=samples_per_bin_s,
        bins_str=bins_str_s,
        target_folder=target_folder_arg,
    )

    print("Plot class distribution in original and cloudy test set.")
    # Overall Class Distributions for original test split and cloudy subset of test split
    a_class_cloudy =np.sum(a_labels_cloudy, axis=0)
    a_class_clear = np.sum(a_labels_clear, axis=0)
    create_bar_plot_classes(
        a_class_clear, a_class_cloudy, target_folder=target_folder_arg
    )


    print("Process finished!")