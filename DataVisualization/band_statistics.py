"""
Visualizes band statistics of SEN12MS and SEN12MSCR samples based on
cloud coverage and class.
"""

import sys
from pathlib import Path
sys.path.append(str(Path().resolve()))
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
import os
import argparse

from DataHandling.data_loader_sen12ms import class_names


def plot_bandwise_fingerprint(
    pkl_file: str, target_folder: str = "./ResultPlots/BandStatistics"
):
    """
    This methods visualizes the band statistics computed in ComputeBandStatistics.py.

    Arguments:
    pkl_file (str):                     Path to pickle file with band statistics.
    target_folder (str, optional):      Path to target folder for the plots.
    """

    os.makedirs(target_folder, exist_ok=True)
    with open(pkl_file, "rb") as f:
        stats = pkl.load(f)

    # Plot for each bin
    cloudy = stats["cloudy"]
    clear = stats["clear"]

    std_frac = 1

    values = []
    std_list = []
    for label in cloudy:
        num_samples = 0
        mu = np.zeros(13, dtype=np.float64)
        mu_sqr = np.zeros(13, dtype=np.float64)

        for coverage in cloudy[label]:
            if coverage < 100:
                continue

            num_samples += cloudy[label][coverage]["num_samples"][0]
            assert (
                cloudy[label][coverage]["num_samples"][0]
                == cloudy[label][coverage]["num_samples"][-1]
            )

            mu += cloudy[label][coverage]["mu"]
            mu_sqr += cloudy[label][coverage]["mu_sqr"]

        if num_samples != 0:

            mu /= num_samples
            std = (mu_sqr / num_samples - mu ** 2) ** 0.5

        assert all(std >= 0)
        values.append(mu)
        std_list.append(std)

    for i in range(10):
        plt.fill_between(
            np.arange(1, 14),
            values[i] - std_frac * std_list[i],
            values[i] + std_frac * std_list[i],
            alpha=0.5,
        )
    for i in range(10):
        plt.plot(np.arange(1, 14), values[i], label=class_names[i])

    plt.legend()
    plt.ylim([-0.1, 8000])
    plt.savefig(os.path.join(target_folder, f"095_coverage_{std_frac}_std_cloudy.pdf"))
    plt.savefig(os.path.join(target_folder, f"095_coverage_{std_frac}_std_cloudy.png"))
    plt.clf()

    values = []
    std_list = []
    for label in clear:
        num_samples = 0
        mu = np.zeros(13, dtype=np.float64)
        mu_sqr = np.zeros(13, dtype=np.float64)

        for coverage in clear[label]:
            if coverage < 100:
                continue

            num_samples += clear[label][coverage]["num_samples"][0]

            mu += clear[label][coverage]["mu"]
            mu_sqr += clear[label][coverage]["mu_sqr"]

        if num_samples != 0:

            mu /= num_samples
            std = (mu_sqr / num_samples - mu ** 2) ** 0.5

        assert all(std >= 0)
        values.append(mu)
        std_list.append(std)

    for i in range(10):
        plt.plot(np.arange(1, 14), values[i], label=class_names[i])
        plt.fill_between(
            np.arange(1, 14),
            values[i] - std_frac * std_list[i],
            values[i] + std_frac * std_list[i],
            alpha=0.5,
        )
    plt.legend()
    plt.ylim([-0.1, 8000])
    plt.savefig(os.path.join(target_folder, f"095_coverage_{std_frac}_std_clear.pdf"))
    plt.savefig(os.path.join(target_folder, f"095_coverage_{std_frac}_std_clear.png"))

    print(f"Plots saved to '{target_folder}':")
    print(f"  (1)  095_coverage_{std_frac}_std_clear.png")
    print(f"  (2)  095_coverage_{std_frac}_std_clear.pdf")
    print(f"  (1)  095_coverage_{std_frac}_std_cloudy.png")
    print(f"  (2)  095_coverage_{std_frac}_std_cloudy.pdf")
    print("")
    plt.clf()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--band_stats_pkl_file", type=str, help="Path to pickle file with band statistics.",
        default="./pkl_files/band_stats.pkl"
    )

    parser.add_argument(
        "--target_folder",
        type=str,
        help="Target folder for plots.",
        default="./ResultPlots/BandStatistics",
    )

    args = parser.parse_args()

    print("Plot bandwise finger prints:")
    plot_bandwise_fingerprint(pkl_file=args.band_stats_pkl_file, target_folder=args.target_folder)

    print("Process finished!")