"""
Compute band statistics over SEN12MS and SEN12MSCR. The results are clustered by cloud coverage
(if cloud coverage pickle file is provided) and classes.
"""
import sys
from pathlib import Path
sys.path.append(str(Path().resolve()))
import numpy as np
from DataHandling.data_loader_sen12ms import SEN12MS
import pickle as pkl
from tqdm import tqdm
from torch.utils.data import DataLoader

import os
import argparse


def compute_statistics(
    label_split_dir: str,
    data_dir: str,
    num_workers: int = None,
    cloud_cover_pkl: str = None,
    save_path_pkl: str = None,
):
    """
    Computes the bandstatiscs over the cloudy and non-cloudy versions of the SEN12MS data set.
    The statistics are computed over the whole data set and class-wise.
    If a pickle file with the cloud coverage of the cloudy images is provided, the statistics are also
    computed bin-wise for different levels of cloud-coverage.
    If a save_pkl path is given, the results are not only printed but also saved into a pickle file.

    Args:
        data_dir (str): Path to the root folder of the SEN12MSCR data set.
        label_split_dir (str): Path to label_splits file listing considered samples.
        num_workers (int, optional): Number of workers used for PyTorch data loaders.
        cloud_cover_pkl (str, optional): Path to a pickle-file containing the cloud coverage information.
        save_pkl (str, optional): Path to a pickle-file where the statistics should be saved (optional).
    """

    cf_label_type = "single_label"
    cf_threshold = None
    batch_size = 128

    cloud_wise = cloud_cover_pkl is not None

    if cloud_wise:
        with open(cloud_cover_pkl, "rb") as f:
            cloud_file = pkl.load(f)

    # load non-cloudy data
    data_gen_clear = SEN12MS(
        data_dir,
        ls_dir=label_split_dir,
        img_transform=lambda x: x,
        label_type=cf_label_type,
        threshold=cf_threshold,
        subset="test_cloudy",
        use_s1=False,
        use_s2=True,
        use_rgb=False,
        use_cloudy=False,
        igbp_s=True,
        label_filter=None,
    )
    data_loader_clear = DataLoader(
        data_gen_clear,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=True,
    )

    # load corresponding cloudy data
    data_gen_cloudy = SEN12MS(
        data_dir,
        ls_dir=label_split_dir,
        img_transform=lambda x: x,
        label_type=cf_label_type,
        threshold=cf_threshold,
        subset="test_cloudy",
        use_s1=False,
        use_s2=True,
        use_rgb=False,
        use_cloudy=True,
        igbp_s=True,
        label_filter=None,
    )

    data_loader_cloudy = DataLoader(
        data_gen_cloudy,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=True,
    )

    res_dict = {"clear": {}, "cloudy": {}}

    # prepare sum of mean value, and sum of squared mean values for each class
    for j in range(10):
        res_dict["clear"][j] = {}
        res_dict["cloudy"][j] = {}

        if cloud_wise:
            for i in range(0, 21):
                res_dict["clear"][j][i * 5] = {
                    "num_samples": np.zeros(13, dtype=np.float64),
                    "mu": np.zeros(13, dtype=np.float64),
                    "mu_sqr": np.zeros(13, dtype=np.float64),
                }
                res_dict["cloudy"][j][i * 5] = {
                    "num_samples": np.zeros(13, dtype=np.float64),
                    "mu": np.zeros(13, dtype=np.float64),
                    "mu_sqr": np.zeros(13, dtype=np.float64),
                }
        else:
            res_dict["clear"][j][-1] = {
                "num_samples": np.zeros(13, dtype=np.float64),
                "mu": np.zeros(13, dtype=np.float64),
                "mu_sqr": np.zeros(13, dtype=np.float64),
            }
            res_dict["cloudy"][j][-1] = {
                "num_samples": np.zeros(13, dtype=np.float64),
                "mu": np.zeros(13, dtype=np.float64),
                "mu_sqr": np.zeros(13, dtype=np.float64),
            }

    # Iterate over batches in data set of clear and cloudy images
    for cloudy_clear, dataloader in zip(
        ["cloudy", "clear"], [data_loader_clear, data_loader_cloudy]
    ):
        print(f"Iterate over {cloudy_clear} data:")
        for _, data in enumerate(tqdm(dataloader)):

            x = data["image"].numpy()
            labels = np.argmax(data["label"].numpy(), axis=-1)

            if cloud_wise:
                coverage = [
                    int(s)
                    for s in 5
                    * np.ceil(
                        np.array(
                            [
                                cloud_file[idx.replace("_s2_", "_s2_cloudy_")]
                                for idx in data["id"]
                            ]
                        )
                        / 5
                    )
                ]
            else:
                coverage = [-1 for idx in data["id"]]

            mu = np.mean(x, axis=(-1, -2))
            mu_sqr = np.mean(x ** 2, axis=(-1, -2))

            for i in range(len(x)):
                res_dict[cloudy_clear][labels[i]][coverage[i]]["num_samples"] += 1
                res_dict[cloudy_clear][labels[i]][coverage[i]]["mu"] += mu[i]
                res_dict[cloudy_clear][labels[i]][coverage[i]]["mu_sqr"] += mu_sqr[i]

    num_samples = np.sum(
        [
            res_dict["clear"][l][c]["num_samples"]
            for l in range(10)
            for c in np.arange(0, 100, 5)
        ]
    )
    mu = (
        np.sum(
            [
                res_dict["clear"][l][c]["mu"]
                for l in range(10)
                for c in np.arange(0, 100, 5)
            ]
        )
        / num_samples
    )
    mu_sqr = (
        np.sum(
            [
                res_dict["clear"][l][c]["mu_sqr"]
                for l in range(10)
                for c in np.arange(0, 100, 5)
            ]
        )
        / num_samples
    )
    print("\n# Statistics #############################################")
    print("  Clear Data")
    print("    Mean:          ", mu)
    print("    Cloudy Data:   ", (mu_sqr - mu_sqr) ** 0.5)

    num_samples = np.sum(
        [
            res_dict["cloudy"][l][c]["num_samples"]
            for l in range(10)
            for c in np.arange(0, 100, 5)
        ]
    )
    mu = (
        np.sum(
            [
                res_dict["cloudy"][l][c]["mu"]
                for l in range(10)
                for c in np.arange(0, 100, 5)
            ]
        )
        / num_samples
    )
    mu_sqr = (
        np.sum(
            [
                res_dict["cloudy"][l][c]["mu_sqr"]
                for l in range(10)
                for c in np.arange(0, 100, 5)
            ]
        )
        / num_samples
    )
    print("  Cloudy Data")
    print("    Mean:          ", mu)
    print("    Cloudy Data:   ", (mu_sqr - mu_sqr) ** 0.5)

    if save_path_pkl is not None:
        os.makedirs(Path(save_path_pkl).parent, exist_ok=True) 

        print("\nSave statistics to: {save_path_pkl}")
        with open(save_path_pkl, "wb") as f:
            pkl.dump(res_dict, f)
        print("saving done!")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()


    parser.add_argument(
        "--data_root_path",
        type=str,
        help="Root path of SEN12MSCR data set.",
    )
    parser.add_argument(
        "--label_split_dir",
        type=str,
        default="./DataHandling/DataSplits",
        help="Path to label split pickel file, for availabel with the officialSEN12MS repository.",
    )
    parser.add_argument(
        "--save_pkl_path",
        type=str,
        help="Save path for bandstatistics pickle file.",
        default="./pkl_files/band_stats.pkl",
    )

    parser.add_argument(
        "--cloud_cover_pkl",
        type=str,
        default="./pkl_files/cloud_coverage.pkl",
        help="Path to a pickle-file containing the cloud coverage information.",
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        help="Number of workers for PyTorch data loader.",
        default=16,
    )

    args = parser.parse_args()

    compute_statistics(
        save_path_pkl=args.save_pkl_path,
        label_split_dir=args.label_split_dir,
        data_dir=args.data_root_path,
        num_workers=args.num_workers,
        cloud_cover_pkl=args.cloud_cover_pkl,
    )
