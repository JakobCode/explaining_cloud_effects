"""
Compute the cloud coverage over SEN12MSCR and save the results in a pickle file.
"""

import sys
from pathlib import Path
sys.path.append(str(Path().resolve()))
import os
import numpy as np
import pickle as pkl
import rasterio
from s2cloudless import S2PixelCloudDetector
import glob
import argparse
from datetime import datetime
import multiprocessing as mp

def load_s2(path: str):
    """
    Utility function for reading s2 data. The samples are
    loaded and normlized to the range of [0,1].

    Args:
        path (str): Path to the SEN12MS Sentinel-2 TIFF-file.
    """

    s2_bands = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]

    with rasterio.open(path) as data:
        s2 = data.read(s2_bands)

    s2 = s2.astype(np.float32)
    s2 = np.clip(s2, 0, 10000)
    s2 /= 10000
    return s2

cloud_detector = S2PixelCloudDetector(threshold=0.4, average_over=4, dilation_size=2, all_bands=True)

def compute_coverage(img_path):

    img_name = img_path.split("/")[-1]
    img = np.transpose(load_s2(img_path))
    preds = cloud_detector.get_cloud_masks(img)
    
    occurrences = np.sum(preds == 1)
    cloud_coverage = (occurrences/(256*256))*100
    
    return (img_name, cloud_coverage)

def compute_cloud_coverage(data_dir: str, save_pkl_path: str = "./cloud_coverage.pkl"):
    """
    Function to compute the sample-wise cloud coverage for a subset of the
    SEN12MSCR data set. The results for all samples are saved into one
    pickle file. For the computation of the cloud coverage, the S2PixelCLoudDetector
    is used.

    Args:
        data_dir (str):         Path to the root folder of the SEN12MSCR dataset.
        save_pkl_path (str):    Path to the file where the results shall be saved to.
    """
    os.makedirs(str(Path(save_pkl_path).parent), exist_ok=True)

    cloud_cov_by_id = {}
    
    img_list = [f for f in glob.glob(os.path.join(data_dir, '*','*','*.tif')) if "_s2_cloudy" in f]

    print(f"{len(img_list)} images found in the data set!")

    pool = mp.Pool(min(mp.cpu_count(), 25))
    start = datetime.now()

    print(f"Start parallel computation of {len(img_list)} samples on {pool._processes} CPUs.")

    res_tupple = []
    num_steps = int(np.ceil(len(img_list)/100))

    for batch_id in range(num_steps):
        print(f"Process batch {batch_id+1} of {num_steps}!                  ", end="\r")
        res_tupple += pool.map(compute_coverage, img_list[batch_id*100:min((batch_id+1)*100,len(img_list))])   
    
    print(f"{datetime.now()-start} for {len(img_list)} files.")
    cloud_cov_by_id = dict(res_tupple)
    
    with open(save_pkl_path, "bw") as myfile:
        pkl.dump(cloud_cov_by_id, myfile)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_root_path",
        type=str,
        help="Root path of SEN12MSCR data set.",
    )

    parser.add_argument(
        "--save_pkl_path",
        type=str,
        default="./pkl_files/cloud_coverage.pkl",
        help="Save path for pickle file containing the cloud coverage information.",
    )

    args = parser.parse_args()

    compute_cloud_coverage(data_dir=args.data_root_path, save_pkl_path=args.save_pkl_path)
