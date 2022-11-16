"""
Evalute the performance of a given Neural Network trained on SEN12MS using the
training process provided with the official repository.
"""
import sys
from pathlib import Path
sys.path.append(str(Path().resolve()))
import argparse
import numpy as np
import pickle as pkl
from tqdm import tqdm
import os 

import torch
from torchvision.transforms import Compose
from torch.utils.data import DataLoader

from DataPreparation.utils.eval_results import evaluate_results
from DataPreparation.utils.metrics import get_dictionary_from_logits

from DataHandling.data_loader_sen12ms import (
    SEN12MS,
    ToTensor,
    Normalize,
    bands_mean,
    bands_std,
)
from Models.vgg import VGG16, VGG19
from Models.resnet import ResNet50, ResNet101, ResNet152
from Models.densenet import DenseNet121, DenseNet161, DenseNet169, DenseNet201


model_choices = [
    "VGG16",
    "VGG19",
    "ResNet50",
    "ResNet101",
    "ResNet152",
    "DenseNet121",
    "DenseNet161",
    "DenseNet169",
    "DenseNet201",
]

# ------------------------ define and parse arguments -------------------------
parser = argparse.ArgumentParser()

# configure
parser.add_argument(
    "--model_type", type=str, default="ResNet50", help="Type of model", choices=model_choices
)

# data directory
parser.add_argument(
    "--data_root_path",
    type=str,
    help="path to SEN12MSCR dataset",
)
parser.add_argument(
    "--label_split_dir",
    type=str,
    default="./DataHandling/DataSplits/",
    help="path to label data and split list",
)
parser.add_argument(
    "--checkpoint_pth",
    type=str,
    help="path to the pretrained weights file",
)

parser.add_argument(
    "--save_folder",
    type=str,
    default="./pkl_files/",
    help="Path to folder where pickle files shall be saved to.",
)

# hyperparameters
parser.add_argument(
    "--batch_size", type=int, default=32, help="mini-batch size (default: 64)"
)
parser.add_argument(
    "--num_workers", type=int, default=16, help="num_workers for data loading in pytorch"
)

args = parser.parse_args()


# -------------------------------- Main Program ------------------------------
def main():
    global args

    model_type = args.model_type

    # load test dataset
    img_transform = Compose([ToTensor(), Normalize(bands_mean, bands_std)])

    target_folder = args.save_folder
    os.makedirs(target_folder, exist_ok=True)

    # load non-cloudy data
    test_data_gen_clear = SEN12MS(
        args.data_root_path,
        ls_dir=args.label_split_dir,
        img_transform=img_transform,
        label_type="single_label",
        threshold=0.1,
        subset="test_cloudy",
        use_s1=False,
        use_s2=True,
        use_rgb=True,
        use_cloudy=False,
        igbp_s=True,
    )

    # load corresponding cloudy data
    test_data_gen_cloudy = SEN12MS(
        args.data_root_path,
        ls_dir=args.label_split_dir,
        img_transform=img_transform,
        label_type="single_label",
        threshold=0.1,
        subset="test_cloudy",
        use_s1=False,
        use_s2=True,
        use_rgb=True,
        use_cloudy=True,
        igbp_s=True,
    )

    # number of input channels
    n_inputs = test_data_gen_clear.n_inputs
    print("input channels =", n_inputs)

    # set up dataloaders
    test_data_loader_clear = DataLoader(
        test_data_gen_clear,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=True,
    )

    test_data_loader_cloudy = DataLoader(
        test_data_gen_cloudy,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=True,
    )

    # -------------------------------- ML setup
    # cuda
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cuda":
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

    # use IGBP simple
    num_classes = 10

    print("num_class: ", num_classes)

    # define model
    if model_type == "VGG16":
        model = VGG16(n_inputs, num_classes)
    elif model_type == "VGG19":
        model = VGG19(n_inputs, num_classes)

    elif model_type == "ResNet50":
        model = ResNet50(n_inputs, num_classes)
    elif model_type == "ResNet101":
        model = ResNet101(n_inputs, num_classes)
    elif model_type == "ResNet152":
        model = ResNet152(n_inputs, num_classes)

    elif model_type == "DenseNet121":
        model = DenseNet121(n_inputs, num_classes)
    elif model_type == "DenseNet161":
        model = DenseNet161(n_inputs, num_classes)
    elif model_type == "DenseNet169":
        model = DenseNet169(n_inputs, num_classes)
    elif model_type == "DenseNet201":
        model = DenseNet201(n_inputs, num_classes)
    else:
        raise NameError("no model")

    # move model to GPU if is available
    model = model.to(device)

    y_true_clear = []
    predicted_probs_clear = []
    logits_clear = []
    predicted_clear = []

    ids_cloudy = []
    ids_clear = []

    y_true_cloudy = []
    predicted_probs_cloudy = []
    logits_cloudy = []
    predicted_cloudy = []

    if "epoch" in args.checkpoint_pth:
        epoch_pre = "_epoch" + args.checkpoint_pth.split("epoch")[-1][:3]
    else:
        epoch_pre = ""

    checkpoint = torch.load(args.checkpoint_pth, map_location=device)
    
    checkpoint["model_state_dict"]["fc.weight"] = checkpoint["model_state_dict"]["FC.weight"]
    checkpoint["model_state_dict"]["fc.bias"] = checkpoint["model_state_dict"]["FC.bias"]

    del checkpoint["model_state_dict"]["FC.weight"]
    del checkpoint["model_state_dict"]["FC.bias"]
    model.load_state_dict(checkpoint["model_state_dict"])
    print(
        "=> loaded checkpoint '{}' (epoch {})".format(
            args.checkpoint_pth, checkpoint["epoch"]
        )
    )

    model.eval()

    with torch.no_grad():
        for batch_idx, data in enumerate(
            tqdm(test_data_loader_clear, desc="test_clear")
        ):

            # unpack sample
            bands = data["image"]
            if batch_idx == 0:
                print(bands.shape)
            labels = data["label"]
            ids_clear += data["id"]

            # move data to gpu if model is on gpu
            bands = bands.to(device)

            # forward pass
            logits = model(bands)

            sm = torch.nn.Softmax(dim=1)
            probs = sm(logits).cpu().numpy()

            logits_clear += list(logits.cpu().numpy())
            labels = labels.cpu().numpy()  # keep true & pred label at same loc.
            predicted_probs_clear += list(probs)
            predicted_clear += list(np.argmax(probs, axis=-1))
            y_true_clear += list(labels)

    with torch.no_grad():
        for batch_idx, data in enumerate(
            tqdm(test_data_loader_cloudy, desc="test_cloudy")
        ):

            # unpack sample
            bands = data["image"]
            labels = data["label"]
            ids_cloudy += data["id"]

            # move data to gpu if model is on gpu
            bands = bands.to(device)

            # forward pass
            logits = model(bands)

            # convert logits to probabilies
            sm = torch.nn.Softmax(dim=1)
            probs = sm(logits).cpu().numpy()

            logits_cloudy += list(logits.cpu().numpy())
            labels = labels.cpu().numpy()  # keep true & pred label at same loc.
            predicted_probs_cloudy += list(probs)
            predicted_cloudy += list(np.argmax(probs, axis=-1))
            y_true_cloudy += list(labels)

    with open(os.path.join(target_folder, "predictions_" + model_type + "_" + epoch_pre + ".pkl"), "wb") as f:
        pkl.dump(
            {
                "sample_id": ids_clear,
                "y_true": y_true_clear,
                "y_pred_cloudy": predicted_probs_cloudy,
                "y_pred_clear": predicted_probs_clear,
            },
            f,
        )

    metric_dict = get_dictionary_from_logits(logits_clear, logits_cloudy)
    with open(os.path.join(target_folder, "performance_" + model_type + "_" + epoch_pre + "_shift_detection.pkl"), "wb") as f:
        pkl.dump(metric_dict, f)

    evaluate_results(
        ids_clear,
        logits_clear,
        y_true_clear,
        save_name=os.path.join(target_folder, "performance_" + model_type + "_" + epoch_pre + "_clear.pkl"),
        from_logits=True,
    )
    evaluate_results(
        ids_cloudy,
        logits_cloudy,
        y_true_cloudy,
        save_name=os.path.join(target_folder, "performance_" + model_type + "_" + epoch_pre + "_cloudy.pkl"),
        from_logits=True,
    )


if __name__ == "__main__":
    main()
