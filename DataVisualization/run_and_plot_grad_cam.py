"""
Run over SEN12MS samples, compute and visualize saliency maps with GradCam
and compute statistics over the saliency maps and save them in a pkl file.
"""

import sys
from pathlib import Path
sys.path.append(str(Path().resolve()))
import argparse
import numpy as np
import pickle as pkl
from os.path import join
import os
import matplotlib.pyplot as plt
from captum.attr import LayerGradCam, LayerAttribution

import torch
import torchvision.transforms as transforms

from DataVisualization.grad_cam_violin_plots import plot_violin_plots
from DataVisualization.HelperFunctions.get_gradcam_heatmap import show_cam_on_image
from Models.vgg import VGG16, VGG19
from Models.resnet import ResNet50, ResNet101, ResNet152
from Models.densenet import DenseNet121, DenseNet161, DenseNet169, DenseNet201

from DataHandling.data_loader_sen12ms import (
    SEN12MS,
    ToTensor,
    Normalize,
    bands_mean,
    bands_std,
    class_names,
)


# Setup matplotlib parameters for plots
from matplotlib import rcParams

rcParams["axes.titlepad"] = 14
rcParams["xtick.labelsize"] = 12
rcParams["ytick.labelsize"] = 12
plt.rc("legend", fontsize=11)  # using a size in points
title_size = 13


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


def run_grad_cam(
    model_type: str,
    results_pkl: str,
    cloud_coverage_pkl_path: str,
    data_dir: str,
    label_split_dir: str,
    save_path: str,
    checkpoint_pth: str,
    num_eval: int = None,
    num_print: int = 100,
):

    """
    Method to compute the GradCam based saliency maps on cloudy and clear samples of SEN12MS.

    Arguments:
    model_type (str):       Model type to be loaded and evaluated.
    results_pkl (str):      Path to evaluation pickle file.
    save_path (str):        Path to pickle file with result dicts.
    data_dir (str           Path to SEN12MS dataset.
    label_split_dir (str):  Path to label data and split list.
    checkpoint_pth (str):   Path to the pretrained weights file.
    num_eval (int):         Number of samples to be evaluated.
    num_print (int):        Number of samples to be visualized and saved.
    """

    result_types = ["both_correct", "cloudy_false", "clear_false", "both_false"]

    map_stats = {
        "both_correct": {
            "clear": {"min": [], "max": [], "mean": [], "std": []},
            "cloudy": {"min": [], "max": [], "mean": [], "std": []},
        },
        "cloudy_false": {
            "clear": {"min": [], "max": [], "mean": [], "std": []},
            "cloudy": {"min": [], "max": [], "mean": [], "std": []},
        },
        "clear_false": {
            "clear": {"min": [], "max": [], "mean": [], "std": []},
            "cloudy": {"min": [], "max": [], "mean": [], "std": []},
        },
        "both_false": {
            "clear": {"min": [], "max": [], "mean": [], "std": []},
            "cloudy": {"min": [], "max": [], "mean": [], "std": []},
        },
    }

    # result dictionary
    with open(results_pkl, "rb") as f:
        res_dict = pkl.load(f)

    with open(cloud_coverage_pkl_path,  "rb") as f:
        cloud_coverage_dict = pkl.load(f)

    # define threshold

    num_eval = num_eval if num_eval > -1 else len(res_dict["sample_id"])
    num_print = min(num_eval, num_print)
    batch_size = min(num_eval, 128)

    rgb_mean = np.reshape(
        np.array([bands_mean["s2_mean"][2], bands_mean["s2_mean"][1], bands_mean["s2_mean"][0]]), [1, 1, 1, -1]
    )
    rgb_std = np.reshape(
        np.array([bands_std["s2_std"][2], bands_std["s2_std"][1], bands_std["s2_std"][0]]), [1, 1, 1, -1]
    )

    # load test dataset
    img_transform = transforms.Compose([ToTensor(), Normalize(bands_mean, bands_std)])

    # load non-cloudy data
    test_data_gen_clear = SEN12MS(
        data_dir,
        ls_dir=label_split_dir,
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
        data_dir,
        ls_dir=label_split_dir,
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

    # -------------------------------- ML setup
    # cuda
    use_cuda = torch.cuda.is_available()

    if use_cuda:
        device = "cuda"
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
    else:
        device = "cpu"

    num_classes = 10  # IGBP simple
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

    # import model weights
    checkpoint = torch.load(checkpoint_pth, map_location=device)
    checkpoint["model_state_dict"]["fc.weight"] = checkpoint["model_state_dict"]["FC.weight"]
    checkpoint["model_state_dict"]["fc.bias"] = checkpoint["model_state_dict"]["FC.bias"]

    del checkpoint["model_state_dict"]["FC.weight"]
    del checkpoint["model_state_dict"]["FC.bias"]
    model.load_state_dict(checkpoint["model_state_dict"])

    # move model to GPU if is available
    if use_cuda:
        model = model.cuda()

    model.zero_grad()
    model.eval()

    all_layers = dict(model.named_modules())

    k_list = list(all_layers.keys())

    for i in range(len(all_layers)):
        if "conv" in k_list[i]:
            if "relu" in k_list[i + 1]:
                last_conv_relu = k_list[i + 1]
            if "relu" in k_list[i + 2]:
                last_conv_relu = k_list[i + 2]

    print("Evaluate: ", last_conv_relu)

    os.makedirs(save_path, exist_ok=True)

    cam = LayerGradCam(model, all_layers[last_conv_relu])

    statdict = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: []}
    statdict_cloudy = {
        0: [],
        1: [],
        2: [],
        3: [],
        4: [],
        5: [],
        6: [],
        7: [],
        8: [],
        9: [],
    }

    saved_images = [[], [], [], [], [], [], [], [], [], []]
    np.random.seed(42)

    samples_counter = 0
    
    print(f"Num Eval: {num_eval}")
    print(f"Num Prints: {num_print}")
    print(f"Num Batches: {len(res_dict['sample_id'])//batch_size}")
    print(f"Batch Size: {batch_size}")

    # for i in range(num_examples):
    for i in np.arange(len(res_dict["sample_id"]) // batch_size):

        if samples_counter >= num_eval:
            break
        print("                                               ", end="\r")
        print(f"{i+1}/{len(res_dict['sample_id']) // batch_size}")

        # ids = np.random.randint(size=batch_size, low=0, high=len(res_dict["sample_id"]))
        ids = np.arange(
            start=i * batch_size,
            stop=np.min([(i + 1) * batch_size, len(res_dict["sample_id"])]),
        )

        pred_cloudy = [
            torch.argmax(torch.from_numpy(res_dict["y_pred_cloudy"][idx]), axis=-1)
            for idx in ids
        ]
        pred_clear = torch.stack(
            [
                torch.argmax(torch.from_numpy(res_dict["y_pred_clear"][idx]), axis=-1)
                for idx in ids
            ],
            dim=0,
        )
        groundtruth = torch.stack(
            [
                torch.argmax(torch.from_numpy(res_dict["y_true"][idx]), axis=-1)
                for idx in ids
            ],
            dim=0,
        )
        coverage_list = [
            int(
                cloud_coverage_dict[
                    res_dict["sample_id"][idx].replace("_s2_", "_s2_cloudy_")
                ]
            )
            for idx in ids
        ]

        sample_clear = []
        rgb_img_clear = []
        rgb_img_cloudy = []
        sample_cloudy = []
        pred_cloudy_prob = []
        pred_clear_prob = []

        # load and print predictions
        for idx in ids:
            print(f"Load and print predictions: {samples_counter} / {num_eval}                 ", end="\r")
            samples_counter += 1

            if samples_counter > num_eval:
                break

            n = (
                f"cloudy_false_{i}_{np.argmax(res_dict['y_true'][idx], axis=-1)}"
                + f"_{np.argmax(res_dict['y_pred_clear'][idx], axis=-1)}"
                + f"_{np.argmax(res_dict['y_pred_cloudy'][idx], axis=-1)}"
                + f"{int(cloud_coverage_dict[res_dict['sample_id'][idx].replace('_s2_', '_s2_cloudy_')])}"
            )

            if samples_counter <= num_print:

                plt.clf()
                plt.bar(
                    np.arange(10) - 0.2,
                    res_dict["y_pred_clear"][idx],
                    width=0.4,
                    label="Clear",
                )
                plt.bar(
                    np.arange(10) + 0.2,
                    res_dict["y_pred_cloudy"][idx],
                    width=0.4,
                    label="Cloudy",
                )
                plt.setp(
                    plt.gca().get_xticklabels(),
                    rotation=45,
                    horizontalalignment="right",
                )
                plt.xticks(np.arange(10), class_names)
                plt.legend()
                plt.ylim(0, 1)
                plt.tight_layout()
                plt.savefig(os.path.join(save_path, n + "_pred.pdf"))
                plt.savefig(os.path.join(save_path, n + "_pred.png"))
                plt.clf()

            sample_clear.append(
                test_data_gen_clear.load_by_img_name(res_dict["sample_id"][idx])
            )

            rgb_img_clear.append(
                np.clip(
                    np.transpose(
                        sample_clear[-1]["image"][1:4, :, :].numpy(), axes=[1, 2, 0]
                    )
                    * rgb_std
                    + rgb_mean,
                    a_min=0,
                    a_max=10000,
                )
                / 10000
            )  #  # normalize

            sample_cloudy.append(
                test_data_gen_cloudy.load_by_img_name(res_dict["sample_id"][idx])
            )

            rgb_img_cloudy.append(
                np.clip(
                    np.transpose(
                        sample_cloudy[-1]["image"][1:4, :, :].numpy(),
                        axes=[1, 2, 0],
                    )
                    * rgb_std
                    + rgb_mean,
                    a_min=0,
                    a_max=10000,
                )
                / 10000
            )  #  # normalize

            pred_cloudy_prob.append(torch.from_numpy(res_dict["y_pred_cloudy"][idx]))
            pred_clear_prob.append(torch.from_numpy(res_dict["y_pred_clear"][idx]))

        print(f"Run and print GradCam Predictions                 ", end="\r")

        # Setup results ids:    0 -> cloudy and clear correct predicted
        #                       1 -> only clear correct predicted
        #                       2 -> only cloudy correct predicted
        #                       3 -> Both false
        result_type_ids = (
            3
            - np.array([pred_cloudy[j] == groundtruth[j] for j in range(len(pred_cloudy))])
            - 2 * np.array([pred_clear[j] == groundtruth[j] for j in range(len(pred_cloudy))])
        )

        x_clear = torch.stack(
            [
                torch.as_tensor(sample["image"], device=device)
                for sample in sample_clear
            ],
            0,
        )
        x_cloudy = torch.stack(
            [
                torch.as_tensor(sample["image"], device=device)
                for sample in sample_cloudy
            ],
            0,
        )

        input_tensor = torch.cat([x_clear, x_cloudy], axis=0)

        groundtruth = torch.cat(
            [
                torch.as_tensor(groundtruth, dtype=torch.int64, device=device),
                torch.as_tensor(groundtruth, dtype=torch.int64, device=device),
            ],
            0,
        )
        pred_clear = torch.cat(
            [
                torch.as_tensor(pred_clear, device=device),
                torch.as_tensor(pred_clear, device=device),
            ],
            0,
        )
        pred_cloudy = torch.cat(
            [
                torch.as_tensor(pred_cloudy, device=device),
                torch.as_tensor(pred_cloudy, device=device),
            ],
            0,
        )

        grayscale_cam_true = (
            cam.attribute(
                inputs=input_tensor,
                target=groundtruth,
                attribute_to_layer_input=False,
                relu_attributions=True,
            )
            .cpu()
            .detach()
        )

        grayscale_cam_pred_clear = (
            cam.attribute(
                inputs=input_tensor,
                target=pred_clear,
                attribute_to_layer_input=False,
                relu_attributions=True,
            )
            .cpu()
            .detach()
        )

        grayscale_cam_pred_cloudy = (
            cam.attribute(
                inputs=input_tensor,
                target=pred_cloudy,
                attribute_to_layer_input=False,
                relu_attributions=True,
            )
            .cpu()
            .detach()
        )

        # Iterate over predictions and filter correct predictions
        sub_counter = -1
        for c1, c2 in zip(groundtruth[:batch_size], pred_clear[:batch_size]):
            sub_counter += 1

            if c1 != c2:
                continue
            saved_images[c1].append(grayscale_cam_true[sub_counter])
            statdict[int(c1.cpu().detach().numpy())].append(
                grayscale_cam_true[sub_counter].cpu().detach().numpy()
            )

        sub_counter = batch_size - 1
        for c1, c2 in zip(groundtruth[batch_size:], pred_clear[batch_size:]):
            sub_counter += 1

            saved_images[c1].append(grayscale_cam_true[sub_counter])

            statdict_cloudy[int(c2.cpu().detach().numpy())].append(
                grayscale_cam_pred_cloudy[sub_counter].cpu().detach().numpy()
            )

        grayscale_cam_true = LayerAttribution.interpolate(
            grayscale_cam_true,
            (input_tensor.shape[-2], input_tensor.shape[-1]),
            interpolate_mode="bilinear",
        )
        grayscale_cam_pred_clear = LayerAttribution.interpolate(
            grayscale_cam_pred_clear,
            (input_tensor.shape[-2], input_tensor.shape[-1]),
            interpolate_mode="bilinear",
        )
        grayscale_cam_pred_cloudy = LayerAttribution.interpolate(
            grayscale_cam_pred_cloudy,
            (input_tensor.shape[-2], input_tensor.shape[-1]),
            interpolate_mode="bilinear",
        )

        grayscale_cam_true_clear_list = grayscale_cam_true[:batch_size]
        grayscale_cam_pred_clear_clear_list = grayscale_cam_pred_clear[:batch_size]
        grayscale_cam_pred_cloudy_clear_list = grayscale_cam_pred_cloudy[:batch_size]

        grayscale_cam_true_cloudy_list = grayscale_cam_true[batch_size:]
        grayscale_cam_pred_clear_cloudy_list = grayscale_cam_pred_clear[batch_size:]
        grayscale_cam_pred_cloudy_cloudy_list = grayscale_cam_pred_cloudy[batch_size:]

        grayscale_cam_true_clear_list = (
            grayscale_cam_true_clear_list.squeeze().unsqueeze(-1).numpy()
        )
        grayscale_cam_pred_clear_clear_list = (
            grayscale_cam_pred_clear_clear_list.squeeze().unsqueeze(-1).numpy()
        )
        grayscale_cam_pred_cloudy_clear_list = (
            grayscale_cam_pred_cloudy_clear_list.squeeze().unsqueeze(-1).numpy()
        )
        grayscale_cam_true_cloudy_list = (
            grayscale_cam_true_cloudy_list.squeeze().unsqueeze(-1).numpy()
        )
        grayscale_cam_pred_clear_cloudy_list = (
            grayscale_cam_pred_clear_cloudy_list.squeeze().unsqueeze(-1).numpy()
        )
        grayscale_cam_pred_cloudy_cloudy_list = (
            grayscale_cam_pred_cloudy_cloudy_list.squeeze().unsqueeze(-1).numpy()
        )

        visualization_true_clear_list = [
            show_cam_on_image(
                img=rgb_img_clear[j],
                mask=grayscale_cam_true_clear_list[j],
            )
            for j in range(batch_size)
        ]
        visualization_clear_clear_list = [
            show_cam_on_image(
                rgb_img_clear[j], grayscale_cam_pred_clear_clear_list[j], 
            )
            for j in range(batch_size)
        ]
        visualization_cloudy_clear_list = [
            show_cam_on_image(
                rgb_img_clear[j], grayscale_cam_pred_cloudy_clear_list[j],
            )
            for j in range(batch_size)
        ]

        visualization_true_cloudy_list = [
            show_cam_on_image(
                rgb_img_cloudy[j], grayscale_cam_true_cloudy_list[j],
            )
            for j in range(batch_size)
        ]
        visualization_clear_cloudy_list = [
            show_cam_on_image(
                rgb_img_cloudy[j],
                grayscale_cam_pred_clear_cloudy_list[j],
            )
            for j in range(batch_size)
        ]
        visualization_cloudy_cloudy_list = [
            show_cam_on_image(
                rgb_img_cloudy[j],
                grayscale_cam_pred_cloudy_cloudy_list[j],
            )
            for j in range(batch_size)
        ]

        counter = 0

        for (
            gt,
            pclear,
            pcloudy,
            result_type_id,
            rgb_img_clear_single,
            rgb_img_cloudy_single,
            grayscale_cam_pred_clear_clear,
            grayscale_cam_pred_cloudy_cloudy,
            visualization_true_clear,
            visualization_clear_clear,
            visualization_cloudy_clear,
            visualization_true_cloudy,
            visualization_clear_cloudy,
            visualization_cloudy_cloudy,
            coverage,
        ) in zip(
            groundtruth,
            pred_clear,
            pred_cloudy,
            result_type_ids,
            rgb_img_clear,
            rgb_img_cloudy,
            grayscale_cam_pred_clear_clear_list,
            grayscale_cam_pred_cloudy_cloudy_list,
            visualization_true_clear_list,
            visualization_clear_clear_list,
            visualization_cloudy_clear_list,
            visualization_true_cloudy_list,
            visualization_clear_cloudy_list,
            visualization_cloudy_cloudy_list,
            coverage_list,
        ):

            idx = ids[counter]
            counter += 1

            n = (
                f"cloudy_false_{i}_{np.argmax(res_dict['y_true'][idx], axis=-1)}"
                + f"_{np.argmax(res_dict['y_pred_clear'][idx], axis=-1)}"
                + f"_{np.argmax(res_dict['y_pred_cloudy'][idx], axis=-1)}"
                + f"_{int(cloud_coverage_dict[res_dict['sample_id'][idx].replace('_s2_', '_s2_cloudy_')])}"
            )

            map_stats[result_types[result_type_id]]["clear"]["min"].append(
                np.min(grayscale_cam_pred_clear_clear)
            )
            map_stats[result_types[result_type_id]]["cloudy"]["min"].append(
                np.min(grayscale_cam_pred_cloudy_cloudy)
            )
            map_stats[result_types[result_type_id]]["clear"]["max"].append(
                np.max(grayscale_cam_pred_clear_clear)
            )
            map_stats[result_types[result_type_id]]["cloudy"]["max"].append(
                np.max(grayscale_cam_pred_cloudy_cloudy)
            )
            map_stats[result_types[result_type_id]]["clear"]["mean"].append(
                np.mean(grayscale_cam_pred_clear_clear)
            )
            map_stats[result_types[result_type_id]]["cloudy"]["mean"].append(
                np.mean(grayscale_cam_pred_cloudy_cloudy)
            )
            map_stats[result_types[result_type_id]]["clear"]["std"].append(
                np.std(grayscale_cam_pred_clear_clear)
            )
            map_stats[result_types[result_type_id]]["cloudy"]["std"].append(
                np.std(grayscale_cam_pred_cloudy_cloudy)
            )

            if samples_counter <= num_print:
                postfix = result_types[result_type_id]
                plt.imsave(
                    join(
                        save_path,
                        f"{postfix}_{i}_{gt}_{pclear}_{pcloudy}_{coverage}_clear_mask_pred_clear.pdf",
                    ),
                    visualization_clear_clear[0],
                )
                plt.imsave(
                    join(
                        save_path,
                        f"{postfix}_{i}_{gt}_{pclear}_{pcloudy}_{coverage}_clear_mask_pred_cloudy.pdf",
                    ),
                    visualization_cloudy_clear[0],
                )
                plt.imsave(
                    join(
                        save_path,
                        f"{postfix}_{i}_{gt}_{pclear}_{pcloudy}_{coverage}_clear_mask_true.pdf",
                    ),
                    visualization_true_clear[0],
                )
                plt.imsave(
                    join(
                        save_path,
                        f"{postfix}_{i}_{gt}_{pclear}_{pcloudy}_{coverage}_clear.pdf",
                    ),
                    rgb_img_clear_single[0],
                )
                plt.imsave(
                    join(
                        save_path,
                        f"{postfix}_{i}_{gt}_{pclear}_{pcloudy}_{coverage}_cloudy_mask_pred_clear.pdf",
                    ),
                    visualization_clear_cloudy[0],
                )
                plt.imsave(
                    join(
                        save_path,
                        f"{postfix}_{i}_{gt}_{pclear}_{pcloudy}_{coverage}_cloudy_mask_pred_cloudy.pdf",
                    ),
                    visualization_cloudy_cloudy[0],
                )
                plt.imsave(
                    join(
                        save_path,
                        f"{postfix}_{i}_{gt}_{pclear}_{pcloudy}_{coverage}_cloudy_mask_true.pdf",
                    ),
                    visualization_true_cloudy[0],
                )
                plt.imsave(
                    join(
                        save_path,
                        f"{postfix}_{i}_{gt}_{pclear}_{pcloudy}_{coverage}_cloudy.pdf",
                    ),
                    rgb_img_cloudy_single[0],
                )

    with open(join(save_path, "stat_dict.pkl"), "wb") as f:
        pkl.dump(statdict, f)

    with open(join(save_path, "stat_dict_cloudy.pkl"), "wb") as f:
        pkl.dump(statdict_cloudy, f)

    plot_violin_plots(
        save_path=save_path,
        stat_dict_clear_path=join(save_path, "stat_dict.pkl"),
        stat_dict_cloudy_path=join(save_path, "stat_dict_cloudy.pkl"),
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # data directory
    parser.add_argument("--data_root_path", type=str, help="path to SEN12MS dataset")
    
    # configure
    parser.add_argument(
        "--model_type",
        type=str,
        default="ResNet50",
        choices=model_choices,
        help="model type to evaluate",
    )

    # results
    parser.add_argument(
        "--target_folder",
        type=str,
        default="./ResultPlots/grad_cam/saliency_and_pred",
        help="Folder where results shall save in."
    )

    # cloud coverage pkl
    parser.add_argument(
        "--cloud_coverage_pkl_path",
        type=str,
        default="./pkl_files/cloud_coverage.pkl",
        help="Path to pickle file with cloud coverae information."
    )


    parser.add_argument(
        "--label_split_dir", type=str, help="Path to label data and split list.",
        default="./DataHandling/DataSplits"
    )
    parser.add_argument(
        "--checkpoint_pth",
        type=str,
        default="./saved_models/ResNet50_pretrained.pth",
        help="Path to the pretrained weights file."
    )

    parser.add_argument(
        "--predictions_pkl_path",
        type=str,
        help="Path to pickle file saved after model evaluation.",
    )

    parser.add_argument(
        "--num_eval", type=int, default=-1, help="Number of samples to be evaluated."
    )

    parser.add_argument(
        "--num_print",
        type=int,
        default=100,
        help="Number of samples to be visualized and saved.",
    )

    args = parser.parse_args()

    run_grad_cam(
        model_type=args.model_type,
        data_dir=args.data_root_path,
        results_pkl=args.predictions_pkl_path,
        cloud_coverage_pkl_path=args.cloud_coverage_pkl_path,
        save_path=args.target_folder,
        label_split_dir=args.label_split_dir,
        checkpoint_pth=args.checkpoint_pth,
        num_eval=args.num_eval,
        num_print=args.num_print,
    )
