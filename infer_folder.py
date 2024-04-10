import os
import argparse

import numpy as np
import torch

# import some helper functions from chrislib (will be installed by the intrinsic repo)
from chrislib.general import show, view, uninvert, view_scale
from chrislib.data_util import load_image

# import model loading and running the pipeline
from intrinsic.pipeline import run_pipeline
from intrinsic.model_util import load_models

from utils.imagefolder_dataset import ImageFolderLoader
from utils import image_util




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple example of a ControlNet training script.")
    parser.add_argument("--dataset",
                        type=str,
                        default=None,
                        required=True,
                        help="Dataset to be evaluated")
    parser.add_argument("--output_dir",
                        type=str,
                        default="out",
                        help="Output directory")
    parser.add_argument("--not_maintain_size",
                        action="store_true",
                        default=False,
                        help="Whether to maintain the size of the input image")
    parser.add_argument("--resize_conf",
                        type=int,
                        default=-1,
                        help="Resize confidence"
                        )
    args = parser.parse_args()

    # download the pretrained weights and return the model (may take a bit to download weights)
    intrinsic_model = load_models('paper_weights')
    dataset_config = {
        "iiw_low_resolution": {
            "data_dir": "data/IIW_test_low_resolution",
            "mode": None,
            "is_linear": False
        },
        "ARAP": {
            "data_dir": "data/ARAP/input",
            "mode": None,
            "is_linear": True
        },
        "comparison_IIW": {
            "data_dir": "data/IIW",
            "mode": None,
            "is_linear": False
        },
        "comparison_internet": {
            "data_dir": "data/comparison-internet/",
            "mode": None,
            "is_linear": False
        },
    }
    # data_dir_paths = [
    #     "data/IIW",
    #     "data/Unsplash",
    #     "data/AdobeStock",
    #     "data/BigTime_v1/0030",
    #     "data/BigTime_v1/0306",
    #     "data/test0",
    #     "data/ARAP/input",
    #     "data/iiw_extracted_test_data/low_resolution",
    #     "data/MIT_test_input",
    #     "data/comparison-internet/"
    # ]
    # loading_data_modes = [
    #     None,
    #     None,
    #     None,
    #     "with_mask",
    #     "with_mask",
    #     None,
    #     None,
    #     None,
    #     "MIT_test",
    #     None,
    # ]
    # is_linear = [
    #     False, False, False, False, False, False, True, False, True, False
    # ]

    os.makedirs(args.output_dir, exist_ok=True)
    for dataset_name in[args.dataset]:
        assert dataset_name in dataset_config, f"Dataset {dataset_name} not found in dataset_config"
        config = dataset_config[dataset_name]
        data_dir_path = config["data_dir"]
        loading_data_mode = config["mode"]
        is_linear = config["is_linear"]
        folder_dataset = ImageFolderLoader(
            data_dir=data_dir_path,
            mode=loading_data_mode
        )
        print(f"{len(folder_dataset)} images (is_linear {is_linear}) in path: {data_dir_path} ")
        for k in range(len(folder_dataset)):
            data = folder_dataset[k]
            img_path = data["img_path"]
            img_dir = os.path.dirname(img_path)
            img_name = data["img_name"]
            srgb_img = data["srgb_img"]
            srgb_img_np = srgb_img.cpu().permute(1, 2, 0).numpy()
            # run the image through the pipeline (use R0 resizing dicussed in the paper)
            max_dim = max(srgb_img_np.shape[:2])
            if args.resize_conf < 0:
                if max_dim < 1024:
                    resize_conf = 0.1
                else:
                    resize_conf = 1024
            else:
                resize_conf = args.resize_conf
            if k == 0:
                print(f"Running {img_path} with resize_conf {resize_conf}, is_linear {is_linear}")
            result = run_pipeline(
                intrinsic_model,
                srgb_img_np,
                resize_conf=resize_conf,
                maintain_size=not args.not_maintain_size,
                linear=is_linear,
                device='cuda'
            )
            # convert the inverse shading to regular shading for visualization
            result['shading'] = uninvert(result['inv_shading'])
            shd = result['shading']
            alb = result['albedo']
            vis_shd = view_scale(shd, p=99.5)
            vis_alb = view_scale(alb, p=99.5)
            # show([srgb_img_np, vis_shd, vis_alb], size=(20, 7))
            out_path = os.path.join(args.output_dir, img_dir)
            file_shd = f"{img_name}_s.png"
            file_alb = f"{img_name}_r.png"
            image_util.save_srgb_image(vis_shd, out_path, file_shd)
            image_util.save_srgb_image(vis_alb, out_path, file_alb)
            np.save(os.path.join(out_path, f"{img_name}_r.npy"), result["albedo"])
            np.save(os.path.join(out_path, f"{img_name}_s.npy"), result["shading"])
            if k % 20 == 0:
                print(f"Saving {k}th shading to {out_path}/{file_shd}")
                print(f"Saving {k}th albedo to {out_path}/{file_alb}")
        break
    print("Done")
