# Train/Fine-Tune SAM 2 on the SAV dataset

import numpy as np
import torch
import os
import time
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sav_dataset.utils.sav_utils import SAVDataset
import pycocotools.mask as mask_util
import cv2
from datetime import datetime
import wandb

# Initialize SAVDataset
sav_dir = os.path.expanduser("~/mldata/sav_000")
sav_dataset = SAVDataset(sav_dir)

# set the seed for numpy and torch

seed = 999
np.random.seed(seed)
torch.manual_seed(seed)

config = {
    "seed": seed,
    "batch_size": 16,
    "learning_rate": 1e-5,
    "weight_decay": 4e-5,
    "num_epochs": 101,
    "model_cfg": "sam2_hiera_t.yaml",
    "sam2_checkpoint": "checkpoints/sam2_hiera_tiny.pt",
}


# Get list of video IDs
video_ids = [f.split(".")[0] for f in os.listdir(sav_dir) if f.endswith(".mp4")]


def read_video(video_ids):
    video_id = np.random.choice(video_ids)
    frames, manual_annot, _ = sav_dataset.get_frames_and_annotations(video_id)

    if frames is None or manual_annot is None:
        return None, None, None, None

    # Randomly select a frame
    frame_idx = np.random.randint(len(frames))
    Img = frames[frame_idx]
    r = np.min([1024 / Img.shape[1], 1024 / Img.shape[0]])  # scalling factor
    Img = cv2.resize(Img, (int(Img.shape[1] * r), int(Img.shape[0] * r)))
    orig_shape = Img.shape
    # Get masks and points from manual annotations
    rles = manual_annot["masklet"][frame_idx]
    masks = [mask_util.decode(rle) for rle in rles]

    points = []
    valid_masks = []
    for mask in masks:
        coords = np.argwhere(mask > 0)
        if len(coords) > 0:
            valid_masks.append(mask)

    if len(valid_masks) == 0:
        return None, None, None

    # Ensure all masks have the same shape
    H, W = orig_shape[:2]
    valid_masks = [
        cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
        for mask in valid_masks
    ]

    if Img.shape[0] < 1024:
        Img = np.concatenate(
            [Img, np.zeros([1024 - Img.shape[0], Img.shape[1], 3], dtype=np.uint8)],
            axis=0,
        )
        valid_masks = [
            np.concatenate(
                [mask, np.zeros([1024 - mask.shape[0], mask.shape[1]], dtype=np.uint8)],
                axis=0,
            )
            for mask in valid_masks
        ]
    if Img.shape[1] < 1024:
        Img = np.concatenate(
            [Img, np.zeros([Img.shape[0], 1024 - Img.shape[1], 3], dtype=np.uint8)],
            axis=1,
        )
        valid_masks = [
            np.concatenate(
                [mask, np.zeros([mask.shape[0], 1024 - mask.shape[1]], dtype=np.uint8)],
                axis=1,
            )
            for mask in valid_masks
        ]

    for mask in valid_masks:
        coords = np.argwhere(mask > 0)
        if len(coords) > 0:
            yx = coords[np.random.randint(len(coords))]
            points.append([[yx[1], yx[0]]])

    idx = 0
    return Img, valid_masks[idx], points[idx]


def read_batch(data, batch_size):
    limage = []
    lmask = []
    linput_point = []
    for i in range(batch_size):
        image = None
        while image is None:
            image, mask, input_point = read_video(data)
        limage.append(image)
        lmask.append(mask)
        linput_point.append(input_point)
    return limage, np.array(lmask), np.array(linput_point), np.ones([batch_size, 1])


def main():
    # initialize wandb
    wandb.init(
        project="segment-anything-2-coreml",
        config=config,
        name=f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )
    # Load model
    sam2_checkpoint = config[
        "sam2_checkpoint"
    ]  # path to model weight (pre model loaded from: https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_small.pt)
    model_cfg = config["model_cfg"]  #  model config
    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device="cuda")  # load model
    predictor = SAM2ImagePredictor(sam2_model)

    # Set training parameters

    predictor.model.sam_mask_decoder.train(True)  # enable training of mask decoder
    predictor.model.sam_prompt_encoder.train(True)  # enable training of prompt encoder
    predictor.model.image_encoder.train(True)
    """
    #The main part of the net is the image encoder, if you have good GPU you can enable training of this part by using:
    predictor.model.image_encoder.train(True)
    #Note that for this case, you will also need to scan the SAM2 code for “no_grad” commands and remove them (“ no_grad” blocks the gradient collection, which saves memory but prevents training).
    """
    optimizer = torch.optim.AdamW(
        params=predictor.model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    scaler = torch.cuda.amp.GradScaler()  # mixed precision

    # Training loop

    for itr in range(config["num_epochs"]):
        start_time = time.time()  # Start time for epoch duration tracking
        with torch.cuda.amp.autocast():  # cast to mix precision
            image, mask, input_point, input_label = read_batch(
                video_ids, batch_size=config["batch_size"]
            )
            if mask.shape[0] == 0:
                continue
            predictor.set_image_batch(image)

            # prompt encoding

            mask_input, unnorm_coords, labels, unnorm_box = predictor._prep_prompts(
                input_point,
                input_label,
                box=None,
                mask_logits=None,
                normalize_coords=True,
            )
            sparse_embeddings, dense_embeddings = predictor.model.sam_prompt_encoder(
                points=(unnorm_coords, labels),
                boxes=None,
                masks=None,
            )

            # mask decoder
            high_res_features = [
                feat_level[-1].unsqueeze(0)
                for feat_level in predictor._features["high_res_feats"]
            ]
            low_res_masks, prd_scores, _, _ = predictor.model.sam_mask_decoder(
                image_embeddings=predictor._features["image_embed"],
                image_pe=predictor.model.sam_prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=True,
                repeat_image=False,
                high_res_features=high_res_features,
            )
            prd_masks = predictor._transforms.postprocess_masks(
                low_res_masks, predictor._orig_hw[-1]
            )  # Upscale the masks to the original image resolution

            # Segmentaion Loss caclulation

            gt_mask = torch.tensor(mask.astype(np.float32)).cuda()
            prd_mask = torch.sigmoid(
                prd_masks[:, 0]
            )  # Turn logit map to probability map
            raw_image = wandb.Image(image[0], caption="Raw Image")
            gt_images = wandb.Image(gt_mask[0], caption="Ground Truth Masks")
            prd_images = wandb.Image(prd_mask[0], caption="Predicted Masks")
            seg_loss = (
                -gt_mask * torch.log(prd_mask + 0.00001)
                - (1 - gt_mask) * torch.log((1 - prd_mask) + 0.00001)
            ).mean()  # cross entropy loss

            # Score loss calculation (intersection over union) IOU

            inter = (gt_mask * (prd_mask > 0.5)).sum(1).sum(1)
            iou = inter / (
                gt_mask.sum(1).sum(1) + (prd_mask > 0.5).sum(1).sum(1) - inter
            )
            score_loss = torch.abs(prd_scores[:, 0] - iou).mean()
            loss = seg_loss + score_loss * 0.05  # mix losses

            # apply back propogation

            predictor.model.zero_grad()  # empty gradient
            scaler.scale(loss).backward()  # Backpropogate
            scaler.step(optimizer)
            scaler.update()  # Mix precision

            if itr % 100 == 0:
                torch.save(predictor.model.state_dict(), f"model_{itr}.torch")
                print(f"save model {itr}")

            # Display results

            if itr == 0:
                mean_iou = 0
            mean_iou = mean_iou * 0.99 + 0.01 * np.mean(iou.cpu().detach().numpy())
            print("step)", itr, "Accuracy(IOU)=", mean_iou)
            epoch_duration = time.time() - start_time
            wandb.log(
                {
                    "image": raw_image,
                    "gt_mask": gt_images,
                    "prd_mask": prd_images,
                    "seg_loss": seg_loss,
                    "score_loss": score_loss,
                    "iou": iou,
                    "loss": loss,
                    "mean_iou": mean_iou,
                    "epoch_duration": epoch_duration,
                }
            )


if __name__ == "__main__":
    main()
