# Train/Fine Tune SAM 2 on LabPics 1 dataset
# This mode use several images in a single batch
# Labpics can be downloaded from: https://zenodo.org/records/3697452/files/LabPicsV1.zip?download=1

import numpy as np
import torch
import cv2
import os

from torch.onnx.symbolic_opset11 import hstack

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# Read data

data_dir = r"LabPicsV1//"  # Path to dataset (LabPics 1)
data = []  # list of files in dataset
for ff, name in enumerate(
    os.listdir(data_dir + "Simple/Train/Image/")
):  # go over all folder annotation
    data.append(
        {
            "image": data_dir + "Simple/Train/Image/" + name,
            "annotation": data_dir + "Simple/Train/Instance/" + name[:-4] + ".png",
        }
    )


def read_single(data):  # read random image and single mask from  the dataset (LabPics)

    #  select image

    ent = data[np.random.randint(len(data))]  # choose random entry
    Img = cv2.imread(ent["image"])[..., ::-1]  # read image
    ann_map = cv2.imread(ent["annotation"])  # read annotation

    # resize image

    r = np.min([1024 / Img.shape[1], 1024 / Img.shape[0]])  # scalling factor
    Img = cv2.resize(Img, (int(Img.shape[1] * r), int(Img.shape[0] * r)))
    ann_map = cv2.resize(
        ann_map,
        (int(ann_map.shape[1] * r), int(ann_map.shape[0] * r)),
        interpolation=cv2.INTER_NEAREST,
    )
    if Img.shape[0] < 1024:
        Img = np.concatenate(
            [Img, np.zeros([1024 - Img.shape[0], Img.shape[1], 3], dtype=np.uint8)],
            axis=0,
        )
        ann_map = np.concatenate(
            [
                ann_map,
                np.zeros(
                    [1024 - ann_map.shape[0], ann_map.shape[1], 3], dtype=np.uint8
                ),
            ],
            axis=0,
        )
    if Img.shape[1] < 1024:
        Img = np.concatenate(
            [Img, np.zeros([Img.shape[0], 1024 - Img.shape[1], 3], dtype=np.uint8)],
            axis=1,
        )
        ann_map = np.concatenate(
            [
                ann_map,
                np.zeros(
                    [ann_map.shape[0], 1024 - ann_map.shape[1], 3], dtype=np.uint8
                ),
            ],
            axis=1,
        )

    # merge vessels and materials annotations

    mat_map = ann_map[:, :, 0]  # material annotation map
    ves_map = ann_map[:, :, 2]  # vessel  annotaion map
    mat_map[mat_map == 0] = ves_map[mat_map == 0] * (mat_map.max() + 1)  # merge maps

    # Get binary masks and points

    inds = np.unique(mat_map)[1:]  # load all indices
    if inds.__len__() > 0:
        ind = inds[np.random.randint(inds.__len__())]  # pick single segment
    else:
        return read_single(data)

    # for ind in inds:
    mask = (mat_map == ind).astype(
        np.uint8
    )  # make binary mask corresponding to index ind
    coords = np.argwhere(mask > 0)  # get all coordinates in mask
    yx = np.array(
        coords[np.random.randint(len(coords))]
    )  # choose random point/coordinate
    return Img, mask, [[yx[1], yx[0]]]


def read_batch(data, batch_size=4):
    limage = []
    lmask = []
    linput_point = []
    for i in range(batch_size):
        image, mask, input_point = read_single(data)
        limage.append(image)
        lmask.append(mask)
        linput_point.append(input_point)

    return limage, np.array(lmask), np.array(linput_point), np.ones([batch_size, 1])


# Load model

sam2_checkpoint = "sam2_hiera_small.pt"  # path to model weight
model_cfg = "sam2_hiera_s.yaml"  #  model config
sam2_model = build_sam2(model_cfg, sam2_checkpoint, device="cuda")  # load model
predictor = SAM2ImagePredictor(sam2_model)

# Set training parameters

predictor.model.sam_mask_decoder.train(True)  # enable training of mask decoder
predictor.model.sam_prompt_encoder.train(True)  # enable training of prompt encoder
predictor.model.image_encoder.train(
    True
)  # enable training of image encoder: For this to work you need to scan the code for "no_grad" and remove them all
optimizer = torch.optim.AdamW(
    params=predictor.model.parameters(), lr=1e-5, weight_decay=4e-5
)
scaler = torch.cuda.amp.GradScaler()  # mixed precision

# Training loop

for itr in range(100000):
    with torch.cuda.amp.autocast():  # cast to mix precision
        image, mask, input_point, input_label = read_batch(
            data, batch_size=4
        )  # load data batch
        if mask.shape[0] == 0:
            continue  # ignore empty batches
        predictor.set_image_batch(image)  # apply SAM image encoder to the image
        # predictor.get_image_embedding()
        # prompt encoding

        mask_input, unnorm_coords, labels, unnorm_box = predictor._prep_prompts(
            input_point, input_label, box=None, mask_logits=None, normalize_coords=True
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
        prd_mask = torch.sigmoid(prd_masks[:, 0])  # Turn logit map to probability map
        seg_loss = (
            -gt_mask * torch.log(prd_mask + 0.00001)
            - (1 - gt_mask) * torch.log((1 - prd_mask) + 0.00001)
        ).mean()  # cross entropy loss

        # Score loss calculation (intersection over union) IOU

        inter = (gt_mask * (prd_mask > 0.5)).sum(1).sum(1)
        iou = inter / (gt_mask.sum(1).sum(1) + (prd_mask > 0.5).sum(1).sum(1) - inter)
        score_loss = torch.abs(prd_scores[:, 0] - iou).mean()
        loss = seg_loss + score_loss * 0.05  # mix losses

        # apply back propogation

        predictor.model.zero_grad()  # empty gradient
        scaler.scale(loss).backward()  # Backpropogate
        scaler.step(optimizer)
        scaler.update()  # Mix precision

        if itr % 1000 == 0:
            torch.save(predictor.model.state_dict(), "model.torch")  # save model

        # Display results

        if itr == 0:
            mean_iou = 0
        mean_iou = mean_iou * 0.99 + 0.01 * np.mean(iou.cpu().detach().numpy())
        print("step)", itr, "Accuracy(IOU)=", mean_iou)
