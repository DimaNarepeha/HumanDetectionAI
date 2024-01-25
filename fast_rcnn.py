# -*- coding: utf-8 -*-
"""Image & Speech Recognition - Human Detection
# Setup
"""

import os
import sys

saving_dir = '.'
vgg_type = 'D'
rcnn_fn = os.path.join(saving_dir, f"rcnn_bs2_backbone_D.pt")

"""# Definitions

## imports
"""

from typing import Union, TypedDict
from pathlib import Path
import multiprocessing
import xml.etree.ElementTree
import json

import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import trange

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data

import torchvision.models.vgg as vgg
import torchvision.datasets.voc as voc

from torchvision.ops import box_iou, roi_pool
from torchvision.transforms import v2 as T
from torchvision.tv_tensors import BoundingBoxes
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks

from imutils.object_detection import non_max_suppression

DS = Union[voc.VOCDetection, torch.utils.data.Subset]
matplotlib.use('Agg')
cv2.setUseOptimized(True);
cv2.setNumThreads(multiprocessing.cpu_count());
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

"""## plotting"""

def plot(imgs, row_title=None, **imshow_kwargs):
    if not isinstance(imgs[0], list):
        # Make a 2d grid even if there's just 1 row
        imgs = [imgs]

    num_rows = len(imgs)
    num_cols = len(imgs[0])
    _, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False)
    for row_idx, row in enumerate(imgs):
        for col_idx, img in enumerate(row):
            boxes = None
            masks = None
            if isinstance(img, tuple):
                img, target = img
                if isinstance(target, dict):
                    boxes = target.get("boxes")
                    masks = target.get("masks")
                elif isinstance(target, BoundingBoxes):
                    boxes = target
                else:
                    raise ValueError(f"Unexpected target type: {type(target)}")
            img = T.functional.to_image(img)
            if img.dtype.is_floating_point and img.min() < 0:
                # Poor man's re-normalization for the colors to be OK-ish. This
                # is useful for images coming out of Normalize()
                img -= img.min()
                img /= img.max()

            img = T.functional.to_dtype(img, torch.uint8, scale=True)
            if boxes is not None:
                img = draw_bounding_boxes(img, boxes, colors="yellow", width=3)
            if masks is not None:
                img = draw_segmentation_masks(img, masks.to(torch.bool), colors=["green"] * masks.shape[0], alpha=.65)

            ax = axs[row_idx, col_idx]
            ax.imshow(img.permute(1, 2, 0).numpy(), **imshow_kwargs)
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    if row_title is not None:
        for row_idx in range(num_rows):
            axs[row_idx, 0].set(ylabel=row_title[row_idx])

    plt.tight_layout()

"""## modeling

### R-CNN
"""

def init_weights(module: nn.Module):
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.constant_(m.bias, 0)


def make_header(in_features: int, hidden_features: int, out_feautures: int, dropout: float) -> nn.Module:
    return nn.Sequential(
        nn.Linear(in_features, hidden_features),
        nn.ReLU(True),
        nn.Dropout(p=dropout),
        nn.Linear(hidden_features, hidden_features),
        nn.ReLU(True),
        nn.Dropout(p=dropout),
        nn.Linear(hidden_features, out_feautures)
    )


class Classifier(nn.Module):
    def __init__(self, in_channels: int = 512, pool_size: int = 7, dropout: float = 0.5, num_classes=1):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((pool_size, pool_size))
        self.header = make_header(in_channels * pool_size * pool_size, 4096, num_classes, dropout)
        init_weights(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.header(x)
        return x


class RCNN(nn.Module):
    def __init__(self,
                 encoder: nn.Module,
                 in_channels: int = 512, pool_size: int = 7, dropout: float = 0.5, num_classes=2):
        super().__init__()
        self.pool_size = pool_size
        self.encoder = encoder
        self.classifier = make_header(in_channels * pool_size * pool_size, 4096, num_classes, dropout)
        self.regressor = make_header(in_channels * pool_size * pool_size, 4096, num_classes*4, dropout)
        init_weights(self.classifier)
        init_weights(self.regressor)

    def forward(self, inputs: torch.Tensor, rois: list[torch.Tensor]):
        x = self.encoder(inputs)
        # print(x.shape, inputs.shape[2]/x.shape[2])
        x = roi_pool(x, rois, self.pool_size, spatial_scale=x.shape[2]/inputs.shape[2])
        x = torch.flatten(x, 1)
        cls = self.classifier(x)
        bbx = self.regressor(x)
        return cls, bbx


class MultiTaskLoss(nn.Module):
    def __init__(self, lambda_: float = 1):
        super().__init__()
        self._lambda = lambda_
        self.cls = nn.CrossEntropyLoss(reduction='none')
        self.loc = nn.SmoothL1Loss(reduction='none')

    def forward(self, input_logits, input_loc, target_cls, target_loc):
        # input_logits: [N, K] predicted classes logits
        # input_log   : [N, K*4] predicted per class location offsets
        # target_cls  : [N, K] target class lables
        # target_loc  : [N, 4] target class offsets
        not_background = (target_cls[:, 0] == 0).type(input_logits.dtype)
        input_cls_idx = target_cls.argmax(dim=1)
        input_cls_loc = torch.concat([
            input_loc[[i], cls_idx*4: (cls_idx+1)*4] for i, cls_idx in enumerate(input_cls_idx)
        ])
        cls = self.cls(input_logits, target_cls)
        loc = self.loc(input_cls_loc, target_loc).sum(1)
        return (cls + self._lambda * not_background * loc).mean()

"""## data

### voc preprocessing
"""

voc_lables = [
    'aeroplane',
    'bicycle',
    'bird',
    'boat',
    'bottle',
    'bus',
    'car',
    'cat',
    'chair',
    'cow',
    'diningtable',
    'dog',
    'horse',
    'motorbike',
    'person',
    'pottedplant',
    'sheep',
    'sofa',
    'train',
    'tvmonitor'
]

person_idx = voc_lables.index('person')

def annotation_to_target(ann: dict, lables: list[str] | None = None):
    w = int(ann['size']['width'])
    h = int(ann['size']['height'])
    b_arr = []
    l_arr = []
    d_arr = []
    for obj in ann['object']:
        if lables is not None and obj['name'] not in lables:
            continue
        b_arr.append([int(obj['bndbox'][k]) for k in ['xmin', 'ymin', 'xmax', 'ymax']])
        l_arr.append(voc_lables.index(obj['name']))
        d_arr.append(int(int(obj['truncated']) or int(obj['occluded']) or int(obj['difficult'])))
    bboxes = BoundingBoxes(b_arr, format='XYXY', canvas_size=(h, w))
    return bboxes, torch.tensor(l_arr), torch.tensor(d_arr)


class AnnotationToTarget(nn.Module):
    def forward(self, img, target):
        return img, annotation_to_target(target['annotation'])


class BoxesToArray(nn.Module):
    def forward(self, img, target: tuple[BoundingBoxes, torch.Tensor]):
        bboxes, labels = target
        boxes = bboxes.data
        boxes[:, [0, 2]] / bboxes.canvas_size[1]
        boxes[:, [1, 3]] / bboxes.canvas_size[0]
        return img, (boxes, labels)


def make_transforms(transforms: list) -> T.Compose:
    return T.Compose([
        # AnnotationToTarget(),
        *(transforms or []),
        T.PILToTensor(),
        T.ConvertImageDtype(torch.float),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

"""### RoI computing"""

def get_item(ds: DS, index: int, lables: list[str] | str = None, pil=False, load_im=True, difficulties=False):
    if not isinstance(ds, voc.VOCDetection):
        ds = ds.dataset
    path = ds.images[index]
    im = None
    if load_im:
        im = Image.open(path) if pil else cv2.imread(path)
    target = ds.parse_voc_xml(
        xml.etree.ElementTree.parse(ds.annotations[index]).getroot()
    )
    target = annotation_to_target(
        target['annotation'],
        lables=lables
    )
    if not difficulties:
        target = target[:2]
    return (
        path,
        im,
        *target
    )


def item_rois(im, quality=False):
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(im)
    if quality:
        ss.switchToSelectiveSearchQuality()
    else:
        ss.switchToSelectiveSearchFast()

    rois = ss.process() # x, y, w, h
    rois[:, 2] += rois[:, 0] # x += w
    rois[:, 3] += rois[:, 1] # y += h
    rois = torch.tensor(rois)
    return rois


class ROIsDict(TypedDict):
    filepath:        str
    shape:           tuple[int, int]
    human_bboxes:    list[tuple[int, int, int, int]]
    human_rois:      list[tuple[int, int, int, int, int]]
    background_rois: list[tuple[int, int, int, int]]


def generate_roi_samples(ds: DS, i: int) -> ROIsDict | None:
    path, im, bboxes, lables = get_item(ds, i, ['person'])
    if not len(lables): # no humans
        return
    rois = item_rois(im)
    ious = box_iou(bboxes.data, rois).T
    thresholded = (ious < 0.5).all(axis=1)
    background_rois = rois[thresholded]
    return {
        'filepath': path,
        'shape': im.shape[:2],
        'human_bboxes': bboxes.tolist(),
        'human_rois': torch.concat([
            rois[~thresholded],                               # roi bboxes
            ious[~thresholded].argmax(axis=1, keepdims=True)  # index of target human box
        ], axis=1).tolist(),
        'background_rois': background_rois.tolist(),
    }

def _run(inp):
    ds, b, e = inp
    result = []
    for i in trange(b, e, desc=f'{b}:{e}'):
        if (samples := generate_roi_samples(ds, i)):
            result.append(samples)
    return result

def run_multiproc(ds: DS):
    n = multiprocessing.cpu_count()
    h = len(ds) // n
    with multiprocessing.Pool(processes=n) as pool:
        result = []
        for r in pool.map(_run, [(ds, i*h, (i+1)*h) for i in range(n-1)] + [(ds, (n-1)*h, len(ds))]):
            result += r
    return result

def get_ds_rois(ds: DS, fn: str):
    if not os.path.exists(fn):
        rois = run_multiproc(ds)
        with open(fn, 'w') as f:
            json.dump(rois, f)
    else:
        with open(fn, 'r') as f:
            rois = json.load(f)
    return rois

"""### RoI dataset"""

ROISample = tuple[Image.Image, tuple[BoundingBoxes, BoundingBoxes, BoundingBoxes]]
# image, [neg, pos, target]

class ROIDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            rois: list[ROIsDict],
            total_rois: int = 64, # number of rois in resulted sample
            pos_proposals_ratio: float = 0.25, # precentage of total availabe positive rois
                                               # (with human) to use in sample
            balanced_rois: bool = False, # ignore pos_proposals_ratio and try to
                                         # include same amount of negative and positive rois
            transforms: T.Transform = None,
            ) -> None:
        super().__init__()
        # TODO: remove this filtering and increase number of positive rois
        # or ensure there is at least one for each target bbox
        self.rois = [x for x in rois if x['human_rois']]
        self.total_rois = total_rois
        self.pos_proposals_ratio = pos_proposals_ratio
        self.balanced_rois = balanced_rois
        self.transforms = transforms or T.Transform()

    def select_random(self, data: ROIsDict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        data['human_bboxes'] = np.asarray(data['human_bboxes'])
        data['human_rois'] = np.asarray(data['human_rois'])
        data['background_rois'] = np.asarray(data['background_rois'])
        data['shape'] = tuple(data['shape'])

        pos_n = len(data['human_rois'])
        neg_n = len(data['background_rois'])

        pos_select = (
            self.total_rois // 2
            if self.balanced_rois
            else int(pos_n * self.pos_proposals_ratio)
        )
        # TODO: maybe we should pass batches without positive rois?
        if not pos_select:
            pos_select = pos_n
        pos_rois = data['human_rois'][
            np.random.choice(
                list(range(pos_n)),
                min(pos_select, pos_n)
            ).tolist()
        ]
        pos_targets = data['human_bboxes'][pos_rois[:, 4]]
        pos_rois = pos_rois[:, :4]

        neg_rois = data['background_rois'][
            np.random.choice(
                list(range(neg_n)),
                max(self.total_rois - len(pos_rois), 0)
            ).tolist()
        ]
        return neg_rois, pos_rois, pos_targets

    def get_base(self, index) -> ROISample:
        data = self.rois[index]
        im = Image.open(data['filepath']).convert("RGB")
        target = [
            BoundingBoxes(data=bboxes, canvas_size=data['shape'], format='XYXY')
            for bboxes in self.select_random(data)
        ]
        return im, target

    def __len__(self):
        return len(self.rois)


class ROIClassifierDataset(ROIDataset):
    def __getitem__(self, index) -> tuple[list[torch.Tensor], torch.Tensor]:
        im, (neg, pos, _) = self.get_base(index)
        target = torch.zeros(len(neg)+len(pos))
        target[-len(pos):] = 1
        return (
            [
                self.transforms(im.crop((x1,y1,x2,y2)))
                for (x1,y1,x2,y2) in (
                    neg.data.type(torch.int32).tolist() +
                    pos.data.type(torch.int32).tolist()
                )
            ],
            target
        )

    def collate(self, batch: list[tuple[list[torch.Tensor], torch.Tensor]]):
        images = []
        lables = []
        for imgs, targets in batch:
            images += imgs
            lables += [targets]
        return torch.stack(images), torch.concat(lables)


class ROIDetectionDataset(ROIDataset):
    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        im, (neg, pos, target) = self.transforms(*self.get_base(index))
        lables = torch.zeros(len(neg)+len(pos), 2)
        lables[:len(neg), 0] = 1
        lables[-len(pos):, 1] = 1
        bboxes = torch.zeros(len(neg)+len(pos), 4)
        bboxes[-len(pos):] = target.data - pos.data
        rois = torch.concat([neg.data, pos.data])
        return (
            im, rois, bboxes, lables
        )

    def collate(self, batch: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]):
        sizes = []
        images = []
        rois = []
        bboxes = []
        lables = []
        for img, roi, bbx, lbl in batch:
            images += [img]
            sizes += [list(img.shape[1:])]
            rois += [roi]
            bboxes += [bbx]
            lables += [lbl]
        sizes = np.asarray(sizes)
        h = sizes[:, 0].max()
        w = sizes[:, 1].max()
        return (
            torch.stack([F.pad(img, (0, w-img.shape[2], 0, h-img.shape[1])) for img in images]),
            rois,
            torch.concat(bboxes),
            torch.concat(lables)
        )

def fast_rcnn(path,thr = 0.9,overlap_thr = 0.3,):
    im = Image.open(path).convert('RGB')
    rois = item_rois(np.asarray(im), quality=True).type(torch.float32).to(device)
    inputs = valid_det_transforms(im)
    with torch.no_grad():
        output_logits, output_loc = rcnn(torch.unsqueeze(inputs.to(device), 0), [rois])
        confs = F.softmax(output_logits.cpu(), 1)[:, 1]
        boxes = torch.round(rois + output_loc[:, 4:]).type(torch.int32).cpu()
        valid = (boxes[:, 0] <= boxes[:, 2]) & (boxes[:, 1] <= boxes[:, 3])
        confs = confs[valid]
        boxes = boxes[valid]
        order = (1-confs).argsort()
        confs = confs[order]
        boxes = boxes[order]

    selected = non_max_suppression(
        boxes[confs > thr].numpy(),
        confs[confs > thr].numpy(),
        overlapThresh=overlap_thr
    )
    selected[:,0] = np.clip(selected[:,0], 0, im.width)
    selected[:,2] = np.clip(selected[:,2], 0, im.width)
    selected[:,1] = np.clip(selected[:,1], 0, im.height)
    selected[:,3] = np.clip(selected[:,3], 0, im.height)
    return im, selected


if __name__ == '__main__':
    valid_det_transforms = make_transforms([])

    encoder = nn.Sequential(*vgg.make_layers(vgg.cfgs[vgg_type], batch_norm=True))
    rcnn = RCNN(encoder).to(device)

    rcnn_state = torch.load(rcnn_fn, map_location=device)
    rcnn.load_state_dict(rcnn_state['model_state_dict'])

    im, selected = fast_rcnn(sys.argv[1])
    plot([(
        im,
        BoundingBoxes(selected, format='XYXY', canvas_size=(im.width, im.height))
    )])
    out_f = str(Path(sys.argv[1]).with_suffix('.rcnn.png'))
    print(out_f)
    plt.savefig(out_f)
