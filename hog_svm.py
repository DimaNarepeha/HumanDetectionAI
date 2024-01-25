# -*- coding: utf-8 -*-
"""Image & Speech Recognition - Human Detection
"""

import os
import sys

saving_dir = '.'

"""# Definitions

## imports
"""

from typing import Union
from pathlib import Path
import multiprocessing

import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import torch
import torch.utils.data

import torchvision.datasets.voc as voc

from torchvision.transforms import v2 as T
from torchvision.tv_tensors import BoundingBoxes
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks

from skimage.feature import hog
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

winSize = (32, 96)
blockSize = (32, 32)
blockStride = (16, 16)
cellSize = (16, 16)
nbins = 9

def hog_svm(path, scales=(0.1, 0.2, 0.3, 0.4, 0.5), thr=4.4, overlap_thr=0.1):
    img = cv2.imread(path)
    all_boxes = []
    all_weights = []
    for scale in scales:
        scaled = cv2.resize(img, (int(scale*img.shape[1]), int(scale*img.shape[0])))
        if scaled.shape[0] < winSize[0]*2 or scaled.shape[1] < winSize[1]*2:
            continue
        locs, weights = hog.detect(scaled, thr)
        if not len(locs):
            continue
        all_boxes.append(np.concatenate([locs/scale, (locs + winSize)/scale], axis=1))
        all_weights.append(weights)

    all_weights = np.concatenate(all_weights)
    all_boxes = np.concatenate(all_boxes)

    selected = non_max_suppression(
        all_boxes,
        all_weights,
        overlapThresh=overlap_thr
    )
    selected[:,0] = np.clip(selected[:,0], 0, img.shape[1])
    selected[:,2] = np.clip(selected[:,2], 0, img.shape[1])
    selected[:,1] = np.clip(selected[:,1], 0, img.shape[0])
    selected[:,3] = np.clip(selected[:,3], 0, img.shape[0])
    return img, selected

if __name__ == '__main__':

    svm = cv2.ml.SVM_load(os.path.join(saving_dir,'svm_model.yml'))
    hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)
    hog.setSVMDetector(svm.getSupportVectors()[0])


    im, boxes = hog_svm(sys.argv[1])
    plot([(
        im[:,:,[2,1,0]],
        BoundingBoxes(boxes, format='XYXY', canvas_size=im.shape[:2])
    )])
    
    out_f = str(Path(sys.argv[1]).with_suffix('.hog.png'))
    print(out_f)
    plt.savefig(out_f)
