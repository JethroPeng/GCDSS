import cv2
import os
import json
import numpy as np
import torch
from typing import Tuple
from tqdm import tqdm
from torch.utils.data.dataloader import DataLoader 

def find_connected_regions(image):
    ret, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity=8)
    return labels, num_labels, centroids

def expand_box(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    expand_ratio: float = 1.0,
    max_h: int = None,
    max_w: int = None,
):
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    w = x2 - x1
    h = y2 - y1
    w = w * expand_ratio
    h = h * expand_ratio
    box = [cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h]
    if max_h is not None:
        box[1] = max(0, box[1])
        box[3] = min(max_h - 1, box[3])
    if max_w is not None:
        box[0] = max(0, box[0])
        box[2] = min(max_w - 1, box[2])
    return [int(b) for b in box]

def crop_with_mask(
    image: torch.Tensor,
    mask: torch.Tensor,
    bbox: torch.Tensor,
    fill: Tuple[float, float, float] = (0, 0, 0),
    expand_ratio: float = 1.0,
):
    #print(bbox)
    l, t, r, b = expand_box(*bbox, expand_ratio)
    _, h, w = image.shape
    l = max(l, 0)
    t = max(t, 0)
    r = min(r, w)
    b = min(b, h)
    new_image = torch.cat(
        [image.new_full((1, b - t, r - l), fill_testue=test) for test in fill]
    )
    return image[:, t:b, l:r] * mask[None, t:b, l:r] + ( ~mask[None, t:b, l:r]) * new_image, mask[None, t:b, l:r]

def read_feature_file(data_loader: DataLoader,
                      feature_path: str,
                      is_label: bool):
    len_f = 0
    feature_data       = []
    feature_label_full = []
    feature_label_max  = []
    feature_label_area = []
    
    with open(feature_path, 'r') as f:
        len_f = sum(1 for line in f)
        
    with open(feature_path, 'r') as f:
        for line in tqdm(f, total=len_f, desc='collecting data and label'):
            line = line.strip('\n')
            image_path, mask_num, feature = line.split(' ')
            label_path = os.path.join(image_path.replace(data_loader.dataset.image_dir_name, data_loader.dataset.mask_label_dir_name).replace(data_loader.dataset.image_suffix, ''), mask_num + '.npy')
            feature = np.array(json.loads(feature))
            label_dic = np.load(label_path, allow_pickle=True).item()
            if label_dic:
                max_key = max(label_dic, key=label_dic.get) # as ground truth label
                feature_data.append(feature)
                feature_label_full.append(label_dic)
                feature_label_max.append(data_loader.dataset.label_map[max_key])
                if is_label:
                    feature_label_area.append((label_dic[max_key])*1.0)
                else:
                    feature_label_area.append(sum(label_dic.values())*1.0) #max_key is not available
    feature_data = np.vstack(feature_data)
    feature_label_full = np.vstack(feature_label_full).squeeze(1)
    feature_label_max = np.vstack(feature_label_max).squeeze(1)
    feature_label_area = np.vstack(feature_label_area).squeeze(1)
    return feature_data, feature_label_full, feature_label_max, feature_label_area