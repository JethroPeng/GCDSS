import os
import random
import argparse
import numpy as np
import cv2
import json
import torch
import torch.cuda
from pyparsing import FollowedBy
from torch.nn import functional as F
from torch.utils.data.dataloader import DataLoader 
from torchvision import transforms as pth_transforms
from PIL import Image
from tqdm import tqdm
from utils import *
from model.model import * 
from util.mask import *
import warnings
warnings.filterwarnings('ignore')

def mask_generate(label_loader: DataLoader,
                  unlabel_loader: DataLoader,
                  device: torch.device,
                  model = 'sam',
                  theta = 200):
    model = load_mask_model(model,device)
    for cur_split in ['label','unlabel']:
        if cur_split == 'label':
            data_loader = label_loader
        else:
            data_loader = unlabel_loader
        for img, lbl, img_name, lbl_name in tqdm(data_loader, desc='generating mask for ' + cur_split):
            # img: [C, H, W], uint8
            # lbl: [1, H, W], (0, 1, ...20)
            # img_name: (str,)
            # lbl_name: (str,)
            img = np.array(img).squeeze(0)
            img_name = img_name[0]
            masks = model.generate(img)
            masks = sorted(masks, key=lambda x: np.sum(x['segmentation']), reverse=True)
            label_counter = 1
            label_map = np.zeros_like(masks[0]['segmentation'], dtype=np.uint16)
            os.makedirs(data_loader.dataset.mask_dir, exist_ok=True)
            mask_path = os.path.join(data_loader.dataset.mask_dir, img_name + '.npy')
            for i in range(len(masks)):
                cur_mask = masks[i]['segmentation']
                nonzero_pixels = np.where(cur_mask > 0)
                if 0 not in np.unique(label_map[nonzero_pixels]):
                    continue
                else:
                    label_map[nonzero_pixels] = label_counter
                    label_counter = label_counter + 1
            rest_labels, num_labels, centroids = find_connected_regions(255 * (label_map == 0).astype(np.uint8))
            for label in range(1, num_labels):
                if np.sum(rest_labels == label) > theta:
                    nonzero_pixels = np.where(rest_labels == label)
                    label_map[nonzero_pixels] = label_counter
                    label_counter = label_counter + 1
                else:
                    i = int(centroids[label][1])
                    j = int(centroids[label][0])
                    cut = label_map[max(0, i - 10):min(i + 10, label_map.shape[0]),max(0, j - 10):min(j + 10, label_map.shape[1])]
                    unique_values, counts = np.unique(cut, return_counts=True)
                    sorted_indices = np.argsort(-counts)
                    sorted_unique_values = unique_values[sorted_indices]
                    if sorted_unique_values[0] != 0:
                        fill = sorted_unique_values[0]
                    else:
                        fill = sorted_unique_values[1]
                    nonzero_pixels = np.where(rest_labels == label)
                    label_map[nonzero_pixels] = fill
            np.save(mask_path, label_map)
            

def mask_label_generate(label_loader: DataLoader,
                        unlabel_loader: DataLoader,
                        device: torch.device):
    for cur_split in ['label','unlabel']:
        if cur_split == 'label':
            data_loader = label_loader
        else:
            data_loader = unlabel_loader
        for img, lbl, img_name, lbl_name in tqdm(data_loader, desc='generating mask label for ' + cur_split):
            lbl = np.array(lbl).squeeze(0)
            img_name = img_name[0]
            lbl_name = lbl_name[0]
            mask_lables_path = os.path.join(data_loader.dataset.mask_label_dir, img_name)
            os.makedirs(mask_lables_path, exist_ok=True)
            masks_path = os.path.join(data_loader.dataset.mask_dir, img_name + '.npy')
            masks = np.load(masks_path)
            mask_num = np.max(masks)
            for mask_num in range(1, mask_num + 1):
                mask_label_path = os.path.join(mask_lables_path, str(mask_num))
                mask_cur = (masks==mask_num)
                mask_dic = {}
                unique_labels, counts = np.unique(lbl[mask_cur], axis=0, return_counts=True)
                for i in range(unique_labels.shape[0]):
                    if unique_labels[i] != data_loader.dataset.ignore_index:
                        mask_dic[unique_labels[i]] = counts[i]
                np.save(mask_label_path, mask_dic)

def mask_to_feature(label_loader: DataLoader,
                    unlabel_loader: DataLoader,
                    device: torch.device,
                    model_name: str,
                    label_feature_path: str,
                    unlabel_feature_path: str,
                    bbox_scale: float):
    model = load_feature_model(model_name,device)
    for cur_split in ['label', 'unlabel']:
        if cur_split == 'label':
            data_loader = label_loader
            feature_path = label_feature_path
        else:
            data_loader = unlabel_loader
            feature_path = unlabel_feature_path
        with open(feature_path, 'w') as f:
            for img, lbl, img_name, lbl_name in tqdm(data_loader, desc='generating feature for ' + cur_split):       
                img = np.array(img).squeeze(0)
                img_name = img_name[0]
                image_path = os.path.join(data_loader.dataset.image_dir, img_name + data_loader.dataset.image_suffix)
                if model_name == 'sam':
                    features = model.set_image(img)
                    features = torch.nn.functional.interpolate(features, size=img.shape[:2], mode='bilinear', align_corners=False)
                    features = features[0]
                    features = features.permute(1, 2, 0)
                    masks_path = os.path.join(data_loader.dataset.mask_dir, img_name + '.npy')
                    masks = np.load(masks_path)
                    mask_num = np.max(masks)
                    for mask_num in range(1, mask_num + 1):
                        mask_cur = (masks==mask_num)
                        if np.sum(mask_cur)== 0:
                            continue
                        feature = torch.mean(features[mask_cur], axis=0).cpu().numpy()
                        f.write(str(image_path))
                        f.write(' ')
                        f.write(str(mask_num))
                        f.write(' [')
                        for i in range(feature.shape[0]-1):
                            f.write(str(float(feature[i])))
                            f.write(',')
                        f.write(str(float(feature[feature.shape[0]-1])))
                        f.write(']\n')              
                elif model_name == 'dinov2' or model_name == 'dinov1':
                    image_size = 224
                    transform = pth_transforms.Compose([
                                pth_transforms.Resize([image_size, image_size]),
                                pth_transforms.ToTensor(),
                                pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
                    masks_path = os.path.join(data_loader.dataset.mask_dir, img_name + '.npy')
                    masks = np.load(masks_path)
                    mask_num = np.max(masks)
                    for mask_num in range(1, mask_num + 1):
                        mask_cur = (masks==mask_num)
                        if np.sum(mask_cur)== 0:
                            continue
                        non_zero_indices = np.argwhere(mask_cur)
                        min_row, min_col = non_zero_indices.min(axis=0)
                        max_row, max_col = non_zero_indices.max(axis=0)
                        width = max_col - min_col + 1
                        height = max_row - min_row + 1
                        x1 = min_row - height * bbox_scale
                        x2 = max_row + height * bbox_scale+1
                        y1 = min_col - width * bbox_scale
                        y2 = max_col + width * bbox_scale+1
                        mask_img = img[int(max(x1,0)):int(min(x2,img.shape[0])),int(max(y1,0)):int(min(y2,img.shape[1]))]
                        mask_ = np.expand_dims(mask_cur[int(max(x1,0)):int(min(x2,img.shape[0])),int(max(y1,0)):int(min(y2,img.shape[1]))], axis=2)
                        mask_img = mask_img * mask_ + np.tile(~mask_ *np.mean(mask_img),(1, 1, 3))
                        image = transform(Image.fromarray(np.uint8(mask_img))).unsqueeze(0).to(device)
                        feature = model(image) 
                        feature = torch.nn.functional.normalize(feature, dim=-1).cpu().numpy().squeeze()
                        f.write(str(image_path))
                        f.write(' ')
                        f.write(str(mask_num))
                        f.write(' [')
                        for i in range(feature.shape[0]-1):
                            f.write(str(float(feature[i])))
                            f.write(',')
                        f.write(str(float(feature[feature.shape[0]-1])))
                        f.write(']\n')  
                elif model_name == 'ovseg':
                    PIXEL_MEAN = (0.48145466, 0.4578275, 0.40821073)
                    PIXEL_STD = (0.26862954, 0.26130258, 0.27577711)
                    mask_fill = [255.0*c for c in PIXEL_MEAN]
                    pixel_mean = torch.tensor(PIXEL_MEAN).reshape(1, -1, 1, 1)
                    pixel_std = torch.tensor(PIXEL_STD).reshape(1, -1, 1, 1)
                    img = torch.as_tensor(img.astype('float32').transpose(2,0,1))
                    masks_path = os.path.join(data_loader.dataset.mask_dir, img_name+'.npy')
                    masks = np.load(masks_path)
                    mask_num = np.max(masks)
                    for mask_num in range(1, mask_num + 1):
                        mask_cur = (masks==mask_num)
                        if np.sum(mask_cur)== 0:
                            continue
                        mask_cur = [mask_cur[None,:,:]]
                        mask_cur = np.row_stack(mask_cur)
                        mask_cur = BitMasks(mask_cur)
                        bbox = mask_cur.get_bounding_boxes()
                        region, _ = crop_with_mask(img, mask_cur[0][-1], bbox, fill=mask_fill)
                        region = region.unsqueeze(0)
                        reigin = F.interpolate(region.to(torch.float),size=(224,224),mode="bicubic")
                        imgs = (reigin/255.0 - pixel_mean) / pixel_std
                        with torch.no_grad():
                            feature = model.encode_image(imgs.cuda().half())
                            feature /= feature.norm(dim=-1, keepdim=True)
                            feature = feature.cpu().numpy().squeeze()
                        f.write(str(image_path))
                        f.write(' ')
                        f.write(str(mask_num))
                        f.write(' [')
                        for i in range(feature.shape[0]-1):
                            f.write(str(float(feature[i])))
                            f.write(',')
                        f.write(str(float(feature[feature.shape[0]-1])))
                        f.write(']\n')  
                elif model_name == 'clip':
                    from model.CLIP import clip
                    _, preprocess = clip.load('ViT-B/32', device=device)
                    masks_path = os.path.join(data_loader.dataset.mask_dir, img_name+'.npy')
                    masks = np.load(masks_path)
                    mask_num = np.max(masks)
                    for mask_num in range(1, mask_num + 1):
                        mask_cur = (masks==mask_num)
                        if np.sum(mask_cur)== 0:
                            continue
                        non_zero_indices = np.argwhere(mask_cur)
                        min_row, min_col = non_zero_indices.min(axis=0)
                        max_row, max_col = non_zero_indices.max(axis=0)
                        width = max_col - min_col + 1
                        height = max_row - min_row + 1
                        x1 = min_row - height * bbox_scale
                        x2 = max_row + height * bbox_scale+1
                        y1 = min_col - width * bbox_scale
                        y2 = max_col + width * bbox_scale+1
                        mask_img = img[int(max(x1,0)):int(min(x2,img.shape[0])),
                                       int(max(y1,0)):int(min(y2,img.shape[1]))]
                        image = preprocess(Image.fromarray(mask_img)).unsqueeze(0).to(device)
                        with torch.no_grad():
                            image_features = model.encode_image(image)
                        feature = image_features.cpu().numpy().squeeze()
                        f.write(str(image_path))
                        f.write(' ')
                        f.write(str(mask_num))
                        f.write(' [')
                        for i in range(feature.shape[0]-1):
                            f.write(str(float(feature[i])))
                            f.write(',')
                        f.write(str(float(feature[feature.shape[0]-1])))
                        f.write(']\n')  

def get_dis_array(label_loader: DataLoader,
                  unlabel_loader: DataLoader,
                  device: torch.device,
                  K: int,
                  save_path: str,
                  dis_name: str,
                  ground_truth_name: str,
                  label_data, 
                  label_label_full, 
                  label_label_max, 
                  label_label_area,
                  unlabel_data, 
                  unlabel_label_full, 
                  unlabel_label_max, 
                  unlabel_label_area):
    list_data = []
    combined_data = np.concatenate((label_data, unlabel_data))
    unlabel_data = torch.from_numpy(unlabel_data).float().cuda(device=device)
    combined_data =  torch.from_numpy(combined_data).float().cuda(device=device)
    for i in tqdm(range(len(unlabel_data)), desc='getting dis array'):
        dis_array = (- unlabel_data[i].reshape(1,-1) @ combined_data.T).squeeze().cpu().numpy().squeeze()
        indices = np.argpartition(dis_array, K)[:K]
        sorted_indices = indices[np.argsort(dis_array[indices])]
        list_data.append(sorted_indices)
    list_data_np = np.vstack(list_data)
    np.save(os.path.join(save_path, dis_name), list_data_np, allow_pickle=True)
    all_label = np.concatenate((label_label_max, unlabel_label_max))
    all_label = np.vstack(all_label)
    np.save(os.path.join(save_path, ground_truth_name), all_label, allow_pickle=True)

def filter_sample(label_loader: DataLoader, 
                  unlabel_loader: DataLoader,
                  device: torch.device,
                  save_path: str,
                  dis_name: str,
                  ground_truth_name: str,
                  pred_name: str,
                  k: int,
                  pred_new: int,
                  threshold: float,
                  max_iterations: int,
                  label_data, 
                  label_label_full, 
                  label_label_max, 
                  label_label_area,
                  unlabel_data, 
                  unlabel_label_full, 
                  unlabel_label_max, 
                  unlabel_label_area):
                  
    dis_matrix = np.load(os.path.join(save_path, dis_name))
    label_all = np.load(os.path.join(save_path, ground_truth_name)).squeeze()
    
    null_char=255
    
    #init
    split_len = dis_matrix.shape[0]
    plabel = np.full(label_all.shape[0], null_char, dtype=int)
    plabel[:-split_len] = label_all[:-split_len]
    weight = np.full(label_all.shape[0], 0, dtype=float)
    weight[:-split_len] = 1.0
    
    array = np.zeros(k)
    array[0] = 1.0
    for i in range(1, k):
        array[i] = array[i - 1] * 0.9
    sum_of_elements = np.sum(array)
    array = array * k / np.sum(array)
    
    for iteration in tqdm(range(max_iterations), desc='filter sample'):
        all_num = 0
        new_plabel = np.copy(plabel)
        new_weight = np.copy(weight)
        for i in range(split_len):
            nearest_nodes = dis_matrix[i][1:1+k]
            if sum(weight[nearest_nodes]) != 0:
                label_counts = np.bincount(plabel[nearest_nodes], weights=weight[nearest_nodes]*array[:k], minlength=unlabel_loader.dataset.class_base_num+1)
                new_label = np.argmax(label_counts)
                if label_counts[new_label] >= threshold*k:
                    new_plabel[-split_len+i] = new_label
                    new_weight[-split_len+i] = label_counts[new_label] *1.0 / (nearest_nodes.shape[0]) 
                    all_num = all_num + 1
        if np.array_equal(new_plabel, plabel):
            break
        plabel = new_plabel
        weight = new_weight
    old_label  = np.copy(plabel)
            
    rest_list = []
    rest_table = []
    for i in tqdm(range(split_len)):
        if (plabel[-split_len+i] == null_char):
            rest_list.append(label_all[-split_len+i])
            rest_table.append(i) 
            plabel[-split_len+i]  = null_char
            weight[-split_len+i] = 1
        else:
            weight[-split_len+i] = 0
    rest_list = np.vstack(rest_list).squeeze()
    rest_counts = np.bincount(rest_list, minlength=unlabel_loader.dataset.class_base_num+1)
    #refine
    new_plabel = np.copy(plabel)
    new_weight = np.copy(weight)
    for i in range(split_len):
        nearest_nodes = dis_matrix[i][1:1+k]
        if sum(weight[nearest_nodes]) != 0 and (i not in rest_table):
            label_counts = np.bincount(plabel[nearest_nodes], weights=weight[nearest_nodes]*array[:k], minlength=unlabel_loader.dataset.class_base_num+1)
            new_label = np.argmax(label_counts)
            new_plabel[-split_len+i]   = new_label
            new_weight[-split_len + i] = label_counts[new_label] * 1.0 / (nearest_nodes.shape[0])
    plabel = new_plabel
    weight = new_weight
    rest_list = []
    for i in tqdm(range(split_len)):
        if (plabel[-split_len+i] == null_char) and (weight[-split_len+i]>threshold):
            rest_list.append(label_all[-split_len+i])  
    rest_counts = np.bincount(rest_list, minlength=unlabel_loader.dataset.class_base_num+1)
    from sklearn.cluster import KMeans
    data = unlabel_data[rest_list]
    weights = unlabel_label_area[rest_list]    
    initial_centers = []
    initial_centers.append(np.random.choice(len(data), 1, p=weights/weights.sum())[0])
    for _ in range(1, pred_new ):
        distances = [min([np.linalg.norm(x-c)**2 for c in initial_centers]) for x in data]
        probabilities = distances / sum(distances)
        cumulative_probabilities = probabilities.cumsum()
        r = np.random.rand()
        i = 0
        for j, p in enumerate(cumulative_probabilities):
            if r < p:
                i = j
                break
        initial_centers.append(i)
    kmeans = KMeans(n_clusters=pred_new, init=np.array([data[i] for i in initial_centers]))
    kmeans.fit(data,sample_weight=weights)
    preds = kmeans.labels_
    j =0
    for i in tqdm(range(split_len)):
        if  plabel[-split_len+i] == null_char and (weight[-split_len+i]>threshold):
            old_label[-split_len+i] = preds[j] + unlabel_loader.dataset.class_base_num
            j = j +1
    print('saving pred npy')
    np.save(os.path.join(save_path, pred_name), old_label[-split_len:])
    print('saved pred npy')

def feature_clustering(label_loader: DataLoader,
                       unlabel_loader: DataLoader,
                       device: torch.device,
                       save_path: str,
                       feature_path: str,
                       pred_name: str,
                       pred_new: int,
                       label_data, 
                       label_label_full, 
                       label_label_max, 
                       label_label_area,
                       unlabel_data, 
                       unlabel_label_full, 
                       unlabel_label_max, 
                       unlabel_label_area):
    
    from util import gcd_clustering_alg as sskmeans
    
    
    label_data =  torch.from_numpy(label_data).to(device)
    unlabel_data  =  torch.from_numpy(unlabel_data).to(device)
    label_label_max  = torch.from_numpy(label_label_max).to(device)
    label_label_area = torch.from_numpy(label_label_area).to(device)
    unlabel_label_area = torch.from_numpy(unlabel_label_area).to(device)

    print('clustering')
    km = sskmeans.K_Means(k=unlabel_loader.dataset.class_base_num + pred_new,
                          tolerance=0, 
                          max_iterations=100, 
                          init='k-means++', 
                          n_init=5, 
                          random_state=None, 
                          n_jobs=None, 
                          pairwise_batch_size=1024, 
                          mode=None)
    km.fit(u_feats=unlabel_data, 
           u_weight=unlabel_label_area, 
           l_feats=label_data, 
           l_weight=label_label_area, 
           l_targets=label_label_max,
           momentum=0)
    preds = km.labels_.cpu()
    print('clustered')
    np.save(os.path.join(save_path, pred_name), preds[-unlabel_data.shape[0]:].cpu().numpy())
    print('saved pred npy')


def compute_miou(label_loader: DataLoader,
                 unlabel_loader: DataLoader,
                 device: torch.device,
                 save_path: str,
                 feature_path: str,
                 pred_name: str,
                 pred_new: int,
                 label_data, 
                 label_label_full, 
                 label_label_max, 
                 label_label_area,
                 unlabel_data, 
                 unlabel_label_full, 
                 unlabel_label_max, 
                 unlabel_label_area):
                 
    from util.cluster_utils import match_cluster_miou

    print('loading preds')
    pred = np.load(os.path.join(save_path, pred_name))

    label_map = unlabel_loader.dataset.label_map    
    class_num = unlabel_loader.dataset.class_num
    novel_class_num = len(unlabel_loader.dataset.novel_class_map_indices)
    base_class_num = unlabel_loader.dataset.class_base_num
    
    sum_unlabel_label_area = np.zeros([class_num])
    for i in range(class_num):
        sum_unlabel_label_area[i] = sum(unlabel_label_area[unlabel_label_max==i])
        unlabel_label_area[unlabel_label_max==i] = unlabel_label_area[unlabel_label_max==i] / sum_unlabel_label_area[i]
    #import pdb
    #pdb.set_trace()
    ind = match_cluster_miou(y_true=unlabel_label_max,
                             y_pred=pred,
                             y_weight=unlabel_label_area,
                             base_c=base_class_num,
                             class_num=class_num,
                             over_c=pred_new-novel_class_num,
                             fix=True)
                             
    for i in range(1,len(ind)):   
        if ind[i]== 0:
            ind[i] = 19      
              
    image_path_curr = ''
    data_count = np.zeros([unlabel_loader.dataset.class_num+1, 4]) # dim 0---ground truth area,dim 1---pred area,dim 2---intersection area,dim 3---union area
    num_count  = np.zeros([unlabel_loader.dataset.class_num+1, 2]) # dim 0---area,dim 1---pic num
    
    len_f = 0
    with open(feature_path, 'r') as f:
        len_f = sum(1 for line in f)
        
    i = 0
    with open(feature_path, 'r') as f:
        for line in tqdm(f, total=len_f, desc='calculating miou'):
            line = line.strip('\n')
            image_path, mask_num, feature = line.split(' ')
            label_path = os.path.join(image_path.replace('leftImg8bit_gcd', 'maskLabels').replace('_leftImg8bit.png', ''), mask_num + '.npy')
            label_dic = np.load(label_path, allow_pickle=True).item()
            if label_dic:
                if image_path_curr != image_path:
                    for j in range(unlabel_loader.dataset.class_num):
                        if data_count[j, 0] > 0:
                            num_count[j, 0] = num_count[j, 0] + 1
                            num_count[j, 1] = num_count[j, 1] + data_count[j, 2]*100.0/data_count[j,3]
                    data_count = np.zeros([unlabel_loader.dataset.class_num+1, 4])
                    image_path_curr = image_path
                max_key = max(label_dic, key=label_dic.get)
                for k, v in label_dic.items():
                    if label_map[k] != ind[pred[i]]:
                        data_count[label_map[k], 0] = data_count[label_map[k], 0] + v
                        data_count[label_map[k], 3] = data_count[label_map[k], 3] + v
                        data_count[ind[pred[i]], 3] = data_count[ind[pred[i]], 3] + v
                    if label_map[k] == ind[pred[i]]:
                        data_count[label_map[k], 0] = data_count[label_map[k], 0] + v
                        data_count[label_map[k], 1] = data_count[label_map[k], 1] + v
                        data_count[label_map[k], 2] = data_count[label_map[k], 2] + v
                        data_count[label_map[k], 3] = data_count[label_map[k], 3] + v
                i = i + 1

    novel_class_name = []
    sum_iou = 0.0
    base_iou = 0.0 
    novel_iou = 0.0
    print('base class: ')
    for j in range(unlabel_loader.dataset.class_base_num):
        print(str(j) + ': ' + unlabel_loader.dataset.class_name_map[j] + ': ', end='')
        print(round(num_count[j,1]*1.0/num_count[j,0], 2))
        sum_iou += num_count[j,1]*1.0/num_count[j,0]
        base_iou += num_count[j,1]*1.0/num_count[j,0]
    for j in unlabel_loader.dataset.novel_class_map_indices:
        novel_class_name.append(unlabel_loader.dataset.class_name_map[j])
    print('new class: ')
    for j in range(unlabel_loader.dataset.class_num - len(unlabel_loader.dataset.novel_class_map_indices), \
                   unlabel_loader.dataset.class_num):
        print(str(j) + ': ' + unlabel_loader.dataset.class_name_map[j] + ': ', end='')
        print(round(num_count[j,1]*1.0/num_count[j,0], 2))
        sum_iou += num_count[j,1]*1.0/num_count[j,0]
        novel_iou += num_count[j,1]*1.0/num_count[j,0]
    print('miou: ', round(sum_iou/unlabel_loader.dataset.class_num, 2))
    print('base miou: ', round(base_iou/(unlabel_loader.dataset.class_num - len(unlabel_loader.dataset.novel_class_map_indices)), 2))
    print('novel miou: ', round(novel_iou/len(unlabel_loader.dataset.novel_class_map_indices), 2))

