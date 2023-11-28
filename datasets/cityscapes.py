import os
import numpy as np
from torch.utils.data.dataset import Dataset
from PIL import Image
from torchvision.transforms import functional as F

class Cityscapes_GCD(Dataset):
    def __init__(self,
                 root,
                 split_path,
                 novel_class_list=[3, 15, 16, 17],
                 scale_ratio=0.5,
                 image_set='label',
                 debug=True
                 ):
        
        self.data_root = root
        self.image_set = image_set
        self.root = os.path.join(self.data_root, 'Cityscapes')
        self.ignore_index = 255
        self.image_dir_name = 'leftImg8bit_gcd'
        self.image_dir = os.path.join(self.root, self.image_dir_name)
        self.label_dir_name = 'gtFine_gcd'
        self.label_dir = os.path.join(self.root, self.label_dir_name)
        
        novel_class_str = ''
        for num in novel_class_list:
            novel_class_str += '_' + str(num)

        if split_path != None:
            self.split_path = split_path
        else:
            self.split_path = os.path.join(self.root, 'split', self.image_set + '_gcd_novel' + novel_class_str + '.txt')
        
        if not os.path.exists(self.split_path):
            raise ValueError("Wrong split_path")
        with open(os.path.join(self.split_path), 'r') as f:
            file_name_list = [x.strip() for x in f.readlines()]
        
        self.mask_dir_name = 'maskImages'
        self.mask_dir = os.path.join(self.root, self.mask_dir_name)
        self.mask_label_dir_name = 'maskLabels'
        self.mask_label_dir = os.path.join(self.root, self.mask_label_dir_name)
            
        self.file_name_list = file_name_list
        self.image_suffix = '_leftImg8bit.png'
        self.label_suffix = '_gtFine_labelTrainIds.png'
        self.images = [os.path.join(self.image_dir, x + self.image_suffix) for x in file_name_list]
        self.labels = [os.path.join(self.label_dir, x + self.label_suffix) for x in file_name_list]
        
        if debug:
            self.images = self.images[:2]
            self.labels = self.labels[:2]
            
        self.origin_class_name_map = {0: 'road', 1: 'sidewalk', 2: 'building', 3: 'wall', 4: 'fence',
                           5: 'pole', 6: 'traffic light', 7: 'traffic sign', 8: 'vegetation', 
                           9: 'terrain', 10: 'sky', 11: 'person', 12: 'rider', 13: 'car', 
                           14: 'truck', 15: 'bus', 16: 'train', 17: 'motorcycle', 18: 'bicycle'}
        self.class_num = 19

        self.novel_class_list = np.array(novel_class_list, dtype=np.int64)
        self.class_name_map = {}
        self.label_map = {}
        self.novel_class_map_indices = []
        base_class_name = []
        novel_class_name = []
        base_label = []
        novel_label = []
        
        for i in range(self.class_num):
            if i in self.novel_class_list:
                novel_class_name.append(self.origin_class_name_map[i])
                novel_label.append(i)
            else:
                base_class_name.append(self.origin_class_name_map[i])
                base_label.append(i)
        for i in range(len(base_class_name)):
            self.class_name_map[i] = base_class_name[i]
            self.label_map[base_label[i]] = i
        for i in range(len(novel_class_name)):
            self.class_name_map[len(base_class_name) + i] = novel_class_name[i]
            self.label_map[novel_label[i]] = len(base_class_name) + i
            self.novel_class_map_indices.append(len(base_class_name) + i)
            
        self.novel_class_map_indices = np.array(self.novel_class_map_indices, dtype=np.int64)
        self.scale_ratio = scale_ratio
        
        assert (len(self.images) == len(self.labels))

    def __getitem__(self, index):
        '''
        Args:
            index (int): Index
        Returns:
            tuple: (img, lbl, img_name, lbl_name).
        '''
        img_name = self.file_name_list[index]
        lbl_name = img_name
        # HWC
        img = Image.open(self.images[index]).convert('RGB')
        # HW
        lbl = Image.open(self.labels[index])
        raw_w, raw_h = img.size
        resized_size = (int(raw_h * self.scale_ratio), int(raw_w * self.scale_ratio))
        img = F.resize(img, resized_size, Image.BILINEAR)
        lbl = F.resize(lbl, resized_size, Image.NEAREST)
        img = np.array(img)
        lbl = np.array(lbl)
        return img, lbl, img_name, lbl_name
    
    def __len__(self):
        return len(self.images)
