import os
from PIL import Image
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

def preprocessing_cs():
    cwd = os.getcwd()

    class_name = {0: 'road', 1: 'sidewalk', 2: 'building', 3: 'wall', 4: 'fence',
                  5: 'pole', 6: 'traffic light', 7: 'traffic sign', 8: 'vegetation', 
                  9: 'terrain', 10: 'sky', 11: 'person', 12: 'rider', 13: 'car', 
                  14: 'truck', 15: 'bus', 16: 'train', 17: 'motorcycle', 18: 'bicycle'}

    gtFine_dir = os.path.join(cwd, 'datasets/Cityscapes/gtFine')
    leftImg8bit_dir = os.path.join(cwd, 'datasets/Cityscapes/leftImg8bit')
    gtFine_gcd = os.path.join(cwd, 'datasets/Cityscapes/gtFine_gcd')
    leftImg8bit_gcd = os.path.join(cwd, 'datasets/Cityscapes/leftImg8bit_gcd')

    os.makedirs(gtFine_gcd, exist_ok=True)
    os.makedirs(leftImg8bit_gcd, exist_ok=True)

    #for split in ['train', 'val']:
    #    for split in os.listdir(leftImg8bit_dir):
    #        split_dir = os.path.join(leftImg8bit_dir, split)
    #        for city in tqdm(os.listdir(split_dir), desc='copying img in {}'.format(split)):
    #            city_dir = os.path.join(split_dir, city)
    #            for img_name_suffix in os.listdir(city_dir):
    #                img_path = os.path.join(city_dir, img_name_suffix)
    #                img = Image.open(img_path)
    #                img_path_gcd = os.path.join(leftImg8bit_gcd, img_name_suffix)
    #                img.save(img_path_gcd)

    for split in ['train', 'val']:
        split_dir = os.path.join(gtFine_dir, split)
        for city in tqdm(os.listdir(split_dir), desc='copying img in {}'.format(split)):
            city_dir = os.path.join(split_dir, city)
            for lbl_name_suffix in os.listdir(city_dir):
                if lbl_name_suffix.__contains__('_labelTrainIds'):
                    lbl_path = os.path.join(city_dir, lbl_name_suffix)
                    lbl = Image.open(lbl_path)
                    lbl_path_gcd = os.path.join(gtFine_gcd, lbl_name_suffix)
                    lbl.save(lbl_path_gcd)

if __name__ == '__main__':
    preprocessing_cs()



        
    

