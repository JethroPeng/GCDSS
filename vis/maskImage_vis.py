import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def overlay_mask_with_colorbar(image_path, mask_path, output_path=None):
    mask = np.load(mask_path)
    image = cv2.imread(image_path)
    image = cv2.resize(image, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_LINEAR)
    num_classes = np.max(mask)
    colormap = plt.get_cmap('tab20', num_classes)
    mask_colors = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for class_idx in range(num_classes):
        class_mask = (mask == class_idx)
        if np.sum(class_mask) == 0:
            continue
        color = (np.array(colormap(class_idx)[:3]) * 255).astype(np.uint8)
        mask_colors[class_mask] = color

    colorbar = plt.cm.ScalarMappable(cmap=colormap)
    colorbar.set_array(range(0, num_classes))

    combined_image = np.zeros((image.shape[0], 2*image.shape[1], 3), dtype=np.uint8)
    combined_image[:, :image.shape[1], :] = image

    combined_image[:, image.shape[1]:, :] = mask_colors

    if output_path:
        cv2.imwrite(output_path, combined_image)
    else:
        plt.imshow(cv2.cvtColor(combined_image, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()

if __name__ == "__main__":
    image_path = "../datasets/Cityscapes/leftImg8bit_gcd/aachen_000000_000019_leftImg8bit.png"
    mask_path  = "../datasets/Cityscapes/maskImages/aachen_000000_000019.npy"
    output_path = "../result/vis_mask.jpg"
    overlay_mask_with_colorbar(image_path, mask_path, output_path)
