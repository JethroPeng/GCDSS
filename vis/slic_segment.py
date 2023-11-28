import matplotlib.pyplot as plt
from skimage import io, segmentation, color, graph
from skimage.segmentation import slic
import numpy as np
from PIL import Image

image_path = "../datasets/Cityscapes/leftImg8bit_gcd/aachen_000000_000019_leftImg8bit.png"
image = io.imread(image_path)
labels = slic(image, n_segments=200, compactness=0.5, start_label=1)
print(np.unique(labels))
cmap = plt.get_cmap('tab20', len(np.unique(labels)))
colored_labels = cmap(labels)
plt.imshow(colored_labels)
plt.axis('off')
plt.show()
