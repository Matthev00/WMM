from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np


images_dir = Path("LAB5/monochrome")

img_paths = sorted(list(images_dir.glob("*.png")))

id_number = 325203
number_of_imgs = len(img_paths)
img_nr = id_number % number_of_imgs
print(number_of_imgs)

img = cv2.imread(str(img_paths[img_nr]))

plt.figure(figsize=(15, 10))
plt.title('image')
plt.imshow(img)
plt.show()


def calc_entropy(hist):
    pdf = hist/hist.sum()
    # entropy = -(pdf*np.log2(pdf)).sum() ### zapis na tablicach, ale problem z '/0'
    entropy = -sum([x*np.log2(x) for x in pdf if x != 0])
    return entropy


"""
cv2.calcHist() zwraca histogram w postaci tablicy 2D,
do dalszego przetwarzania wygodniejsza może być tablica jednowymiarowa -> flatten().
"""
hist_image = cv2.calcHist([img], [0], None, [256], [0, 256])
hist_image = hist_image.flatten()
print(hist_image.sum(), img.shape[0]*img.shape[1]) ### dla sprawdzenia: suma wartości histogramu powinna być równa liczbie pikseli w obrazie

H_image = calc_entropy(hist_image)
print(f"Entropia = {H_image:.4f}")