from pathlib import Path
import cv2
import numpy as np
import os


def cv_imshow(img, img_title="image"):
    """
    Funkcja do wyświetlania obrazu w wykorzystaniem okna OpenCV.
    Wykonywane jest przeskalowanie obrazu z rzeczywistymi lub 16-bitowymi całkowitoliczbowymi wartościami pikseli,
    żeby jedną funkcją wywietlać obrazy różnych typów.
    """
    # cv2.namedWindow(img_title, cv2.WINDOW_AUTOSIZE) # cv2.WINDOW_NORMAL

    if (img.dtype == np.float32) or (img.dtype == np.float64):
        img_ = img / 255
    elif img.dtype == np.int16:
        img_ = img * 128
    else:
        img_ = img
    cv2.imshow(img_title, img_)
    cv2.waitKey(0)


def calc_entropy(hist):
    pdf = hist / hist.sum()
    # entropy = -(pdf*np.log2(pdf)).sum() ### zapis na tablicach, ale problem z '/0'
    entropy = -sum([x * np.log2(x) for x in pdf if x != 0])
    return entropy


def printi(img, img_title="image"):
    """Pomocnicza funkcja do wypisania informacji o obrazie."""
    print(
        f"{img_title}, wymiary: {img.shape}, typ danych: {img.dtype}, wartości: {img.min()} - {img.max()}"
    )


images_dir = Path("LAB5/monochrome")

img_paths = sorted(list(images_dir.glob("*.png")))

id_number = 325203
number_of_imgs = len(img_paths)
img_nr = id_number % number_of_imgs
print(number_of_imgs)

img = cv2.imread(str(img_paths[img_nr]))

bitrate = 8 * os.stat(str(img_paths[img_nr])).st_size / (img.shape[0] * img.shape[1])
print(f"bitrate: {bitrate:.4f}")
