from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt


def printi(img, img_title="image"):
    """Pomocnicza funkcja do wypisania informacji o obrazie."""
    print(
        f"{img_title}, wymiary: {img.shape}, typ danych: {img.dtype}, wartości: {img.min()} - {img.max()}"
    )


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


images_dir = Path("LAB5/color")

img_paths = sorted(list(images_dir.glob("*.png")))

id_number = 325203
number_of_imgs = len(img_paths)
img_nr = id_number % number_of_imgs

img = cv2.imread(str(img_paths[img_nr]))
cv_imshow(img)

image_YCrCb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb) ### albo: cv2.COLOR_BGR2YUV
printi(image_YCrCb, "image_YCrCb")

hist_Y = cv2.calcHist([image_YCrCb[:, :, 0]], [0], None, [256], [0, 256]).flatten()
hist_Cr = cv2.calcHist([image_YCrCb[:, :, 1]], [0], None, [256], [0, 256]).flatten()
hist_Cb = cv2.calcHist([image_YCrCb[:, :, 2]], [0], None, [256], [0, 256]).flatten()

H_Y = calc_entropy(hist_Y)
H_Cr = calc_entropy(hist_Cr)
H_Cb = calc_entropy(hist_Cb)
print(f"Entropia(Y) = {H_Y:.4f} \nEntropia(Cr) = {H_Cr:.4f} \nEntropia(Cb) = {H_Cb:.4f} \nEntropia_śr = {(H_Y+H_Cr+H_Cb)/3:.4f}")

cv_imshow(image_YCrCb[:, :, 0], "image_Y")
cv_imshow(image_YCrCb[:, :, 1], "image_Cr")
cv_imshow(image_YCrCb[:, :, 2], "image_Cb")
plt.figure()
plt.plot(hist_Y, color="gray")
plt.plot(hist_Cr, color="red")
plt.plot(hist_Cb, color="blue")
plt.title("hist YCrCb")
plt.xlim([0, 255])
plt.show()