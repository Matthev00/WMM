from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


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


def calc_mse_psnr(img1, img2):
    """Funkcja obliczająca MSE i PSNR dla różnicy podanych obrazów, zakładana wartość pikseli z przedziału [0, 255]."""

    imax = 255.0**2  ### maksymalna wartość sygnału -> 255
    """
    W różnicy obrazów istotne są wartości ujemne, dlatego img1 konwertowany jest do typu np.float64 (liczby rzeczywiste)
    aby nie ograniczać wyniku do przedziału [0, 255].
    """
    mse = (
        (img1.astype(np.float64) - img2) ** 2
    ).sum() / img1.size  ###img1.size - liczba elementów w img1, ==img1.shape[0]*img1.shape[1] dla obrazów mono, ==img1.shape[0]*img1.shape[1]*img1.shape[2] dla obrazów barwnych
    psnr = 10.0 * np.log10(imax / mse)
    return (mse, psnr)


images_dir = Path("LAB5/color")

img_paths = sorted(list(images_dir.glob("*.png")))

id_number = 325203
number_of_imgs = len(img_paths)
img_nr = id_number % number_of_imgs

img = cv2.imread(str(img_paths[img_nr]))

print(8 * os.stat(str(img_paths[img_nr])).st_size / (img.shape[0] * img.shape[1]))


xx = []  ### tablica na wartości osi X -> bitrate
ym = []  ### tablica na wartości osi Y dla MSE
yp = []  ### tablica na wartości osi Y dla PSNR

for quality in [
    100,
    90,
    90,
    70,
    60,
    50,
    40,
    30,
    20,
    10,
]:  ### wartości dla parametru 'quality' należałoby dobrać tak, aby uzyskać 'gładkie' wykresy...
    out_file_name = f"LAB5/out_images/q{quality:03d}.jpg"
    """ Zapis do pliku w formacie .jpg z ustaloną 'jakością' """
    cv2.imwrite(out_file_name, img, (cv2.IMWRITE_JPEG_QUALITY, quality))
    """ Odczyt skompresowanego obrazu, policzenie bitrate'u i PSNR """
    image_compressed = cv2.imread(out_file_name, cv2.IMREAD_UNCHANGED)
    bitrate = (
        8 * os.stat(out_file_name).st_size / (img.shape[0] * img.shape[1])
    )  ### image.shape == image_compressed.shape
    mse, psnr = calc_mse_psnr(img, image_compressed)
    """ Zapamiętanie wyników do pózniejszego wykorzystania """
    xx.append(bitrate)
    ym.append(mse)
    yp.append(psnr)

""" Narysowanie wykresów """
fig = plt.figure()
fig.set_figwidth(fig.get_figwidth() * 2)
plt.suptitle("Charakterystyki R-D")
plt.subplot(1, 2, 1)
plt.plot(xx, ym, "-.")
plt.title("MSE(R)")
plt.xlabel("bitrate")
plt.ylabel("MSE", labelpad=0)
plt.subplot(1, 2, 2)
plt.plot(xx, yp, "-o")
plt.title("PSNR(R)")
plt.xlabel("bitrate")
plt.ylabel("PSNR [dB]", labelpad=0)
plt.show()

print(max(xx))

cv2.waitKey(0)

