from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np


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


images_dir = Path("LAB5/monochrome")

img_paths = sorted(list(images_dir.glob("*.png")))

id_number = 325203
number_of_imgs = len(img_paths)
img_nr = id_number % number_of_imgs
print(number_of_imgs)

img = cv2.imread(str(img_paths[img_nr]))


def printi(img, img_title="image"):
    """Pomocnicza funkcja do wypisania informacji o obrazie."""
    print(
        f"{img_title}, wymiary: {img.shape}, typ danych: {img.dtype}, wartości: {img.min()} - {img.max()}"
    )


"""
Predykcja w kierunku poziomym:
od wartości danego piksela odejmowana jest wartość piksela z lewej strony - 'lewego sąsiada' (operacje na kolumnach).
Operację taką można wykonać dla pikseli leżących w drugiej i kolejnych kolumnach obrazu, z pominięciem skrajnie lewej kolumny.
"""
img_tmp1 = img[:, 1:]
img_tmp2 = img[:, :-1]

"""
W wyniku odejmowania pojawią się wartości ujemne - zakres wartości pikseli w obrazie różnicowym to będzie [-255, 255],
dlatego trzeba zminić typ wartości pikseli, żeby zakres wartości nie ograniczał się do [0, 255];
może to być np. cv2.CV_16S (odpowiednio np.int16 w NumPy), żeby pozostać w domenie liczb całkowitych.
"""
image_hdiff = cv2.addWeighted(img_tmp1, 1, img_tmp2, -1, 0, dtype=cv2.CV_16S)
"""
image_hdiff ma o jedną kolumnę mniej - dla skrajnie lewej kolumny nie było danych do odejmowania,
kolumnę tę można potraktować oddzielnie i 'połączyć' wyniki.
"""
image_hdiff_0 = cv2.addWeighted(img[:, 0], 1, 0, 0, -127, dtype=cv2.CV_16S).reshape(
    512, 1, 3
)  # Zmiana wymiarów na 3D
image_hdiff = np.hstack((image_hdiff_0, image_hdiff))
printi(image_hdiff, "image_hdiff")

cv_imshow(
    image_hdiff, "image_hdiff"
)  ### zdefiniowana funkcja pomocnicza odpowiednio 'obsługuje' obrazy z 16-bitowymi wartościami

"""
cv2.calcHist() wymaga danych w formacie liczb całkowitych bez znaku (8- lub 16-bitowych) lub 32-bitowych liczb rzeczywistych,
dlatego wartości pikseli są przesuwane z zakresu [-255, 255] do [0, 510] (-> '+255')
oraz konwertowane na typ np.uint16 (-> astype()).
"""
image_tmp = (image_hdiff + 255).astype(np.uint16)
hist_hdiff = cv2.calcHist([image_tmp], [0], None, [511], [0, 511]).flatten()

hist_image = cv2.calcHist([img], [0], None, [256], [0, 256])
hist_image = hist_image.flatten()
H_image = calc_entropy(hist_image)

""" Wypisanie entropii """
H_hdiff = calc_entropy(hist_hdiff)
print(f"Entropia obrazu predykcją poziomą: {H_hdiff:.4f}")
print(f"Entropia obrazu oryginalnego: {H_image:.4f}")

""" Wyświetlenie histogramów z wykorzystaniem matplotlib.pyplot """
plt.figure()
plt.subplot(1, 2, 1)
plt.plot(hist_image, color="blue")
plt.title("hist_image")
plt.xlim([0, 255])
plt.subplot(1, 2, 2)
plt.plot(
    np.arange(-255, 256, 1), hist_hdiff, color="red"
)  ### jawne podane wartości 'x' i 'y', żeby zmienić opisy na osi poziomej
plt.title("hist_hdiff")
plt.xlim([-255, 255])
plt.show()
