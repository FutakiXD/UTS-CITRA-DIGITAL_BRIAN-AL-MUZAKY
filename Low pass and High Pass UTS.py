import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1️⃣ Baca gambar dalam grayscale
img = cv2.imread('Babahlil.jpg', 0)

# 2️⃣ Lakukan Transformasi Fourier
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)  # pusatkan spektrum ke tengah
magnitude_spectrum = 20 * np.log(np.abs(fshift))

# 3️⃣ Buat Mask Filter (Low-Pass dan High-Pass)
rows, cols = img.shape
crow, ccol = rows // 2, cols // 2

# Radius filter
radius = 30

# Filter Low-Pass (melewatkan frekuensi rendah)
mask_low = np.zeros((rows, cols), np.uint8)
cv2.circle(mask_low, (ccol, crow), radius, 1, thickness=-1)

# Filter High-Pass (melewatkan frekuensi tinggi)
mask_high = 1 - mask_low

# 4️⃣ Terapkan filter pada domain frekuensi
fshift_low = fshift * mask_low
fshift_high = fshift * mask_high

# 5️⃣ Kembalikan ke domain spasial (invers Fourier)
img_low = np.abs(np.fft.ifft2(np.fft.ifftshift(fshift_low)))
img_high = np.abs(np.fft.ifft2(np.fft.ifftshift(fshift_high)))

# 6️⃣ Tampilkan hasil
plt.figure(figsize=(12,8))
plt.subplot(2,3,1), plt.imshow(img, cmap='gray'), plt.title('Citra Asli')
plt.subplot(2,3,2), plt.imshow(magnitude_spectrum, cmap='gray'), plt.title('Spektrum Frekuensi')
plt.subplot(2,3,3), plt.imshow(mask_low*255, cmap='gray'), plt.title('Low-Pass Filter')
plt.subplot(2,3,4), plt.imshow(img_low, cmap='gray'), plt.title('Hasil Low-Pass')
plt.subplot(2,3,5), plt.imshow(mask_high*255, cmap='gray'), plt.title('High-Pass Filter')
plt.subplot(2,3,6), plt.imshow(img_high, cmap='gray'), plt.title('Hasil High-Pass')
plt.tight_layout()
plt.show()
