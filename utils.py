import numpy as np
import cv2
import os
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt

IMG_SIZE = 256

def load_images_from_folder(folder):
    images = []
    for filename in sorted(os.listdir(folder)):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            images.append(img)
    return np.array(images, dtype=np.float32) / 255.0

def add_mixed_noise(images):
    noisy = np.empty_like(images)
    for i in range(len(images)):
        gauss = np.random.normal(0, np.random.uniform(5, 60)/255., images[i].shape)
        poisson = np.random.poisson(images[i]*np.random.uniform(5, 30))/255.
        noisy[i] = np.clip(images[i] + gauss + poisson, 0, 1)
    return noisy

def evaluate_denoiser(model, test_noisy, test_clean, num_samples=5):
    denoised = model.predict(test_noisy[:num_samples])
    plt.figure(figsize=(20, 15))

    for i in range(num_samples):
        clean = test_clean[i]
        noisy = test_noisy[i]
        denoised_img = denoised[i]

        psnr_val = psnr(clean, denoised_img, data_range=1.0)
        ssim_val = ssim(clean, denoised_img, channel_axis=-1, data_range=1.0)

        plt.subplot(3, num_samples, i+1)
        plt.imshow(noisy)
        plt.title(f"Noisy\nPSNR: {psnr(clean, noisy, data_range=1.0):.2f}")
        plt.axis('off')

        plt.subplot(3, num_samples, i+num_samples+1)
        plt.imshow(clean)
        plt.title("Original")
        plt.axis('off')

        plt.subplot(3, num_samples, i+2*num_samples+1)
        plt.imshow(denoised_img)
        plt.title(f"Denoised\nPSNR: {psnr_val:.2f}\nSSIM: {ssim_val:.3f}")
        plt.axis('off')

    plt.tight_layout()
    plt.savefig("results/sample_denoised.png")
    plt.show()
