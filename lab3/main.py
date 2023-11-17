import os
import pandas as pd
import numpy as np
import skimage.metrics
from matplotlib import pyplot as plt

from skimage.io import imread, imshow, imsave
from PIL import Image
from scipy.signal import convolve2d

import lab2.main as lab2

SEED = 42
### VAR 22 CONSTANTS ###
area = 1/4
DECOMPOSITION_LEVELS = 3
COMPONENT_NUM_FOR_INJECT = 2  # LL - 0, LH - 1, HL - 2, HH - 3
########################

def cycle_shift(img, r):
    size = np.shape(img)
    new_img = np.roll(img, int(r * size[1]), axis=1)
    new_img = np.roll(new_img, int(r * size[0]), axis=0)
    return new_img

def rotate(img, angle):
    new_img = Image.fromarray(img, mode="L")
    new_img = np.asarray(new_img.rotate(angle))
    return new_img

def blur_img(img, s):
    M = 2 * int(3 * s) + 1
    impulse_characteristic = np.vectorize(calc_g, signature="(),(n),()->(k)")
    g = impulse_characteristic(m1=np.arange(0, M, 1), m2=np.arange(0, M, 1), sigma=s)
    g /= np.sum(g)
    new_img = convolve2d(img, g, mode="same")
    return new_img

def calc_g(m1, m2, sigma):
    M = 2 * np.floor(3 * sigma) + 1
    tmp1 = (m1 - M / 2) ** 2
    tmp2 = (m2 - M / 2) ** 2
    power = -(tmp1 + tmp2) / (2 * sigma)
    GaussH = np.exp(power)
    return GaussH

def to_jpg(img, quality):
    imsave("tmpFile.jpg", img, quality=int(quality))
    new_img = imread("tmpFile.jpg")
    os.remove("tmpFile.jpg")
    return new_img

def print_ro_graph(title, ro_array):
    plt.plot(ro_array)

def main():
    lab2.ROUND = True
    lab2.CLIP = True

    C = imread("barb.tif")

    n = int(0.25 * (C.shape[0] // (2 ** DECOMPOSITION_LEVELS)) * (C.shape[1] // (2 ** DECOMPOSITION_LEVELS)))
    omega = np.random.default_rng(seed=SEED).random(n)

    components = lab2.dwt2_wrapper(C, DECOMPOSITION_LEVELS)

    alpha = 1.25
    components_w = lab2.inject_omega_to_container(omega, components, alpha, COMPONENT_NUM_FOR_INJECT)
    Cw = lab2.compile_Cw(components_w)

    components_w = lab2.dwt2_wrapper(Cw, DECOMPOSITION_LEVELS)
    new_omega = lab2.rate_omega(components_w[COMPONENT_NUM_FOR_INJECT], components[COMPONENT_NUM_FOR_INJECT], alpha)[:n]

    ro = lab2.detector(omega, new_omega)
    psnr = skimage.metrics.peak_signal_noise_ratio(C, Cw)

    print(f"Cw without changes: ro = {ro} psnr = {psnr}\n")

    shifted_imgs = []
    cycleShift = np.arange(0.1, 1, 0.1)
    shifted_ro = []
    for shift in cycleShift:
        Cw_damaged = cycle_shift(Cw, shift)
        components_damaged = lab2.dwt2_wrapper(Cw_damaged, DECOMPOSITION_LEVELS)
        new_omega_damaged = lab2.rate_omega(components_damaged[COMPONENT_NUM_FOR_INJECT], components[COMPONENT_NUM_FOR_INJECT], alpha)[:n]
        ro_damaged = lab2.detector(omega, new_omega_damaged)
        psnr_damaged = skimage.metrics.peak_signal_noise_ratio(C, Cw_damaged)
        
        shifted_ro.append(ro_damaged)
        shifted_imgs.append(Cw_damaged)
        print(f"Cycle shift {shift:.1f}:\t\tro damaged: {ro_damaged:.5f}\t\tpsnr damaged: {psnr_damaged:.3f}\t\talpha: {alpha}")
    print("\n")

    rotated_imgs = []
    rotation_angles = np.arange(1, 91, 8.9)
    rotated_ro = []
    for rot_a in rotation_angles:
        Cw_damaged = rotate(Cw, rot_a)
        components_damaged = lab2.dwt2_wrapper(Cw_damaged, DECOMPOSITION_LEVELS)
        new_omega_damaged = lab2.rate_omega(components_damaged[COMPONENT_NUM_FOR_INJECT],
                                            components[COMPONENT_NUM_FOR_INJECT], alpha)[:n]
        ro_damaged = lab2.detector(omega, new_omega_damaged)
        psnr_damaged = skimage.metrics.peak_signal_noise_ratio(C, Cw_damaged)
        
        rotated_ro.append(ro_damaged)
        rotated_imgs.append(Cw_damaged)
        print(f"Rotation angle {rot_a:.1f}:\t\tro damaged: {ro_damaged:.5f}\t\tpsnr damaged: {psnr_damaged:.3f}\t\talpha: {alpha}")
    print("\n")


    blurred_images = []
    blur_coeffs = np.arange(1, 4.1, 0.5)
    blurred_ro = []
    for sigma in blur_coeffs:
        Cw_damaged = blur_img(Cw, sigma)
        components_damaged = lab2.dwt2_wrapper(Cw_damaged, DECOMPOSITION_LEVELS)
        new_omega_damaged = lab2.rate_omega(components_damaged[COMPONENT_NUM_FOR_INJECT],
                                            components[COMPONENT_NUM_FOR_INJECT], alpha)[:n]
        ro_damaged = lab2.detector(omega, new_omega_damaged)
        psnr_damaged = skimage.metrics.peak_signal_noise_ratio(C, Cw_damaged)

        blurred_ro.append(ro_damaged)
        blurred_images.append(Cw_damaged)
        print(f"Gaussian Blur {sigma}:\t\tro damaged: {ro_damaged:.5f}\t\tpsnr damaged: {psnr_damaged:.3f}\t\talpha: {alpha}")
    print("\n")

    jpeg_imgs = []
    jpeg_qf = np.arange(30, 100, 10)
    jpeg_ro = []
    for QF in jpeg_qf:
        Cw_damaged = to_jpg(Cw, QF)
        components_damaged = lab2.dwt2_wrapper(Cw_damaged, DECOMPOSITION_LEVELS)
        new_omega_damaged = lab2.rate_omega(components_damaged[COMPONENT_NUM_FOR_INJECT],
                                            components[COMPONENT_NUM_FOR_INJECT], alpha)[:n]
        ro_damaged = lab2.detector(omega, new_omega_damaged)
        psnr_damaged = skimage.metrics.peak_signal_noise_ratio(C, Cw_damaged)
        
        jpeg_ro.append(ro_damaged)
        jpeg_imgs.append(Cw_damaged)
        print(f"JPEG {QF}:\t\tro damaged: {ro_damaged:.5f}\t\tpsnr damaged: {psnr_damaged:.3f}\t\talpha: {alpha}")
    print("\n")

    print("### RESULTS ###")
    print(f"CyclicShift:\t{np.mean(shifted_ro)}\n")
    print(f"Rotation:\t{np.mean(rotated_ro)}\n")
    print(f"Gaussian Blur:\t{np.mean(blurred_ro)}\n")
    print(f"JPEG:\t{np.mean(jpeg_ro)}\n")
    print("###############")

    fig, axes = plt.subplots(2, 2)
    axes[0, 0].plot(shifted_ro)
    axes[0, 0].semilogy()
    axes[0, 0].axhline(y=ro, color='r', linestyle='-')
    axes[0, 0].set_title('CyclicShift (iteration - ro value)')

    axes[0, 1].plot(rotated_ro)
    axes[0, 1].semilogy()
    axes[0, 1].axhline(y=ro, color='r', linestyle='-')
    axes[0, 1].set_title('Rotation (iteration - ro value)')

    axes[1, 0].plot(blurred_ro)
    axes[1, 0].semilogy()
    axes[1, 0].axhline(y=ro, color='r', linestyle='-')
    axes[1, 0].set_title('Gaussian Blur (iteration - ro value)')

    axes[1, 1].plot(jpeg_ro)
    axes[1, 1].semilogy()
    axes[1, 1].axhline(y=ro, color='r', linestyle='-')
    axes[1, 1].set_title('JPEG (iteration - ro value)')

    plt.show()

    fig, axes = plt.subplots(2, 2)
    axes[0, 0].plot(shifted_ro)
    axes[0, 0].semilogy()
    axes[0, 0].axhline(y=psnr, color='r', linestyle='-')
    axes[0, 0].set_title('CyclicShift (iteration - psnr value)')

    axes[0, 1].plot(rotated_ro)
    axes[0, 1].semilogy()
    axes[0, 1].axhline(y=psnr, color='r', linestyle='-')
    axes[0, 1].set_title('Rotation (iteration - ro value)')

    axes[1, 0].plot(blurred_ro)
    axes[1, 0].semilogy()
    axes[1, 0].axhline(y=psnr, color='r', linestyle='-')
    axes[1, 0].set_title('Gaussian Blur (iteration - ro value)')

    axes[1, 1].plot(jpeg_ro)
    axes[1, 1].semilogy()
    axes[1, 1].axhline(y=psnr, color='r', linestyle='-')
    axes[1, 1].set_title('JPEG (iteration - ro value)')

    df = pd.DataFrame(np.zeros((len(jpeg_qf), len(rotation_angles))))
    for i, qf in enumerate(jpeg_qf):
        for j, rot_a in enumerate(rotation_angles):
            Cw_damaged = to_jpg(Cw, QF)
            Cw_damaged = rotate(Cw_damaged, rot_a)

            components_damaged = lab2.dwt2_wrapper(Cw_damaged, DECOMPOSITION_LEVELS)
            new_omega_damaged = lab2.rate_omega(components_damaged[COMPONENT_NUM_FOR_INJECT],
                                                components[COMPONENT_NUM_FOR_INJECT], alpha)[:n]
            ro_damaged = lab2.detector(omega, new_omega_damaged)
            psnr_damaged = skimage.metrics.peak_signal_noise_ratio(C, Cw_damaged)


            df.iloc[i, j] = ro_damaged


    print(df.to_string())


if __name__ == '__main__':
    main()