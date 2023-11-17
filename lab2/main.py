import copy
from time import sleep

import numpy as np
import pywt
import skimage.metrics

import skimage as si

from matplotlib import pyplot as plt
from skimage.io import imread, imshow, imsave

#### CONSTANTS ###
SEED = 42
ROUND = True
CLIP = True
##################

### VAR 22 CONSTANTS ###
area = 1/4
DECOMPOSITION_LEVELS = 3
COMPONENT_NUM_FOR_INJECT = 2  # LL - 0, LH - 1, HL - 2, HH - 3
########################

def idwt2_wrapper(coeffs):
    *LL, LH, HL, HH = coeffs
    if len(LL) == 1:
        LL = LL[0]
    else:
        LL = idwt2_wrapper(LL)

    return pywt.idwt2((LL, (LH, HL, HH)), 'haar')

def dwt2_wrapper(data_2d, decomposition_levels):
    LL, (LH, HL, HH) = pywt.dwt2(data_2d, 'haar')
    if decomposition_levels == 1:
        return [LL, LH, HL, HH]
    else:
        return [*dwt2_wrapper(LL, decomposition_levels - 1), LH, HL, HH]

def compile_F(components, current_decomposition_lvl):
    components = [copy.copy(i) for i in components]
    HH = components.pop()
    HL = components.pop()
    LH = components.pop()
    if current_decomposition_lvl == 1:
        LL = components.pop()
    else:
        LL = compile_F(components, current_decomposition_lvl - 1)

    L = np.concatenate((LL, LH,), axis=1)
    H = np.concatenate((HL, HH,), axis=1)
    F = np.concatenate((L, H,), axis=0)
    return F

def inject_omega(component, omega, alpha):
    fmean = np.mean(component)

    component_copy = copy.copy(component)
    component_vector_repr = component_copy.reshape(1, -1)
    omega_len = len(omega)
    component_vector_repr[0][:omega_len] = fmean + (component_vector_repr[0][:omega_len] - fmean) * (1 + alpha * omega)

    return component_vector_repr.reshape(component.shape)

def rate_omega(f_w, f, alpha):
    fmean = np.mean(f)
    omega = (f_w - f) / (alpha * (f - fmean))
    return omega.reshape(1, -1)[0]

def detector(w, wnew):
    w_ = wnew[0:len(w)]
    sum = np.sum(w*w_)
    delimiter = np.sum(np.square(w_)) * np.sum(np.square(w))
    p = sum/np.sqrt(delimiter)
    return np.abs(p)

def inject_omega_to_container(omega, container_components, alpha, component_number):
    components_copy = [copy.copy(i) for i in container_components]
    component_for_inject = components_copy[component_number]
    components_copy[component_number] = inject_omega(component_for_inject, omega, alpha)
    return components_copy

def compile_Cw(components):
    Cw = idwt2_wrapper(components)

    if ROUND:
        Cw = Cw.round()

    if CLIP:
        Cw = np.clip(Cw, 0, 255)

    if ROUND and CLIP:
        Cw = Cw.astype(np.uint8)

    return Cw

def main():
    C = imread("bridge.tif")

    # Task 1
    n = int(area * (C.shape[0] // (2 ** DECOMPOSITION_LEVELS)) * (C.shape[1] // (2 ** DECOMPOSITION_LEVELS)))
    omega = np.random.default_rng(seed=SEED).random(n)

    # Task 2
    components = dwt2_wrapper(C, DECOMPOSITION_LEVELS)
    F = compile_F(components, DECOMPOSITION_LEVELS)

    # Task 3
    alpha = 0.10
    components_copy = inject_omega_to_container(omega, components, alpha, COMPONENT_NUM_FOR_INJECT)

    # Task 4
    Cw = compile_Cw(components_copy)
    imsave("Cw.tif", Cw)

    # Task 5
    Cw = imread("Cw.tif")
    components_w = dwt2_wrapper(Cw, DECOMPOSITION_LEVELS)

    # Task 6
    new_omega = rate_omega(components_w[COMPONENT_NUM_FOR_INJECT], components[COMPONENT_NUM_FOR_INJECT], alpha)[:n]
    ro = detector(omega, new_omega)
    print(f"[Task 6] ro = {ro}")

    # Task 7
    alpha = 0
    ro = 0.000001
    prev_max_ro = 0
    best_ro = 0
    best_psnr = 0
    best_alpha = 0
    while prev_max_ro <= ro:
        alpha += 0.25

        components_copy = inject_omega_to_container(omega, components, alpha, COMPONENT_NUM_FOR_INJECT)
        Cw = compile_Cw(components_copy)
        components_w = dwt2_wrapper(Cw, DECOMPOSITION_LEVELS)

        new_omega = rate_omega(components_w[COMPONENT_NUM_FOR_INJECT], components[COMPONENT_NUM_FOR_INJECT], alpha)[:n]
        ro = detector(omega, new_omega)
        psnr = skimage.metrics.peak_signal_noise_ratio(C, Cw)

        if ro > 0.9:
            if psnr > best_psnr:
                best_ro = ro
                best_psnr = psnr
                best_alpha = alpha


        if prev_max_ro < ro:
            prev_max_ro = ro

    print(f"[Task 7] best_ro = {best_ro} best_psnr = {best_psnr} best_alpha = {best_alpha}")

    # Task 8
    results = []
    for i in range(0, 4):  # LL - 0, LH - 1, HL - 2, HH - 3
        components_copy = inject_omega_to_container(omega, components, best_alpha, COMPONENT_NUM_FOR_INJECT)
        Cw = compile_Cw(components_copy)
        imsave(f"Cw_{i}.tif", Cw)
        Cw = imread(f"Cw_{i}.tif")
        components_w = dwt2_wrapper(Cw, DECOMPOSITION_LEVELS)

        new_omega = rate_omega(components_w[COMPONENT_NUM_FOR_INJECT], components[COMPONENT_NUM_FOR_INJECT], best_alpha)[:n]
        ro = detector(omega, new_omega)
        psnr = skimage.metrics.peak_signal_noise_ratio(C, Cw)
        results.append((ro, psnr))

    for i, r in enumerate(results):
        print(f"{i}: ro = {r[0]} psnr = {r[1]}")



if __name__ == '__main__':
    main()




# fig, axes = plt.subplots(1, 1)
    # axes.imshow(F, cmap='gray')
    # LL, (LH, HL, HH) = coeffs
    # L = np.concatenate((LL, LH,), axis=1)
    # H = np.concatenate((HL, HH,), axis=1)
    # F = np.concatenate((L, H,), axis=0)
    # fig, axes = plt.subplots(1, 1)
    # axes.imshow(F, cmap='gray')
    #
    # fig, axes = plt.subplots(2, 2)
    #
    # axes[0, 0].imshow(LL, cmap='gray')
    # axes[0, 1].imshow(LH, cmap='gray')
    # axes[1, 0].imshow(HL, cmap='gray')
    # axes[1, 1].imshow(HH, cmap='gray')
    # axes[0,0].scatter = LL
    # axes[0,1] = LH
    # axes[1,0] = LL
    # axes[1,1] = HH

    # plt.show()

    # F_w = compile_F(components_copy, DECOMPOSITION_LEVELS)
    #
    # fig = plt.figure()
    # fig.add_subplot(1, 2, 1)
    # imshow(F, cmap="gray")
    # fig.add_subplot(1, 2, 2)
    # imshow(F_w, cmap="gray")

    # fig = plt.figure()
    # fig.add_subplot(1, 2, 1)
    # imshow(C, cmap="gray")
    # fig.add_subplot(1, 2, 2)
    # imshow(Cw, cmap="gray")
    #
    # plt.show()