import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')

xor_func = np.vectorize(lambda lhs, rhs: lhs ^ rhs)

def read_image(path: str, flag=cv2.IMREAD_UNCHANGED):
    image = cv2.imread(path, flag)
    return image


def return_channel(image: np.ndarray, num: int):
    b, g, r = cv2.split(image)
    return [b, g, r][num-1]


def prepare_wm(image: np.ndarray):
    return image // 255


def save_image(image: np.ndarray, name: str):
    cv2.imwrite(name, image)


# region Task1


def to_bin(num):
    return format(num, '08b')


def to_dec(num):
    return np.uint8(int(num, 2))


def invade(dec_value, new_value):
    value_in_bin_format = to_bin(dec_value)
    value_in_bin_format = list(value_in_bin_format)
    value_in_bin_format[2] = str(new_value)

    return int(''.join(value_in_bin_format), 2)



def extract(channel, pos: int):
    tmp = list(channel)[pos]
    return int(tmp)


def get_byte_plane(img, channel, position):
    blue, green, red = cv2.split(img)
    to_bin_func = np.vectorize(to_bin)
    reveal_func = np.vectorize(extract)

    if channel == "green":
        binary_channel = to_bin_func(green)
    elif channel == "red":
        binary_channel = to_bin_func(red)
    elif channel == "blue":
        binary_channel = to_bin_func(blue)
    else:
        raise RuntimeError(f"Chanel should be in range 'red', 'green', blue'. Provided: {channel}")

    return reveal_func(binary_channel, position)



def hide_img(channel, wm, pos):
    to_bin_func = np.vectorize(to_bin)
    to_dec_func = np.vectorize(to_dec)
    hide_func = np.vectorize(invade)
    channel = to_bin_func(channel)
    wm = to_bin_func(wm)
    channel = hide_func(channel, wm, pos)
    channel = to_dec_func(channel)
    return channel


def receive_img(image, pos, ch_pos):
    blue, green, red = cv2.split(image)
    channels = [blue, green, red]
    to_bin_func = np.vectorize(to_bin)
    reveal_func = np.vectorize(extract)
    channels[ch_pos] = to_bin_func(channels[ch_pos])
    result = reveal_func(channels[ch_pos], pos)
    return result * 255


def show_func(images):
    plt.subplot(221)
    plt.imshow(images[0])
    plt.subplot(222)
    plt.imshow(images[1])
    plt.subplot(223)
    plt.imshow(images[2], cmap='gray', vmin=0, vmax=255)
    plt.subplot(224)
    plt.imshow(images[3], cmap='gray', vmin=0, vmax=255)
    plt.show()


def task1():
    baboon = read_image("baboon.tif")
    ornament = read_image("ornament.tif", cv2.IMREAD_GRAYSCALE) // 255

    green2_channel_from_baboon = get_byte_plane(baboon, "green", 2)
    red3_channel_from_baboon = get_byte_plane(baboon, "red", 3)

    def allign_to_ornament(source, lhs, rhs):
        if rhs == lhs:
            return source
        else:
            return int(not bool(source))

    allign_to_ornament_vect = np.vectorize(allign_to_ornament)

    xored_virtual_plane = xor_func(green2_channel_from_baboon, red3_channel_from_baboon)

    bin_injected_plane = allign_to_ornament_vect(green2_channel_from_baboon, xored_virtual_plane, ornament)
    # bin_injected_plane = xor_func(xored_virtual_plane, ornament)
    # dec_injected_plane = bin_injected_plane * (2 ** 1) # Подставляем в 2 плоскость, поэтому в степени 1


    orig_blue, orig_green, orig_red = cv2.split(baboon)

    invade_vect = np.vectorize(invade)
    injected_green = invade_vect(orig_green, bin_injected_plane).astype(np.uint8)
    final_baboon = cv2.merge([orig_blue, injected_green,orig_red])

    save_image(final_baboon, "final_baboon.tif")



def task2():
    baboon_hacked = read_image("final_baboon.tif")

    green2_new_channel_from_hacked_baboon = get_byte_plane(baboon_hacked, "green", 2)
    red3_channel_from_hacked_baboon = get_byte_plane(baboon_hacked, "red", 3)


    watermark = xor_func(green2_new_channel_from_hacked_baboon, red3_channel_from_hacked_baboon) * 255
    save_image(watermark, "final_ornament.tif")

# endregion

# region Task2


def check_for_channel(channel, delta):
    checker = np.floor(channel / delta) % 2
    return np.uint8(checker)


def task2_show(images):
    plt.subplot(321)
    plt.imshow(images[0])
    plt.subplot(322)
    plt.imshow(images[1])
    plt.subplot(325)
    plt.imshow(images[4], cmap='gray', vmin=0, vmax=255)
    plt.subplot(326)
    plt.imshow(images[5], cmap='gray', vmin=0, vmax=255)
    plt.subplot(323)
    plt.imshow(images[2], cmap='gray', vmin=0, vmax=255)
    plt.subplot(324)
    plt.imshow(images[3], cmap='gray', vmin=0, vmax=255)
    plt.show()


def simple_qim(channel, wm):
    delta = 8
    two_delta = delta * 2
    new_data = np.floor(channel/two_delta) * two_delta + wm * delta
    return np.uint8(new_data)


def hide2(image, wm):
    blue, green, red = cv2.split(image)
    hide_funk = np.vectorize(simple_qim)
    red = hide_funk(red, wm)
    gauss = np.random.normal(0, 0.03, image.size)
    gauss = gauss.reshape(image.shape[0], image.shape[1], image.shape[2]).astype('uint8')
    image = cv2.merge((blue, green, red))
    image = cv2.add(image, gauss)
    return image


def find2(image):
    blue, green, red = cv2.split(image)
    find_funk = np.vectorize(check_for_channel)
    image = find_funk(red, 8)
    return image * 255


# def task2():
#     baboon_default = read_image("baboon.tif")
#     b, g, r1 = cv2.split(baboon_default)
#     baboon_show = cv2.merge((r1, g, b))
#
#     wm_default = read_image("ornament.tif", cv2.IMREAD_GRAYSCALE)
#     wm_prepared = prepare_wm(wm_default)
#
#     new_image = hide2(baboon_default, wm_prepared)
#     save_image(new_image, "stega2.tif")
#     b, g, r2 = cv2.split(new_image)
#     new_image_show = cv2.merge((r2, g, b))
#
#     new_wm = find2(new_image)
#
#     task2_show([baboon_show, new_image_show, cv2.imread("asas.tif"), new_wm, r1, r2])

# endregion


# Варинат 22
#

if __name__ == '__main__':
    task1()
    task2()
    # task2()