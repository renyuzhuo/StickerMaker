import os

import cv2
import numpy as np
from PIL import Image

in_image_path = './1/'
out_1_image_path = './out_1/'
out_2_image_path = './out_2/'
out_3_image_path = './out_3/'


def run(name):
    image_path = in_image_path + name

    # Load the image
    original_image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    # Convert image to grayscale
    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

    # Apply thresholding to get the binary image
    _, binary_image = cv2.threshold(gray_image, 240, 255, cv2.THRESH_BINARY_INV)

    # Find contours from the binary image
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Assume the largest external contour in image is the main object
    # Create an all white image
    alpha_channel = np.ones(binary_image.shape, dtype=original_image.dtype) * 255

    # Fill the detected contours with black color in white image
    cv2.drawContours(alpha_channel, contours, contourIdx=-1, color=(0, 0, 0), thickness=-1)

    # Create a mask for the main object
    mask = alpha_channel != 255

    # Create a new image that contains the main object with an alpha channel
    extracted_image = original_image.copy()

    # Set the alpha channel to the mask we created
    extracted_image = cv2.cvtColor(extracted_image, cv2.COLOR_BGR2BGRA)
    extracted_image[:, :, 3] = np.where(mask, 255, 0)

    # Save the extracted image
    extracted_image_path = out_1_image_path + name
    cv2.imwrite(extracted_image_path, extracted_image)


def run2(name):
    blur_radius = 3
    border_size = 5
    image_path = out_1_image_path + name

    img = Image.open(image_path).convert("RGBA")
    img_np = np.array(img)

    # 提取alpha通道
    alpha_channel = img_np[:, :, 3]

    # 高斯模糊，平滑边缘
    blurred_alpha = cv2.GaussianBlur(alpha_channel, (blur_radius * 2 + 1, blur_radius * 2 + 1), 0)

    # 使用OpenCV寻找边缘
    edges = cv2.Canny(blurred_alpha, 100, 200)

    # 创建用于膨胀的核，增加边缘宽度
    kernel = np.ones((border_size, border_size), np.uint8)
    dilation = cv2.dilate(edges, kernel, iterations=1)

    # 将膨胀的边缘应用于原始图像
    img_np[(dilation == 255) & (img_np[:, :, 3] < 255)] = [255, 255, 255, 255]

    # 使用高斯模糊平滑边缘，再次应用
    smooth_edges = cv2.GaussianBlur(dilation, (blur_radius * 2 + 1, blur_radius * 2 + 1), 0)
    alpha_channel[smooth_edges > 0] = 255

    # 将处理后的alpha通道重新合并到图像
    img_np[:, :, 3] = alpha_channel

    # 保存结果
    output_image_path = out_2_image_path + name
    Image.fromarray(img_np).save(output_image_path)


def run3(name):
    image_path = out_2_image_path + name
    output_path = out_3_image_path + name
    target_size = 240

    if name.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
        # 打开并调整图片大小
        with Image.open(image_path) as img:
            img = img.convert("RGBA")  # 确保图片为RGBA格式，以处理透明度
            # 计算新尺寸
            ratio = target_size / max(img.size)
            new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
            resized_img = img.resize(new_size, Image.Resampling.LANCZOS)

            # 保存到输出目录
            resized_img.save(output_path)
            print(output_path)


if __name__ == "__main__":
    files = os.listdir(in_image_path)
    print(files)
    for file in files:
        print(file)
        run(file)
        run2(file)
        run3(file)
