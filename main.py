import os

import cv2
import numpy as np
from PIL import Image

in_image_path = './1/'
out_1_image_path = './out_1/'
out_2_image_path = './out_2/'
out_3_image_path = './out_3/'
out_5_image_path = './out_5/'
out_6_image_path = './out_6/'


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

def run4():
    # 打开四张图片
    background = Image.open("cover/background.png").convert("RGBA")
    image1 = Image.open("cover/mid.png").convert("RGBA")
    image2 = Image.open("cover/left.png").convert("RGBA")
    image3 = Image.open("cover/right.png").convert("RGBA")

    output_folder = "out_4"
    os.makedirs(output_folder, exist_ok=True)

    # 获取背景图片的尺寸
    bg_width, bg_height = background.size

    # 缩小图片
    resize_factor_image1 = 0.625
    resize_factor_image2 = 0.525
    resize_factor_image3 = 0.525

    image1 = image1.resize((int(image1.width * resize_factor_image1), int(image1.height * resize_factor_image1)))
    image2 = image2.resize((int(image2.width * resize_factor_image2), int(image2.height * resize_factor_image2)))
    image3 = image3.resize((int(image3.width * resize_factor_image3), int(image3.height * resize_factor_image3)))

    # 获取前景图片的尺寸
    image1_width, image1_height = image1.size
    image2_width, image2_height = image2.size
    image3_width, image3_height = image3.size

    # 计算前景图片的位置
    x_offset_image1 = (bg_width - image1_width) // 2
    y_offset_image1 = (bg_height - image1_height) * 3 // 4

    x_offset_image2 = 0
    y_offset_image2 = (bg_height - image2_height) * 3 // 4

    x_offset_image3 = bg_width - image3_width
    y_offset_image3 = (bg_height - image3_height) * 3 // 4

    # 创建每张图的 mask
    mask_image1 = image1.split()[3]  # 获取 alpha 通道
    mask_image2 = image2.split()[3]
    mask_image3 = image3.split()[3]

    # 将前景图片叠加到背景图片上，并使用 mask
    background.paste(image1, (x_offset_image1, y_offset_image1), mask=mask_image1)
    background.paste(image2, (x_offset_image2, y_offset_image2), mask=mask_image2)
    background.paste(image3, (x_offset_image3, y_offset_image3), mask=mask_image3)

    # 保存结果
    output_path = os.path.join(output_folder, "cover.png")
    background.save(output_path)

    print("图片已保存到:", output_path)

def run6(name):
    image_path = out_1_image_path + name
    output_path = out_5_image_path + name
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

def run7(name):
    image_path = out_1_image_path + name
    output_path = out_6_image_path + name
    target_size = 50

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

def run5():
    image_path = './out_4/cover.png'
    output_path = './out_4/cover_out.png'
    target_size = 750

    # 打开并调整图片大小
    with Image.open(image_path) as img:
        img = img.convert("RGBA")  # 确保图片为RGBA格式，以处理透明度
        # 计算新尺寸
        ratio = target_size / max(img.size)
        new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
        resized_img = img.resize(new_size, Image.Resampling.LANCZOS)

        width, height = resized_img.size

        # 定义截取区域的坐标
        left = 0
        top = height - 400
        right = width
        bottom = height

        cropped_image = resized_img.crop((left, top, right, bottom))

        # 保存到输出目录
        cropped_image.save(output_path)
        print(output_path)


def run7(filename):
    # 创建输出文件夹
    output_folder = 'thanks'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    output_folder_1 = 'thanks_1'
    if not os.path.exists(output_folder_1):
        os.makedirs(output_folder_1)

    # 打开背景图
    background_2 = Image.open('cover/background_2.png')
    background_3 = Image.open('cover/background_3.png')

    # 读取 out_2 文件夹中的每一张图片
    out_folder = 'out_2'
    # 打开图片
    image = Image.open(os.path.join(out_folder, filename))

    # 调整图片大小
    max_width = 500
    width_percent = max_width / float(image.size[0])
    new_height = int(float(image.size[1]) * float(width_percent))
    image = image.resize((max_width, new_height), Image.Resampling.LANCZOS)

    # 创建新图像
    new_image_bg2 = background_2.copy()
    new_image_bg3 = background_3.copy()

    # 将图片粘贴到背景图上
    offset = ((new_image_bg2.width - image.width) // 2, (new_image_bg2.height - image.height) // 2)
    new_image_bg2.paste(image, offset, image)

    offset = ((new_image_bg3.width - image.width) // 2, (new_image_bg3.height - image.height) // 2)
    new_image_bg3.paste(image, offset, image)

    # 保存合成后的图像到输出文件夹
    new_filename = os.path.splitext(filename)[0] + '_background_2.png'
    new_image_bg2.save(os.path.join(output_folder, new_filename))

    new_filename = os.path.splitext(filename)[0] + '_background_3.png'
    new_image_bg3.save(os.path.join(output_folder_1, new_filename))


if __name__ == "__main__":
    files = os.listdir(in_image_path)
    print(files)
    for file in files:
        print(file)
        run(file)
        run2(file)
        run3(file)
        run6(file)
        run7(file)
        run7(file)
    run4()
    run5()
