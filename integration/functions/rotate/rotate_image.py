import concurrent
import cv2
import numpy as np
import os
import shutil
from concurrent.futures import ThreadPoolExecutor

import sys
from pathlib import Path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]  # root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

def get_rotation_angle(image, template_folder):
    # 图像增强操作
    # 先将图像放大
    height, width = image.shape[:2]
    new_width = 600  # 设定新的宽度
    new_height = int((new_width / width) * height)  # 根据新的宽度调整高度
    image_resized = cv2.resize(image, (new_width, new_height))  # 放大图像

    # 将放大的图像灰度化
    gray_image = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)

    # 设置阈值进行二值化
    threshold_value = 200  # 阈值可以根据需要调整
    _, img_binarized = cv2.threshold(gray_image, threshold_value, 255, cv2.THRESH_BINARY)

    # 将处理后的图像作为待匹配图像
    image_gray = img_binarized

    # 定义匹配的阈值
    thresh = 0.5

    # 获取模板文件夹中的所有模板文件
    template_paths = []
    for subfolder in os.listdir(template_folder):
        subfolder_path = os.path.join(template_folder, subfolder)
        if os.path.isdir(subfolder_path):
            for filename in os.listdir(subfolder_path):
                if filename.endswith('.png'):
                    template_paths.append(os.path.join(subfolder_path, filename))

    # 用于存储匹配结果
    max_val = -1  # 初始最大匹配度
    best_angle = None
    best_template = None
    best_template_path = None  # 用于保存最佳模板的路径

    # 获取待检测图像的尺寸
    image_h, image_w = image_gray.shape

    # 遍历每个模板进行匹配
    for template_path in template_paths:
        template = cv2.imread(template_path)
        template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

        # 获取模板图像的尺寸
        template_h, template_w = template_gray.shape

        # 如果模板图像大于待检测图像，则跳过该模板
        if template_h > image_h or template_w > image_w:
            continue

        # 使用归一化交叉相关方法执行模板匹配
        result = cv2.matchTemplate(image_gray, template_gray, cv2.TM_CCOEFF_NORMED)

        # 获取匹配度的最大值
        min_val, max_val_temp, min_loc, max_loc = cv2.minMaxLoc(result)

        # 如果当前匹配度高于阈值并且大于最大匹配度，则更新最佳角度
        if max_val_temp >= thresh and max_val_temp > max_val:
            max_val = max_val_temp
            best_angle = os.path.basename(os.path.dirname(template_path))  # 获取文件夹名作为角度
            best_template = template  # 保存最佳模板
            best_template_path = template_path  # 保存最佳模板路径

    # 如果没有找到匹配模板，返回原图
    rotated_image = image  

    if best_angle is not None:
        # 将角度转换为整数
        rotation_angle = int(best_angle)

        # 根据匹配到的角度进行旋转恢复
        if rotation_angle == 0:
            rotated_image = image  # 不进行旋转，返回原图
        elif rotation_angle == 90:
            # 如果匹配到90度，旋转-90度恢复
            rotated_image = rotate_image(image, -90)
        elif rotation_angle == 45:
            # 如果匹配到45度，旋转-45度恢复
            rotated_image = rotate_image(image, -45)
        else:
            # 如果角度不是0、45或90，则不做旋转
            rotated_image = image

    # 返回增强的图像和旋转的图像
    return best_angle, rotated_image, img_binarized  # 这里返回增强后的二值图像

def rotate_image(image, angle):
    # 获取图像的尺寸
    h, w = image.shape[:2]
    
    # 获取旋转矩阵
    rotation_matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
    
    # 计算旋转后图像的大小，确保图像完整显示
    abs_cos = abs(rotation_matrix[0, 0])
    abs_sin = abs(rotation_matrix[0, 1])
    
    # 计算新的图像大小
    new_w = int(h * abs_sin + w * abs_cos)
    new_h = int(h * abs_cos + w * abs_sin)
    
    # 调整旋转矩阵，使得图像完全显示（以白色填充背景）
    rotation_matrix[0, 2] += (new_w / 2) - w / 2
    rotation_matrix[1, 2] += (new_h / 2) - h / 2
    
    # 执行旋转并填充白色背景
    rotated_image = cv2.warpAffine(image, rotation_matrix, (new_w, new_h), borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    
    return rotated_image

def process_image(image_path, test_images_folder, template_folder, output_folder, show_results=0):
    image = cv2.imread(image_path)

    # 获取每张图片的旋转角度和增强后的图像
    rotation_angle, rotated_image, enhanced_image = get_rotation_angle(image, template_folder)

    # 旋转增强的二值图像
    enhanced_rotated_image = rotate_image(enhanced_image, -int(rotation_angle))

    # 如果需要展示结果，打印旋转角度信息
    if show_results == 1:
        print(f"{os.path.basename(image_path)} 的旋转角度为: {rotation_angle}")

    # 保存旋转后的增强二值图像，文件名加上旋转角度
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    output_image_path = os.path.join(output_folder, f"{base_name}_{rotation_angle}.png")
    cv2.imwrite(output_image_path, enhanced_rotated_image)  # 保存旋转后的增强图像

    if show_results == 1:
        print(f"保存旋转后的增强图像至: {output_image_path}")
    
    return 1  # 返回1表示处理完成

def process_images(test_images_folder, template_folder, output_folder, show_results=0):
    # 清空rotated_images文件夹中的所有文件
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder)  # 重新创建空的rotated_images文件夹

    # 获取所有图片文件路径
    image_paths = [os.path.join(test_images_folder, filename) for filename in os.listdir(test_images_folder) if filename.endswith(('.jpg', '.png', '.jpeg'))]

    total_count = len(image_paths)  # 获取总的图片数量

    # 使用多线程处理图片
    processed_count = 0  # 已处理图片的数量

    def update_progress(result):
        nonlocal processed_count
        processed_count += result
        print(f"进度：{processed_count}/{total_count}")

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_image, image_path, test_images_folder, template_folder, output_folder, show_results) for image_path in image_paths]
        for future in futures:
            future.add_done_callback(lambda future: update_progress(future.result()))
    print("\n已处理完成")


# 使用
test_images_folder = ROOT /'result/crops/data'  # 测试图片文件夹路径
template_folder =  ROOT /'functions/rotate/templates'  # 模板文件夹路径
output_folder =  ROOT /'result/rotated_crop'  # 存储旋转后图片的文件夹

# 处理所有测试图像，show_results=0 表示不展示结果，启用多线程
process_images(test_images_folder, template_folder, output_folder, show_results=0)
