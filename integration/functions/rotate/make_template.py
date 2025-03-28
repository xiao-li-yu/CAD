import os
import shutil
from PIL import Image
import cv2
import numpy as np

import sys
from pathlib import Path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]  
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

def process_images(src_folder, dest_folder):
    # 清空 templates 文件夹
    if os.path.exists(dest_folder):
        shutil.rmtree(dest_folder)
    os.makedirs(dest_folder)
    
    # 创建子文件夹
    subfolders = ['0', '45', '90']
    for subfolder in subfolders:
        os.makedirs(os.path.join(dest_folder, subfolder))
    
    # 读取 ori_templates 文件夹中的所有图片
    for filename in os.listdir(src_folder):
        file_path = os.path.join(src_folder, filename)
        
        # 检查文件是否是图片（可以根据需要添加更多的图像格式）
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):

            try:
                # 打开图片
                img = Image.open(file_path)
                
                # 将原始图像保存到 0 文件夹
                img.save(os.path.join(dest_folder, '0', filename))
                
                # 旋转45度并保存到 45 文件夹
                img_45 = rotate_and_resize_image(img, 45)
                img_45.save(os.path.join(dest_folder, '45', filename))
                
                # 旋转90度并保存到 90 文件夹
                img_90 = rotate_and_resize_image(img, 90)
                img_90.save(os.path.join(dest_folder, '90', filename))
                
            except Exception as e:
                print(f"无法处理文件 {filename}: {e}")

def rotate_and_resize_image(img, angle):
    """
    旋转并调整图像大小，确保旋转后的图像显示完整，并填充白色。
    """
    # 获取原图的尺寸
    width, height = img.size
    
    # 计算旋转后的图像尺寸
    radians = np.deg2rad(angle)
    new_width = int(abs(width * np.cos(radians)) + abs(height * np.sin(radians)))
    new_height = int(abs(width * np.sin(radians)) + abs(height * np.cos(radians)))
    
    # 旋转图像并填充白色背景
    img_rotated = img.rotate(angle, resample=Image.BICUBIC, expand=True, fillcolor=(255, 255, 255))
    
    return img_rotated

def resize_and_save_image(image, scale_factor, save_path, threshold_value=200):
    # 二值化图像，阈值为200
    _, binary_image = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)

    # 计算新的尺寸
    new_width = int(binary_image.shape[1] * scale_factor)
    new_height = int(binary_image.shape[0] * scale_factor)
    new_size = (new_width, new_height)

    # 缩放图像
    resized_image = cv2.resize(binary_image, new_size, interpolation=cv2.INTER_LINEAR)

    # 保存缩放后的图像
    cv2.imwrite(save_path, resized_image)

def delete_images_with_underscore(parent_folder):
    # 获取父文件夹中的所有子文件夹
    subfolders = ['0', '45', '90']
    
    for subfolder in subfolders:
        # 获取当前子文件夹的路径
        folder_path = os.path.join(parent_folder, subfolder)
        
        # 确保该路径是一个有效的文件夹
        if os.path.isdir(folder_path):
            # 获取当前文件夹中的所有文件
            files = os.listdir(folder_path)
            
            # 遍历文件夹中的所有文件
            for file in files:
                # 如果文件名包含下划线"_"并且是png文件，则删除
                if '_' in file and file.endswith('.png'):
                    file_path = os.path.join(folder_path, file)
                    try:
                        os.remove(file_path)
                        print(f"已删除文件: {file_path}")
                    except Exception as e:
                        print(f"删除文件 {file_path} 时发生错误: {e}")
        else:
            print(f"文件夹不存在: {folder_path}")

def delete_images_without_underscore(parent_folder):
    """
    删除文件夹中没有下划线的图片文件（0, 45, 90 文件夹中的图片）
    """
    subfolders = ['0', '45', '90']
    
    for subfolder in subfolders:
        # 获取当前子文件夹的路径
        folder_path = os.path.join(parent_folder, subfolder)
        
        # 确保该路径是一个有效的文件夹
        if os.path.isdir(folder_path):
            # 获取当前文件夹中的所有文件
            files = os.listdir(folder_path)
            
            # 遍历文件夹中的所有文件
            for file in files:
                # 如果文件名不包含下划线"_"并且是png文件，则删除
                if '_' not in file and file.endswith('.png'):
                    file_path = os.path.join(folder_path, file)
                    try:
                        os.remove(file_path)
                        print(f"已删除文件: {file_path}")
                    except Exception as e:
                        print(f"删除文件 {file_path} 时发生错误: {e}")
        else:
            print(f"文件夹不存在: {folder_path}")

def resize_templates(template_folder, scale_factors=[1,1.5,1.8,2,2.2,2.5,3]):
    # 遍历角度文件夹
    angles = [0,  45,  90]
    for angle in angles:
        # 获取每个角度文件夹的路径
        angle_folder = os.path.join(template_folder, str(angle))
        
        if not os.path.exists(angle_folder):
            print(f"文件夹 {angle_folder} 不存在，跳过")
            continue

        # 获取文件夹中的所有图像文件
        image_files = [f for f in os.listdir(angle_folder) if f.endswith('.png')]
        
        for image_file in image_files:
            image_path = os.path.join(angle_folder, image_file)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # 直接加载为灰度图像

            if image is None:
                print(f"无法加载图像 {image_path}，跳过")
                continue

            # 遍历所有的缩放因子
            for scale_factor in scale_factors:
                # 为缩放后的图像创建一个新的文件名
                new_image_name = f"{os.path.splitext(image_file)[0]}_x{scale_factor}.png"
                save_path = os.path.join(angle_folder, new_image_name)

                # 调用函数缩放图像并保存
                resize_and_save_image(image, scale_factor, save_path)
                print(f"保存图像: {save_path}")

# 设置原模板文件夹路径
src_folder = ROOT / 'functions/rotate/ori_templates'
# 设置生成模板文件夹路径
template_folder = ROOT / 'functions/rotate/templates'


# 调用函数，参数分别是原始图片所在文件夹和目标文件夹
process_images(src_folder, template_folder)

# 删除之前的文件
delete_images_with_underscore(template_folder)

# 执行缩放操作
resize_templates(template_folder)

# 删除没有下划线的图片
delete_images_without_underscore(template_folder)
