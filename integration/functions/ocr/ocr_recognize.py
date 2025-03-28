import os
import cv2  # OpenCV用于读取和处理图像
import numpy as np
import matplotlib.pyplot as plt
import re
from paddleocr import PaddleOCR
from pathlib import Path
import sys
FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
import torch
torch.cuda.empty_cache()
# 创建 PaddleOCR 实例
ocr = PaddleOCR(use_angle_cls=False, lang='ch', rec_dict='custom_dict.txt',use_gpu=False)

def process_image_and_save_result(img_path, result_file):
    """处理图像并将 OCR 结果追加到同一个文件的每一行"""
    # 读取图像
    image0 = cv2.imread(str(img_path))

    # 图像预处理：将图像转换为灰度并应用阈值处理，确保数字部分突出
    img_gray = cv2.cvtColor(image0, cv2.COLOR_BGR2GRAY)
    _, img_threshold = cv2.threshold(img_gray, 150, 255, cv2.THRESH_BINARY_INV)  # 调整阈值以提高对比度

    # 使用高斯模糊减少噪声
    img_blurred = cv2.GaussianBlur(img_threshold, (5, 5), 0)

    def get_ocr_results(image):
        """获取 OCR 识别结果并返回文本和置信度，忽略空格，并合并被分割的数字"""
        result = ocr.ocr(image, cls=True)
        if result is None:
            return [], 0  # 如果没有识别结果，返回空列表和置信度为 0
        text_results = []
        highest_confidence = 0
        for line in result:
            if line:  # 检查 line 是否为 None
                for word_info in line:
                    if word_info:  # 检查 word_info 是否为 None
                        text = word_info[1][0].replace(' ', '')  # 去掉空格
                        confidence = word_info[1][1]  # 识别置信度
                        text_results.append((text, confidence))
                        if confidence > highest_confidence:
                            highest_confidence = confidence

        # 后处理：合并分割的数字（如12和5合并为12.5）
        processed_text_results = []
        merged_text = ""
        for i, (text, confidence) in enumerate(text_results):
            if re.match(r'\d', text):  # 当前文本是数字
                if merged_text and '.' in merged_text and re.match(r'\d', text):
                    # 如果前一部分已经有小数点，且当前部分是数字，则合并它们
                    merged_text += text
                elif merged_text and re.match(r'\d', merged_text[-1]) and not '.' in merged_text:
                    # 连续数字合并
                    merged_text += text
                else:
                    if merged_text:
                        processed_text_results.append((merged_text, highest_confidence))
                    merged_text = text
            else:
                if merged_text:
                    processed_text_results.append((merged_text, highest_confidence))
                merged_text = text
        if merged_text:
            processed_text_results.append((merged_text, highest_confidence))

        # 删除以小数点结尾的文本
        if processed_text_results and processed_text_results[-1][0].endswith('.'):
            processed_text_results[-1] = (processed_text_results[-1][0][:-1], highest_confidence)

        return processed_text_results, highest_confidence

    # 获取原图的 OCR 识别结果
    result_texts, original_confidence = get_ocr_results(img_blurred)

    # 如果原图的识别置信度大于0.99，则直接使用原图的结果
    if original_confidence >= 0.99 and result_texts:
        best_result_text = "&".join([text for text, _ in result_texts])
    else:
        best_result_text = "&".join([text for text, _ in result_texts])

    # 如果没有识别到任何结果，写入 "NAN"
    if not best_result_text:
        best_result_text = "NAN"

    # 提取旋转角度从文件名（例如 auimage128_0.png 提取 0， auimage12811_90.png 提取 90）
    match = re.search(r'_(\d+)(?=\.\w+$)', img_path.name)  # 查找以 _ 后跟数字并且是文件扩展名之前的部分
    rotation_angle = match.group(1) if match else "未知"  # 如果找不到旋转角度，使用 "未知"

    # 输出最佳识别结果
    print(f"\n图像: {img_path.name}, 识别结果: {best_result_text}, 旋转角度: {rotation_angle}")

    # 将 OCR 结果写入统一的 result_file 文件中，同时写入旋转角度
    with open(result_file, 'a', encoding='utf-8') as f:  # 'a'模式表示追加
        f.write(f"{best_result_text} {rotation_angle}\n")

def natural_sort_key(text):
    """根据数字进行排序"""
    return [float(t) if t.isdigit() else t.lower() for t in re.split('(\d+)', text)]

def process_images_in_folder(images_folder, result_file):
    """遍历 images_folder 中的所有图像并处理，将结果写入统一的 result_file 文件"""
    # 获取文件夹中的所有文件并按文件名数字顺序排序
    image_files = sorted(images_folder.iterdir(), key=lambda x: natural_sort_key(x.name))
    
    # 遍历排序后的文件
    for img_path in image_files:
        if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:  # 只处理特定格式的图片
            # 处理当前图像并保存结果
            process_image_and_save_result(img_path, result_file)
    print("\n识别结果已写入", img_path, "\\", result_file, "\n")

def merge_txt_files(file1, file2):
    # 打开1.txt和2.txt
    with open(file1, 'r', encoding='utf-8') as f1, open(file2, 'r', encoding='utf-8') as f2:
        lines1 = f1.readlines()
        lines2 = f2.readlines()

    # 确保两个文件行数相同
    if len(lines1) != len(lines2):
        print("Warning: The two files have different number of lines.")
    
    # 合并内容并写回1.txt
    with open(file1, 'w', encoding='utf-8') as f1:
        for i in range(len(lines1)):
            # 合并每行
            merged_line = lines1[i].strip() + ' ' + lines2[i].strip() + "\n"
            f1.write(merged_line)
    
    # 删除2.txt
    import os
    os.remove(file2)
    print("\nlabel文件已被修改\n")


# 获取当前脚本文件所在的目录
images_folder = ROOT / 'result/rotated_crop'  # 图像文件夹
result_file = ROOT / 'result/ocr_results.txt'  # 识别结果文件
label_folder = ROOT / 'result/labels'  # labels文件夹路径
txt_files = list(label_folder.glob('*.txt'))  # 获取其中的txt文件路径
label_file = txt_files[0]  # 赋值
# 处理文件夹中的所有图像，并将结果写入文件
process_images_in_folder(images_folder, result_file)

# 示例调用
merge_txt_files(label_file, result_file)
