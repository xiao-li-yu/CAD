import cv2
import os
import matplotlib.pyplot as plt

import sys
from pathlib import Path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]  
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

def display_image_with_annotations(root_path):
    """显示root路径下的data/images中的图片并在其上框选区域,显示标注数据"""
    
    # 图片路径
    image_path = os.path.join(root_path, 'data/images', os.listdir(os.path.join(root_path, 'data/images'))[0])
    image = cv2.imread(image_path)

    # 获取标注文件路径
    label_file_path = os.path.join(root_path, 'result/labels', os.listdir(os.path.join(root_path, 'result/labels'))[0])
    #print("Label file path:", label_file_path)  # 打印路径以检查是否正确
    
    # 读取标注数据
    with open(label_file_path, 'r',encoding='utf-8') as f:
        lines = f.readlines()

    # 遍历每一行标注
    for line in lines:
        # 拆分每行数据
        data = line.strip().split()
        
        # 提取各个数据
        class_id = int(data[0])  # 类别ID
        center_x = float(data[1])  # 归一化后的中心点x
        center_y = float(data[2])  # 归一化后的中心点y
        width = float(data[3])  # 归一化后的宽度
        height = float(data[4])  # 归一化后的高度
        recognition_value = data[5]  # 识别的结果值
        rotation_angle = int(data[6])  # 旋转角度
        
        # 获取图像的宽高
        img_height, img_width, _ = image.shape
        
        # 计算框的位置和尺寸
        x_min = int((center_x - width / 2) * img_width)
        y_min = int((center_y - height / 2) * img_height)
        x_max = int((center_x + width / 2) * img_width)
        y_max = int((center_y + height / 2) * img_height)
        
        # 绘制矩形框
        color = (0, 255, 0)  # 绿色框，颜色可以自定义
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)

        # 显示类别ID和识别结果文本
        label_text = f"{recognition_value}  {rotation_angle}"

        # 在框的右侧显示文本
        text_x = x_max + 10  # 文本的x位置，稍微偏右
        text_y = y_min + 20  # 文本的y位置，稍微偏下，确保不会与框重叠
        cv2.putText(image, label_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # 使用matplotlib显示图像，以便更好地显示颜色
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')  # 不显示坐标轴
    plt.show()


display_image_with_annotations(ROOT)
