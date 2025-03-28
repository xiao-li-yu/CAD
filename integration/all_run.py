import subprocess
import sys
import os
from pathlib import Path
import time

# 记录总的开始时间
start_time = time.time()

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

# 用来存储每个脚本的执行时长
script_durations = {}

def run_script(script_path):
    """运行一个 Python 脚本并记录其运行时长。"""
    try:
        print('\n')
        print("###############################################################################################")
        print(f"正在运行脚本: {script_path}")
        print('\n')

        # 记录单个脚本的开始时间
        script_start_time = time.time()

        # 执行脚本
        subprocess.check_call([sys.executable, script_path])  # 使用 sys.executable 确保使用相同的 Python 解释器

        # 记录单个脚本的结束时间
        script_end_time = time.time()

        # 计算单个脚本的运行时间
        script_duration = script_end_time - script_start_time
        script_durations[script_path.name] = script_duration  # 保存每个脚本的运行时间

        print(f"脚本 {script_path} 执行时长: {script_duration:.4f}秒")
        
    except subprocess.CalledProcessError as e:
        print(f"运行 {script_path} 时发生错误: {e}")
        sys.exit(1)  # 如果出错，则退出程序

if __name__ == "__main__":
    # 使用相对路径指定要执行的 .py 文件
    scripts = [
        ROOT / "functions/yolo/detect.py",  # yolo检测
        #ROOT / "functions/rotate/make_template.py",  # 生成模板
        ROOT / "functions/rotate/rotate_image.py",  # rotate旋转
        ROOT / "functions/ocr/ocr_recognize.py",  # ocr识别
        ROOT / "functions/show/show_result.py",  # 识别结果展示
    ]
    
    # 遍历所有脚本并运行
    for script in scripts:
        run_script(script)
    
    # 记录总的结束时间
    total_end_time = time.time()
    
    # 计算总运行时间
    total_duration = total_end_time - start_time

    # 统一输出所有脚本的时间统计
    print("##########################################################")
    print("\n所有脚本执行完毕！")
    print("\n各个脚本的执行时长统计：")
    for script_name, duration in script_durations.items():
        print(f"{script_name}: {duration:.4f}秒")
    print("##########################################################")
    print(f"总运行时长: {total_duration:.4f}秒")
    print("##########################################################")
