import subprocess
import sys
import os
from pathlib import Path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

def run_script(script_path):
    """运行一个 Python 脚本。"""
    try:
        print('\n')
        print("###############################################################################################")
        print(f"正在运行脚本: {script_path}")
        print('\n')
        subprocess.check_call([sys.executable, script_path])  # 使用 sys.executable 确保使用相同的 Python 解释器
    except subprocess.CalledProcessError as e:
        print(f"运行 {script_path} 时发生错误: {e}")
        sys.exit(1)  # 如果出错，则退出程序

if __name__ == "__main__":
    # 使用相对路径指定要执行的 .py 文件
    scripts = [
        ROOT / "functions/yolo/detect.py",  # yolo检测
        ROOT / "functions/rotate/make_template.py",  # 生成模板
        ROOT / "functions/rotate/rotate_image.py",  # rotate旋转
        ROOT / "functions/ocr/ocr_recognize.py",  # ocr识别
        ROOT / "functions/show/show_result.py",  # 识别结果展示
    ]

    for script in scripts:
        run_script(script)
    
    print("所有脚本执行成功！")
