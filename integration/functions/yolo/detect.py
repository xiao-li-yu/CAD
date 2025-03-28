import argparse
import csv
import os
import platform
import shutil
import sys
from pathlib import Path

import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from ultralytics.utils.plotting import Annotator, colors, save_one_box

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (
    LOGGER,
    Profile,
    check_file,
    check_img_size,
    check_imshow,
    check_requirements,
    colorstr,
    cv2,
    increment_path,
    non_max_suppression,
    print_args,
    scale_boxes,
    strip_optimizer,
    xyxy2xywh,
)
from utils.torch_utils import select_device, smart_inference_mode


@smart_inference_mode()
def run(
    weights=ROOT / "functions/yolo/best.pt",  # model path or triton URL
    source=ROOT / "data/images",  # file/dir/URL/glob/screen/0(webcam)
    data=ROOT / "data/mydata.yaml",  # dataset.yaml path
    imgsz=(1024, 1024),  # inference size (height, width)
    conf_thres=0.5,  # confidence threshold
    iou_thres=0.45,  # NMS IOU threshold #!检测框容忍交并比
    max_det=100,  # maximum detections per image
    device="0",  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    view_img=False,  # show results
    save_txt=False,  # save results to *.txt
    save_format=0,  # save boxes coordinates in YOLO format or Pascal-VOC format (0 for YOLO and 1 for Pascal-VOC)
    save_csv=False,  # save results in CSV format
    save_conf=False,  # save confidences in --save-txt labels
    save_crop=False,  # save cropped prediction boxes
    nosave=False,  # do not save images/videos
    classes=None,  # filter by class: --class 0, or --class 0 2 3
    agnostic_nms=False,  # class-agnostic NMS
    augment=False,  # augmented inference
    visualize=False,  # visualize features
    update=False,  # update all models
    project=ROOT / "result",  # save results to project/name
    name="exp",  # save results to project/name
    exist_ok=False,  # existing project/name ok, do not increment
    line_thickness=3,  # bounding box thickness (pixels)
    hide_labels=False,  # hide labels
    hide_conf=False,  # hide confidences
    half=False,  # use FP16 half-precision inference
    dnn=False,  # use OpenCV DNN for ONNX inference
    vid_stride=1,  # video frame-rate stride
):
    source = str(source)
    save_img = not nosave and not source.endswith(".txt")  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(("rtsp://", "rtmp://", "http://", "https://"))
    webcam = source.isnumeric() or source.endswith(".streams") or (is_url and not is_file)
    screenshot = source.lower().startswith("screen")
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / "labels" if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    bs = 1  # batch_size
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(device=device), Profile(device=device), Profile(device=device))
    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim
            if model.xml and im.shape[0] > 1:
                ims = torch.chunk(im, im.shape[0], 0)

        # Inference
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            if model.xml and im.shape[0] > 1:
                pred = None
                for image in ims:
                    if pred is None:
                        pred = model(image, augment=augment, visualize=visualize).unsqueeze(0)
                    else:
                        pred = torch.cat((pred, model(image, augment=augment, visualize=visualize).unsqueeze(0)), dim=0)
                pred = [pred, None]
            else:
                pred = model(im, augment=augment, visualize=visualize)
        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Define the path for the CSV file
        csv_path = save_dir / "predictions.csv"

        def clear_folder(folder_path):
            """Clears all files in a given folder."""
            for filename in os.listdir(folder_path):
                file_path = os.path.join(folder_path, filename)
                try:
                    if os.path.isdir(file_path):
                        shutil.rmtree(file_path)  # Delete subdirectories and their contents
                    else:
                        os.remove(file_path)  # Delete files
                except Exception as e:
                    print(f"Error while deleting file {file_path}: {e}")

        def write_to_csv(image_name, prediction, confidence):
            """Writes prediction data for an image to a CSV file, appending if the file exists."""
            data = {"Image Name": image_name, "Prediction": prediction, "Confidence": confidence}
            file_exists = os.path.isfile(csv_path)
            with open(csv_path, mode="a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=data.keys())
                if not file_exists:
                    writer.writeheader()
                writer.writerow(data)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f"{i}: "
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, "frame", 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / "labels" / p.stem) + ("" if dataset.mode == "image" else f"_{frame}")  # im.txt
            s += "{:g}x{:g} ".format(*im.shape[2:])  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))

            # Clear the content of labels and crops folders before saving new content
            if save_txt:
                clear_folder(str(save_dir / "labels"))
            if save_crop:
                clear_folder(str(save_dir / "crops"))

            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # ! 坐标框信息
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)  # integer class
                    label = names[c] if hide_conf else f"{names[c]}"
                    confidence = float(conf)
                    confidence_str = f"{confidence:.2f}"

                    if save_csv:
                        write_to_csv(p.name, label, confidence_str)

                    if save_txt:  # ! Write to file
                        # 获取四个角点坐标
                        x1, y1, x2, y2 = map(int, xyxy)
                        w_img, h_img = im0.shape[1], im0.shape[0]  # 获取图像宽高

                        # 计算归一化坐标
                        x_center = (x1 + x2) / 2 / w_img
                        y_center = (y1 + y2) / 2 / h_img
                        width = (x2 - x1) / w_img
                        height = (y2 - y1) / h_img

                        # 格式化输出，保留6位小数
                        line = f"{int(cls)} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"

                        # 写入文件
                        with open(f"{txt_path}.txt", "a") as f:
                            f.write(line)

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f"{names[c]} {conf:.2f}")
                        annotator.box_label(xyxy, label, color=colors(c, True))

                    if save_crop:
                        save_one_box(xyxy, imc, file=save_dir / "crops" / names[c] / f"{p.stem}.jpg", BGR=True)

            # Stream results
            im0 = annotator.result()
            if view_img:
                if platform.system() == "Linux" and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == "image":
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix(".mp4"))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                    vid_writer[i].write(im0)

        # Print time (inference-only)
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1e3:.1f}ms")

    # Print results
    t = tuple(x.t / seen * 1e3 for x in dt)  # speeds per image
    LOGGER.info(f"Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}" % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ""
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)


def parse_opt():
#! #######################################################################################################################################
#? 配置区域
    #data的images文件中存放待识别图片
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", nargs="+", type=str, default=ROOT / "functions/yolo/best.pt", help="model path or triton URL") #!这里修改权重文件
    parser.add_argument("--source", type=str, default=ROOT / "data/images", help="file/dir/URL/glob/screen/0(webcam)")
    parser.add_argument("--data", type=str, default=ROOT / "data/mydata.yaml", help="(optional) dataset.yaml path") #!这里修改data参数
    parser.add_argument("--imgsz", "--img", "--img-size", nargs="+", type=int, default=[1024], help="inference size h,w")
    parser.add_argument("--conf-thres", type=float, default=0.5, help="confidence threshold")       #!这里修改置信率 0.5以上会被标注识别出来
    parser.add_argument("--iou-thres", type=float, default=0.1, help="NMS IoU threshold")             #!框的交并比阈值
    parser.add_argument("--max-det", type=int, default=200, help="maximum detections per image")
    parser.add_argument("--device", default="cpu", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--view-img", action="store_true", help="show results")                     #  展示结果
    parser.add_argument("--save-txt", action="store_true",default=True, help="save results to *.txt")            #! 保存框的位置信息
    parser.add_argument(
        "--save-format",
        type=int,
        default=0,
        help="whether to save boxes coordinates in YOLO format or Pascal-VOC format when save-txt is True, 0 for YOLO and 1 for Pascal-VOC",
    )
#! #######################################################################################################################################
#? 功能区域
    parser.add_argument("--save-csv", action="store_true", help="save results in CSV format")  
# 作用：保存检测结果为 CSV 格式的文件  
# 参数类型：布尔值（True/False），使用时加上 --save-csv 选项即启用  

    parser.add_argument("--save-conf", action="store_true", help="save confidences in --save-txt labels")  
# !作用：在 --save-txt 输出的标签文件中保存置信度（confidence）  
# 参数类型：布尔值（True/False），使用 --save-conf 选项即启用  

    parser.add_argument("--save-crop", action="store_true",default=True, help="save cropped prediction boxes")  
# !作用：保存检测到的目标裁剪后的图像  
# 参数类型：布尔值（True/False），使用 --save-crop 选项即启用  

    parser.add_argument("--nosave", action="store_true", default=True,help="do not save images/videos")  
# 作用：不保存检测结果（图像或视频）  
# 参数类型：布尔值（True/False），使用 --nosave 选项即启用  

    parser.add_argument("--classes", nargs="+", type=int, help="filter by class: --classes 0, or --classes 0 2 3")  
# 作用：按照类别 ID 进行筛选，仅检测指定类别的目标  
# 参数类型：整数列表（如 --classes 0 2 3 仅检测类别 0、2、3）  

    parser.add_argument("--agnostic-nms", action="store_true", help="class-agnostic NMS")  
# 作用：使用类别无关的非极大值抑制（NMS），即不同类别的目标也会进行 NMS 处理  
# 参数类型：布尔值（True/False），使用 --agnostic-nms 选项即启用  

    parser.add_argument("--augment", action="store_true", help="augmented inference")  
# 作用：在推理时使用数据增强，提高模型的鲁棒性  
# 参数类型：布尔值（True/False），使用 --augment 选项即启用  

    parser.add_argument("--visualize", action="store_true", help="visualize features")  
# 作用：可视化特征图，以便分析模型内部的特征提取情况  
# 参数类型：布尔值（True/False），使用 --visualize 选项即启用  

    parser.add_argument("--update", action="store_true", help="update all models")  
# 作用：更新所有模型（通常用于重新转换 ONNX 或 TensorRT 格式等）  
# 参数类型：布尔值（True/False），使用 --update 选项即启用  

    parser.add_argument("--project", default=ROOT / "result", help="save results to project/name")  
# 作用：指定结果保存的根目录  
# 参数类型：字符串，默认保存到 runs/detect  

    parser.add_argument("--name", default="", help="save results to project/name")  
# 作用：指定保存结果的文件夹名称，最终路径为 project/name  
# 参数类型：字符串，默认名称为 exp  

    parser.add_argument("--exist-ok", action="store_true", default=True,help="existing project/name ok, do not increment")  
# 作用：允许使用已存在的 project/name 目录，不会自动增加后缀编号（如 exp2）  
# 参数类型：布尔值（True/False），使用 --exist-ok 选项即启用  

    parser.add_argument("--line-thickness", default=1, type=int, help="bounding box thickness (pixels)")  
# !作用：设置检测框的边界线宽度（像素）   

    parser.add_argument("--hide-labels", default=False, action="store_true", help="hide labels")  
# 作用：隐藏检测框中的类别标签  
# 参数类型：布尔值（True/False），使用 --hide-labels 选项即启用  

    parser.add_argument("--hide-conf", default=False, action="store_true", help="hide confidences")  
# 作用：隐藏检测框中的置信度数值  
# 参数类型：布尔值（True/False），使用 --hide-conf 选项即启用  

    parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")  
# 作用：使用 FP16（半精度浮点数）进行推理，提高推理速度并减少显存占用（仅支持某些 GPU）  
# 参数类型：布尔值（True/False），使用 --half 选项即启用  

    parser.add_argument("--dnn", action="store_true", help="use OpenCV DNN for ONNX inference")  
# 作用：使用 OpenCV DNN 进行 ONNX 模型推理，而非默认的 PyTorch 推理方式  
# 参数类型：布尔值（True/False），使用 --dnn 选项即启用  

    parser.add_argument("--vid-stride", type=int, default=1, help="video frame-rate stride")  
# 作用：设置视频帧率步长，即每隔 vid-stride 帧进行一次检测（提高处理长视频的效率）  
# 参数类型：整数，默认值为 1（即逐帧检测）  
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(ROOT / "functions/yolo/requirements.txt", exclude=("tensorboard", "thop"))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)