import time
import os
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm

from LRTIIS import LRTIIS


def parse_opt():
    parser = argparse.ArgumentParser(description="LRTIIS Model Prediction & FPS Testing")

    # ------------------ 核心模式设置 ------------------
    parser.add_argument('--mode', type=str, default='dir_predict',
                        choices=['predict', 'fps', 'dir_predict'],
                        help='运行模式: predict(单张), fps(测速), dir_predict(批量)')

    # ------------------ 预测配置 ------------------
    parser.add_argument('--count', action='store_true',
                        help='是否进行目标的像素点计数和比例计算 (仅predict模式有效)')

    # 注意：类别通常与模型权重绑定，建议固定在代码中，或者通过配置文件读取
    # 这里为了保持简洁，依旧保留在代码列表里，未做成命令行参数

    # ------------------ 路径配置 ------------------
    parser.add_argument('--dir_origin_path', type=str, default=r"C:\Users\14185\Desktop\CAM\img",
                        help='输入图片文件夹路径 (dir_predict模式)')
    parser.add_argument('--dir_save_path', type=str, default='img_out/',
                        help='输出图片文件夹路径 (dir_predict模式)')

    # ------------------ FPS 测试配置 ------------------
    parser.add_argument('--fps_image_path', type=str, default=r"VOCdevkit_NEU_Seg-main/VOC2007/JPEGImages/000001.jpg",
                        help='用于测试FPS的图片路径')
    parser.add_argument('--test_interval', type=int, default=100,
                        help='FPS测试循环次数')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    opt = parse_opt()

    # 模型实例化 (LRTIIS)
    lrtiis_model = LRTIIS()

    # 固定的类别列表
    name_classes = ["_background_", "cable", "tower", "dog", "boat", "bottle", "bus"]

    # ------------------ 模式 1: 单张预测 ------------------
    if opt.mode == "predict":
        while True:
            img = input('Input image filename:')
            try:
                image = Image.open(img)
            except:
                print('Open Error! Try again!')
                continue
            else:
                r_image = lrtiis_model.detect_image(image, count=opt.count, name_classes=name_classes)
                r_image.show()

    # ------------------ 模式 2: FPS 测试 ------------------
    elif opt.mode == "fps":
        try:
            img = Image.open(opt.fps_image_path)
        except Exception as e:
            print(f"Load FPS image error: {e}")
            exit()

        tact_time = lrtiis_model.get_FPS(img, opt.test_interval)
        print(str(tact_time) + ' seconds, ' + str(1 / tact_time) + 'FPS, @batch_size 1')

    # ------------------ 模式 3: 文件夹批量预测 ------------------
    elif opt.mode == "dir_predict":
        img_names = os.listdir(opt.dir_origin_path)

        if not os.path.exists(opt.dir_save_path):
            os.makedirs(opt.dir_save_path)

        for img_name in tqdm(img_names):
            if img_name.lower().endswith(
                    ('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                image_path = os.path.join(opt.dir_origin_path, img_name)
                image = Image.open(image_path)

                r_image = lrtiis_model.detect_image(image)

                r_image.save(os.path.join(opt.dir_save_path, img_name))

    else:
        raise AssertionError("Please specify the correct mode.")