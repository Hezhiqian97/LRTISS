import os
import argparse
from PIL import Image
from tqdm import tqdm

from LRTIIS import LRTIIS
from utils.utils_metrics import compute_mIoU, show_results


def parse_opt():
    parser = argparse.ArgumentParser(description="LRTIIS mIoU Evaluation")

    # 0: 预测+计算, 1: 仅预测, 2: 仅计算
    parser.add_argument('--miou_mode', type=int, default=0, choices=[0, 1, 2],
                        help='Mode: 0=Predict&Calc, 1=PredictOnly, 2=CalcOnly')

    parser.add_argument('--num_classes', type=int, default=7, help='Number of classes including background')

    parser.add_argument('--voc_path', type=str, default=r'F:\dataset\VOCdevkit_Magnetic',
                        help='Path to VOC dataset root')

    parser.add_argument('--output_dir', type=str, default='miou_out',
                        help='Directory to save results')

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    opt = parse_opt()

    # 这里的类别名称需要和你训练时保持一致
    name_classes = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus"]

    image_ids_path = os.path.join(opt.voc_path, "VOC2007/ImageSets/Segmentation/test.txt")
    gt_dir = os.path.join(opt.voc_path, "VOC2007/SegmentationClass/")
    pred_dir = os.path.join(opt.output_dir, 'detection-results')

    image_ids = open(image_ids_path, 'r').read().splitlines()

    if opt.miou_mode == 0 or opt.miou_mode == 1:
        if not os.path.exists(pred_dir):
            os.makedirs(pred_dir)

        print("Load model.")
        lrtiis_model = LRTIIS()
        print("Load model done.")

        print("Get predict result.")
        for image_id in tqdm(image_ids):
            image_path = os.path.join(opt.voc_path, "VOC2007/JPEGImages", image_id + ".jpg")
            image = Image.open(image_path)
            # 注意：这需要 unet.py 里有 get_miou_png 方法
            image = lrtiis_model.get_miou_png(image)
            image.save(os.path.join(pred_dir, image_id + ".png"))
        print("Get predict result done.")

    if opt.miou_mode == 0 or opt.miou_mode == 2:
        print("Get miou.")
        hist, IoUs, PA_Recall, Precision = compute_mIoU(gt_dir, pred_dir, image_ids, opt.num_classes, name_classes)
        print("Get miou done.")
        show_results(opt.output_dir, hist, IoUs, PA_Recall, Precision, name_classes)