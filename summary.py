import argparse
from torchstat import stat
from nets.v3 import lenet


def parse_opt():
    parser = argparse.ArgumentParser(description="LRTIIS Model Analysis with torchstat")

    parser.add_argument('--input_shape', type=int, nargs='+', default=[224, 224],
                        help='Input image shape: [Height, Width]')
    parser.add_argument('--num_classes', type=int, default=4,
                        help='Number of classes')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    opt = parse_opt()

    # torchstat 默认在 CPU 上运行
    lrtiis_model = lenet(num_classes=opt.num_classes)

    # stat 接收的 input_size 为 (Channels, Height, Width)，不需要 Batch 维度
    stat(lrtiis_model, (3, opt.input_shape[0], opt.input_shape[1]))