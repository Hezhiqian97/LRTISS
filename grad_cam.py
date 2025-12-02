import os
import warnings
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import argparse

try:
    from pytorch_grad_cam import (
        GradCAM, GradCAMPlusPlus, XGradCAM,
        HiResCAM, LayerCAM
    )
    from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
    from pytorch_grad_cam.utils.model_targets import SemanticSegmentationTarget
except ImportError:
    try:
        from ccam.grad_cam import (
            GradCAM, GradCAMPlusPlus, XGradCAM,
            HiResCAM, LayerCAM
        )
        from ccam.image import show_cam_on_image, preprocess_image
        from ccam.model_targets import SemanticSegmentationTarget
    except ImportError:
        raise ImportError("Please install grad-cam: pip install grad-cam")

from nets.LRTIIS import LRTIIS as model

warnings.filterwarnings('ignore')
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

CAM_ALGORITHMS = {
    "LayerCAM": LayerCAM,
    "GradCAM++": GradCAMPlusPlus,
    "GradCAM": GradCAM,
    "XGradCAM": XGradCAM,
    "HiResCAM": HiResCAM,
}

def parse_args():
    parser = argparse.ArgumentParser(description='CAM Visualization for Segmentation')
    parser.add_argument('--img-root', type=str, required=True, help='Path to image folder')
    parser.add_argument('--model-path', type=str, required=True, help='Path to .pth model weights')
    parser.add_argument('--save-path', type=str, required=True, help='Path to save results')
    parser.add_argument('--classes', type=str, nargs='+', required=True, help='List of class names (e.g. BW HD PF WR)')
    parser.add_argument('--device', type=str, default='', help='Device to use (cpu/cuda)')
    return parser.parse_args()

def letterbox_image(image, size):
    iw, ih = image.size
    w, h = size
    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)

    image = image.resize((nw, nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128, 128, 128))
    new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
    return new_image

def auto_find_target_layer(model):
    for name, module in reversed(list(model.named_modules())):
        if isinstance(module, torch.nn.Conv2d):
            return [module]
    return [list(model.children())[-1]]

def create_comparison_grid(results_list, save_path, main_title):
    num_imgs = len(results_list)
    cols = 3
    rows = (num_imgs + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4.5))
    fig.suptitle(main_title, fontsize=16, y=0.99)

    if num_imgs == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

    for i, ax in enumerate(axes):
        if i < num_imgs:
            title, img = results_list[i]
            ax.imshow(img)
            ax.set_title(title)
            ax.axis('off')
        else:
            ax.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)

class SegmentationModelOutputWrapper(torch.nn.Module):
    def __init__(self, model):
        super(SegmentationModelOutputWrapper, self).__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)

def process_single_image(img_path, model_wrapper, device, save_base_dir, sem_classes):
    filename_base = os.path.basename(img_path).split('.')[0]

    try:
        img_pil = Image.open(img_path).convert('RGB')
    except Exception as e:
        print(f"Error reading {img_path}: {e}")
        return

    image_resized = letterbox_image(img_pil, [224, 224])
    image_np = np.array(image_resized)
    rgb_img_float = np.float32(image_np) / 255.0

    input_tensor = preprocess_image(rgb_img_float, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if str(device) != 'cpu':
        input_tensor = input_tensor.to(device)

    with torch.no_grad():
        output = model_wrapper(input_tensor)
        prediction = torch.nn.functional.softmax(output, dim=1).cpu()[0].argmax(dim=0).numpy()

    target_indices = [idx for idx in np.unique(prediction) if idx != 0]

    if not target_indices:
        print(f"Skip {filename_base}: Background only.")
        return

    detected_names = [sem_classes[i] for i in target_indices]
    print(f"Detected in {filename_base}: {detected_names}")

    target_layers = auto_find_target_layer(model_wrapper.model)

    for class_idx in target_indices:
        class_name = sem_classes[class_idx]

        class_save_dir = os.path.join(save_base_dir, f"{filename_base}_{class_name}")
        if not os.path.exists(class_save_dir):
            os.makedirs(class_save_dir)

        binary_mask = np.float32(prediction == class_idx)
        targets = [SemanticSegmentationTarget(class_idx, binary_mask)]

        collection_for_grid = [("Original", image_np)]

        for algo_name, AlgoClass in CAM_ALGORITHMS.items():
            try:
                cam_algo = AlgoClass(model=model_wrapper, target_layers=target_layers)
                grayscale_cam = cam_algo(input_tensor=input_tensor, targets=targets)[0, :]
                grayscale_cam[grayscale_cam < 0.2] = 0

                cam_image = show_cam_on_image(rgb_img_float, grayscale_cam, use_rgb=True, image_weight=0.5)

                save_path = os.path.join(class_save_dir, f"{algo_name}.png")
                Image.fromarray(cam_image).save(save_path)

                collection_for_grid.append((algo_name, cam_image))

            except Exception as e:
                print(f"Algo {algo_name} failed: {e}")
            finally:
                if 'cam_algo' in locals(): del cam_algo
                torch.cuda.empty_cache()

        grid_save_path = os.path.join(save_base_dir, f"{filename_base}_{class_name}_GRID.png")
        create_comparison_grid(collection_for_grid, grid_save_path, f"{filename_base} | {class_name}")
        print(f"Saved grid: {grid_save_path}")

    del input_tensor
    torch.cuda.empty_cache()

if __name__ == '__main__':
    args = parse_args()

    sem_classes = args.classes
    num_classes = len(sem_classes)
    print(f"Classes: {num_classes} ({sem_classes})")

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = model(num_classes=num_classes)

    try:
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print("Weights loaded.")
    except Exception as e:
        print(f"Weights loading failed: {e}")
        exit()

    model.to(device)
    model.eval()
    model_wrapper = SegmentationModelOutputWrapper(model)

    image_files = [f for f in os.listdir(args.img_root) if f.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp'))]
    total_imgs = len(image_files)
    print(f"Found {total_imgs} images.\n")

    for i, filename in enumerate(image_files):
        print(f"[{i + 1}/{total_imgs}] Processing: {filename}")
        full_path = os.path.join(args.img_root, filename)
        process_single_image(full_path, model_wrapper, device, args.save_path, sem_classes)

    print("\nDone.")