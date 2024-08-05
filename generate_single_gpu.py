import os
import json
from argparse import ArgumentParser
from omegaconf import OmegaConf
from tqdm import trange
import copy

import torch

from models.maskdit import Precond_models
from sample_single_gpu import generate_with_net
from utils import parse_float_none, parse_int_list

def generate(args):
    if args.tf32: # True: fast but may lead to some small numerical differences
        tf32 = True
        torch.backends.cudnn.allow_tf32 = bool(tf32)
        torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
        torch.set_float32_matmul_precision('high' if tf32 else 'highest')
        print(f"Fast inference mode is enabled🏎️🏎️🏎️. TF32: {tf32}")
    else:
        print("Fast inference mode is disabled🐢🐢🐢, you may enable it by passing the '--tf32' flag!")

    config = OmegaConf.load(args.config)
    label_dict = json.load(open(args.label_dict, 'r'))

    # Use all available GPUs
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Setup model
    model = Precond_models[config.model.precond](
        img_resolution=config.model.in_size,
        img_channels=config.model.in_channels,
        num_classes=config.model.num_classes,
        model_type=config.model.model_type,
        use_decoder=config.model.use_decoder,
        mae_loss_coef=config.model.mae_loss_coef,
        pad_cls_token=config.model.pad_cls_token,
        use_encoder_feat=config.model.self_cond,
    ).to(device)

    # Wrap model with DataParallel
    model.eval()
    model = torch.compile(model)

    print(f"{config.model.model_type} ((use_decoder: {config.model.use_decoder})) Model Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f'extras: {model.model.extras}, cls_token: {model.model.cls_token}')

    ckpt = torch.load(args.ckpt_path, map_location=device)
    model.load_state_dict(ckpt['ema'])

    # Create a tqdm progress bar for classes
    class_pbar = trange(config.model.num_classes, desc="Class Progress")
    print(f"Generation starts🔥🔥🔥!")

    for class_idx in class_pbar:
        class_label = label_dict[str(class_idx)][1]
        class_pbar.set_description(f"Class {class_label}")

        # Setup directory for this class
        sample_dir = os.path.join(args.results_dir, class_label)
        os.makedirs(sample_dir, exist_ok=True)
        args.outdir = sample_dir

        # Modify args for this specific class
        args_copy = copy.deepcopy(args)
        args_copy.class_idx = class_idx
        # args_copy.num_expected = args.images_per_class

        # Generate images for this class
        total_images = args.num_images_per_class
        for i in range(0, total_images, args.max_batch_size):
            print(f"Generating images {i} to {min(i + args.max_batch_size, total_images)} for class {class_label}")
            generate_with_net(args_copy, model, device, class_idx=class_idx)

    print(f'Sampling done!')
  

if __name__ == '__main__':
    parser = ArgumentParser('Sample from a trained model')
    parser.add_argument('--config', type=str, required=True, help='path to config file')
    parser.add_argument('--label_dict', type=str, default='assets/imagenet_label.json', help='path to label dict')
    parser.add_argument("--results_dir", type=str, default="samples", help='path to save samples')
    parser.add_argument('--ckpt_path', type=str, default=None, help='path to ckpt')

    # sampling
    parser.add_argument('--seeds', type=parse_int_list, default='100-131', help='Random seeds (e.g. 1,2,5-10)')
    parser.add_argument('--subdirs', action='store_true', help='Create subdirectory for every 1000 seeds')
    parser.add_argument("--cfg_scale", type=parse_float_none, default=None, help='None = no guidance, by default = 4.0')

    parser.add_argument('--num_steps', type=int, default=40, help='Number of sampling steps')
    parser.add_argument('--S_churn', type=int, default=0, help='Stochasticity strength')
    parser.add_argument('--solver', type=str, default=None, choices=['euler', 'heun'], help='Ablate ODE solver')
    parser.add_argument('--discretization', type=str, default=None, choices=['vp', 've', 'iddpm', 'edm'], help='Ablate ODE solver')
    parser.add_argument('--schedule', type=str, default=None, choices=['vp', 've', 'linear'], help='Ablate noise schedule sigma(t)')
    parser.add_argument('--scaling', type=str, default=None, choices=['vp', 'none'], help='Ablate signal scaling s(t)')
    parser.add_argument('--pretrained_path', type=str, default='assets/autoencoder_kl.pth', help='Autoencoder ckpt')

    parser.add_argument('--max_batch_size', type=int, default=10, help='Maximum batch size per GPU during sampling')
    parser.add_argument('--num_expected', type=int, default=32, help='Number of images to use')

    parser.add_argument('--tf32', action='store_true', help='Enable fast inference mode')
    parser.add_argument('--num_images_per_class', type=int, default=50, help='Number of images to generate per class')

    args = parser.parse_args()

    generate(args)