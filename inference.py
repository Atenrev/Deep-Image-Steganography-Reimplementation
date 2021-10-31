import argparse
from enum import Enum
import os
from PIL import Image

import torch
from torchvision import transforms
from torchvision.utils import save_image

from model import AutoSteganographer


class InferenceMode(Enum):
    HIDE = 'h'
    RECOVER = 'r'


def _parse_args() -> object:
    parser = argparse.ArgumentParser(
        description='Steganography inference parser')

    parser.add_argument('--load_model', type=str,
                        help='model to load and do inference')
    parser.add_argument('--mode', type=str,
                        choices=[InferenceMode.HIDE.value,
                                 InferenceMode.RECOVER.value],
                        help='inference mode. h for hiding, r for recovering')
    parser.add_argument('--im', type=str,
                        help='path of the image used as cover')
    parser.add_argument('--imh', type=str,
                        help='path of the image to hide')
    parser.add_argument('--out', type=str, default="./",
                        help='save path for the output')
    parser.add_argument('--image_size', type=int, default=512,
                        help='image size')

    args = parser.parse_args()
    return args


def main(args) -> None:
    transform = transforms.Compose([
        transforms.Resize([args.image_size, args.image_size]),
        transforms.ToTensor(),
    ])

    model = AutoSteganographer()
    model.load_state_dict(torch.load(args.load_model))

    image_original = Image.open(args.im).convert('RGB')
    image_original = transform(image_original)[None, :]

    if args.mode == InferenceMode.HIDE.value:
        image_to_hide = Image.open(args.imh).convert('RGB')
        image_to_hide = transform(image_to_hide)[None, :]
        result = model.merge(image_original, image_to_hide)
        save_image(result, os.path.join(args.out, 'merged.png'))
    else:
        result = model.revealer(image_original)
        save_image(result, os.path.join(args.out, 'recovered.png'))


if __name__ == "__main__":
    args = _parse_args()
    main(args)
