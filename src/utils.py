import PIL
import numpy as np
import torch
from torchvision import transforms as transforms


# Tensor and PIL utils
def checkin(img, out_path):
    save_img(img, str(out_path))
    return out_path


def save_img(img, file_name):
    img = np.transpose(img, (1, 2, 0))
    img = np.clip(img, 0, 1)
    img = np.uint8(img * 254)
    pimg = PIL.Image.fromarray(img, mode="RGB")
    pimg.save(file_name)


def pil_resize_long_edge_to(pil, trg_size):
    short_w = pil.width < pil.height
    ar_resized_long = (trg_size / pil.height) if short_w else (trg_size / pil.width)
    resized = pil.resize((int(pil.width * ar_resized_long), int(pil.height * ar_resized_long)), PIL.Image.BICUBIC)
    return resized


def np_to_pil(npy):
    return PIL.Image.fromarray(npy.astype(np.uint8))


def pil_to_np(pil):
    return np.array(pil)


def np_to_tensor(npy, space):
    if space == 'vgg':
        return np_to_tensor_correct(npy)
    return (torch.Tensor(npy.astype(np.float) / 127.5) - 1.0).permute((2, 0, 1)).unsqueeze(0)


def np_to_tensor_correct(npy):
    pil = np_to_pil(npy)
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                    ])
    return transform(pil).unsqueeze(0)


def rgb_to_yuv(rgb):
    C = (
        torch.Tensor(
            [[0.577350, 0.577350, 0.577350], [-0.577350, 0.788675, -0.211325], [-0.577350, -0.211325, 0.788675]]
        ).to(rgb.device)
     )
    yuv = torch.mm(C, rgb)
    return yuv


def get_image_augmentation(use_normalized_clip):
    augment_trans = transforms.Compose([
        transforms.RandomPerspective(fill=1, p=1, distortion_scale=0.5),
        transforms.RandomResizedCrop(224, scale=(0.7, 0.9)),
    ])

    if use_normalized_clip:
        augment_trans = transforms.Compose([
            transforms.RandomPerspective(fill=1, p=1, distortion_scale=0.5),
            transforms.RandomResizedCrop(224, scale=(0.7, 0.9)),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ])
    return augment_trans