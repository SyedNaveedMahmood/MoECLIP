import os
import json
import math
import random
import torch
from torch.utils.data import Dataset
from utils import AddGaussianNoise
from torchvision import transforms
from PIL import Image
from .constants import CLASS_NAMES, DATA_PATH, DOMAINS


class BaseDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        meta_path: str,
        img_size: int,
        text: bool = False,
    ):
        self.data_path = data_path
        self.img_size = img_size
        self.text = text
        self.meta = []
        self.full_shot = "full-shot" in meta_path
        with open(meta_path, "r") as f:
            for line in f:
                self.meta.append(json.loads(line))

        self.transforms_list = [
            transforms.RandomApply(
                [transforms.RandomRotation(degrees=math.degrees(math.pi / 6))], p=0.5
            ),
            transforms.RandomApply(
                [transforms.RandomAffine(degrees=0, translate=(0.15, 0.15))], p=0.5
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
        ]

        transform_x = []
        if not text:
            transform_x.append(
                transforms.RandomApply([transforms.ColorJitter(brightness=0.5)], p=0.7)
            )
            transform_x.append(
                transforms.RandomApply([transforms.ColorJitter(contrast=0.5)], p=0.7)
            )
            transform_x.append(
                transforms.RandomApply([transforms.ColorJitter(saturation=0.5)], p=0.7)
            )
        self.transform_x = transforms.Compose(
            transform_x
            + [
                transforms.Resize((img_size, img_size), Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711),
                ),
            ],
        )
        self.transform_mask = transforms.Compose(
            [
                transforms.Resize((img_size, img_size), Image.NEAREST),
                transforms.ToTensor(),
            ]
        )

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        meta = self.meta[idx]
        data_path = self.data_path
        img_path = os.path.join(data_path, meta["image_path"])
        img = Image.open(img_path).convert("RGB")

        img = self.transform_x(img)
        if meta["label"]:
            mask_path = os.path.join(data_path, meta["mask_path"])
            mask = Image.open(mask_path).convert("L")
            mask = self.transform_mask(mask)
            mask = (mask != 0).float()
        else:
            mask = torch.zeros([1, self.img_size, self.img_size])

        random_transform = transforms.Compose(self.transforms_list)
        transform_tensor = torch.cat([img, mask], dim=0)
        assert transform_tensor.shape[0] == 4
        transform_tensor = random_transform(transform_tensor)
        img = transform_tensor[0:3, :, :]
        mask = transform_tensor[3:4, :, :]

        inputs = {
            "image": img,
            "mask": mask,
            "label": torch.tensor(meta["label"]).to(torch.int64),
            "file_name": meta["image_path"],
            "class_name": meta["class_name"],
        }
        return inputs


class BaseSingleClassDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        meta_path: str,
        img_size: int,
        class_name: str,
        logger=None,
    ):

        assert class_name is not None, "class_name should be provided"
        self.data_path = data_path
        self.img_size = img_size
        self.meta = []
        with open(meta_path, "r") as f:
            for line in f:
                m = json.loads(line.strip())
                if m["class_name"] == class_name:
                    self.meta.append(m)

        self.transform_x = transforms.Compose(
            [
                transforms.Resize((img_size, img_size), Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize( 
                    mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )
        self.transform_mask = transforms.Compose(
            [
                transforms.Resize((img_size, img_size), Image.NEAREST),
                transforms.ToTensor(),
            ]
        )

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        meta = self.meta[idx]
        img_path = os.path.join(self.data_path, meta["image_path"])
        img = Image.open(img_path).convert("RGB")
        img = self.transform_x(img)
        if meta["label"]:

            if "mask_path" in meta and meta["mask_path"]:

                mask_path = os.path.join(self.data_path, meta["mask_path"])
                mask = Image.open(mask_path).convert("L")
                mask = self.transform_mask(mask)
                mask = (mask != 0).float()
            else:

                mask = torch.zeros([1, self.img_size, self.img_size])

        else:
            mask = torch.zeros([1, self.img_size, self.img_size])
        inputs = {
            "image": img,
            "mask": mask,
            "label": meta["label"],
            "file_name": meta["image_path"],
            "class_name": meta["class_name"],
        }
        return inputs


class PairedBaseDataset(BaseDataset):
    """BaseDataset + a registered thermal channel for MoE-TwinCLIP.

    Thermal files live under a parallel root (data/<name>_T) and mirror the
    RGB relative paths exactly. The thermal image is concatenated into the
    joint geometric transform tensor (img 3ch + mask 1ch + thermal 1ch = 5ch)
    so rotation/affine/flip augmentations stay perfectly registered.
    """

    def __init__(
        self,
        data_path: str,
        thermal_path: str,
        meta_path: str,
        img_size: int,
        text: bool = False,
    ):
        super().__init__(data_path, meta_path, img_size, text)
        self.thermal_root = thermal_path
        self.transform_thermal = transforms.Compose(
            [
                transforms.Resize((img_size, img_size), Image.BICUBIC),
                transforms.ToTensor(),
            ]
        )

    def __getitem__(self, idx):
        meta = self.meta[idx]
        img_path = os.path.join(self.data_path, meta["image_path"])
        pil_img = Image.open(img_path).convert("RGB")
        img = self.transform_x(pil_img)

        # thermal counterpart (grayscale, single channel)
        thermal_file = os.path.join(self.thermal_root, meta["image_path"])
        if os.path.exists(thermal_file):
            th_img = Image.open(thermal_file).convert("L")
        else:
            th_img = Image.new("L", pil_img.size)
        th = self.transform_thermal(th_img)  # (1, H, W)

        if meta["label"]:
            mask_path = os.path.join(self.data_path, meta["mask_path"])
            mask = Image.open(mask_path).convert("L")
            mask = self.transform_mask(mask)
            mask = (mask != 0).float()
        else:
            mask = torch.zeros([1, self.img_size, self.img_size])

        random_transform = transforms.Compose(self.transforms_list)
        transform_tensor = torch.cat([img, mask, th], dim=0)  # (5, H, W)
        assert transform_tensor.shape[0] == 5
        transform_tensor = random_transform(transform_tensor)
        img = transform_tensor[0:3, :, :]
        mask = transform_tensor[3:4, :, :]
        th = transform_tensor[4:5, :, :]

        inputs = {
            "image": img,
            "thermal": th,
            "mask": mask,
            "label": torch.tensor(meta["label"]).to(torch.int64),
            "file_name": meta["image_path"],
            "class_name": meta["class_name"],
        }
        return inputs


def get_dataset(
    dataset_name: str,
    img_size: int,
    training_mode: str,
    shot: int = -1,
    stage: str = "train",
    logger=None,
    use_thermal: bool = False,
):
    if "Med" not in dataset_name:
        assert dataset_name in DATA_PATH, (
            f"Dataset {dataset_name} not found; available datasets: {list(DATA_PATH.keys())}"
        )

    if stage == "train":
        if training_mode == "few_shot":
            assert shot > 0, "shot should be positive"
            meta_path = os.path.join(
                "./dataset/metadata", dataset_name, f"{shot}-shot.jsonl"
            )
        else:
            meta_path = os.path.join(
                "./dataset/metadata", dataset_name, "full-shot.jsonl"
            )

        data_path = DATA_PATH[dataset_name.split("-")[0]]
        text_dataset = None
        image_dataset = None
        if use_thermal:
            from .constants import THERMAL_PATH

            thermal_path = THERMAL_PATH[dataset_name.split("-")[0]]
            text_dataset = PairedBaseDataset(
                data_path, thermal_path, meta_path, img_size, text=True
            )
            image_dataset = PairedBaseDataset(
                data_path, thermal_path, meta_path, img_size, text=True
            )
        else:
            text_dataset = BaseDataset(data_path, meta_path, img_size, text=True)
            image_dataset = BaseDataset(data_path, meta_path, img_size, text=True)
        return text_dataset, image_dataset
    elif stage == "test":
        meta_path = os.path.join("./dataset/metadata", dataset_name, "full-shot.jsonl")
        class_names = CLASS_NAMES[dataset_name]
        datasets = {}
        for class_name in class_names:
            image_dataset = BaseSingleClassDataset(
                data_path=DATA_PATH[dataset_name],
                meta_path=meta_path,
                img_size=img_size,
                class_name=class_name,
                logger=logger,
            )
            datasets[class_name] = image_dataset
        return datasets
    elif stage == "visualize":
        class_names = CLASS_NAMES[dataset_name]
        meta_path = os.path.join("./dataset/metadata", dataset_name, "full-shot.jsonl")
        datasets = {}
        for class_name in class_names:
            image_dataset = BaseSingleClassDataset(
                data_path=DATA_PATH[dataset_name],
                meta_path=meta_path,
                img_size=img_size,
                class_name=class_name,
                logger=None,
            )
            datasets[class_name] = image_dataset
        return datasets
    else:
        raise ValueError(f"stage {stage} not found; available stages: train, test")