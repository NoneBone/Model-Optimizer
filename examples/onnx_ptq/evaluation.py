# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Module to evaluate a device model for the specified task."""

import random
from pathlib import Path
from typing import Final

import torch
from datasets import load_dataset
from timm.data import ImageNetInfo
from tqdm import tqdm

from modelopt.torch._deploy.device_model import DeviceModel

ACCURACY: Final[str] = "accuracy"
TINY_IMAGENET_SYNSET_REMAP: Final[dict[str, str]] = {
    "n02666347": "n02666196",
    "n03373237": "n02190166",
    "n04465666": "n04465501",
    "n04598010": "n04597913",
    "n07056680": "n04067472",
    "n07646821": "n01855672",
    "n07647870": "n03250847",
    "n07657664": "n07579787",
    "n07975909": "n02206856",
    "n08496334": "n02730930",
    "n08620881": "n03976657",
    "n08742578": "n02085620",
    "n12520864": "n02906734",
    "n13001041": "n07734744",
    "n13652335": "n03804744",
    "n13652994": "n02999410",
    "n13719102": "n01945685",
    "n14991210": "n07747607",
}


class ImageNetWrapper(torch.utils.data.Dataset):
    """Wrapper for the ILSVRC/imagenet-1k Hugging Face dataset."""

    def __init__(self, hf_dataset, transform=None, label_names=None):
        """Initialize the wrapper.

        Args:
            hf_dataset: The Hugging Face dataset object.
            transform: Optional transform to apply to images.
            label_names: Optional WordNet label names to map to ImageNet-1K indices.
        """
        self.dataset = hf_dataset
        self.transform = transform
        self.label_indices = None
        if label_names:
            imagenet_label_indices = {
                label_name: index
                for index, label_name in enumerate(ImageNetInfo().label_names())
            }
            self.label_indices = [
                imagenet_label_indices[TINY_IMAGENET_SYNSET_REMAP.get(label_name, label_name)]
                for label_name in label_names
            ]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item["image"]

        # Convert to RGB if needed
        if image.mode != "RGB":
            image = image.convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = item["label"]
        if self.label_indices:
            label = self.label_indices[label]
        return image, label


def evaluate(
    model: torch.nn.Module | DeviceModel,
    transform,
    evaluation_type: str = ACCURACY,
    batch_size=1,
    num_examples=None,
    device="cuda",
    dataset_path="ILSVRC/imagenet-1k",
):
    """Evaluate a model for the given dataset.

    Args:
        model: PyTorch model or DeviceModel to evaluate.
        transform: Transform to apply to the dataset images.
        evaluation_type: Type of evaluation to perform. Currently only accuracy is supported.
        batch_size: Batch size to use for evaluation. Currently only batch_size=1 is supported.
        num_examples: Number of examples to evaluate on. If None, evaluate on the entire dataset.
        device: Device to run evaluation on. Supported devices: "cpu" and "cuda". Defaults to "cuda".
        dataset_path: HF dataset card or local path to the imagenet dataset. Defaults to "ILSVRC/imagenet-1k".
    Returns:
        The evaluation result.
    """

    if Path(dataset_path).is_dir():
        dataset = load_dataset(dataset_path, split="valid")
        label_names = dataset.features["label"].names
        val_dataset = ImageNetWrapper(dataset, transform=transform, label_names=label_names)
    else:
        # Load imagenet-1k from Hugging Face
        dataset = load_dataset(
            dataset_path,
            split="validation",
            data_files={
                "validation": "data/validation*",
            },
            verification_mode="no_checks",
        )
        val_dataset = ImageNetWrapper(dataset, transform=transform)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=True, num_workers=4
    )

    # TODO: Add support for segmentation tasks.
    if evaluation_type == ACCURACY:
        return evaluate_accuracy(
            model, val_loader, num_examples, batch_size, topk=(1, 5), device=device
        )
    else:
        raise ValueError(f"Unsupported evaluation type: {evaluation_type}")


def evaluate_accuracy(
    model, val_loader, num_examples, batch_size, topk=(1,), random_seed=None, device="cuda"
):
    """Evaluate the accuracy of the model on the validation dataset.

    Args:
        model: Model to evaluate.
        val_loader: DataLoader for the validation dataset.
        num_examples: Number of examples to evaluate on. If None, evaluate on the entire dataset.
        batch_size: Batch size to use for evaluation.
        topk: function support topk accuracy. Return list of accuracy equal to topk length.
            example of usage `top1, top5 = evaluate_accuracy(..., topk=(1,5))`
            `top1, top5, top10 = evaluate_accuracy(..., topk=(1,5,10))`
        random_seed: Random seed to use for evaluation.
        device: Device to run evaluation on. Supported devices: "cpu" and "cuda". Defaults to "cuda".

    Returns:
        The accuracy of the model on the validation dataset.
    """

    if random_seed:
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
        random.seed(random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if isinstance(model, torch.nn.Module):
        model.eval()
        model = model.to(device)
    total = 0
    corrects = [0] * len(topk)
    for _, (inputs, labels) in tqdm(
        enumerate(val_loader),
        total=num_examples // batch_size if num_examples is not None else len(val_loader),
        desc="Evaluation progress: ",
    ):
        if num_examples is not None and total >= num_examples:
            break
        # Forward pass
        if not isinstance(model, torch.nn.Module):
            inputs = [inputs]
        else:
            inputs = inputs.to(device)
        outputs = model(inputs)

        # Calculate accuracy
        outputs = outputs[0] if isinstance(outputs, list) else outputs.data
        labels_size = labels.size(0)
        outputs = outputs[:labels_size]

        total += labels_size

        labels = labels.to(outputs.device)

        for ind, k in enumerate(topk):
            _, predicted = torch.topk(outputs, k, dim=1)
            corrects[ind] += (predicted == labels.unsqueeze(1)).any(dim=1).sum().item()

    res = [100 * corr / total for corr in corrects]
    return res
