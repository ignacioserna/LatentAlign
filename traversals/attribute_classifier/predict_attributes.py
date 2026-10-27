"""Fig. 3 -- facial-attribute probabilities of the traversed images.

ResNet-18 attribute classifier trained with the protocol and code of
Lingenfelter & Hand, "Improving Evaluation of Facial Attribute Prediction
Models", FG 2021 (https://github.com/blingenf/celeba-baselines), on the 47
attribute labels of MAAD-Face (Terhörst et al.), one sigmoid output per
attribute (column order of ``ATTRIBUTES``). Download the weights
(``resnet18_unaligned_pretrain_maadface``, PyTorch state dict, 45 MB, sha256
dbcfe459...6fbf476) from the v1.0.0 release of the repository into this folder.

For every image in ``--images`` it writes one CSV row with the 47 probabilities;
``traversals/fig3_aggregate.py`` averages them per traversal strength (encoded in
the file name as ``..._<strength>.png``). The CSVs used in the paper are in
``data/traversals/classifier_outputs``.

Preprocessing is the original one: images read with OpenCV (BGR), scaled to
[0, 1], resized to 224 x 224, no mean/std normalisation.

    python traversals/attribute_classifier/predict_attributes.py \
        --images results/traversals/arc2face_positive \
        --out data/traversals/classifier_outputs/arc2face_positive.csv
"""
from __future__ import annotations

import argparse
import csv
import glob
import os

import cv2
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from tqdm import tqdm

ATTRIBUTES = [
    "Male", "Young", "Middle_Aged", "Senior", "Asian", "White", "Black",
    "Rosy_Cheeks", "Shiny_Skin", "Bald", "Wavy_Hair", "Receding_Hairline", "Bangs", "Sideburns",
    "Black_Hair", "Blond_Hair", "Brown_Hair", "Gray_Hair", "No_Beard", "Mustache", "5_o_Clock_Shadow",
    "Goatee", "Oval_Face", "Square_Face", "Round_Face", "Double_Chin", "High_Cheekbones", "Chubby",
    "Obstructed_Forehead", "Fully_Visible_Forehead", "Brown_Eyes", "Bags_Under_Eyes", "Bushy_Eyebrows",
    "Arched_Eyebrows", "Mouth_Closed", "Smiling", "Big_Lips", "Big_Nose", "Pointy_Nose", "Heavy_Makeup",
    "Wearing_Hat", "Wearing_Earrings", "Wearing_Necktie", "Wearing_Lipstick", "No_Eyewear",
    "Eyeglasses", "Attractive",
]


class AttributeNN(nn.Module):
    """ResNet-18 whose last layer maps to ``n_labels`` sigmoid outputs."""

    def __init__(self, n_labels: int):
        super().__init__()
        self.resnet = models.resnet18()
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, n_labels, bias=True)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        return self.activation(self.resnet(x))


class ImageFolderFlat(torch.utils.data.Dataset):
    def __init__(self, img_dir: str):
        self.root = img_dir
        self.files = sorted(f for f in glob.glob(os.path.join(img_dir, "*")) if os.path.isfile(f))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        image = cv2.imread(path)  # BGR, as in the original script
        if image is None:
            return None
        image = T.Resize((224, 224))(T.functional.to_tensor(image))
        # "<folder>/<file>", the format of the Filename column of the shipped CSVs
        return image, f"{os.path.basename(os.path.normpath(self.root))}/{os.path.basename(path)}"


def collate(batch):
    batch = [b for b in batch if b is not None]
    return torch.utils.data.dataloader.default_collate(batch)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--images", required=True, help="folder with the traversed images")
    p.add_argument("--weights", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                     "resnet18_unaligned_pretrain_maadface"),
                   help="trained classifier (state dict)")
    p.add_argument("--out", required=True, help="output CSV")
    p.add_argument("--batch-size", type=int, default=500)
    args = p.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = AttributeNN(len(ATTRIBUTES))
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval().to(device)

    loader = torch.utils.data.DataLoader(ImageFolderFlat(args.images), batch_size=args.batch_size, collate_fn=collate)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Filename"] + ATTRIBUTES)
        for batch, names in tqdm(loader):
            with torch.no_grad():
                probs = model(batch.to(device)).cpu().numpy()
            for name, row in zip(names, probs):
                writer.writerow([name, *row.tolist()])
    print("Saved", args.out)


if __name__ == "__main__":
    main()
