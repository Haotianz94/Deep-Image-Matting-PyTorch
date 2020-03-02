import os

import cv2 as cv
import numpy as np
import torch
from torchvision import transforms
from tqdm import tqdm
import sys

from config import device
from data_gen import data_transforms
from utils import ensure_folder


IMG_FOLDER = 'data/image/{}'.format(sys.argv[1])
TRIMAP_FOLDER = 'data/trimap/{}'.format(sys.argv[1])
OUTPUT_FOLDER = 'output/{}'.format(sys.argv[1])
num_split = sys.argv[2]
split_id = sys.argv[3]


if __name__ == '__main__':
    checkpoint = 'BEST_checkpoint.tar'
    checkpoint = torch.load(checkpoint)
    model = checkpoint['model'].module
    model = model.to(device)
    model.eval()

    transformer = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    files = [f for f in os.listdir(IMG_FOLDER) if f.endswith('.jpg')]

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    files = [files[i] for i in range(split_id, len(files), num_split)]

    for file in tqdm(files):

        filename = os.path.join(IMG_FOLDER, file)
        img = cv.imread(filename)
        h, w = img.shape[:2]

        x = torch.zeros((1, 4, h, w), dtype=torch.float)
        image = img[..., ::-1]  # RGB
        image = transforms.ToPILImage()(image)
        image = transformer(image)
        x[0:, 0:3, :, :] = image

        filename = os.path.join(TRIMAP_FOLDER, file)
        print('reading {}...'.format(filename))
        trimap = cv.imread(filename, 0)
        x[0:, 3, :, :] = torch.from_numpy(trimap.copy() / 255.)

        # Move to GPU, if available
        x = x.type(torch.FloatTensor).to(device)

        with torch.no_grad():
            pred = model(x)

        pred = pred.cpu().numpy()
        pred = pred.reshape((h, w))

        pred[trimap == 0] = 0.0
        pred[trimap == 255] = 1.0

        out = (pred.copy() * 255).astype(np.uint8)

        filename = os.path.join(OUTPUT_FOLDER, file)
        cv.imwrite(filename, out)
        print('wrote {}.'.format(filename))