import os
import sys
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm
import argparse
from astropy.io import fits
import matplotlib.pyplot as plt

# リサイズ、標準化、正規化
# 27000枚で2時間くらい
# python3 5_preprocessing.py path/to/source/dir --out_dir= --resize=512 --scaling_method=sqrt

parser = argparse.ArgumentParser(description='Resize, Scaling, Normalize')
parser.add_argument('source', type=str)
parser.add_argument('--out_dir', type=str, default='.')
parser.add_argument('--resize', type=int, required=True)
parser.add_argument('--scaling_method', type=str, required=True)
parser.add_argument('--max', type=int, default=10000)
parser.add_argument('--min', type=int, default=0 )
args = parser.parse_args()

# 負の値は0に切り上げ、最大値は指定の値に切り捨て
def normalize_sqrt(img, min_v, max_v):
    img = np.where(img < min_v, min_v, img)
    img = np.where(img > max_v, max_v, img)
    img -= min_v
    img = (np.sqrt(img) / np.sqrt(max_v - min_v)).astype(np.float64)
    return img

def normalize_simple(img, min_v, max_v):
    img = np.minimum(max_v,np.maximum(min_v,img))
    img = (img / max_v).astype(np.float64)
    return img

def normalize_log(img, min_v, max_v):
    #img = np.where(img < min_v, min_v, img)
    img = np.maximum(min_v, img)
    #img = np.where(img > max_v, max_v, img)
    img = np.minimum(max_v, img)
    img = img - min_v + 1
    img = np.log(img)
    img = (img / np.log(max_v - min_v)).astype(np.float64)
    img = np.minimum(img,1)
    img = np.maximum(img,0)
    return img

def normalize_gamma(img, min_v, max_v):
    gamma = 0.45
    img = np.where(img < min_v, min_v, img)
    img = np.where(img > max_v, max_v, img)
    img = img - min_v
    img = ((img ** gamma) / ((max_v - min_v) ** gamma)).astype(np.float64)
    return img


def normalize_log(img, min_v, max_v):
    img = np.minimum(max_v,np.maximum(min_v,img))  + 1 #負の値対策で1をたす
    img = (np.log(img) / np.log(max_v)).astype(np.float64)
    return img

npy_files = sorted(glob(args.source + '/*'))

for file in tqdm(npy_files):

    img = np.load(file)
    img = cv2.resize(img, dsize=(args.resize, args.resize),interpolation=cv2.INTER_CUBIC)
    img = img[..., np.newaxis]

    if args.scaling_method == 'log':
        img = normalize_log(img, args.min, args.max)
    elif args.scaling_method == 'simple':
        img = normalize_simple(img, args.min, args.max)
    elif args.scaling_method == 'gamma':
        img = normalize_gamma(img, args.min, args.max)
    elif args.scaling_method == 'sqrt':
        img = normalize_sqrt(img, args.min, args.max)

    basename_without_ext = os.path.splitext(os.path.basename(file))[0]
    np.save(os.path.join(args.out_dir,basename_without_ext), img)
