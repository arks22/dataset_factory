import os
import sys
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm
import argparse
from astropy.io import fits
import matplotlib.pyplot as plt

# (3) FITSの配列要素を指定の方法で正規化し0~1のfloat64に変換
# 指定サイズにリサイズし
# 1枚のfitsにつき一枚のnpyファイルに保存
# 画像の破損もチェック
# 27000枚で2時間くらい

parser = argparse.ArgumentParser(description='normalize fits')
parser.add_argument('source', type=str)
parser.add_argument('--out_dir', type=str, default='.')
parser.add_argument('--broken_dir', type=str, default='.')
parser.add_argument('--method', type=str, required=True)
parser.add_argument('--max', type=int, default=10000)
parser.add_argument('--min', type=int, default=0 )
parser.add_argument('--resize', type=int, required=True)
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

# 画像の破損をチェックする
def is_image_broken(img, file):
    img = np.minimum(10000,np.maximum(0,img))
    img = (np.sqrt(img) / np.sqrt(10000)) * 255
    img = np.where(img <= 10, 0, img)
    img = np.minimum(img,64)
    img = (np.log10(img + 1) / np.log10(64) * 255).astype(np.uint8)

    limb_zero_count = np.count_nonzero((190 <= len_from_center) & (len_from_center <= 205) & (img == 0))
    limb_size = np.count_nonzero((190 <= len_from_center) & (len_from_center <= 205))

    if limb_zero_count > limb_size * 0.2:
        basename = os.path.splitext(os.path.basename(file))[0] + '.jpg'
        cv2.imwrite(os.path.join(args.broken_dir,basename), img)
        return True
    else:
        return False

# 画像破損判定のための変数
len_from_center = np.zeros((args.resize, args.resize))
center = ((len_from_center.shape[0] - 1) / 2, (len_from_center.shape[1] - 1) / 2)

# len_from_center に中心からの距離を格納
for i in range(len_from_center.shape[0]):
    for j in range(len_from_center.shape[1]):
        len_from_center[i,j] = ((i - center[0]) ** 2 + (j - center[1]) ** 2 ) ** 0.5

fits_files = sorted(glob(args.source + '/*'))

for file in tqdm(fits_files):
    try: 
        fits_data = fits.open(file)
        img = np.array(fits_data[1].data)

    except KeyboardInterrupt:
        exit(1)
    except:
        print('fits loading error')
        print('skip:', file)
        continue

    img = cv2.resize(img, dsize=(args.resize, args.resize),interpolation=cv2.INTER_CUBIC)

    if is_image_broken(img, file):
        print('broken image', file)
    else:
        if args.method == 'log':
            img = normalize_log(img, args.min, args.max)
        elif args.method == 'simple':
            img = normalize_simple(img, args.min, args.max)
        elif args.method == 'gamma':
            img = normalize_gamma(img, args.min, args.max)
        elif args.method == 'sqrt':
            img = normalize_sqrt(img, args.min, args.max)

        img = img[..., np.newaxis]
        basename_without_ext = os.path.splitext(os.path.basename(file))[0]
        np.save(os.path.join(args.out_dir,basename_without_ext), img)
