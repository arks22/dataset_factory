import os
import sys
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
from astropy.io import fits

# (3) FITS(.fits) -> ndarrayバイナリファイル(.npy)　
# チャンネルの次元も追加している
# 27000枚で約2時間

parser = argparse.ArgumentParser(description='convert 1 fits file to 1 npy file')
parser.add_argument('src_dir', type=str)
parser.add_argument('--dst_dir', type=str, default='.')
args = parser.parse_args()

fits_files = sorted(glob(args.src_dir + '/*'))
error_count = 0

for file in tqdm(fits_files):
    try: 
        fits_data = fits.open(file)
        img = np.array(fits_data[1].data)

    except KeyboardInterrupt:
        exit(1)
    except:
        print('fits loading error')
        print('skip:', file)
        error_count += 1
        continue

    img = img[..., np.newaxis]
    basename_without_ext = os.path.splitext(os.path.basename(file))[0]
    np.save(os.path.join(args.dst_dir,basename_without_ext), img)

print(f'load error count: {error_count}')
