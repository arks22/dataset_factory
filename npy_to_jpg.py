import numpy as np
import argparse
import os
import cv2 

# 単体のnpyをjpgに変換
# デバッグなどに使う

parser = argparse.ArgumentParser(description='convert fits to npy file')
parser.add_argument('source', type=str)
parser.add_argument('--name', type=str)
args = parser.parse_args()

if not args.name:
    args.name = os.path.splitext(os.path.basename(args.source))[0] + '.jpg'

img = np.load(args.source)
img = (img * 255).astype(np.uint8)
cv2.imwrite(args.name, img)
