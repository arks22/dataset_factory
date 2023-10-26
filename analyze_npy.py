import os
import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description='convert fits to npy file')
parser.add_argument('source', type=str)
parser.add_argument('--name', type=str)
args = parser.parse_args()

if not args.name:
    args.name = os.path.splitext(os.path.basename(args.source))[0] + '-hist.png'

img = np.load(args.source)
#img = cv2.resize(img, dsize=(512, 512),interpolation=cv2.INTER_CUBIC)
img = img[..., np.newaxis] #resizeによってチャンネルの次元が消されるので追加
values = np.ravel(img)

print('max: ', np.amax(values))
print('min: ', np.amin(values))
print('mean: ', np.mean(values))

plt.ticklabel_format(style='plain',axis='y')
plt.hist(values, bins=40, range=[-10, 10000], log=False)
plt.savefig(args.name)
