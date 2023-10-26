import numpy as np
import argparse
import os
import cv2
from tqdm import tqdm
import skvideo.io

# 動画予測形式のnpyをmp4に変換
# 特に何かに使うわけではない

parser = argparse.ArgumentParser(description='visualize video prediction datasets')
parser.add_argument('source', type=str)
parser.add_argument('--name', type=str, default='test')
parser.add_argument('--num_samples', type=int, default=100)
args = parser.parse_args()

data = np.load(args.source)

if data.ndim == 4:
    data = data[..., np.newaxis]

shape = data.shape
len_seq  = shape[0]
len_data = shape[1]
height   = shape[2]
width    = shape[3]
channel  = shape[4]

if len_data < args.num_samples:
    raise ValueError('num_samples must be smaller than data length')

print('{} data * {} frames * {} px * {} px * {} channel'.format(len_data, len_seq, height, width, channel))

for i in tqdm(range(args.num_samples)):
    if i > len_data: break
    video_name = str(args.name) + '_' + str(i+1) + '.mp4'
    video = np.repeat(data[:,i], 3).reshape(len_seq,height,width,3)
    writer = skvideo.io.FFmpegWriter(video_name, inputdict={'-r':'1'}, outputdict={'-r':'1','-pix_fmt':'yuv420p','-vcodec': 'libx264'})
    writer.writeFrame(video)
    writer.close()
