import numpy as np
import argparse
import os
import cv2
from tqdm import tqdm
import skvideo.io

# 動画予測形式のnpyをmp4に変換
# sourceのshapeは(データ数, フレーム数, 高さ, 幅, チャンネル数)
# データ数とチャンネルの次元はあってもなくても良い
# 特に何かに使うわけではない

parser = argparse.ArgumentParser(description='visualize video prediction datasets')
parser.add_argument('source', type=str)
parser.add_argument('--name', type=str, default='sample')
parser.add_argument('--num_samples', type=int, default=1)
parser.add_argument('--no_data_axis', action='store_true', help='データ数の次元が存在しない場合に指定')
parser.add_argument('--no_channel_axis', action='store_true', help='チャンネルの次元が存在しない場合に指定')
parser.add_argument('--MAU_fmt', action='store_false', help='MAUから出力される形式の場合に指定')
args = parser.parse_args()

def make_video(video_data, video_name):
    writer = skvideo.io.FFmpegWriter(video_name, inputdict={'-r':'1'}, outputdict={'-r':'1','-pix_fmt':'yuv420p','-vcodec': 'libx264'})
    writer.writeFrame(video_data)
    writer.close()


data = np.load(args.source)

# チャンネルの次元が存在しない場合は追加
if args.no_channel_axis:
    data = data[..., np.newaxis]

# チャンネルの次元が1の場合はvideo writerの仕様に合わせて3倍にする
if data.shape[-1] == 1:
    data = np.repeat(data, 3, axis=-1)
elif data.shape[-1] != 3:
    raise ValueError('channel axis must be 1 or 3 but got {}'.format(data.shape[-1]))

# データの次元がない場合はデータの先頭に列1の次元を追加
if args.no_data_axis:
    data = data[np.newaxis, ...]

# MAUの出力形式の場合はフレームとデータの次元を入れ替える
print(data.shape)
if args.MAU_fmt:
    data = data.transpose(1, 0, 2, 3, 4)
    #0-10000に正規化されているので0-255に戻す
    data = data / 10000 * 255

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
    video_name = f'{args.name}_{i+1}.mp4'
    video_data = data[:, i].reshape(len_seq,height,width,3) # videowriterの仕様に合わせて次元を並び替え
    make_video(video_data, video_name)