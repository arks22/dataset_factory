import os
from glob import glob
from tqdm import tqdm
import cv2
import argparse
import numpy as np
import shutil


# 画像の破損をチェックする
# ギャンギャンに彩度(?)をあげて、リム部分の要素を見て判断。
# 破損していた場合該当の画像を削除。
# 27000枚で50分くらい

parser = argparse.ArgumentParser(description='Validate images')
parser.add_argument('source', type=str)
parser.add_argument('--broken_dir', type=str)
parser.add_argument('--out_dir', type=str)
args = parser.parse_args()


def download(range_head, range_end):
    series = 'aia.lev1_euv_12s'
    email  = 'sasaki.10222022@gmail.com'
    client = jsoc.JSOCClient()  

    range_head = parse_time(range_head)
    range_end  = parse_time(range_end)
    response = client.search(a.Time(range_head,range_end),
                             a.jsoc.Series(series),
                             a.jsoc.Segment('image'),
                             a.Wavelength(args.wave*u.AA),
                             a.Sample(1*u.day),
                             a.jsoc.Notify(email))
    print(response)
    if not len(response) == 0:
        requests = client.fetch(response, path=args.source, progress=True)


def is_image_broken(img, file):
    # リサイズ
    # 平方根変換, 255階調に正規化
    # 10以下を0に切り下げ
    # 64以上を64に切り捨て
    # 対数変換
    # 255階調に正規化

    img = cv2.resize(img, dsize=(judge_size, judge_size), interpolation=cv2.INTER_CUBIC)
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


def filename_to_date(filename):
    basename = os.path.basename(filename)
    year = int(basename[17:21])
    month = int(basename[22:24])
    day = int(basename[25:27])
    hour = int(basename[28:30])
    minute = int(basename[30:32])
    second = int(basename[32:34])
    date = datetime(year, month, day, hour, minute, second)

    return date


def handle_image(file):
    # 
    try: 
        img = np.load(file)

        if is_image_broken(img, file):
            print('broken image', file)
            os.remove(file) # 壊れた画像を削除

    except KeyboardInterrupt:
        print('Keyboard Interrupt')
        exit(1)

    except:
        print('loading error', file)
        os.remove(file) # 壊れた画像を削除



def main(args):
    judge_size = 512
    # 画像破損判定のための配列
    len_from_center = np.zeros((judge_size, judge_size))
    center = ((len_from_center.shape[0] - 1) / 2, (len_from_center.shape[1] - 1) / 2)

    # len_from_center に中心からの距離を格納
    for i in range(len_from_center.shape[0]):
        for j in range(len_from_center.shape[1]):
            len_from_center[i,j] = ((i - center[0]) ** 2 + (j - center[1]) ** 2 ) ** 0.5


    npy_files = sorted(glob(args.out_dir + '/*'))

    for file in tqdm(npy_files):
        # 画像を処理
        handle_image(file)



if __name__ == '__main__':
    main(args)
