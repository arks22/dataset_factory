import numpy as np
import argparse
import glob
from datetime import datetime, timedelta
from tqdm import tqdm
import cv2
import os

from dataset_utils import filename_to_date

# (4)  指定したディレクトリ内の全てのnpyファイルを一つの動画予測形式のndarrayバイナリファイルに変換
# タイムドリブンでまとめ、画像のない、または破損したデータの含まれる期間はスキップする
# 3分くらい

def find_date_in_range(dates, start_date, end_date):
    """
    開始日と終了日の範囲内に初めて含まれる日付のインデックスを探す関数

    Parameters
    ----------
    dates: list of date
        日付のリスト（dateオブジェクト）
    start_date : date
        開始日（dateオブジェクト）
    end_date : date
        終了日（dateオブジェクト）

    Returns
    -------
    index : int
        範囲内に初めて含まれる日付のインデックス、範囲内の日付がない場合はNone
    """

    # dates内の各日付を開始日と終了日と比較
    for i, date in enumerate(dates):
        if start_date <= date <= end_date:
            return i  # 範囲内の日付が見つかったらそのインデックスを返す

    return None  # 範囲内の日付が見つからなかった場合


def handle_sequence(seq_len, width, channels_n, delta, now_frame):
    seq = np.zeros((seq_len, width, width, channels_n), dtype=np.float64)
    seq_files = [] 

    for channel in range(channels_n): #チャンネルごとに探す
        files = files_list[channel]
        dates = dates_list[channel]

        for j in range(seq_len): 
            start_date_search = now_frame - timedelta(minutes=2) #ダウンロードするわけではないので広めに許容誤差を取る
            end_date_search   = now_frame + timedelta(minutes=2)

            #rangeに収まる壊れていないファイルを探す
            date_index = find_date_in_range(dates, start_date_search, end_date_search)
            if date_index is not None:
                file = files.pop(k)
                dates.pop(k)
                try:
                    img = np.load(file) #npyファイルを読み込み
                except: #外部ファイル操作関連は失敗が多いのでエラーキャッチを用意しておく
                    print('\nLoad failed',file)
                    return None

                seq[j, :, :, channel] = img
                seq_files.append(os.path.basename(file)) #割り当て表のために保存
                #3波長の仕様にあわせて要変更
                now_frame += timedelta(hours=delta) #時間を次に進める

            else: #rangeに収まるファイルがない場合
                print('\nNo image: ',now_frame)
                return None

    return seq, seq_files


def main(args):
    src_dirs = [args.source_1, args.source_2, args.source_3]
    channels_n = 3 # 3波長

    files_list = []
    dates_list = []
    for i in range(channels_n):
        files = sorted(glob.glob(src_dirs[i] + "/*"))
        files_list.append[files]

        dates = []
        for file in files:
            dates.append(filename_to_date(file))

        dates_list.append(sorted(dates))

    start_date = dates_list[0][0]  #一つ目のチャンネルのソース基準で決定
    end_date   = dates_list[0][-1] #一つ目のチャンネルのソース基準で決定
    day_per_seq = (args.delta * args.seq_len / 24)
    data_len = int((end_date - start_date).days // day_per_seq)

    print('start date:', start_date)
    print('end date:', end_date)
    print('----------------------')

    sample_img = np.load(files_list[0][0]) #サンプルのnpyファイルを読み込み
    width = sample_img.shape[1]
    lack_count = 0
    i = 0
    assignment_text = []
    now_seq = start_date
    data = np.zeros((args.seq_len, data_len, width, width, channels_n), dtype=np.float64)


    while now_seq + timedelta(days=day_per_seq) < end_date:
        print('\r', i, '/', data_len - lack_count,end='')

        seq_res = handle_sequence()
        if seq_res is not None:
            data[:, i, :, :, :] = seq_res[0] #欠損がなければdataにseqを追加
            assignment_text.append(f'{seq_res[1]}\n') # 割り当て表のためのテキスト追加
            i += 1
        else:
            lack_count += 1

        now_seq += timedelta(hours=args.delta * args.seq_len) #次のseqの始まりまで日付を進める

    data = data[:, :data_len - lack_count]

    print('----------------------')
    print('lacks: ',lack_count )
    print('----------------------')

    print('data shape:',  data.shape)
    print('data len:',  data.shape[1], 'seqs')
    print('Please input data length')
    print('train: ', end='')
    train_len  = int(input())
    print('val: ', end='')
    val_len  = int(input())
    print('test: ', end='')
    test_len  = int(input())

    train_data = data[:,                     : train_len]
    val_data   = data[:, train_len           : train_len + val_len]
    test_data  = data[:, train_len + val_len : train_len + val_len + test_len]

    np.save(args.dataset_name + '_train', train_data)
    np.save(args.dataset_name + '_val', val_data)
    np.save(args.dataset_name + '_test', test_data)

    print('Data generation finished')

    for i in range(train_len):
        assignment_text[i] = f'train {text[i]}'
    for i in range(val_len):
        assignment_text[train_len + i] = f'val {text[train_len + i]}'
    for i in range(test_len):
        assignment_text[train_len + val_len + i] = f'test {text[train_len + val_len + i]}'
        
    with open('dataset_date_assignment.txt', mode='w') as f:
        f.writelines(assignment_text)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='convert multi npy to single npy file')
    parser.add_argument('--src_dir_1', default='/mnt/hdd1/sasaki/MAU_data/1_downloded_fits/211', type=str)
    parser.add_argument('--src_dir_2', default='/mnt/hdd1/sasaki/MAU_data/1_downloded_fits/193', type=str)
    parser.add_argument('--src_dir_3', default='/mnt/hdd1/sasaki/MAU_data/1_downloded_fits/171', type=str)
    parser.add_argument('--dataset_name', type=str, required=True)
    parser.add_argument('--seq_len', type=int, required=True)
    parser.add_argument('--delta', type=int, required=True)
    args = parser.parse_args()
