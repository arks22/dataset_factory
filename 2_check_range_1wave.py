import os
from glob import glob
from datetime import timedelta
import argparse

import astropy.units as u
from sunpy.net import jsoc
from sunpy.net import attrs as a
from sunpy.time import parse_time

# (2) 重複や欠落がないかチェック

from dataset_utils import filename_to_date
from dataset_utils import date_to_filename


def fetch_closest_file(target_datetime, time_range=5):
    """
    指定されたdatetimeオブジェクトに最も近いAIA 211のデータファイルを取得する関数
    
    Parameters:
    - target_datetime (datetime.datetime): 中心となるdatetimeオブジェクト
    - time_range (int): 指定されたdatetimeオブジェクトを中心とした時間範囲（単位: 分）
    
    Returns:
    - None: ダウンロードが完了するとファイルがローカルに保存される
    """
    
    # JSOCクライアントを初期化
    client = jsoc.JSOCClient()
    
    # 時間範囲を設定
    tr = TimeRange(target_datetime - datetime.timedelta(minutes=time_range),
                   target_datetime + datetime.timedelta(minutes=time_range))
    
    # AIA 211データを検索
    res = client.search(a.Time(tr), a.jsoc.Series('aia.lev1_euv_12s'), a.jsoc.Wavelength(211*u.angstrom))
    
    # 検索結果がない場合
    if len(res) == 0:
        print("No AIA 211 files found for the specified datetime and time range.")
        return False
    
    # 各ファイルのdatetimeと指定されたdatetimeとの差を計算
    time_diffs = [abs((r.time.datetime - target_datetime).total_seconds()) for r in res]
    
    # 最も近いデータを特定
    closest_idx = np.argmin(time_diffs)
    
    # ダウンロード
    client.fetch(res[closest_idx])
    return True




def search_files_in_range(dates, range_head ,range_end):
    file_dates_in_range = []
    for file_date in dates:
        if range_head <= file_date <= range_end:
            file_dates_in_range.append(file_date)

    return file_dates_in_range


def delete_dupl(file_dates_in_range, now):
    min_diff = timedelta(hours=1)
    min_diff_date_index = 0

    # 最もnowに近いファイルのインデックスを探す
    for i, file_date in enumerate(file_dates_in_range):
        if abs(file_date - now) < min_diff:
            min_diff_date_index = i

    file_dates_in_range.pop(min_diff_date_index)

    #ファイルを削除
    for file_date in file_dates_in_range:
        file = date_to_filename(file_date, args.wave, 'fits' )
        os.remove(os.path.join(args.source, file))

    return file_dates_in_range


def main(args):
    all_files = sorted(glob(args.source + '/*'))

    dates = []
    for file in all_files:
        dates.append(filename_to_date(file))

    dates = sorted(dates)
    start_date = dates[0]
    end_date   = dates[-1]
    print('start date:', start_date)
    print('end date:', end_date)

    lack_count = 0 #欠落の数
    downloaded_files_count = 0 #欠落によってダウンロードされたファイルの数
    dupl_count = 0 #重複しているレンジの数
    deleted_files_count = 0 #重複で消されたファイルの数

    now = start_date

    while now < end_date:
        range_head = now - timedelta(minutes=5)
        range_end  = now + timedelta(minutes=5)

        # レンジ内ファイル数カウント
        file_dates_in_range = search_files_in_range(dates, range_head, range_end)
        file_count = len(file_dates_in_range)

        if not file_count == 1:

            if file_count == 0: #欠落
                print(f'Lacking found: {range_head} to {range_end}')
                lack_count += 1
                if download(range_head, range_end, args.wave): 
                    downloaded_files_count += 1

            if file_count >= 2: #重複　
                print(f'Duplicate found: {range_head} < {range_end}')
                deleted_files = delete_dupl(file_dates_in_range, now)
                deleted_files_count += len(deleted_files)
                dupl_count += 1
                print(f'Deleted: {deleted_files}')

        now += timedelta(hours=args.delta)

    print('---------------------------')
    print(f'{lack_count} Lacking were found and {downloaded_files_count} files were downloaded.')
    print(f'{dupl_count} Duplication were found and {deleted_files_count} files were deleted.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('source', type=str)
    parser.add_argument('--delta', type=int, required=True)
    parser.add_argument('--wave', type=int, required=True)
    args = parser.parse_args()

    main(args)
