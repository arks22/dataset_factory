import os
from glob import glob
from datetime import timedelta
from datetime import datetime
import argparse

import numpy as np

import astropy.units as u
from sunpy.net import jsoc
from sunpy.net import attrs as a
from sunpy.time import parse_time
from sunpy.time import TimeRange

# (2) 重複や欠落がないかチェック
#　欠落枚数にもよるが1時間から3時間くらい

from dataset_utils import filename_to_date
from dataset_utils import date_to_filename


def fetch_closest_file(target_datetime, wave, time_range):
    """
    指定されたdatetimeオブジェクトに最も近いAIA 211のデータファイルを取得する関数
    
    Parameters:
    - target_datetime (datetime.datetime): 中心となるdatetimeオブジェクト
    - time_range (int): 指定されたdatetimeオブジェクトを中心とした時間範囲（単位: 分）
    
    Returns:
    - bool: ダウンロードの可否
    """
    
    # JSOCクライアントを初期化
    client = jsoc.JSOCClient()
    
    # 時間範囲を設定
    tr = TimeRange(target_datetime - timedelta(minutes=time_range),
                   target_datetime + timedelta(minutes=time_range))
    
    # AIA データを検索
    res = client.search(a.Time(tr),
                        a.jsoc.Series('aia.lev1_euv_12s'), 
                        a.jsoc.Wavelength(wave*u.angstrom))
    print(res)

    # 検索結果がない場合
    if len(res) == 0:
        print('***** No data matches')
        return False

    # 各ファイルのdatetimeと指定されたdatetimeとの差を計算
    time_diffs = [abs((datetime.strptime(r['T_REC'], '%Y-%m-%dT%H:%M:%SZ') - target_datetime).total_seconds()) for r in res]

    # 最も近いデータを特定
    closest_idx = np.argmin(time_diffs)
    closest_time = datetime.strptime(res[closest_idx]['T_REC'], '%Y-%m-%dT%H:%M:%SZ')

    # 再検索
    tr_closest = TimeRange(closest_time, closest_time)
    res_closest = client.search(a.Time(tr_closest),
                                a.jsoc.Series('aia.lev1_euv_12s'),
                                a.jsoc.Segment('image'),
                                a.jsoc.Wavelength(wave*u.angstrom),
                                a.jsoc.Notify("sasaki.10222022@gmail.com"))
    #a.jsoc.Notify("f22c017d@mail.cc.niigata-u.ac.jp"))
    print(res_closest)
    
    # ダウンロード
    if len(res_closest) > 0:
        client.fetch(res_closest, path=args.src_dir, overwrite=True, progress=True)
    else:
        print("No file found for the closest time.")

    return True



def search_files_in_range(dates, target_datetime, time_range):
    file_dates_in_range = []
    range_head = target_datetime - timedelta(minutes=time_range)
    range_end  = target_datetime + timedelta(minutes=time_range)

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
        os.remove(os.path.join(args.src_dir, file))

    return file_dates_in_range


def main(args):
    all_files = sorted(glob(args.src_dir + '/*'))

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

    time_range = 1 #許容誤差範囲

    now = start_date

    while now < end_date:

        # レンジ内ファイル数カウント
        file_dates_in_range = search_files_in_range(dates, now, time_range)
        file_count = len(file_dates_in_range)

        if not file_count == 1:

            if file_count == 0: #欠落
                print(f'Lacking found around : {now}')
                lack_count += 1
                if fetch_closest_file(now, args.wave, time_range):
                    downloaded_files_count += 1

            if file_count >= 2: #重複　
                print(f'Duplicate found around : {now}')
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
    parser.add_argument('src_dir', type=str)
    parser.add_argument('--delta', type=int, required=True)
    parser.add_argument('--wave', type=int, required=True)
    args = parser.parse_args()

    main(args)
