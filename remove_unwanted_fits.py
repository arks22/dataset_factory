import os
import re
import argparse
from datetime import timedelta, datetime

# 実際にファイルを削除するので注意
# 通常のデータセット作成工程では使用しない

def is_near_target_time(time, target_hours):
    """
    指定された時間（target_hours）に近いかどうか判定する。
    ±1分以内であればTrueを返す。
    """
    for th in target_hours:
        target_time = time.replace(hour=th, minute=0, second=0)
        delta = abs((time - target_time).total_seconds())
        if delta <= 60 or delta >= 86340:  # 86400 - 60 = 86340
            return True
    return False


def remove_unwanted_fits_files(target_dir):
    """
    指定されたディレクトリ内の不要なfitsファイルを削除する。
    """
    fits_files = [f for f in os.listdir(target_dir) if f.endswith('.fits')]
    count = 0

    time_pattern = re.compile(r'(\d{4})-(\d{2})-(\d{2})T(\d{2})(\d{2})(\d{2})Z')
    target_hours = [i for i in range(0, 24, 4)]  # 0時, 4時, 8時, ..., 20時

    # ソート
    fits_files.sort(key=lambda x: time_pattern.search(x).groups() if time_pattern.search(x) else x)

    for file_name in fits_files:
        match = time_pattern.search(file_name)
        if match:
            year, month, day, hour, minute, second = map(int, match.groups())
            time = datetime(year, month, day, hour, minute, second)

            if is_near_target_time(time, target_hours):
                continue  # 削除しない
            
            #os.remove(os.path.join(target_dir, file_name))
            count += 1
            print(f"削除されたファイル: {file_name}")

    print(f"合計で {count} 個のファイルが削除されました。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ミスってダウンロードした誤差の大きいfitsファイルを削除するスクリプト")
    parser.add_argument("target_dir", type=str, help="検査対象のディレクトリのパス")
    
    args = parser.parse_args()
    remove_unwanted_fits_files(args.target_dir)

