import ast
import os
import shutil
import re
from tqdm import tqdm

# 3波長でデータセットを作ると、1波長で作ったデータセットよりも欠損箇所が多くなるため、
# 3波長で作った割り当て表をもとに1波長のFITSファイルをコピーする

def txt_to_array_dataset(file_path):
    """
    指定されたタイプ（例: 'train', 'val' など）に対応する行を読み出し、Pythonのリスト形式に変換する。
    :param file_path: データセットの情報が含まれるテキストファイルのパス
    :param dataset_type: 読み出したいデータセットのタイプ（例: 'train', 'val'）
    :return: Pythonリスト形式に変換されたデータセット
    """

    # ファイルを開き、行ごとに読み出す
    with open(file_path, 'r') as file:
        dataset_lines = file.readlines()

    # 選択された行をPythonのリストに変換
    datasets = []

    for line in dataset_lines:
        # 'train: ','val: 'などのプレフィックスの文字を正規表現でマッチして分割,さらにstripで前後の空白を削除
        list_str = re.split('train|val|test', line)[1].strip()

        # 文字列をPythonのリストに変換
        dataset = ast.literal_eval(list_str)
        datasets.append(dataset)

    return datasets

def transfer_fits(source, assignment_path, dst_dir):

    npy_list = txt_to_array_dataset(assignment_path)
    print('Number of total sequence:', len(npy_list))
    
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir, exist_ok=True)

    for npy_seq in tqdm(npy_list):

        # FITSファイルをコピー
        for npy_file in npy_seq[0]: # 1波長目(211Å)のFITSファイルをコピー
            shutil.copy(os.path.join(source, npy_file), os.path.join(dst_dir, npy_file))

transfer_fits('/mnt/hdd1/sasaki/MAU_data/3_npy_converted_from_fits/211/2023', '/mnt/hdd1/sasaki/MAU_data/6_datasets/aia3wave_4h_512px_sqrt_2023/dataset_date_assignment.txt', '/mnt/hdd1/sasaki/MAU_data/3_npy_converted_from_fits/211_tuned_3wave')