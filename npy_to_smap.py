import numpy as np
from sunpy.map import Map
import matplotlib.pyplot as plt
from sunpy.map import Map
import matplotlib.animation as animation

data = np.load('/home/sasaki/evaluate_factory/202310311754/tmp/01.npy')
print(data.shape)
len_frames = data.shape[0]
width = data.shape[1]
height = data.shape[2]

meta = {
    'telescop': 'SDO/AIA',
    'detector': 'AIA',
    'wavelnth': 211,
    'waveunit': 'angstrom',
    'observatory': 'SDO',
    'exptime': 1.0,
    'CTYPE1': 'HPLN-TAN',  # 水平方向の座標軸
    'CTYPE2': 'HPLT-TAN',  # 垂直方向の座標軸
    'CUNIT1': 'arcsec',    # 水平方向の単位
    'CUNIT2': 'arcsec',    # 垂直方向の単位
    'CRVAL1': 0,           # 参照ピクセルの水平座標
    'CRVAL2': 0,           # 参照ピクセルの垂直座標
    'CRPIX1': width // 2,  # 参照ピクセルの水平位置
    'CRPIX2': height // 2, # 参照ピクセルの垂直位置
    'CDELT1': 1,           # ピクセルあたりの水平方向のスケール
    'CDELT2': 1            # ピクセルあたりの垂直方向のスケール
    # 必要に応じて追加のメタデータをここに入力
}
aia211_cmap = plt.get_cmap('sdoaia211')

# ndarrayの各要素をSunPy Mapに変換し、PNG形式で保存します。
for i in range(len_frames):
    data_i = data[i, :, :]
    vmin = np.percentile(data_i, 0.1)  # 下限を0.1パーセンタイルに設定
    vmax = np.percentile(data_i, 99.9)  # 上限を99.9パーセンタイルに設定
 
    # smapのデータをクリップ 
    clipped_data = np.clip(data_i, vmin, vmax) # Mapオブジェクトを作成
    
    smap = Map(clipped_data, meta)
    
    # プロット
    plt.figure()
    smap.plot(cmap=aia211_cmap)  # カラーマップを適用
    
    # ファイル名を設定（例: "solar_image_0.png", "solar_image_1.png", ...）
    filename = f"solar_image_{i}.png"
    
    # PNGとして保存
    plt.savefig(filename)
    plt.close()
    
    print(f"{filename} に保存しました。")