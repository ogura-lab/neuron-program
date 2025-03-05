import pickle
import matplotlib.pyplot as plt
import torch
import shutil
import numpy as np

# tstrを指定
tstr = "20241218-172645"
k = [0] #時間
time_num = 0 #対象時間番号
fire_cate = 1 #RSなど
alphamin, alphamax = -100.0, 100.0
# 保存したpklファイルのパス
for n in k:
    save_path = fr'dir_{fire_cate}_{tstr}_normsave/input_data_{fire_cate}_{time_num}_mid_input_data_{fire_cate}_{time_num}_epoch_{n}.pkl'  # 必要に応じて修正
    # pickleファイルの読み込み
    with open(save_path, 'rb') as f:
        data = pickle.load(f)

    # 保存したデータを取得
    ts = data['ts']  # 時間ステップ
    theta = data['theta']  # sde.theta[i] のリスト
    fnum = data['fnum']  # fnum の値
    l = data['l']  # l の値
    epochNum = data['epochNum']  # epochNum の値

    # theta の選択
    result = np.concatenate([theta[i] for i in range(len(theta))], axis=0)
    scaled_result = alphamin + (alphamax - alphamin) * 0.5 * (np.tanh(result) + 1.0)

    # グラフの描画
    plt.plot(ts, scaled_result, color=(0.2, 0.2, 0.2), linewidth=2.5)
    #plt.title('input')
    plt.xlabel('$t$',fontsize=24)
    plt.ylabel('$l(t)$',fontsize=24)
    plt.tick_params(axis='both', labelsize=20)
    plt.xlim(-0.5,10.5)
    plt.ylim(-105,105)
    plt.xticks([0, 5, 10])  # 指定した位置に目盛りを配置

    # グリッド線を引きたい特定のy位置
    grid_x_positions = [1,3,5,7,9]  # グリッドの位置
    #grid_x_positions = [1.6, 2.0, 3.5, 4.2, 9.0]
    for x_pos in grid_x_positions:
        plt.vlines(x=x_pos,ymin=-105, ymax=105, color='gray',linestyle='--',  linewidth=0.5)  # 水平線をグリッドとして扱う


    # グリッド線を引きたい特定のy位置
    grid_y_positions = [-100, 0,100]  # グリッドの位置
    for y_pos in grid_y_positions:
        plt.axhline(y=y_pos, color='gray',linestyle='--',  linewidth=0.5)  # 水平線をグリッドとして扱う
    plt.tight_layout()
    # グラフの保存
    plt.savefig(f"図1_ver6/input_RS_{l}_{n}_開始 1.pdf")
    #plt.savefig(f"図1_ver6/input_RS_{l}_{n}_中間 1.pdf")
    #plt.savefig(f"図1_ver6/input_RS_{l}_{n}_後半 1.pdf")
    plt.close()
