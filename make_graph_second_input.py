import pickle
import matplotlib.pyplot as plt
tstr = "20241218-160406"
fnum = "4" #1:RS, 2:IB, 3:CH, 4:FS
# 保存した pickle ファイルを読み込む
save_path = f'dir_1_{tstr}_randomsave/input_data_{fnum}.pkl'
with open(save_path, 'rb') as f:
    data = pickle.load(f)

# 保存したデータを取り出す
ts = data['ts']  # 時間ステップ
Iext = data['Iext']  # 入力データ Iext

# 再度プロットする
plt.plot(ts, Iext,color=(0.2, 0.2, 0.2), linewidth=2.5)
#plt.title('input')
plt.xlabel('$t$',fontsize=24)
plt.ylabel('$l(t)$',fontsize=24)
plt.xlim(-0.5,10.5)
plt.ylim(-105,105)
plt.xticks([0, 5, 10])  # 指定した位置に目盛りを配置

# グリッド線を引きたい特定のy位置
#grid_x_positions = [1,3,5,7,9]  # グリッドの位置
grid_x_positions = [1.6, 2.0, 3.5, 4.2, 9.0]
for x_pos in grid_x_positions:
    plt.vlines(x=x_pos,ymin=-105, ymax=105, color='gray',linestyle='--',  linewidth=0.5)  # 水平線をグリッドとして扱う

# グリッド線を引きたい特定のy位置
grid_y_positions = [-100, 0,100]  # グリッドの位置
for y_pos in grid_y_positions:
    plt.axhline(y=y_pos, color='gray',linestyle='--',  linewidth=0.5)  # 水平線をグリッドとして扱う

plt.tick_params(axis='both', labelsize=20)
plt.tight_layout()
plt.savefig(f"図4_ver4/input_rand_FS.pdf")  # 保存
plt.show()
plt.close()
