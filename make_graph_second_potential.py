import pickle
import matplotlib.pyplot as plt
# pklファイルを読み込む
#1:RS, 2:IB, 3:CH, 4:FS
with open('/Users/sashie/alb/dir_1_20241218-160406_randomsave/saveplot_potential_4.pkl', 'rb') as f:
    data = pickle.load(f)

# 必要なデータを抽出
time_steps = data['time_steps']
processed_samples = data['processed_samples']
fire_time = data['fire_time']
threshold = data['threshold']
c_value = data['c_value']

# プロット作成
plt.figure(figsize=(6.4, 4.8))

# 処理されたサンプルデータをプロット（最初の10個のサンプル）
for i, sample in enumerate(processed_samples):
    # 発火タイミングを記録
    for k in range(len(sample)):
        if k != 0 and sample[k - 1] > 10 and sample[k] < -10:
            fire_time.append(time_steps[k - 1])

    if i < 10:  # 最初の10個のサンプルのみプロット
        plt.plot(time_steps, sample, label=f'Sample {i}')

# グラフのタイトルとラベル
#plt.title('Processed Samples with Fire Time')
plt.xlabel('$t$' ,fontsize=24)
plt.ylabel('$v(t)$',fontsize=24)
plt.ylim(-70,35)
plt.yticks([-65,30])
plt.xticks([0, 5, 10])  # 指定した位置に目盛りを配置

plt.tick_params(axis='both', labelsize=20)
x_positions=[1.6, 2.0, 3.5, 4.2, 9.0]
for x_pos in x_positions:
    plt.axvline(x=x_pos,ymin=-70, ymax=35, color='black',linestyle='--', linewidth=2, zorder=1)

plt.tight_layout()
# 凡例を表示
#plt.legend(fontsize=20)

# グリッドを表示
#plt.grid(True)

# プロットを保存
plt.savefig('図4_ver2/potential_rand_FS.pdf', dpi=300)

# プロットを表示
plt.show()