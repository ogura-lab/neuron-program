import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# CSVファイルのパス
file_paths = {
    'RS': 'time_difference_groups_1.csv',   # RSニューロンタイプのCSVファイルパス
    'IB': 'time_difference_groups_2.csv',   # IBニューロンタイプのCSVファイルパス
    'CH': 'time_difference_groups_3.csv',   # CHニューロンタイプのCSVファイルパス
    'FS': 'time_difference_groups_4.csv'    # FSニューロンタイプのCSVファイルパス
}

new_columns = ['mod_0', 'mod_1', 'mod_2', 'mod_3', 'mod_4']  # 元の列名
time_values = ['1.6', '2.0', '3.5', '4.2', '9.0']  # timeの値

# CSVファイルを読み込み、辞書に保存
dfs = {}
for neuron_type, file_path in file_paths.items():
    dfs[neuron_type] = pd.read_csv(file_path)
    # 列名をtimeの値に置き換える
    dfs[neuron_type].columns = time_values


# 必要なtime列のみ抽出（time1, time5, time9）
dfs_filtered = {}
for neuron_type, df in dfs.items():
    # 各DataFrameにNeuronType列を追加
    df['NeuronType'] = neuron_type
    # 'time1', 'time5', 'time9'列を抽出し、長い形式に変換
    dfs_filtered[neuron_type] = df[['1.6', '2.0', '3.5', '4.2', '9.0', 'NeuronType']].melt(
        id_vars=['NeuronType'], value_vars=['1.6', '2.0', '3.5', '4.2', '9.0'], var_name='Time', value_name='SpikeTimeError'
    )

# すべてのデータを1つのDataFrameに統合
df_all = pd.concat(dfs_filtered.values())


# ボックスプロットの作成
plt.figure(figsize=(10, 6))
palette = {'1.6': 'blue', '2.0': 'cyan', '3.5': 'green', '4.2': 'orange', '9.0': 'red'}

sns.violinplot(x='NeuronType', y='SpikeTimeError', hue='Time', data=df_all,
               palette=palette, split=True, inner="quartile", density_norm="width")

# グラフの設定
plt.xlabel('Neuron Type', fontsize=12)
plt.ylabel('Spike Time Error', fontsize=12)
plt.legend(title='Time', loc='upper left')

# PDFとして保存
plt.savefig("boxplot_neuron_type_time_error_rand.pdf", format='pdf')  # PDFとして保存

# プロットの表示
plt.close()