import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Path to the CSV file
file_paths = {
    'RS': 'time_difference_groups_1.csv',   
    'IB': 'time_difference_groups_2.csv',   
    'CH': 'time_difference_groups_3.csv',   
    'FS': 'time_difference_groups_4.csv'
}

new_columns = ['mod_0', 'mod_1', 'mod_2', 'mod_3', 'mod_4']  # Original column names
time_values = ['1.6', '2.0', '3.5', '4.2', '9.0']

# Read the CSV file and save it in a dictionary
dfs = {}
for neuron_type, file_path in file_paths.items():
    dfs[neuron_type] = pd.read_csv(file_path)
    # Replace the column names with the values of 'time_values'
    dfs[neuron_type].columns = time_values


# Extracting and organizing the required data
dfs_filtered = {}
for neuron_type, df in dfs.items():
    df['NeuronType'] = neuron_type
    dfs_filtered[neuron_type] = df[['1.6', '2.0', '3.5', '4.2', '9.0', 'NeuronType']].melt(
        id_vars=['NeuronType'], value_vars=['1.6', '2.0', '3.5', '4.2', '9.0'], var_name='Time', value_name='SpikeTimeError'
    )

# Consolidating all the data into a single DataFrame
df_all = pd.concat(dfs_filtered.values())


# Creating a box plot
plt.figure(figsize=(10, 6))
palette = {'1.6': 'blue', '2.0': 'cyan', '3.5': 'green', '4.2': 'orange', '9.0': 'red'}

sns.violinplot(x='NeuronType', y='SpikeTimeError', hue='Time', data=df_all,
               palette=palette, split=True, inner="quartile", density_norm="width")

plt.xlabel('Neuron Type', fontsize=12)
plt.ylabel('Spike Time Error', fontsize=12)
plt.legend(title='Time', loc='upper left')

# Save the box plot
plt.savefig("boxplot_neuron_type_time_error_rand.pdf", format='pdf')  # PDFとして保存
plt.close()