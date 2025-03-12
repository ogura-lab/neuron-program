import pickle
import matplotlib.pyplot as plt
import torch
import shutil
import numpy as np

#Specify the file to be loaded
tstr = "20241218-172645"
k = [0] 
time_num = 0
fire_cate = 1

alphamin, alphamax = -100.0, 100.0
for n in k:
    save_path = fr'dir_{fire_cate}_{tstr}_normsave/input_data_{fire_cate}_{time_num}_mid_input_data_{fire_cate}_{time_num}_epoch_{n}.pkl'  # 必要に応じて修正
    # Loading the saved intermediate input data
    with open(save_path, 'rb') as f:
        data = pickle.load(f)

    # Extract the data
    ts = data['ts']  # Time steps
    theta = data['theta']  # A list of sde.theta[i]
    fnum = data['fnum']
    l = data['l']
    epochNum = data['epochNum']

    result = np.concatenate([theta[i] for i in range(len(theta))], axis=0)
    scaled_result = alphamin + (alphamax - alphamin) * 0.5 * (np.tanh(result) + 1.0)

    # Plot the graph
    plt.plot(ts, scaled_result, color=(0.2, 0.2, 0.2), linewidth=2.5)
    plt.xlabel('$t$',fontsize=24)
    plt.ylabel('$l(t)$',fontsize=24)
    plt.tick_params(axis='both', labelsize=20)
    plt.xlim(-0.5,10.5)
    plt.ylim(-105,105)
    plt.xticks([0, 5, 10])  # Place ticks at specified positions

    # Draw grid lines at specified x positions
    grid_x_positions = [1,3,5,7,9]
    for x_pos in grid_x_positions:
        plt.vlines(x=x_pos,ymin=-105, ymax=105, color='gray',linestyle='--',  linewidth=0.5) 

    # Draw grid lines at specified y positions
    grid_y_positions = [-100, 0,100]
    for y_pos in grid_y_positions:
        plt.axhline(y=y_pos, color='gray',linestyle='--',  linewidth=0.5)
    plt.tight_layout()

    # Save the graph
    plt.savefig(f"図1_ver6/input_RS_{l}_{n}_開始 1.pdf")
    #plt.savefig(f"図1_ver6/input_RS_{l}_{n}_中間 1.pdf")
    #plt.savefig(f"図1_ver6/input_RS_{l}_{n}_後半 1.pdf")
    plt.close()
