import pickle
import matplotlib.pyplot as plt

# Specify the file to be loaded
tstr = "20241218-160406"
fnum = "4" 
save_path = f'dir_1_{tstr}_randomsave/input_data_{fnum}.pkl'

# Load the saved input data
with open(save_path, 'rb') as f:
    data = pickle.load(f)

# Extract the data
ts = data['ts']  # Time steps
Iext = data['Iext']  # Input data Iext

# Plot the graph
plt.plot(ts, Iext,color=(0.2, 0.2, 0.2), linewidth=2.5)
plt.xlabel('$t$',fontsize=24)
plt.ylabel('$l(t)$',fontsize=24)
plt.xlim(-0.5,10.5)
plt.ylim(-105,105)
plt.xticks([0, 5, 10])  # Place ticks at specified positions

# Draw grid lines at specified x positions
grid_x_positions = [1.6, 2.0, 3.5, 4.2, 9.0]
for x_pos in grid_x_positions:
    plt.vlines(x=x_pos,ymin=-105, ymax=105, color='gray',linestyle='--',  linewidth=0.5)

# Draw grid lines at specified y positions
grid_y_positions = [-100, 0,100]
for y_pos in grid_y_positions:
    plt.axhline(y=y_pos, color='gray',linestyle='--',  linewidth=0.5)

plt.tick_params(axis='both', labelsize=20)
plt.tight_layout()
plt.savefig(f"図4_ver4/input_rand_FS.pdf")
plt.show()
plt.close()
