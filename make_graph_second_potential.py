import pickle
import matplotlib.pyplot as plt
# Load the potential data
with open('/Users/sashie/alb/dir_1_20241218-160406_randomsave/saveplot_potential_4.pkl', 'rb') as f:
    data = pickle.load(f)

# Extract the data
time_steps = data['time_steps']
processed_samples = data['processed_samples']
fire_time = data['fire_time']
threshold = data['threshold']
c_value = data['c_value']

# Create the graph
plt.figure(figsize=(6.4, 4.8))

# Plot the sample data (first 10 samples)
for i, sample in enumerate(processed_samples):
    # Record the firing timings
    for k in range(len(sample)):
        if k != 0 and sample[k - 1] > 10 and sample[k] < -10:
            fire_time.append(time_steps[k - 1])

    if i < 10:  # Plot only the first 10 samples
        plt.plot(time_steps, sample, label=f'Sample {i}')

# Plot the graph
plt.xlabel('$t$' ,fontsize=24)
plt.ylabel('$v(t)$',fontsize=24)
plt.ylim(-70,35)
plt.yticks([-65,30])
plt.xticks([0, 5, 10])  # Place ticks at specified positions

plt.tick_params(axis='both', labelsize=20)
x_positions=[1.6, 2.0, 3.5, 4.2, 9.0]
for x_pos in x_positions:
    plt.axvline(x=x_pos,ymin=-70, ymax=35, color='black',linestyle='--', linewidth=2, zorder=1)

plt.tight_layout()

# Save the graph
plt.savefig('図4_ver2/potential_rand_FS.pdf', dpi=300)
plt.show()