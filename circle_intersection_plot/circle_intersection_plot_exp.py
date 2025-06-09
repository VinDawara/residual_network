
"""
This script reads HDF5 (.mat) files from different experimental or simulation test folders,
extracts radius and bond count data, computes averages and standard deviations,
normalizes radii, and plots the results for comparative analysis. It uses Matplotlib for visualization.
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt

# Specify the custom mplstyle file for plot styling
mplstyle_file = r'C:\Users\vinee\OneDrive\Documents\vscode\stressed network model\article_preprint.mplstyle'
plt.style.use(f'{mplstyle_file}')

filename = 'circle_intersect_count'

# Load data from multiple test directories (Test1, Test2, ..., Test8)
def load_data(path, filename):
    """Helper function to load radius and count data from an HDF5 file."""
    with h5py.File(path + f'/{filename}.mat', 'r') as f:
        radius = np.squeeze(f['radius'][:])
        count = np.squeeze(f['count'][:])
    return radius, count

# Data loading
radius_1, nbonds_1 = load_data(r'C:\Users\vinee\OneDrive\Documents\MATLAB\Test1\analysis', filename)
radius_2, nbonds_2 = load_data(r'C:\Users\vinee\OneDrive\Documents\MATLAB\Test2\analysis', filename)
radius_3, nbonds_3 = load_data(r'C:\Users\vinee\OneDrive\Documents\MATLAB\Test3\analysis', filename)
radius_6, nbonds_6 = load_data(r'C:\Users\vinee\OneDrive\Documents\MATLAB\Test6\analysis', filename)
radius_7, nbonds_7 = load_data(r'C:\Users\vinee\OneDrive\Documents\MATLAB\Test7\analysis', filename)
radius_8, nbonds_8 = load_data(r'C:\Users\vinee\OneDrive\Documents\MATLAB\Test8\analysis', filename)

radius_glass1, nbonds_glass1 = load_data(r'C:\Users\vinee\OneDrive\Documents\MATLAB\glass_1\analysis\circle_intersection_frags_count', filename)
radius_glass2, nbonds_glass2 = load_data(r'C:\Users\vinee\OneDrive\Documents\MATLAB\glass_2\analysis\circle_intersection_frags_count', filename)
radius_burg, nbonds_burg = load_data(r'C:\Users\vinee\OneDrive\Documents\MATLAB\Burggraaf\analysis\circle_intersection_frags', filename)

# Compute average and std for v=35 m/s group (Test1, 2, 3)
radius123 = np.mean([radius_1, radius_2, radius_3], axis=0)
radius123 = radius123 / radius123.max()
nbonds123 = np.mean([nbonds_1, nbonds_2, nbonds_3], axis=0)
nbonds123_std = np.std([nbonds_1, nbonds_2, nbonds_3], axis=0)

# Compute average and std for v=20 m/s group (Test6, 7, 8)
radius678 = np.mean([radius_6, radius_7, radius_8], axis=0)
radius678 = radius678 / radius678.max()
nbonds678 = np.mean([nbonds_6, nbonds_7, nbonds_8], axis=0)
nbonds678_std = np.std([nbonds_6, nbonds_7, nbonds_8], axis=0)

# Normalize radii for literature data
radius_glass1 = radius_glass1 / radius_glass1.max()
radius_glass2 = radius_glass2 / radius_glass2.max()
radius_burg = radius_burg / radius_burg.max()

# Create plot
fig, ax = plt.subplots()

# Highlight special regions in the plot
ax.axvspan(0, 1.2/30, color='lightgray', alpha=0.3, label='projectile region')
ax.axvspan(1.2/30, 4/30, color='lightyellow', alpha=1, label='minimum un-accessible region')

# Plot data with shaded standard deviation areas
ax.plot(radius678, nbonds678, label=r'$v = 20$ m/s', color='tab:blue')
ax.fill_between(radius678, nbonds678 - nbonds678_std, nbonds678 + nbonds678_std, alpha=0.2, color='tab:blue')

ax.plot(radius123, nbonds123, label=r'$v = 35$ m/s', color='tab:orange')
ax.fill_between(radius123, nbonds123 - nbonds123_std, nbonds123 + nbonds123_std, alpha=0.2, color='tab:orange')

# Plot literature data
ax.plot(radius_glass1, nbonds_glass1, label='glass 1', color='tab:green', linestyle='--')
ax.plot(radius_glass2, nbonds_glass2, label='glass 2', color='tab:purple', linestyle='--')
ax.plot(radius_burg, nbonds_burg, label='Burggraaf', color='tab:red', linestyle='--')

# Axis labels and limits
ax.set_xlabel(r'$2r/L\rightarrow$')
ax.set_ylabel(r'$N_c\rightarrow$')
ax.set_xlim(left=0)
ax.set_ylim(bottom=0)

# Save and show plot
fig.savefig(r'C:\Users\vinee\OneDrive\Documents\MATLAB\collated_circle_frag_inter.png', bbox_inches='tight', dpi=300)
plt.show()
