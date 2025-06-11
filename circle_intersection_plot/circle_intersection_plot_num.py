"""
Script to plot the number of intersecting circle fragments as a function of their radius
for different simulation and experimental parameters.

Author: Vineet Dawara
Last Updated: 2025-06-09

Description:
-------------
This script loads precomputed data on fragment intersection from both simulations and
experiments. It compares results across different material strains, shape sizes,
bond numbers (m), and experimental tests. The plots are styled using a custom
Matplotlib style file.

Dependencies:
-------------
- Python 3
- matplotlib
- numpy
- h5py
- pickle

"""

import pickle
import matplotlib.pyplot as plt
import numpy as np
import h5py

# Load custom plot style
mplstyle_file = r'C:\Users\vinee\OneDrive\Documents\vscode\stressed network model\article_preprint.mplstyle'
plt.style.use(f'{mplstyle_file}')

# -------------------------------------------------------
# Load simulation data: Variation with strain (ε_b)
# -------------------------------------------------------
filename = 'circle_fragments_inter'

# ε_b = 0.030
data_dir_path = r'C:\MyData\Stressed network work\Data\hole5_d5t125_et03_dec_m5_R108_e02_254X220\circle_fragment_intersection'
with open(data_dir_path + f'/{filename}', 'rb') as f:
    radius_e3, nbonds_e3 = pickle.load(f)

# ε_b = 0.035
data_dir_path = r'C:\MyData\Stressed network work\Data\m5et\hole5_d5t125_et035_dec_m5_R108_e02_254X220\Circle fragments intersection'
with open(data_dir_path + f'/{filename}', 'rb') as f:
    radius_e35, nbonds_e35 = pickle.load(f)

# ε_b = 0.040
data_dir_path = r'C:\MyData\Stressed network work\Data\m5et\hole5_d5t125_et04_dec_m5_R108_e02_254X220\Circle fragment intersection'
with open(data_dir_path + f'/{filename}', 'rb') as f:
    radius_e4, nbonds_e4 = pickle.load(f)

# -------------------------------------------------------
# Load simulation data: Variation with bond connectivity (m)
# -------------------------------------------------------
# m = 2
data_dir_path = r'C:\MyData\Stressed network work\Data\hole5_d5t125_et03_dec_m2_R108_e02_254X220\circle_fragments_intersection'
with open(data_dir_path + f'/{filename}', 'rb') as f:
    radius_m2, nbonds_m2 = pickle.load(f)

# m = 3
data_dir_path = r'C:\MyData\Stressed network work\Data\hole5_d5t125_et03_dec_m3_R108_e02_220X254\circle_fragments_intersection'
with open(data_dir_path + f'/{filename}', 'rb') as f:
    radius_m3, nbonds_m3 = pickle.load(f)

# m = 5
data_dir_path = r'C:\MyData\Stressed network work\Data\hole5_d5t125_et03_dec_m5_R108_e02_254X220\circle_fragment_intersection'
with open(data_dir_path + f'/{filename}', 'rb') as f:
    radius_m5, nbonds_m5 = pickle.load(f)

# -------------------------------------------------------
# Load simulation data: Variation with shape size
# -------------------------------------------------------
# small circle
with open(r'C:\MyData\Stressed network work\Data\hole5_d5t125_et03_dec_m5_R108_e02_254X220\circle_fragment_intersection\circle_fragments_inter', 'rb') as f:
    radius_sc, nbonds_sc = pickle.load(f)

# big circle
with open(r'C:\MyData\Stressed network work\Data\shape_size\hole10_d5t125_et03_dec_m5_R108_e02_254X220\circle_fragment_intersection\circle_fragments_inter', 'rb') as f:
    radius_bc, nbonds_bc = pickle.load(f)

# small ellipse
with open(r'C:\MyData\Stressed network work\Data\shape_size\ellipse_10X5_d5t125_et03_dec_m5_R108_e02_254X220\circle_fragment_intersection\circle_fragments_inter', 'rb') as f:
    radius_se, nbonds_se = pickle.load(f)

# big ellipse
with open(r'C:\MyData\Stressed network work\Data\shape_size\ellipse_15X5_d5t125_et03_dec_m5_R108_e02_254X220\circle_fragment_intersection\circle_fragments_inter', 'rb') as f:
    radius_be, nbonds_be = pickle.load(f)

# -------------------------------------------------------
# Load experimental data from HDF5 files
# -------------------------------------------------------
def load_data(path, filename):
    """
    Load data from an HDF5 (.mat) file.

    Parameters:
    - path: Full directory path
    - filename: File name without extension

    Returns:
    - radius: Array of normalized fragment radii
    - count: Array of bond counts
    """
    with h5py.File(path + f'/{filename}.mat', 'r') as f:
        radius = np.squeeze(f['radius'][:])
        count = np.squeeze(f['count'][:])
    return radius, count

# Load tests
filename = 'circle_intersect_count'
radius_1, nbonds_1 = load_data(r'C:\Users\vinee\OneDrive\Documents\MATLAB\Test1\analysis', filename)
radius_2, nbonds_2 = load_data(r'C:\Users\vinee\OneDrive\Documents\MATLAB\Test2\analysis', filename)
radius_3, nbonds_3 = load_data(r'C:\Users\vinee\OneDrive\Documents\MATLAB\Test3\analysis', filename)
radius_6, nbonds_6 = load_data(r'C:\Users\vinee\OneDrive\Documents\MATLAB\Test6\analysis', filename)
radius_7, nbonds_7 = load_data(r'C:\Users\vinee\OneDrive\Documents\MATLAB\Test7\analysis', filename)
radius_8, nbonds_8 = load_data(r'C:\Users\vinee\OneDrive\Documents\MATLAB\Test8\analysis', filename)

# Literature data
radius_glass1, nbonds_glass1 = load_data(r'C:\Users\vinee\OneDrive\Documents\MATLAB\glass_1\analysis\circle_intersection_frags_count', filename)
radius_glass2, nbonds_glass2 = load_data(r'C:\Users\vinee\OneDrive\Documents\MATLAB\glass_2\analysis\circle_intersection_frags_count', filename)
radius_burg, nbonds_burg = load_data(r'C:\Users\vinee\OneDrive\Documents\MATLAB\Burggraaf\analysis\circle_intersection_frags', filename)

# -------------------------------------------------------
# Process and normalize experimental data
# -------------------------------------------------------
# Average for velocity ~35 m/s
radius123 = np.mean([radius_1, radius_2, radius_3], axis=0)
radius123 /= radius123.max()
nbonds123 = np.mean([nbonds_1, nbonds_2, nbonds_3], axis=0)
nbonds123_std = np.std([nbonds_1, nbonds_2, nbonds_3], axis=0)

# Average for velocity ~20 m/s
radius678 = np.mean([radius_6, radius_7, radius_8], axis=0)
radius678 /= radius678.max()
nbonds678 = np.mean([nbonds_6, nbonds_7, nbonds_8], axis=0)
nbonds678_std = np.std([nbonds_6, nbonds_7, nbonds_8], axis=0)

# Normalize literature data  
# radius_glass1 /= radius_glass1.max()
# radius_glass2 /= radius_glass2.max()
# radius_burg /= radius_burg.max()

radius_glass1 /= 0.58
radius_glass2 /= 0.58
radius_burg /= 1.5875

# -------------------------------------------------------
# Plotting
# -------------------------------------------------------
# L = 110  # Reference length (half size of the network)
L = 5   # Normalization length for radius (initial hole radius)

fig, ax = plt.subplots()
# Simulation curves
ax.plot(radius_e3 / L, nbonds_e3, label=r'$\epsilon_{b} = 0.030$', alpha=0.5, linestyle = 'None', marker='o', markersize=8,
    markeredgecolor='black')
ax.plot(radius_e35 / L, nbonds_e35, label=r'$\epsilon_{b} = 0.035$', alpha=0.5, linestyle = 'None', marker='o', markersize=8,
    markeredgecolor='black')
ax.plot(radius_e4 / L, nbonds_e4, label=r'$\epsilon_{b} = 0.040$', alpha=0.5, linestyle = 'None', marker='o', markersize=8,
    markeredgecolor='black')
# ax.plot(radius_m2 / L, nbonds_m2, label=r'$m=2$', alpha=0.5, linestyle='None', marker='o', markersize=8, markeredgecolor='black')   
# ax.plot(radius_m3 / L, nbonds_m3, label=r'$m=3$', alpha=0.5, linestyle='None', marker='o', markersize=8, markeredgecolor='black')
# ax.plot(radius_m5 / L, nbonds_m5, label=r'$m=5$', alpha=0.5, linestyle='None', marker='o', markersize=8, markeredgecolor='black')

# Shape-size comparison
# ax.plot(radius_sc / L, nbonds_sc, label=r'small circle $2D_c/L=0.09$', color='tab:blue', alpha=0.5, linestyle='None', marker='o', markersize=8,
#      markeredgecolor='black')
# ax.plot(radius_bc / L, nbonds_bc, label=r'big circle $2D_c/L=0.18$', color='tab:orange', alpha=0.5, linestyle='None', marker='o', markersize=8,
#      markeredgecolor='black')
# ax.plot(radius_se / L, nbonds_se, label=r'small ellipse $2D_x/L=0.09, 2D_y/L=0.04$', color='tab:green', alpha=0.5, linestyle = 'None', marker='o', markersize=8,
#      markeredgecolor='black')
# ax.plot(radius_be / L, nbonds_be, label=r'big ellipse $2D_x/L=0.13, 2D_y/L=0.04$', color='tab:cyan', alpha=0.5, linestyle = 'None', marker='o', markersize=8,
#      markeredgecolor='black')

# Optional highlighting
# ax.axvspan(0, 1.2/30, color='lightgray', alpha=0.3, label='projectile region')
# ax.axvspan(1.2/30, 4/30, color='lightyellow', alpha=1, label='minimum un-accessible region')

# Plot literature data with contrasting colors
ax.plot(radius_glass1, nbonds_glass1, label='glass 1', color='lightgray', linestyle='None', marker='s', markersize=8, markeredgecolor='black')
ax.plot(radius_glass2, nbonds_glass2, label='glass 2', color='lightgray', linestyle='None', marker='^', markersize=8, markeredgecolor='black')
ax.plot(radius_burg, nbonds_burg, label='Burggraaf', color='lightgray', linestyle='None', marker='d', markersize=8, markeredgecolor='black')


# Final formatting (optional, depending on your style file)
ax.set_xlabel(r'$2r/L\rightarrow$')
ax.set_ylabel(r'$N_c\rightarrow$')
# ax.legend()
ax.set_xlim(left=0)
ax.set_ylim(bottom=0)

# Save and show plot
fig.savefig(r'C:\Users\vinee\OneDrive\Documents\MATLAB\collated_circle_frag_inter_num_exp_scatter.png', bbox_inches='tight', dpi=300)
plt.show()

