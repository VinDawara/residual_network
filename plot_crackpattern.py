"""
Script to plot and compare crack patterns from two simulations 
using the springlattice model. The crack patterns are overlaid 
for visual comparison of shape/geometry effects on fracture.

Author: Vineet Dawara
Last Modified: 2025-06-09
"""

# === Import Libraries ===
import springlattice as sl                      # Main spring-lattice modeling package
import springlattice.output_1 as so             # Module to handle output processing and visualization
import numpy as np                              # Numerical operations
import matplotlib.pyplot as plt                 # Plotting
import pickle                                   # Not used here, but included if needed later
from matplotlib.collections import LineCollection  # For efficient plotting of line segments

# === Apply Custom Matplotlib Style ===
mplstyle_file = r'C:\Users\vinee\OneDrive\Documents\vscode\stressed network model\article_preprint.mplstyle'
plt.style.use(mplstyle_file)  # Use predefined plot style for consistency

# === Define Data Paths ===
# Data1: Circle with hole geometry
data1 = r'C:\MyData\Stressed network work\Data\hole5_d5t125_et03_dec_m5_R108_e02_254X220'

# Data2: Elliptical shape for comparison
data2 = r'C:\MyData\Stressed network work\Data\shape_size\ellipse_10X5_d5t125_et03_dec_m5_R108_e02_254X220'

# === Load Mesh and Crack Pattern for Data1 ===
mesh = so.load_mesh(data1)      # Load simulation mesh
mesh.folder = data1             # Assign folder manually (required for downstream functions)
crack_pattern1 = so.get_crack_pattern(mesh, time=200)  # Extract crack pattern at specified time step

# === Load Mesh and Crack Pattern for Data2 ===
mesh2 = so.load_mesh(data2)
mesh2.folder = data2
crack_pattern2 = so.get_crack_pattern(mesh2, time=200)

# === Set Plot Limits Based on Node Positions ===
pos = mesh.pos                                 # Node positions from first mesh
xmin, xmax = np.min(pos[:, 0]), np.max(pos[:, 0])
ymin, ymax = np.min(pos[:, 1]), np.max(pos[:, 1])

# === Create Plot ===
fig, ax = plt.subplots()

# Add first crack pattern (black, solid)
line1 = LineCollection(crack_pattern1, linewidths=2, color='black')

# Add second crack pattern (red, transparent)
line2 = LineCollection(crack_pattern2, linewidths=2, color='red', alpha=0.5)

# Add line collections to axes
ax.add_collection(line1)
ax.add_collection(line2)

# Set plot limits with some padding
ax.set_xlim(xmin - 1, xmax + 1)
ax.set_ylim(ymin - 1, ymax + 1)

# Maintain equal aspect ratio and remove axes
ax.set_aspect('equal')
ax.set_axis_off()

# === Save and Show Plot ===
# Save figure to file
fig.savefig(
    r'C:\MyData\Stressed network work\Data\shape_size\shape_vary.png',
    dpi=300,
    bbox_inches='tight'
)

# Show plot on screen
plt.show()
