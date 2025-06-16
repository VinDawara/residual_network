from scipy.optimize import curve_fit
import pickle
import numpy as np
import matplotlib.pyplot as plt

# specify the mplstyle file for the plot
mplstyle_file = r'C:\Users\vinee\OneDrive\Documents\vscode\stressed network model\article_preprint.mplstyle'
plt.style.use(f'{mplstyle_file}')

# filename for the area data
filename = 'burggraaf_area'

# read data for profile m = 5
data_dir_path = r'C:\Users\vinee\OneDrive\Documents\MATLAB\Burggraaf\analysis\area'
with open(data_dir_path + f'/{filename}.txt', 'r') as f:
    size = f.readlines()

size = [float(s.strip()) for s in size]

print(f'total fragments: {len(size)}')
# remove all zero values from size
size = [s for s in size if s >= 2.86]

print(f'fragments after removing zero values: {len(size)}')
max_size = max(size)
min_size = min(size)

# size = [s for s in size if s!=max_size]
# max_size = max(size)
# min_size = min(size)
# size = [s for s in size]
print(f'amin: {min_size}, amax: {max_size}')

# find cumulative area distribution
cum_area = np.arange(min_size, max_size, 1)
cum_count = np.zeros_like(cum_area)
for i in range(len(cum_area)):
    cum_count[i] = len([s for s in size if s >= cum_area[i]])


total_count = cum_count[0]
print(total_count)

# fit exponential function to the data
def exp_fit(x, a, b):
    return a*np.exp(b*x)

def powerlaw_fit(x, a, b):
    return a*x**b   

p, p_conv, _, msg, flag = curve_fit(exp_fit, cum_area, cum_count/total_count, p0=[0.5, -0.1],full_output=True)
print(f'Parameters: {p}')
print(f'Covariance: {p_conv}')
print(f'Flag: {flag}, Message: {msg}')



fig, ax = plt.subplots()
ax.plot(cum_area, cum_count/total_count, color='tab:blue', linestyle = 'none', markersize = 5, marker = 'o')
ax.plot(cum_area, exp_fit(cum_area, p[0], p[1]), color='tab:blue')
# ax.set_xlabel(r'$d_s/d_{s,{\rm{max}}}\rightarrow$')
ax.set_xlabel(r'$A (\rm{mm}^2)\rightarrow$')
ax.set_ylabel(r'fraction of cum. fragments')
# ax.set_xlim([0, 1])
fig.savefig(data_dir_path+ '/exp_cum_area_cut01.png', dpi=300, bbox_inches='tight')
plt.show()

with open(data_dir_path + f'/{filename}_exp_cum_area', 'wb') as f:
    pickle.dump([cum_area, cum_count, p, p_conv, size], f)
