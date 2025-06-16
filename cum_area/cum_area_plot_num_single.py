from scipy.optimize import curve_fit
import pickle
import numpy as np
import matplotlib.pyplot as plt
# specify the mplstyle file for the plot
mplstyle_file = r'C:\Users\vinee\OneDrive\Documents\vscode\stressed network model\article_preprint.mplstyle'
plt.style.use(f'{mplstyle_file}')

# filename for the area data
filename = 'area_size'

# read data for profile m = 5
data_dir_path = r'C:\MyData\Stressed network work\Data\hole5_d5t125_et03_dec_m2_R108_e02_254X220\area_data'
with open(data_dir_path + f'/{filename}.txt', 'r') as f:
    size = f.readlines()

size = [float(s.strip()) for s in size]

# scale the area
size = [s*0.27**2 for s in size]
# remove all values less than 1
size = [s for s in size if s >=2]
max_size = max(size)
min_size = min(size)
print(f'amin: {min_size}, amax: {max_size}')

# find cumulative area distribution
cum_area = np.linspace(min_size, max_size, 200)
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

p, p_conv, _, msg, flag = curve_fit(exp_fit, cum_area, cum_count/total_count, p0=[1, -0.1], full_output=True)
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
fig.savefig(data_dir_path+ '/exp_cum_area.png', dpi=300, bbox_inches='tight')
plt.show()

# save data
with open(data_dir_path + f'/exp_cum_area', 'wb') as f:
    pickle.dump([cum_area, cum_count, p, p_conv], f)
