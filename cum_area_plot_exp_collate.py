
import pickle  
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

"""Collate all by taking averaging of the data"""
# For v = 34
data_dir_path = r'C:\Users\vinee\OneDrive\Documents\MATLAB\Test1\analysis'
with open(data_dir_path + f'/test1_area_exp_cum_area', 'rb') as f:
    cum_area1, cum_count1, p1, p_conv1, area1 = pickle.load(f)

data_dir_path = r'C:\Users\vinee\OneDrive\Documents\MATLAB\Test2\analysis'
with open(data_dir_path + f'/test2_area_exp_cum_area', 'rb') as f:
    cum_area2, cum_count2, p2, p_conv2, area2 = pickle.load(f)

data_dir_path = r'C:\Users\vinee\OneDrive\Documents\MATLAB\Test3\analysis'
with open(data_dir_path + f'/test3_area_exp_cum_area', 'rb') as f:
    cum_area3, cum_count3, p3, p_conv3, area3 = pickle.load(f)

# For v = 20
data_dir_path = r'C:\Users\vinee\OneDrive\Documents\MATLAB\Test6\analysis'
with open(data_dir_path + f'/test6_area_exp_cum_area', 'rb') as f:
    cum_area6, cum_count6, p6, p_conv6, area6 = pickle.load(f)

data_dir_path = r'C:\Users\vinee\OneDrive\Documents\MATLAB\Test7\analysis'
with open(data_dir_path + f'/test7_area_exp_cum_area', 'rb') as f:
    cum_area7, cum_count7, p7, p_conv7, area7 = pickle.load(f)

data_dir_path = r'C:\Users\vinee\OneDrive\Documents\MATLAB\Test8\analysis'
with open(data_dir_path + f'/test8_area_exp_cum_area', 'rb') as f:
    cum_area8, cum_count8, p8, p_conv8, area8 = pickle.load(f)

# find average of (1,2,3)
min_area = min([min(area1), min(area2), min(area3)])
max_area = max([max(area1), max(area2), max(area3)])

cum_area123 = np.arange(0, 70, 1)
cum_count123 = np.zeros_like(cum_area123)
cum_count123_std = np.zeros_like(cum_area123)
for i in range(len(cum_area123)):
    count1 = len([s for s in area1 if s >= cum_area123[i]])
    count2 = len([s for s in area2 if s >= cum_area123[i]])
    count3 = len([s for s in area3 if s >= cum_area123[i]])
    cum_count123[i] = (count1 + count2 + count3)/3
    cum_count123_std[i] = np.std([count1, count2, count3], axis=0)

# find average of (6,7,8)
min_area = min([min(area6), min(area7), min(area8)])
max_area = max([max(area6), max(area7), max(area8)])
cum_area678 = np.arange(0, 70, 1)
cum_count678 = np.zeros_like(cum_area678)
cum_count678_std = np.zeros_like(cum_area678)
for i in range(len(cum_area678)):
    count6 = len([s for s in area6 if s >= cum_area678[i]])
    count7 = len([s for s in area7 if s >= cum_area678[i]])
    count8 = len([s for s in area8 if s >= cum_area678[i]])
    cum_count678[i] = (count6 + count7 + count8)/3
    cum_count678_std[i] = np.std([count6, count7, count8], axis=0)

# fit exponential function to the data
def exp_fit(x, a, b):
     return a*np.exp(b*x)

p1, p_conv1 = curve_fit(exp_fit, cum_area123, cum_count123/cum_count123[0], p0=[1, -0.1])
print(p1, p_conv1)

p2, p_conv2 = curve_fit(exp_fit, cum_area678, cum_count678/cum_count678[0], p0=[1, -0.1])
print(p2, p_conv2)

fig, ax = plt.subplots()
ax.plot(cum_area678, cum_count678/cum_count678[0], color='tab:blue', linestyle = 'none', markersize = 5, marker = 'o', label=r'$v = 20$ m/s')
ax.plot(cum_area123, cum_count123/cum_count123[0], color ='tab:orange', linestyle = 'none', markersize = 5, marker = 'o', label=r'$v = 34$ m/s')
ax.fill_between(cum_area123, (cum_count123 - cum_count123_std)/cum_count123[0], (cum_count123 + cum_count123_std)/cum_count123[0], color='tab:orange', alpha=0.3)
ax.fill_between(cum_area678, (cum_count678 - cum_count678_std)/cum_count678[0], (cum_count678 + cum_count678_std)/cum_count678[0], color='tab:blue', alpha=0.3)
ax.plot(cum_area123, exp_fit(cum_area123, p1[0], p1[1]), color='tab:orange')
ax.plot(cum_area678, exp_fit(cum_area678, p2[0], p2[1]), color='tab:blue')

ax.legend()
ax.set_xlabel(r'$A (\rm{mm}^2)\rightarrow$')
ax.set_ylabel(r'fraction of cum. fragments')
# fig.savefig(data_dir_path+ '/exp_cum_area_avg.png', dpi=300, bbox_inches='tight')
plt.show()

