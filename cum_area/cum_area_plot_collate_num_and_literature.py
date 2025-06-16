import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# specify the mplstyle file for the plot
mplstyle_file = r'C:\Users\vinee\OneDrive\Documents\vscode\stressed network model\article_preprint.mplstyle'
plt.style.use(f'{mplstyle_file}')


def exp_fit(x, a, b):
    return a*np.exp(b*x)

filename = r'exp_cum_area'

data_dir_path = r'C:\MyData\Stressed network work\Data\hole5_d5t125_et03_dec_m5_R108_e02_254X220\area_data'
with open(data_dir_path + f'/{filename}', 'rb') as f:
    cum_area1, cum_count1, p1, p_conv1 = pickle.load(f)


data_dir_path = r'C:\MyData\Stressed network work\Data\m5et\hole5_d5t125_et035_dec_m5_R108_e02_254X220\area_data'
with open(data_dir_path + f'/{filename}', 'rb') as f:
    cum_area2, cum_count2, p2, p_conv2 = pickle.load(f)

data_dir_path = r'C:\MyData\Stressed network work\Data\m5et\hole5_d5t125_et04_dec_m5_R108_e02_254X220\area_data'
with open(data_dir_path + f'/{filename}', 'rb') as f:
    cum_area3, cum_count3, p3, p_conv3 = pickle.load(f)

fig, ax = plt.subplots()
A_num = 60*60
ax.plot(cum_area1/A_num, cum_count1/cum_count1[0], linestyle = 'none', markersize = 5, marker = 'o', color='tab:blue', label = r'$\epsilon_b = 0.03$')
ax.plot(cum_area2/A_num, cum_count2/cum_count2[0], linestyle = 'none', markersize = 5, marker = 'o', color='tab:orange', label = r'$\epsilon_b = 0.035$')
ax.plot(cum_area3/A_num, cum_count3/cum_count3[0], linestyle = 'none', markersize = 5, marker = 'o', color='tab:cyan', label = r'$\epsilon_b = 0.04$')
ax.plot(cum_area1/A_num, exp_fit(cum_area1, p1[0], p1[1]), color='tab:blue', alpha = 0.8)
ax.plot(cum_area2/A_num, exp_fit(cum_area2, p2[0], p2[1]), color='tab:orange', alpha = 0.8)
ax.plot(cum_area3/A_num, exp_fit(cum_area3, p3[0], p3[1]), color='tab:cyan', alpha = 0.8)



# literature data
data_dir_path = r'C:\Users\vinee\OneDrive\Documents\MATLAB\glass_1\analysis\area' 
with open(data_dir_path + f'/glass1_area_exp_cum_area', 'rb') as f:
    cum_area_glass1, cum_count_glass1, p_glass1, p_conv_glass1, area_glass1 = pickle.load(f)

data_dir_path = r'C:\Users\vinee\OneDrive\Documents\MATLAB\glass_2\analysis\area'
with open(data_dir_path + f'/glass2_area_exp_cum_area', 'rb') as f:
    cum_area_glass2, cum_count_glass2, p_glass2, p_conv_glass2, area_glass2 = pickle.load(f)

data_dir_path = r'C:\Users\vinee\OneDrive\Documents\MATLAB\Burggraaf\analysis\area'
with open(data_dir_path + f'/burggraaf_area_exp_cum_area', 'rb') as f:
    cum_area_burggraaf, cum_count_burggraaf, p_burggraaf, p_conv_burggraaf, area_burggraaf = pickle.load(f)

# fit the data for glass 1
p_glass1, p_conv_glass1 = curve_fit(exp_fit, cum_area_glass1, cum_count_glass1/cum_count_glass1[0], p0=[1, -0.1])
print(f'Param for glass 1: {p_glass1}')

# fit the data for glass 2
p_glass2, p_conv_glass2 = curve_fit(exp_fit, cum_area_glass2, cum_count_glass2/cum_count_glass2[0], p0=[1, -0.1])
print(f'Param for glass 2: {p_glass2}')

# fit the data for burggraaf
p_burggraaf, p_conv_burggraaf = curve_fit(exp_fit, cum_area_burggraaf, cum_count_burggraaf/cum_count_burggraaf[0], p0=[1, -0.1])
print(f'Param for burggraaf: {p_burggraaf}')

A_glass = (np.pi/4)*25.4**2
# plot the data for glass 1
ax.plot(cum_area_glass1/A_glass, cum_count_glass1/cum_count_glass1[0], color='tab:green', linestyle = 'none', markersize = 5, marker = 'o', label=r'glass 1')
ax.plot(cum_area_glass1/A_glass, exp_fit(cum_area_glass1, p_glass1[0], p_glass1[1]), color='tab:green')
# plot the data for glass 2
ax.plot(cum_area_glass2/A_glass, cum_count_glass2/cum_count_glass2[0], color='tab:red', linestyle = 'none', markersize = 5, marker = 'o', label=r'glass 2')
ax.plot(cum_area_glass2/A_glass, exp_fit(cum_area_glass2, p_glass2[0], p_glass2[1]), color='tab:red')

A_burggraaf = (np.pi/4)*135**2
# plot the data for burggraaf
ax.plot(cum_area_burggraaf/A_burggraaf, cum_count_burggraaf/cum_count_burggraaf[0], color='tab:purple', linestyle = 'none', markersize = 5, marker = 'o', label=r'Burggraaf')
ax.plot(cum_area_burggraaf/A_burggraaf, exp_fit(cum_area_burggraaf, p_burggraaf[0], p_burggraaf[1]), color='tab:purple')


# ax.set_xlabel(r'$A (\rm{mm}^2)\rightarrow$')
ax.set_xlabel(r'$A/A_{\rm{sample}}\rightarrow$')
ax.set_ylabel(r'fraction of cum. fragments')
ax.legend()
fig.savefig(r'C:\MyData\Stressed network work\Data\m5et'+ '/collated_cum_area_vary_e_and_literature.png', dpi=300, bbox_inches='tight')
plt.show()
