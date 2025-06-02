"""Bond state superimposed over initial lattice"""

# import pickle
# import springlattice as sl
# import matplotlib.pyplot as plt

# from springlattice.crack import update_bond_state
# from springlattice.mesh import mesh_plot

# plt.style.use('./article_preprint.mplstyle')

# folder = r'C:\Users\Admin\Documents\VScode\scripts\Residual Stress Work\model 2\edgecrack_u125_c01_sb04_case4_erf_TF_115X400'
# mesh = sl.LoadMesh(folder)
# mesh.folder = folder
# bond_list = []
# with open(folder + '/delbonds', 'rb') as f:
#     while True:
#         try:
#             data = pickle.load(f) 
#             bond_list.append(data[1])
#         except EOFError:
#             break

# for i in range(len(bond_list)):
#     for id, neigh in bond_list[i]:
#         update_bond_state(mesh, id, neigh)


# mesh_plot(mesh, save=True, filename='crackpattern.png', title='reference lattice')


"""save mesh in reference state"""
# from springlattice.output import meshview
# import numpy as np
# import matplotlib.pyplot as plt
# plt.style.use('./article_preprint.mplstyle')
# folder = r'D:\Residual Stress Work\Data\New Mid plane\decreasing power-law profile\eo005\hole5_d5t125_eb06_dec_powerlaw_m2_R108_e06_254X220'

# meshview(dir_path=folder, time = np.arange(10,30,0.5), save=True)

"""Field plots"""
# import springlattice as sl
# import springlattice.output as so
# import numpy as np
# import matplotlib.pyplot as plt
# import pickle

# plt.style.use('./article_preprint.mplstyle')
# folder = r'C:\Users\Admin\Documents\VScode\scripts\Residual Stress Work\model 2\hole5_eb010_inc_powerlaw_m1_R110_e05_254X220'
# # mesh = sl.LoadMesh(folder)
# # mesh.folder = folder
# # so.triplot_displacement_field(folder, comp ='abs', time = np.arange(0,50,5))

# so.triplot_strain_field(dir_path=folder, comp='max', time = np.arange(0,200,10))

# # so.triplot_stress_field(dir_path=folder, comp='max', time = np.arange(0,500,10), cbarlim = (-0.1, 0.04))

# # x_segs,y_segs = so.generate_crack_path(folder, till_time=20)

# # # fig, ax = plt.subplots()
# # # for x, y in zip(x_segs,y_segs):
# # #     ax.plot(x,y, color = 'red')
# # # ax.axis('equal')   
# # # plt.show()


"""Field plots"""
import springlattice as sl
import springlattice.output_1 as so
import numpy as np
import matplotlib.pyplot as plt
import pickle

plt.style.use('./article_preprint.mplstyle')
folder = r'C:\MyData\Stressed network work\Data\hole5_d5t125_et03_dec_m5_R108_e02_254X220'
# mesh = sl.LoadMesh(folder)
# mesh.folder = folder


# so.triplot_strain_field(dir_path=folder, comp='max', time = np.arange(0,200,10), cbarlim = (-0.03, 0.05))
so.triplot_stress_field(dir_path=folder, comp='max', time = np.arange(0,50,10))