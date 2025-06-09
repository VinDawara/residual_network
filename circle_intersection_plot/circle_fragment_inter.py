import pickle
import springlattice as sl
import matplotlib.pyplot as plt
import matplotlib as mpl
import networkx as nx
from springlattice.crack import update_bond_state
from springlattice.mesh import mesh_plot
import numpy as np
from shapely.geometry import Polygon
from shapely import coverage_union
from shapely import unary_union
from shapely.plotting import plot_polygon
from shapely import area, length
import matplotlib.colors as mcolors
import os

# specify the mplstyle file for the plot
mplstyle_file = r'C:\Users\vinee\OneDrive\Documents\vscode\stressed network model\article_preprint.mplstyle'
plt.style.use(f'{mplstyle_file}')


def get_hexagonal_vertices(xy):
    vertices = np.zeros((6,2))
    vertices[0] = xy + np.array([0.51, 0.51/np.sqrt(3)])
    vertices[1] = xy + np.array([0, 1.1/np.sqrt(3)])
    vertices[2] = xy + np.array([-0.51, 0.51/np.sqrt(3)])
    vertices[3] = xy + np.array([-0.51, -0.51/np.sqrt(3)])
    vertices[4] = xy + np.array([0.1, -1.1/np.sqrt(3)])
    vertices[5] = xy + np.array([0.51, -0.51/np.sqrt(3)])

    return vertices

def extract_reference_mesh(path, time):
    mesh = sl.LoadMesh(path)
    mesh.folder = path
    bond_list = []
    with open(path + '/delbonds', 'rb') as f:
        while True:
            try:
                tt, bond_ids = pickle.load(f)                
            except EOFError:
                break
            if tt>time:
                break
            bond_list.append(bond_ids)

    for i in range(len(bond_list)):
        for id, neigh in bond_list[i]:
            update_bond_state(mesh, id, neigh)

    return mesh    

def network(mesh_obj):
    # Creating empty graph
    G = nx.Graph()
    Edgelist = []

    for id in range(0,len(mesh_obj.pos)):
        G.add_node(id, pos = mesh_obj.pos[id])
        for neigh, ns, aph in zip(mesh_obj.neighbors[id], mesh_obj.normal_stiffness[id], mesh_obj.angles[id]):
            if ns:
                if aph in [0, 60, 120]:
                    Edgelist.append([id, neigh])
                    
    G.add_edges_from(Edgelist)
    
    return G

def update_mesh_extra_bonds(mesh, folder):
    extra_bond_broken_ids = []
    extra_bond_file = folder + '/extra_bond_broken_ids'

    if os.path.exists(extra_bond_file):
        with open(extra_bond_file, 'rb') as f:
            while True:
                try:
                    bond_ids = pickle.load(f)
                except EOFError:
                    break
            extra_bond_broken_ids.extend(bond_ids)

    for id, neigh in extra_bond_broken_ids:
        update_bond_state(mesh, id, neigh) 

    return mesh    


"""Find number of fragment intersecting the circle for varying radius and center of circle"""
# data_dir_path = r'C:\MyData\Stressed network work\Data\m5et\hole5_d5t125_et035_dec_m5_R108_e02_254X220'

# mesh = extract_reference_mesh(data_dir_path, time=250)

# # remove extra bonds
# mesh = update_mesh_extra_bonds(mesh, data_dir_path)

# graph = network(mesh)

# graph_pos = nx.get_node_attributes(graph, 'pos')

# frag_poly = []
# for i, c in enumerate(sorted(nx.connected_components(graph), key=len, reverse=True)):
#         if len(c)>=2:
#             s = graph.subgraph(c)
#             pos = nx.get_node_attributes(s, 'pos')

#             merged_hexagon = []
#             for node in s.nodes():
#                 vertices = get_hexagonal_vertices(graph_pos[node])
#                 hexagon = Polygon(vertices)
#                 merged_hexagon.append(hexagon)

#             merged_hexagon = unary_union(merged_hexagon)
#             frag_poly.append(merged_hexagon)

# # circle
# center = (110, 110)
# radius = np.arange(0,100,0.5)
# theta = np.linspace(0, 2*np.pi, 100)

# count = np.zeros(len(radius))

# for n in range(len(radius)):
#     x = center[0] + radius[n]*np.cos(theta)
#     y = center[1] + radius[n]*np.sin(theta)
#     circle_polygon = Polygon(np.column_stack((x, y)))

#     for poly in frag_poly:
#         # Check if merged_hexagon intersects the circle boundary but not inside the circle
#         if poly.area*0.27**2 > 0: #2
#             if poly.intersects(circle_polygon) and not poly.within(circle_polygon):
#                 count[n] += 1

#     print(f'radius: {radius[n]}, count: {count[n]}')
    
            

# # save the data
# with open(data_dir_path + f'/circle_fragments_inter', 'wb') as f:
#     pickle.dump([radius, count], f)

# # plot the number of intersecting bonds
# fig, ax = plt.subplots()
# ax.plot(radius, count)
# ax.set_xlabel(r'$r$')
# ax.set_ylabel(r'$\# \rm{fragments}$')
# plt.show()
# fig.savefig(data_dir_path + '/circle_fragments_inter.png', bbox_inches='tight', dpi = 300)



"""Checking for given radius and center of circle, if the fragments intersect the circle boundary but not inside the circle"""
# folder = r'C:\MyData\Stressed network work\Data\shape_size\ellipse_10X5_d5t125_et03_dec_m5_R108_e02_254X220'

# mesh = extract_reference_mesh(folder, time=250)

# # remove extra bonds
# mesh = update_mesh_extra_bonds(mesh, folder)

# graph = network(mesh)

# # circle
# center = (108, 110)
# radius = 4
# theta = np.linspace(0, 2*np.pi, 100)
# x = center[0] + radius*np.cos(theta)
# y = center[1] + radius*np.sin(theta)
# circle_polygon = Polygon(np.column_stack((x, y)))

# fig, ax = plt.subplots()
# ax.set_aspect('equal')
# ax.set_xlim([mesh.domain[0]-10, mesh.domain[1]+10])
# ax.set_ylim([mesh.domain[2]-10, mesh.domain[3]+10])


# # Define n distinct colors
# n = len(list(nx.connected_components(graph)))
# color_set = list(mcolors.CSS4_COLORS.values())
# total_colors = len(color_set)
# colors = [color_set[i % total_colors] for i in range(n)]


# graph_pos = nx.get_node_attributes(graph, 'pos')

# Total_area = (mesh.nx-1)*(mesh.ny-1)*np.sqrt(3)/2
# count = 0

# poly = []
# for i, c in enumerate(sorted(nx.connected_components(graph), key=len, reverse=True)):
#     if len(c)>=2:
#         s = graph.subgraph(c)
#         pos = nx.get_node_attributes(s, 'pos')

#         merged_hexagon = []
#         for node in s.nodes():
#             vertices = get_hexagonal_vertices(graph_pos[node])
#             hexagon = Polygon(vertices)
#             merged_hexagon.append(hexagon)

#         merged_hexagon = unary_union(merged_hexagon)
        
#         # circle_polygon = Polygon(circle.get_path().vertices * radius + center)
        


#         # Check if merged_hexagon intersects the circle boundary but not inside the circle
#         if merged_hexagon.area > 1.5:
#             if merged_hexagon.intersects(circle_polygon) and not merged_hexagon.within(circle_polygon):
#                 plot_polygon(merged_hexagon, ax=ax, alpha=0.5, linewidth=2, color='gray',edgecolor='black', add_points=False)
#                 count += 1
#                 poly.append(merged_hexagon)
  
        



# circle = plt.Circle(center, radius, color='r', fill=False)
# ax.add_patch(circle)        
# print(f'count: {count}')
# plt.show()

# # fig, ax = plt.subplots()
# # ax.set_aspect('equal')
# # ax.set_xlim([mesh.domain[0]-10, mesh.domain[1]+10])
# # ax.set_ylim([mesh.domain[2]-10, mesh.domain[3]+10])
# # for i, p in enumerate(poly):
# #     plot_polygon(p, ax=ax, alpha=0.7, linewidth=0.5, color=colors[i], add_points=False)
# #     ax.set_title(f'Fragment {i+1}')
# #     plt.draw()
# #     plt.pause(2)  # Pause for 2 seconds to display each polygon
    
# # circle = plt.Circle(center, radius, color='r', fill=False)
# # ax.add_patch(circle)
# # plt.show()


"""Collated results in single plot for varying breaking strain"""
# filename = 'circle_fragments_inter'
# data_dir_path = r'C:\MyData\Stressed network work\Data\hole5_d5t125_et03_dec_m5_R108_e02_254X220\circle_fragment_intersection'
# with open(data_dir_path + f'/{filename}', 'rb') as f:
#     radius_2, nbonds_2 = pickle.load(f)

# data_dir_path = r'C:\MyData\Stressed network work\Data\m5et\hole5_d5t125_et035_dec_m5_R108_e02_254X220\Circle fragments intersection'
# with open(data_dir_path + f'/{filename}', 'rb') as f:
#     radius_3, nbonds_3 = pickle.load(f)

# data_dir_path = r'C:\MyData\Stressed network work\Data\m5et\hole5_d5t125_et04_dec_m5_R108_e02_254X220\Circle fragment intersection'
# with open(data_dir_path + f'/{filename}', 'rb') as f:
#     radius_5, nbonds_5 = pickle.load(f)

# L = 110
# fig, ax = plt.subplots()
# ax.plot(radius_2/L, nbonds_2, label = r'$\epsilon_{b} = 0.030$')
# ax.plot(radius_3/L, nbonds_3, label = r'$\epsilon_{b} = 0.035$')
# ax.plot(radius_5/L, nbonds_5, label = r'$\epsilon_{b} = 0.040$')
# ax.set_xlabel(r'$2r/L\rightarrow$')
# ax.set_ylabel(r'$N_c\rightarrow$')
# ax.legend()
# ax.set_xlim([0,1])
# ax.set_ylim(bottom=0)
# plt.show()
# fig.savefig(data_dir_path + '/circle_fragments_inter_et_collated.png', bbox_inches='tight', dpi = 300)


"""Collated results in single plot for varying profile"""
# data_dir_path = r'C:\MyData\Stressed network work\Data\hole5_d5t125_et03_dec_m2_R108_e02_254X220\circle_fragments_intersection'
# with open(data_dir_path + f'/circle_fragments_inter', 'rb') as f:
#     radius_2, nbonds_2 = pickle.load(f)

# data_dir_path = r'C:\MyData\Stressed network work\Data\hole5_d5t125_et03_dec_m3_R108_e02_220X254\circle_fragments_intersection'
# with open(data_dir_path + f'/circle_fragments_inter', 'rb') as f:
#     radius_3, nbonds_3 = pickle.load(f)

# data_dir_path = r'C:\MyData\Stressed network work\Data\hole5_d5t125_et03_dec_m5_R108_e02_254X220\circle_fragment_intersection'
# with open(data_dir_path + f'/circle_fragments_inter', 'rb') as f:
#     radius_5, nbonds_5 = pickle.load(f)

# fig, ax = plt.subplots()
# ax.plot(radius_2/110, nbonds_2, label = r'$m=2$')
# ax.plot(radius_3/110, nbonds_3, label = r'$m=3$')
# ax.plot(radius_5/110, nbonds_5, label = r'$m=5$')
# ax.set_xlabel(r'$2r/L\rightarrow$')
# ax.set_ylabel(r'$N_c\rightarrow$')
# ax.legend()
# ax.set_xlim([0,1])
# ax.set_ylim(bottom=0)
# plt.show()
# fig.savefig(data_dir_path + '/circle_fragments_inter_m_collated.png', bbox_inches='tight', dpi = 300)


"""Collate data in single plot for varying hole sizes and shape"""
# data_dir_path = r'C:\MyData\Stressed network work\Data\hole5_d5t125_et03_dec_m5_R108_e02_254X220\circle_fragment_intersection'
# with open(data_dir_path + f'/circle_fragments_inter', 'rb') as f:
#     radius_1, nbonds_1 = pickle.load(f)

# data_dir_path = r'C:\MyData\Stressed network work\Data\shape_size\hole10_d5t125_et03_dec_m5_R108_e02_254X220\circle_fragment_intersection'
# with open(data_dir_path + f'/circle_fragments_inter', 'rb') as f:
#     radius_3, nbonds_3 = pickle.load(f)

# data_dir_path = r'C:\MyData\Stressed network work\Data\shape_size\ellipse_10X5_d5t125_et03_dec_m5_R108_e02_254X220\circle_fragment_intersection'
# with open(data_dir_path + f'/circle_fragments_inter', 'rb') as f:
#     radius_4, nbonds_4 = pickle.load(f)


# data_dir_path = r'C:\MyData\Stressed network work\Data\shape_size\ellipse_15X5_d5t125_et03_dec_m5_R108_e02_254X220\circle_fragment_intersection'
# with open(data_dir_path + f'/circle_fragments_inter', 'rb') as f:
#     radius_5, nbonds_5 = pickle.load(f)

# max_radius = 100
# fig, ax = plt.subplots()
# ax.plot(radius_1/max_radius, nbonds_1, label = r'small circle $2D_c/L=0.09$', color = 'tab:blue')
# ax.plot(radius_3/max_radius, nbonds_3, label = r'big circle $2D_c/L=0.18$', color = 'tab:orange')
# ax.plot(radius_4/max_radius, nbonds_4, label = r'small ellipse $2D_x/L=0.09, 2D_y/L=0.04$', color = 'tab:green')
# ax.plot(radius_5/max_radius, nbonds_5, label = r'big ellipse $2D_x/L=0.13, 2D_y/L=0.04$', color = 'tab:cyan')

# ax.set_xlabel(r'$2r/L\rightarrow$')
# ax.set_ylabel(r'$N_c\rightarrow$')
# ax.set_xlim([0,1])
# ax.set_ylim(bottom=0)
# # ax.legend()
# plt.show()
# fig.savefig(data_dir_path + '/collated_circle_frag_inter_hole_sizes_shape_norm.png', bbox_inches='tight', dpi = 300)

"""Collate data separately for different loads"""
# data_dir_path = r'C:\MyData\Stressed network work\Data\hole5_d5t125_et03_dec_m5_R108_e02_254X220\circle_fragment_intersection'
# with open(data_dir_path + f'/circle_fragments_inter', 'rb') as f:
#     radius_1, nbonds_1 = pickle.load(f)


# data_dir_path = r'C:\MyData\Stressed network work\Data\shape_size\Hole5_d05t25_et03_dec_m5_R108_e02_254X220\circle_fragment_intersection'
# with open(data_dir_path + f'/circle_fragments_inter', 'rb') as f:
#     radius_2, nbonds_2 = pickle.load(f)


# max_radius = 110
# fig, ax = plt.subplots()
# ax.plot(radius_2/max_radius, nbonds_2, label = r'impulse $\mathbf{\dot{u}}_i = 0.02 c_r$', color = 'tab:blue')
# ax.plot(radius_1/max_radius, nbonds_1, label = r'ramp $\mathbf{\dot{u}}_i = 0.04 c_r$', color = 'tab:orange')


# ax.set_xlabel(r'$2r/L\rightarrow$')
# ax.set_ylabel(r'$N_c\rightarrow$')
# ax.set_xlim([0,1])
# ax.set_ylim(bottom=0)
# ax.legend()
# plt.show()
# fig.savefig(data_dir_path + '/collated_circle_frag_inter_loads_flipcolor.png', bbox_inches='tight', dpi = 300)


"""Collate data in single plot for experimental fracture pattern"""
import h5py

filename = 'circle_intersect_count'

data_dir_path = r'C:\Users\vinee\OneDrive\Documents\MATLAB\Test1\analysis'
with h5py.File(data_dir_path + f'/{filename}.mat', 'r') as f:
    radius_1 = f['radius'][:]; radius_1 = np.squeeze(radius_1)
    nbonds_1 = f['count'][:]; nbonds_1 = np.squeeze(nbonds_1)

data_dir_path = r'C:\Users\vinee\OneDrive\Documents\MATLAB\Test2\analysis'
with h5py.File(data_dir_path + f'/{filename}.mat', 'r') as f:
    radius_2 = f['radius'][:]; radius_2 = np.squeeze(radius_2)
    nbonds_2 = f['count'][:]; nbonds_2 = np.squeeze(nbonds_2)

data_dir_path = r'C:\Users\vinee\OneDrive\Documents\MATLAB\Test3\analysis'
with h5py.File(data_dir_path + f'/{filename}.mat', 'r') as f:
    radius_3 = f['radius'][:]; radius_3 = np.squeeze(radius_3)
    nbonds_3 = f['count'][:]; nbonds_3 = np.squeeze(nbonds_3)    

data_dir_path = r'C:\Users\vinee\OneDrive\Documents\MATLAB\Test6\analysis'
with h5py.File(data_dir_path + f'/{filename}.mat', 'r') as f:
    radius_6 = f['radius'][:]; radius_6 = np.squeeze(radius_6)
    nbonds_6 = f['count'][:]; nbonds_6 = np.squeeze(nbonds_6)

data_dir_path = r'C:\Users\vinee\OneDrive\Documents\MATLAB\Test7\analysis'
with h5py.File(data_dir_path + f'/{filename}.mat', 'r') as f:
    radius_7 = f['radius'][:]; radius_7 = np.squeeze(radius_7)
    nbonds_7 = f['count'][:]; nbonds_7 = np.squeeze(nbonds_7)

data_dir_path = r'C:\Users\vinee\OneDrive\Documents\MATLAB\Test8\analysis'
with h5py.File(data_dir_path + f'/{filename}.mat', 'r') as f:
    radius_8 = f['radius'][:]; radius_8 = np.squeeze(radius_8)
    nbonds_8 = f['count'][:]; nbonds_8 = np.squeeze(nbonds_8)      


# average of data 1, 2, and 3
radius123 = np.mean([radius_1, radius_2, radius_3], axis=0)
radius123 = radius123/radius123.max()
nbonds123 = np.mean([nbonds_1, nbonds_2, nbonds_3], axis=0)
nbonds123_std = np.std([nbonds_1, nbonds_2, nbonds_3], axis=0)

# average of data 6, 7, and 8
radius678 = np.mean([radius_6, radius_7, radius_8], axis=0)
radius678 = radius678/radius678.max()
nbonds678 = np.mean([nbonds_6, nbonds_7, nbonds_8], axis=0)
nbonds678_std = np.std([nbonds_6, nbonds_7, nbonds_8], axis=0)

fig, ax = plt.subplots()
# put projectile region
ax.axvspan(0, 1.2/30, color='lightgray', alpha=0.3, label='projectile region')
# put minimum un-accessible region
ax.axvspan(1.2/30, 4/30, color='lightyellow', alpha=1, label='minimum un-accessible region')

ax.plot(radius678, nbonds678, label = r'$v = 20$ m/s', color = 'tab:blue')
ax.fill_between(radius678, nbonds678 - nbonds678_std, nbonds678 + nbonds678_std, alpha=0.2, color='tab:blue')

ax.plot(radius123, nbonds123, label = r'$v = 35$ m/s', color = 'tab:orange')
ax.fill_between(radius123, nbonds123 - nbonds123_std, nbonds123 + nbonds123_std, alpha=0.2, color='tab:orange')


ax.set_xlabel(r'$2r/L\rightarrow$')
ax.set_ylabel(r'$N_c\rightarrow$')
# ax.legend()
ax.set_xlim(left = 0)
ax.set_ylim(bottom = 0)
fig.savefig(data_dir_path + '/collated_circle_frag_inter.png', bbox_inches='tight', dpi = 300)
plt.show()


"""Collated data for line intersection for experimental fracture pattern"""
# import h5py

# filename = 'line_intersect_count'

# data_dir_path = r'C:\Users\vinee\OneDrive\Documents\MATLAB\Test1\analysis'
# with h5py.File(data_dir_path + f'/{filename}.mat', 'r') as f:
#     radius_1 = f['theta'][:]; radius_1 = np.squeeze(radius_1)
#     nbonds_1 = f['count'][:]; nbonds_1 = np.squeeze(nbonds_1)

# data_dir_path = r'C:\Users\vinee\OneDrive\Documents\MATLAB\Test2\analysis'
# with h5py.File(data_dir_path + f'/{filename}.mat', 'r') as f:
#     radius_2 = f['theta'][:]; radius_2 = np.squeeze(radius_2)
#     nbonds_2 = f['count'][:]; nbonds_2 = np.squeeze(nbonds_2)

# data_dir_path = r'C:\Users\vinee\OneDrive\Documents\MATLAB\Test3\analysis'
# with h5py.File(data_dir_path + f'/{filename}.mat', 'r') as f:
#     radius_3 = f['theta'][:]; radius_3 = np.squeeze(radius_3)
#     nbonds_3 = f['count'][:]; nbonds_3 = np.squeeze(nbonds_3)    

# data_dir_path = r'C:\Users\vinee\OneDrive\Documents\MATLAB\Test6\analysis'
# with h5py.File(data_dir_path + f'/{filename}.mat', 'r') as f:
#     radius_6 = f['theta'][:]; radius_6 = np.squeeze(radius_6)
#     nbonds_6 = f['count'][:]; nbonds_6 = np.squeeze(nbonds_6)

# data_dir_path = r'C:\Users\vinee\OneDrive\Documents\MATLAB\Test7\analysis'
# with h5py.File(data_dir_path + f'/{filename}.mat', 'r') as f:
#     radius_7 = f['theta'][:]; radius_7 = np.squeeze(radius_7)
#     nbonds_7 = f['count'][:]; nbonds_7 = np.squeeze(nbonds_7)

# data_dir_path = r'C:\Users\vinee\OneDrive\Documents\MATLAB\Test8\analysis'
# with h5py.File(data_dir_path + f'/{filename}.mat', 'r') as f:
#     radius_8 = f['theta'][:]; radius_8 = np.squeeze(radius_8)
#     nbonds_8 = f['count'][:]; nbonds_8 = np.squeeze(nbonds_8)      


# # average of data 1, 2, and 3
# radius123 = np.mean([radius_1, radius_2, radius_3], axis=0)
# nbonds123 = np.mean([nbonds_1, nbonds_2, nbonds_3], axis=0)
# nbonds123_std = np.std([nbonds_1, nbonds_2, nbonds_3], axis=0)

# # average of data 6, 7, and 8
# radius678 = np.mean([radius_6, radius_7, radius_8], axis=0)
# nbonds678 = np.mean([nbonds_6, nbonds_7, nbonds_8], axis=0)
# nbonds678_std = np.std([nbonds_6, nbonds_7, nbonds_8], axis=0)

# fig, ax = plt.subplots()
# ax.plot(radius678, nbonds678, label = r'$v = 20$ m/s', color = 'tab:orange')
# ax.fill_between(radius678, nbonds678 - nbonds678_std, nbonds678 + nbonds678_std, alpha=0.2, color='tab:orange')

# ax.plot(radius123, nbonds123, label = r'$v = 35$ m/s', color = 'tab:blue')
# ax.fill_between(radius123, nbonds123 - nbonds123_std, nbonds123 + nbonds123_std, alpha=0.2, color='tab:blue')

# ax.set_xlabel(r'$\theta (\rm{degree})\rightarrow$')
# ax.set_ylabel(r'$\# \rm{fragments}\rightarrow$')
# ax.legend()
# fig.savefig(data_dir_path + '/collated_line_frag_inter.png', bbox_inches='tight', dpi = 300)
# plt.show()
