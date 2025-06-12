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

folder = r'C:\MyData\Stressed network work\Data\shape_size\ellipse_10X5_d5t125_et03_dec_m5_R108_e02_254X220'

mesh = extract_reference_mesh(folder, time=250)

# remove extra bonds
mesh = update_mesh_extra_bonds(mesh, folder)

graph = network(mesh)

# Define n distinct colors
n = len(list(nx.connected_components(graph)))
color_set = list(mcolors.CSS4_COLORS.values())
total_colors = len(color_set)
colors = [color_set[i % total_colors] for i in range(n)]


graph_pos = nx.get_node_attributes(graph, 'pos')
size = []
Total_area = (mesh.nx-1)*(mesh.ny-1)*np.sqrt(3)/2

for i, c in enumerate(sorted(nx.connected_components(graph), key=len, reverse=True)):
    if len(c)>=2:
        s = graph.subgraph(c)
        pos = nx.get_node_attributes(s, 'pos')

        merged_hexagon = []
        for node in s.nodes():
            vertices = get_hexagonal_vertices(graph_pos[node])
            hexagon = Polygon(vertices)
            merged_hexagon.append(hexagon)

        merged_hexagon = unary_union(merged_hexagon)
  
        # size.append(np.sqrt(4*merged_hexagon.area/(np.pi)))

        size.append(merged_hexagon.area)

        
        
total_count = len(size)
print(f'total fragments: {total_count}')
fig, ax = plt.subplots()
counts, bins, patches = ax.hist(size, bins=np.arange(3,70,2), edgecolor='black', weights=np.ones(len(size)) / total_count)
ax.set_xlabel(r'$d_s\rightarrow$')
ax.set_ylabel(r'$\%$ of fragments')
ax.set_ylim([0, 0.16])
# fig.savefig(folder + '/a_size.png', dpi=300, bbox_inches='tight')
plt.show()

# write data to file
with open(folder + '/area_size.txt', 'w') as f:
    for i in range(len(size)):
        f.write(f'{size[i]}\n')