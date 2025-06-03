import springlattice as sl
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import tri
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.collections import LineCollection
import pickle
import h5py
from springlattice.output_1 import load_mesh, nodes_inside_hole, generate_crack_segs, compute_evalues
from springlattice.crack import update_bond_state, compute_nodal_stress_tensor
# specify the mplstyle file for the plot
mplstyle_file = r'C:\Users\vinee\OneDrive\Documents\vscode\stressed network model\article_preprint.mplstyle'
plt.style.use(f'{mplstyle_file}')

folder = r'C:\MyData\Stressed network work\Data\hole5_d5t125_et03_dec_m5_R108_e02_254X220'
# mesh = sl.LoadMesh(folder)
# mesh.folder = folder

"""
This script visualizes the stress field for a given time step in a spring lattice model.
It computes the stress tensor at each node, extracts the desired component (xx, xy, yy, max, rr, rt, tt, rt_ratio),
and plots the field using a triangulation of the node positions. The script also handles broken bonds and generates crack segments if necessary.
"""

"""Function to visualize the stress field"""
def triplot_stress_field(dir_path, comp, time, **kwargs):

    def stress_polar_coords(x, y, stress):

        # center
        xc, yc = np.mean(x), np.mean(y)
        x = x - xc; y = y - yc
        # polar coordinates
        theta = np.arctan2(y,x)

        # stress in polar coordinates
        stress_polar = np.zeros_like(stress)
        for id in range(len(x)):
            q = np.array([[np.cos(theta[id]), np.sin(theta[id])], [-np.sin(theta[id]), np.cos(theta[id])]])
            s = np.array([[stress[id,0], stress[id,1]], [stress[id,2], stress[id,3]]])
            s = np.matmul(q, np.matmul(s, q.T))
            stress_polar[id] = np.array([s[0,0], s[0,1], s[1,0], s[1,1]])

        return stress_polar

    # step 0: extract time
    save = kwargs.get('save', True)
    cbarlim = kwargs.get('cbarlim', None)
    
    # step 1: load mesh
    mesh_obj = load_mesh(dir_path)
    mesh_obj.folder = dir_path
    pos = mesh_obj.pos
    dt = mesh_obj.solver['dt']
    skipsteps = mesh_obj.solver['skipsteps']

    # create a mesh grid
    xo, yo = pos[:,0], pos[:,1]
    xo_min, xo_max = xo.min()+1, xo.max()-1
    yo_min, yo_max = yo.min()+0.5, yo.max()-0.5

    # # mask hole ids
    mask_nodes = nodes_inside_hole(mesh_obj, pos) if mesh_obj.circle else []
        
    # handler for displacement data file
    disp_file = h5py.File(mesh_obj.folder + '/disp.h5', 'r')

    # time steps corresponding to the time intervals
    time_step = int((time/(dt*skipsteps)))    
    t = time_step*skipsteps*dt

    # handler for broken bonds info file
    del_bonds_ids = []
    with open(mesh_obj.folder + '/delbonds', 'rb') as del_bond_file:
        while True:
            try:
                tt, bonds_ids = pickle.load(del_bond_file)
            except EOFError:
                print('End of file reached')
                break
            if tt > t-dt:
                break

            del_bonds_ids.extend(bonds_ids)

        if del_bonds_ids:   
            for id, neigh in del_bonds_ids:
                update_bond_state(mesh_obj, node_id=id, neighbor_id=neigh)       
            crack_segs = generate_crack_segs(mesh_obj, del_bonds_ids)
        else:
            crack_segs = None        
    
            
        # compute strain tensor
        mesh_obj.u[:,0] = disp_file['u'][time_step]
        mesh_obj.u[:,1] = disp_file['v'][time_step]

        stress = compute_nodal_stress_tensor(mesh_obj)

        if comp == 'xx':
            field = stress[:,0]
        elif  comp == 'xy':
            field = stress[:,1]
        elif comp == 'yy':
            field = stress[:,3]
        elif comp == 'max':
            field = compute_evalues(stress)   
        elif comp == 'rr':
            field = stress_polar_coords(xo, yo, stress)[:,0]
        elif comp == 'rt':
            field = stress_polar_coords(xo, yo, stress)[:,1]
        elif comp == 'tt':
            field = stress_polar_coords(xo, yo, stress)[:,3]    
        elif comp == 'rt_ratio':
            stress_polar = stress_polar_coords(xo, yo, stress)
            field = np.divide(stress_polar[:,3], stress_polar[:,0], out=np.zeros_like(stress_polar[:,0]), where=stress_polar[:,0]!=0)


        # creating figure
        # trifield(xo,yo, field, xlim = (xo_min,xo_max), 
        #                     ylim = (yo_min, yo_max), save = save, path = path,
        #                     filename = f'{time_steps[i]}', title = f'T = {t[i]}',
        #                     mask = mask_nodes, crackpattern = crack_segs, cbarlim = cbarlim) 
        # temporary fix
        L = 109.5
        xc = np.mean(xo); yc = np.mean(yo)
        if crack_segs is not None:
            crack_segs[:,0,0] = (crack_segs[:,0,0]-xc)/L
            crack_segs[:,0,1] = (crack_segs[:,0,1]-yc)/L
            crack_segs[:,1,0] = (crack_segs[:,1,0]-xc)/L
            crack_segs[:,1,1] = (crack_segs[:,1,1]-yc)/L

            # longitudinal and shear waves
        center = (110,110)
        theta = np.linspace(0, 2*np.pi, 100)
        x_cl = (1.8*t+5)*np.cos(theta)/L
        y_cl = (1.8*t+5)*np.sin(theta)/L
        cl_circle = np.array([x_cl, y_cl]).T
        x_cs = (1.1*t+5)*np.cos(theta)/L
        y_cs = (1.1*t+5)*np.sin(theta)/L
        cs_circle = np.array([x_cs, y_cs]).T
    
        trifield((xo-xc)/L,(yo-yc)/L, field, xlim = (-1,1), 
                            ylim = (-1, 1), save = save, path = dir_path,
                            filename = f'customplot_trans_{time_step}', title = f'$2tc_r/L = {np.round(t/L,2)}$',
                            mask = mask_nodes, crackpattern = crack_segs, cbarlim = cbarlim, cl_circle = cl_circle, cs_circle = cs_circle) 

        
        # trifield((xo-xc)/L,(yo-yc)/L, field, xlim = (0.5,1), 
        #                     ylim = (0.5, 1), save = save, path = dir_path,
        #                     filename = f'customplot_{time_step}_inset', title = f'$2tc_r/L = {np.round(t/L,2)}$',
        #                     mask = mask_nodes, crackpattern = crack_segs, cbarlim = cbarlim, cl_circle = cl_circle, cs_circle = cs_circle) 
    
    disp_file.close()


def trifield(x:np.ndarray,y:np.ndarray,f:np.ndarray, **kwargs):
    # read optional keywords
    title = kwargs.get('title', 'field')
    cbarlim = kwargs.get('cbarlim', None)
    save = kwargs.get('save', False)
    xlim = kwargs.get('xlim', None)
    ylim = kwargs.get('ylim', None)
    path = kwargs.get('path', None)
    filename= kwargs.get('filename', 'disp_field')
    mask = kwargs.get('mask', None)
    crackpattern = kwargs.get('crackpattern', None)
    cs_circle = kwargs.get('cs_circle', None)
    cl_circle = kwargs.get('cl_circle', None)

    triang = tri.Triangulation(x,y)

    if mask is not None:
        f[mask] = np.nan    

    if not xlim:
        xlim = (np.min(x)-2, np.max(x)+2)

    if not ylim:
        ylim = (np.min(y), np.max(y))     

    fig, ax = plt.subplots()
    # set axes properties
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    # ax.xaxis.set_tick_params(which='both', bottom = True, labelbottom=False)
    # ax.set_xlabel(r'$2x/L\rightarrow$')
    # ax.set_ylabel(r'$2y/L\rightarrow$')
    # ax.set_axis_off()
    # ax.set_title(title)
    ax.set_aspect('equal')
    # ax_divider= make_axes_locatable(ax)
    # # add an axes to the right of the main axes
    # cax = ax_divider.append_axes("right", size="7%", pad="2%")
    if not cbarlim:
        ul = np.nanmax(f)
        ll = np.nanmin(f)
        cbarlim = [ll, ul]

    surf = ax.tripcolor(x,y,f,cmap='jet', vmin=cbarlim[0], vmax=cbarlim[1])
    surf.set_clim(cbarlim)
    # cbar = fig.colorbar(surf, cax=cax, ticks = np.round(np.linspace(cbarlim[0], cbarlim[1],6),3))
    # cbar.set_label(r'$\sigma_{\rm{max}}/\rho c_r^2\rightarrow$', rotation=90, labelpad=10)

    # f = np.nan_to_num(f, nan=0.0, posinf=0.0, neginf=0.0)
    # surf = ax.tricontourf(x,y, f, levels=np.array([0.04, 0.045, 0.05, 0.055, 0.06]), cmap='jet', vmin=0.04, vmax=0.06)
    # fig.colorbar(surf, cax=cax, ticks = np.array([0.04, 0.045, 0.05, 0.055, 0.06]))
    # surf.set_clim([0.04, 0.06]) 
    if crackpattern is not None:
        crack_segs = LineCollection(crackpattern, linewidths = 2, color = 'white')
        ax.add_collection(crack_segs)

    if cl_circle is not None:
        ax.plot(cl_circle[:,0], cl_circle[:,1], color = 'magenta', linestyle = '--', linewidth = 2)

    if cs_circle is not None:
        ax.plot(cs_circle[:,0], cs_circle[:,1], color = 'brown', linestyle = '--', linewidth = 2)

    if save:
        fig.savefig(path + f'/{filename}.png', bbox_inches = 'tight' , dpi = 300, transparent=True)
        plt.close()
    else:
        plt.show()  

triplot_stress_field(dir_path=folder, comp='max', time = 100, save=True, cbarlim = (0, 0.1))
