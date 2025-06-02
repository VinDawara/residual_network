# Standard library imports
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from matplotlib.collections import LineCollection
import h5py
import pickle
from scipy.interpolate import griddata

# self defined modules
from . import mesh, crack, solver


def load_mesh(dir, objname='meshobj', arg=None):
    # Load the dictionary file
    with open(f"{dir}/{objname}", 'rb') as f:
        dict_obj = pickle.load(f)

    # Create a mesh_obj using mesh.load_mesh (assuming it's a factory function)
    mesh_obj = mesh.load_mesh(dir, objname) #  main class in module "mesh"

    # Define a mapping of attributes between dictionary keys and subclasses' attributes
    attribute_map = {
        'lbcs': {'ids': 'lbcs.ids', 'fx': 'lbcs.fx', 'fy': 'lbcs.fy', 'fun': 'lbcs.fun'}
    }

    # Assign attributes from the dictionary to the respective sub-classes of mesh_obj
    for subclass, attributes in attribute_map.items():
        if hasattr(mesh_obj, subclass):
            subclass_obj = getattr(mesh_obj, subclass)
            for attr_name, dict_key in attributes.items():
                if dict_key in dict_obj:
                    setattr(subclass_obj, attr_name, dict_obj[dict_key])
                else:
                    print(f"Missing key '{dict_key}' for {subclass}")
        else:
            print(f"Subclass '{subclass}' doesn't exist in mesh_obj")

    return mesh_obj

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
    ax.set_xlabel(r'$2x/L\rightarrow$')
    ax.set_ylabel(r'$2y/L\rightarrow$')
    ax.set_title(title)
    ax.set_aspect('equal')
    ax_divider= make_axes_locatable(ax)
    # add an axes to the right of the main axes
    cax = ax_divider.append_axes("right", size="7%", pad="2%")
    if not cbarlim:
        ul = np.nanmax(f)
        ll = np.nanmin(f)
        cbarlim = [ll, ul]

    surf = ax.tripcolor(x,y,f,cmap='jet', vmin=cbarlim[0], vmax=cbarlim[1])
    surf.set_clim(cbarlim)
    cbar = fig.colorbar(surf, cax=cax, ticks = np.round(np.linspace(cbarlim[0], cbarlim[1],6),3))
    cbar.set_label(r'$\sigma_{\rm{max}}/\rho c_r^2\rightarrow$', rotation=90, labelpad=10)

    # f = np.nan_to_num(f, nan=0.0, posinf=0.0, neginf=0.0)
    # surf = ax.tricontourf(x,y, f, levels=np.array([0.04, 0.045, 0.05, 0.055, 0.06]), cmap='jet', vmin=0.04, vmax=0.06)
    # fig.colorbar(surf, cax=cax, ticks = np.array([0.04, 0.045, 0.05, 0.055, 0.06]))
    # surf.set_clim([0.04, 0.06]) 
    if crackpattern is not None:
        crack_segs = LineCollection(crackpattern, linewidths = 1, color = 'white')
        ax.add_collection(crack_segs)

    if save:
        fig.savefig(path + f'/{filename}.png', bbox_inches = 'tight' , dpi = 300)
        plt.close()
    else:
        plt.show()  

"""Function to plot field""" 
def plotfield(x,y,f, **kwargs):
    # read optional keywords
    title = kwargs.get('title', 'field')
    cbarlim = kwargs.get('cbarlim', None)
    save = kwargs.get('save', False)
    xlim = kwargs.get('xlim', None)
    ylim = kwargs.get('ylim', None)
    path = kwargs.get('path', None)
    filename= kwargs.get('filename', 'field')
    mask = kwargs.get('mask', None)
    crackpattern = kwargs.get('crackpattern', None)

    fig, ax = plt.subplots()
    # set axes properties
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_title(title)
    ax.set_aspect('equal')
    ax_divider= make_axes_locatable(ax)
    # add an axes to the right of the main axes
    cax = ax_divider.append_axes("right", size="7%", pad="2%")
    if not cbarlim:
        ul = np.nanmax(f)
        ll = np.nanmin(f)
        cbarlim = [ll, ul]
        
    surf = ax.pcolormesh(x,y,f,cmap='jet', shading='auto', vmin=cbarlim[0], vmax=cbarlim[1])
    surf.set_clim(cbarlim)
    fig.colorbar(surf, cax=cax, ticks = np.round(np.linspace(cbarlim[0], cbarlim[1],6),4))
    if crackpattern is not None:
        crack_segs = LineCollection(crackpattern, linewidths = 1, color = 'white')
        ax.add_collection(crack_segs)
    
    if save:
        fig.savefig(path + f'/{filename}.png', bbox_inches = 'tight' , dpi = 300)
        plt.close()
    else:
        plt.show()

"""Function to find hole nodes"""
def nodes_inside_hole(mesh_obj: mesh.mesh, pos: np.ndarray):
    center = mesh_obj.circle[0]['center']
    radius = mesh_obj.circle[0]['radius']

    pos_from_center = pos - center
    distances_sq = np.sum(pos_from_center ** 2, axis=1)
    ids = np.where(distances_sq <= radius**2)[0]

    return ids

def compute_evalues(strain):
    return np.array([crack.principal_value(s) for s in strain])

"""Function to reterive the total broken bond ids till given time"""
def get_broken_bonds(mesh_obj:mesh.mesh, time = None):
    dt = mesh_obj.solver['dt']
    skipsteps = mesh_obj.solver['skipsteps']
  
    # time steps corresponding to the time intervals
    if time is not None:
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
            if time is not None and tt > t:
                break
            del_bonds_ids.extend(bonds_ids)

    # remove redundant bonds
    if del_bonds_ids:
        for id, neigh in del_bonds_ids:
            if [neigh, id] in del_bonds_ids:
                del_bonds_ids.remove([neigh, id])

    return del_bonds_ids       



"""Function to update mesh for given time"""
def update_mesh(mesh_obj:mesh.mesh, time=None):
    dt = mesh_obj.solver['dt']
    skipsteps = mesh_obj.solver['skipsteps']
  
    # time steps corresponding to the time intervals
    if time is not None:
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
            if time is not None and tt > t:
                break
            del_bonds_ids.extend(bonds_ids)

    if del_bonds_ids:   
        for id, neigh in del_bonds_ids:
            crack.update_bond_state(mesh_obj, node_id=id, neighbor_id=neigh)

    return mesh_obj        

def generate_crack_segs(mesh_obj, broken_bond_ids):
    n_segs = len(broken_bond_ids)
    segs = np.zeros((n_segs,2,2))
    for i, [id, neigh] in enumerate(broken_bond_ids):
        xc = 0.5*(mesh_obj.pos[id,0] + mesh_obj.pos[neigh,0])
        yc = 0.5*(mesh_obj.pos[id,1] + mesh_obj.pos[neigh,1])
        point = np.array([xc, yc])

        vec = mesh_obj.pos[neigh] - mesh_obj.pos[id]
        perp_vec = np.array([vec[1], -vec[0]])
        r1 = point - 0.288*perp_vec
        r2 = point + 0.288*perp_vec

        segs[i,:,0] = np.array([r1[0], r2[0]])
        segs[i,:,1] = np.array([r1[1], r2[1]])

    return segs  


def generate_crack_segs_list(mesh_obj, broken_bond_ids):
    segs_list = []
    for id, neigh in broken_bond_ids:
        xc = 0.5*(mesh_obj.pos[id,0] + mesh_obj.pos[neigh,0])
        yc = 0.5*(mesh_obj.pos[id,1] + mesh_obj.pos[neigh,1])
        point = np.array([xc, yc])

        vec = mesh_obj.pos[neigh] - mesh_obj.pos[id]
        perp_vec = np.array([vec[1], -vec[0]])
        r1 = point - 0.288*perp_vec
        r2 = point + 0.288*perp_vec

        segs = np.zeros((2,2))
        segs[:,0] = np.array([r1[0], r2[0]])
        segs[:,1] = np.array([r1[1], r2[1]])
        segs_list.append(segs)

    return segs_list

"""Function to visualize the strain field"""
def triplot_strain_field(dir_path, comp, **kwargs):
    # step 0: extract time
    time = kwargs.get('time', None)
    start = kwargs.get('start', None)
    end = kwargs.get('end', None)
    step = kwargs.get('step', None)
    save = kwargs.get('save', True)
    cbarlim = kwargs.get('cbarlim', None)
    
    # step 1: load mesh
    mesh_obj = load_mesh(dir_path)
    mesh_obj.folder = dir_path
    pos = mesh_obj.pos
    dt = mesh_obj.solver['dt']
    skipsteps = mesh_obj.solver['skipsteps']

    # create sub directory to save plots
    path = mesh._sub_directory(f'strain_{comp}', mesh_obj.folder)

    # create a mesh grid
    xo, yo = pos[:,0], pos[:,1]
    xo_min, xo_max = xo.min()+1, xo.max()-1
    yo_min, yo_max = yo.min()+0.5, yo.max()-0.5
    # # create mesh grid
    # nx = 4*mesh_obj.nx
    # ny = 4*mesh_obj.ny
    # x,y = np.meshgrid(np.linspace(xo_min,xo_max,nx), np.linspace(yo_min,yo_max, ny))


    # # mask hole ids
    mask_nodes = nodes_inside_hole(mesh_obj, pos) if mesh_obj.circle else []
        
    # handler for displacement data file
    disp_file = h5py.File(mesh_obj.folder + '/disp.h5', 'r')

    # time steps corresponding to the time intervals
    if time is None:
        time_steps = np.rint((np.arange(start,end,step)/(dt*skipsteps)))
    else:
        time_steps = np.rint((time/(dt*skipsteps)))    
    t = time_steps*skipsteps*dt

    # reading data
    maxsteps = disp_file['u'][:,0].shape[0]
    bucket = 0
    fill_step = int(time_steps[0])
    remain_step = maxsteps - fill_step
    bucket_size = min(10000, maxsteps)

    for i, step in enumerate(time_steps):   
        # mesh connectivity updator
        # handler for broken bonds info file
        del_bonds_ids = []
        with open(mesh_obj.folder + '/delbonds', 'rb') as del_bond_file:
            while True:
                try:
                    tt, bonds_ids = pickle.load(del_bond_file)
                except EOFError:
                    print('End of file reached')
                    break
                if tt > t[i]:
                    break

                del_bonds_ids.extend(bonds_ids)

        if del_bonds_ids:   
            for id, neigh in del_bonds_ids:
                crack.update_bond_state(mesh_obj, node_id=id, neighbor_id=neigh)       
            crack_segs = generate_crack_segs(mesh_obj, del_bonds_ids)
        else:
            crack_segs = None        

        # read displacement data in buckets
        while True:
            if step >= fill_step:
                u = disp_file['u'][fill_step:fill_step+bucket_size]
                v = disp_file['v'][fill_step:fill_step+bucket_size]
                bucket += 1
                idx = fill_step
                fill_step += bucket_size
                remain_step = maxsteps - fill_step 
                if remain_step < bucket_size:
                    bucket_size = remain_step
                print('Data extracted:%4.2f'%((fill_step/maxsteps)*100),
                      'step=',step,'fill=',fill_step,'remain=',remain_step,'idx=',idx,'size=',bucket_size)    
    
            bucket_idx = int(step - idx) 
            if bucket_idx < fill_step-idx:
                break     
            
        # compute strain tensor
        mesh_obj.u[:,0] = u[bucket_idx,:]
        mesh_obj.u[:,1] = v[bucket_idx,:]

        strain = crack.compute_nodal_strain_tensor(mesh_obj)

        if comp == 'xx':
            field = strain[:,0]
        elif  comp == 'xy':
            field = strain[:,1]
        elif comp == 'yy':
            field = strain[:,3]
        elif comp == 'max':
            field = compute_evalues(strain)                

        # fieldq = griddata(pos, field, (x, y), method='cubic')
        # # creating figure
        # plotfield(x,y, fieldq, xlim = (xo_min,xo_max), 
        #                     ylim = (yo_min, yo_max), save = save, path = path,
        #                     filename = f'{time_steps[i]}', title = f'T = {t[i]}',
        #                     mask = mask_nodes, crackpattern = crack_segs, cbarlim = cbarlim)  
        

        # creating figure
        trifield(xo,yo, field, xlim = (xo_min,xo_max), 
                            ylim = (yo_min, yo_max), save = save, path = path,
                            filename = f'{time_steps[i]}', title = f'T = {t[i]}',
                            mask = mask_nodes, crackpattern = crack_segs, cbarlim = cbarlim) 
    
    disp_file.close()

"""Function to visualize the stress field"""
def triplot_stress_field(dir_path, comp, **kwargs):

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
    time = kwargs.get('time', None)
    start = kwargs.get('start', None)
    end = kwargs.get('end', None)
    step = kwargs.get('step', None)
    save = kwargs.get('save', True)
    cbarlim = kwargs.get('cbarlim', None)
    
    # step 1: load mesh
    mesh_obj = load_mesh(dir_path)
    mesh_obj.folder = dir_path
    pos = mesh_obj.pos
    dt = mesh_obj.solver['dt']
    skipsteps = mesh_obj.solver['skipsteps']

    # create sub directory to save plots
    path = mesh._sub_directory(f'stress_{comp}', mesh_obj.folder)

    # create a mesh grid
    xo, yo = pos[:,0], pos[:,1]
    xo_min, xo_max = xo.min()+1, xo.max()-1
    yo_min, yo_max = yo.min()+0.5, yo.max()-0.5

    # # mask hole ids
    mask_nodes = nodes_inside_hole(mesh_obj, pos) if mesh_obj.circle else []
        
    # handler for displacement data file
    disp_file = h5py.File(mesh_obj.folder + '/disp.h5', 'r')

    # time steps corresponding to the time intervals
    if time is None:
        time_steps = np.rint((np.arange(start,end,step)/(dt*skipsteps)))
    else:
        time_steps = np.rint((time/(dt*skipsteps)))    
    t = time_steps*skipsteps*dt

    # reading data
    maxsteps = disp_file['u'][:,0].shape[0]
    bucket = 0
    fill_step = int(time_steps[0])
    remain_step = maxsteps - fill_step
    bucket_size = min(10000, maxsteps)

    for i, step in enumerate(time_steps):   
        # mesh connectivity updator
        # handler for broken bonds info file
        del_bonds_ids = []
        with open(mesh_obj.folder + '/delbonds', 'rb') as del_bond_file:
            while True:
                try:
                    tt, bonds_ids = pickle.load(del_bond_file)
                except EOFError:
                    print('End of file reached')
                    break
                if tt > t[i]-dt:
                    break

                del_bonds_ids.extend(bonds_ids)

        if del_bonds_ids:   
            for id, neigh in del_bonds_ids:
                crack.update_bond_state(mesh_obj, node_id=id, neighbor_id=neigh)       
            crack_segs = generate_crack_segs(mesh_obj, del_bonds_ids)
        else:
            crack_segs = None        

        # read displacement data in buckets
        while True:
            if step >= fill_step:
                u = disp_file['u'][fill_step:fill_step+bucket_size]
                v = disp_file['v'][fill_step:fill_step+bucket_size]
                bucket += 1
                idx = fill_step
                fill_step += bucket_size
                remain_step = maxsteps - fill_step 
                if remain_step < bucket_size:
                    bucket_size = remain_step
                print('Data extracted:%4.2f'%((fill_step/maxsteps)*100),
                      'step=',step,'fill=',fill_step,'remain=',remain_step,'idx=',idx,'size=',bucket_size)    
    
            bucket_idx = int(step - idx) 
            if bucket_idx < fill_step-idx:
                break     
            
        # compute strain tensor
        mesh_obj.u[:,0] = u[bucket_idx,:]
        mesh_obj.u[:,1] = v[bucket_idx,:]

        stress = crack.compute_nodal_stress_tensor(mesh_obj)

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
        L = 110
        xc = np.mean(xo); yc = np.mean(yo)
        if crack_segs is not None:
            crack_segs[:,0,0] = (crack_segs[:,0,0]-xc)/L
            crack_segs[:,0,1] = (crack_segs[:,0,1]-yc)/L
            crack_segs[:,1,0] = (crack_segs[:,1,0]-xc)/L
            crack_segs[:,1,1] = (crack_segs[:,1,1]-yc)/L
        trifield((xo-xc)/L,(yo-yc)/L, field, xlim = (-1,1), 
                            ylim = (-1, 1), save = save, path = path,
                            filename = f'{time_steps[i]}', title = f'$2tc_r/L = {np.round(t[i]/110,2)}$',
                            mask = mask_nodes, crackpattern = crack_segs, cbarlim = cbarlim) 
    
    disp_file.close()

"""Function to reterive crack pattern"""
def get_crack_pattern(mesh_obj, time, return_node_ids=False):
    # mesh objective attributes
    dt = mesh_obj.solver['dt']
    skipsteps = mesh_obj.solver['skipsteps']

    # adjusting time close to the saved time steps
    time_steps = np.rint((time/(dt*skipsteps)))    
    new_time = time_steps*skipsteps*dt

    # handler for broken bonds info file
    del_bonds_ids = []
    with open(mesh_obj.folder + '/delbonds', 'rb') as del_bond_file:
        while True:
            try:
                tt, bonds_ids = pickle.load(del_bond_file)
            except EOFError:
                print('End of file reached')
                break
            if tt > new_time:
                break
            del_bonds_ids.extend(bonds_ids)

    # remove redundant bonds
    if del_bonds_ids:  
        for id, neigh in del_bonds_ids:
            if [neigh, id] in del_bonds_ids:
                del_bonds_ids.remove([neigh, id])

        # for id, neigh in del_bonds_ids:
        #     crack.update_bond_state(mesh_obj, node_id=id, neighbor_id=neigh)       
        crack_segs = generate_crack_segs(mesh_obj, del_bonds_ids)
    else:
        crack_segs = None   

    if return_node_ids:
        return crack_segs, del_bonds_ids
    else:
        return crack_segs  

"""Function to total crack path"""
def get_total_crack_length(mesh_obj, time, return_node_ids=False):
    # mesh objective attributes
    dt = mesh_obj.solver['dt']
    skipsteps = mesh_obj.solver['skipsteps']

    # adjusting time close to the saved time steps
    time_steps = np.rint((time/(dt*skipsteps)))    
    new_time = time_steps*skipsteps*dt

    # handler for broken bonds info file
    del_bonds_ids = []
    with open(mesh_obj.folder + '/delbonds', 'rb') as del_bond_file:
        while True:
            try:
                tt, bonds_ids = pickle.load(del_bond_file)
            except EOFError:
                print('End of file reached')
                break
            if tt > new_time:
                break
            del_bonds_ids.extend(bonds_ids)

    # remove redundant bonds
    if del_bonds_ids:  
        for id, neigh in del_bonds_ids:
            if [neigh, id] in del_bonds_ids:
                del_bonds_ids.remove([neigh, id])

    if return_node_ids:
        return 0.577*len(del_bonds_ids), del_bonds_ids
    else:
        return 0.577*len(del_bonds_ids)

"""Function to estimate the time of crack segments"""
def get_time_crack_segments(mesh_obj, crack_segments, search_time):
    # mesh objective attributes
    dt = mesh_obj.solver['dt']
    skipsteps = mesh_obj.solver['skipsteps']

    # adjusting time close to the saved time steps
    time_steps = np.rint((search_time/(dt*skipsteps)))    
    new_time = time_steps*skipsteps*dt

    # output variables
    total_segs = len(crack_segments)
    time = np.zeros(total_segs)

    # loop through the delbonds file to get the crack segments
    with open(mesh_obj.folder + '/delbonds', 'rb') as del_bond_file:
        while True:
            try:
                tt, bonds_ids = pickle.load(del_bond_file)
            except EOFError:
                print('End of file reached')
                break
            if tt > new_time:
                break


            if bonds_ids:   
                # remove redundant bonds
                for id, neigh in bonds_ids:
                    if [neigh, id] in bonds_ids:
                        bonds_ids.remove([neigh, id])

                curr_crack_segs = generate_crack_segs_list(mesh_obj, bonds_ids)
            else:
                curr_crack_segs = None    

            # check if the crack segments are in the crack pattern
            for i, seg in enumerate(crack_segments):
                for curr_seg in curr_crack_segs:
                    if np.allclose(seg, curr_seg):
                        time[i] = tt
                        break
                    
    # # sort the crack_segs based on time
    # idx = np.argsort(time)
    # time = time[idx]
    # crack_segs = [crack_segments[i] for i in idx]

    return time, crack_segments      

"""Function to get time estimate of the broken bonds"""
def get_time_broken_bond_list(mesh_obj, bond_ids_list, search_time):
    # mesh objective attributes
    dt = mesh_obj.solver['dt']
    skipsteps = mesh_obj.solver['skipsteps']

    # adjusting time close to the saved time steps
    time_steps = np.rint((search_time/(dt*skipsteps)))    
    new_time = time_steps*skipsteps*dt

    # output variables
    total_bonds= len(bond_ids_list)
    time = np.zeros(total_bonds)

    # loop through the delbonds file to get the crack segments
    with open(mesh_obj.folder + '/delbonds', 'rb') as del_bond_file:
        while True:
            try:
                tt, bonds_ids = pickle.load(del_bond_file)
            except EOFError:
                print('End of file reached')
                break
            if tt > new_time:
                print('Search time reached')
                break

            # remove redundant bonds        
            if bonds_ids:   
                for id, neigh in bonds_ids:
                    if [neigh, id] in bonds_ids:
                        bonds_ids.remove([neigh, id]) 

            # check if the crack segments are in the crack pattern
            for i, [id1,id2] in enumerate(bond_ids_list):
                for curr_id1, curr_id2 in bonds_ids:
                    if (curr_id1 == id1 and curr_id2 == id2) or (curr_id1 == id2 and curr_id2 == id1):
                        time[i] = tt
                        break
                    
    return time, bond_ids_list


"""Function to extract strain field for given time"""
def get_strain(mesh_obj, comp, time):

    dt = mesh_obj.solver['dt']
    skipsteps = mesh_obj.solver['skipsteps']
  
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
            crack.update_bond_state(mesh_obj, node_id=id, neighbor_id=neigh)             

    # read displacement data in buckets
    u = disp_file['u'][time_step]
    v = disp_file['v'][time_step]
    
    # compute strain tensor
    mesh_obj.u[:,0] = u
    mesh_obj.u[:,1] = v

    strain = crack.compute_nodal_strain_tensor(mesh_obj)

    if comp == 'xx':
        field = strain[:,0]
    elif  comp == 'xy':
        field = strain[:,1]
    elif comp == 'yy':
        field = strain[:,3]
    elif comp == 'max':
        field = compute_evalues(strain)  
    elif comp == 'all':
        field = strain    

    
    disp_file.close()

    return field

"""Function to extract stress field for given time"""
def get_stress(mesh_obj, comp, time):

    dt = mesh_obj.solver['dt']
    skipsteps = mesh_obj.solver['skipsteps']
  
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
            crack.update_bond_state(mesh_obj, node_id=id, neighbor_id=neigh)             

    # read displacement data in buckets
    u = disp_file['u'][time_step]
    v = disp_file['v'][time_step]
    
    # compute strain tensor
    mesh_obj.u[:,0] = u
    mesh_obj.u[:,1] = v

    stress = crack.compute_nodal_stress_tensor(mesh_obj)

    if comp == 'xx':
        field = stress[:,0]
    elif  comp == 'xy':
        field = stress[:,1]
    elif comp == 'yy':
        field = stress[:,3]
    elif comp == 'max':
        field = compute_evalues(stress)  
    elif comp == 'all':
        field = stress    

    
    disp_file.close()

    return field


"""Function to extract strain energy field for given time"""
def get_strain_energy(mesh_obj, time):
    dt = mesh_obj.solver['dt']
    skipsteps = mesh_obj.solver['skipsteps']
  
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
            crack.update_bond_state(mesh_obj, node_id=id, neighbor_id=neigh)             

    # read displacement data in buckets
    u = disp_file['u'][time_step]
    v = disp_file['v'][time_step]
    
    # compute strain tensor
    mesh_obj.u[:,0] = u
    mesh_obj.u[:,1] = v

    strain = crack.compute_nodal_strain_tensor(mesh_obj)
    stress = crack.compute_nodal_stress_tensor(mesh_obj)
    energy = 0.5*(stress[:,0]*strain[:,0] + stress[:,1]*strain[:,1] + stress[:,2]*strain[:,2] + stress[:,3]*strain[:,3])

    disp_file.close()

    return energy

"""Function to compute the strain energy for given time"""
def get_total_strain_energy(mesh_obj:mesh.mesh, time):
    dt = mesh_obj.solver['dt']
    skipsteps = mesh_obj.solver['skipsteps']
    total_nodes = len(mesh_obj.pos)
    neighbors = mesh_obj.neighbors
    angles = mesh_obj.angles
    norm_stiff = mesh_obj.normal_stiffness
    tang_stiff = mesh_obj.tangential_stiffness
  
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
            crack.update_bond_state(mesh_obj, node_id=id, neighbor_id=neigh)             

    # read displacement data in buckets
    u = disp_file['u'][time_step]
    v = disp_file['v'][time_step]
    
    # compute strain tensor
    mesh_obj.u[:,0] = u
    mesh_obj.u[:,1] = v

    # output variable
    energy = np.zeros(total_nodes)
    for id in range(total_nodes):
        for neigh, ns, ts, aph in zip(neighbors[id], norm_stiff[id], tang_stiff[id], angles[id]):
            ro = np.array([np.cos(np.pi*aph/180), np.sin(np.pi*aph/180)])
            uij = mesh_obj.u[neigh] - mesh_obj.u[id]
            un = np.dot(uij, ro); ut = uij - un*ro; ut = np.linalg.norm(ut)
            energy[id] += 0.5*(ns*un**2 + ts*ut**2)
                

    disp_file.close()

    return np.sum(energy)


"""Function to compute the strain energy for given time"""
def get_total_fracture_energy(mesh_obj:mesh.mesh, till_time = None):
    dt = mesh_obj.solver['dt']
    skipsteps = mesh_obj.solver['skipsteps']
    angles = mesh_obj.angles
    norm_stiff = mesh_obj.normal_stiffness
    tang_stiff = mesh_obj.tangential_stiffness
  
    # handler for displacement data file
    disp_file = h5py.File(mesh_obj.folder + '/disp.h5', 'r')

    # output variable
    tot_energy = []; time = []; energy = 0
    # handler for broken bonds info file
    with open(mesh_obj.folder + '/delbonds', 'rb') as del_bond_file:
        while True:
            try:
                tt, bonds_ids = pickle.load(del_bond_file)
            except EOFError:
                print('End of file reached')
                break
            if till_time is not None and tt > till_time:
                print('Search time reached')
                break            
 
            # remove redundant bonds        
            if bonds_ids:   
                for id, neigh in bonds_ids:
                    if [neigh, id] in bonds_ids:
                        bonds_ids.remove([neigh, id])

            # time steps corresponding to the time 
            time_step = int(tt/(dt*skipsteps))    
            t = time_step*skipsteps*dt
            # read displacement data for corresponding time step
            u = disp_file['u'][time_step]
            v = disp_file['v'][time_step]
        
            # compute strain tensor
            mesh_obj.u[:,0] = u
            mesh_obj.u[:,1] = v

            # energy = 0
            for id1, id2 in bonds_ids:
                idx = mesh_obj.neighbors[id1].index(id2)
                ns = norm_stiff[id1][idx]; ts = tang_stiff[id1][idx]; aph = angles[id1][idx]
                ro = np.array([np.cos(np.pi*aph/180), np.sin(np.pi*aph/180)])
                uij = mesh_obj.u[id2] - mesh_obj.u[id1]
                un = np.dot(uij, ro); ut = uij - un*ro; ut = np.linalg.norm(ut)
                energy += 0.5*(ns*un**2 + ts*ut**2)

            # energy = energy/(0.577*len(bonds_ids))    

            tot_energy.append(energy)
            time.append(t)    
           
    disp_file.close()
    return tot_energy, time

"""Function to work done by residual body forces for given time"""
def get_body_work(mesh_obj:mesh.mesh, time:float):
    dt = mesh_obj.solver['dt']
    skipsteps = mesh_obj.solver['skipsteps']
    total_nodes = len(mesh_obj.pos)
    neighbors = mesh_obj.neighbors
    angles = mesh_obj.angles
    norm_stiff = mesh_obj.normal_stiffness

    # handler for displacement data file
    disp_file = h5py.File(mesh_obj.folder + '/disp.h5', 'r')

    # time steps corresponding to the time intervals
    time_step = int((time/(dt*skipsteps)))    
    t = time_step*skipsteps*dt

    # compute plastic stress
    sxx, syy, sxy = solver.compute_plastic_stress(mesh_obj)

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
            crack.update_bond_state(mesh_obj, node_id=id, neighbor_id=neigh)             

    # read displacement data in buckets
    u = disp_file['u'][time_step]
    v = disp_file['v'][time_step]
    
    # update mesh object
    mesh_obj.u[:,0] = u
    mesh_obj.u[:,1] = v

    # compute work done by residual body forces
    bwd = np.zeros(total_nodes)
    for id in range(total_nodes):
        fx, fy = 0, 0
        for neigh, aph, ns in zip(neighbors[id],angles[id], norm_stiff[id]):    
            if ns!=0:
                rij = np.array([np.cos(np.pi*aph/180), np.sin(np.pi*aph/180)])

                sxx_ij = sxx[neigh] - sxx[id]
                sxy_ij = sxy[neigh] - sxy[id]
                syy_ij = syy[neigh] - syy[id]
                fx += (1/3)*(sxx_ij*rij[0] + sxy_ij*rij[1])
                fy += (1/3)*(sxy_ij*rij[0] + syy_ij*rij[1])
        bwd[id] = fx*mesh_obj.u[id,0] + fy*mesh_obj.u[id,1]        
      

    disp_file.close()

    return np.sum(bwd)

"""Function to compute the work done by surface force for given time"""
def get_surface_work(mesh_obj:mesh.mesh, time, surface = 'all'):
    dt = mesh_obj.solver['dt']
    skipsteps = mesh_obj.solver['skipsteps']
    total_nodes = len(mesh_obj.pos)

    # extract load boundary conditions
    lbcs_ids = mesh_obj.lbcs.ids
    lbcs_fx = mesh_obj.lbcs.fx
    lbcs_fy = mesh_obj.lbcs.fy
    lbcs_fun = mesh_obj.lbcs.fun

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
            crack.update_bond_state(mesh_obj, node_id=id, neighbor_id=neigh)             

    # read displacement data in buckets
    u = disp_file['u'][time_step]
    v = disp_file['v'][time_step]
    disp_file.close()

    # compute strain tensor
    mesh_obj.u[:,0] = u
    mesh_obj.u[:,1] = v

    load = np.zeros(shape=(total_nodes,2))

    for i, fun in enumerate(lbcs_fun):
        id = np.array(lbcs_ids[i])
        if fun == 'parser':
            load[id,0] = eval(lbcs_fx[i])
            load[id,1] = eval(lbcs_fy[i])
        elif fun == 'impulse' and t == 0:
            load[id,0] = eval(lbcs_fx[i])
            load[id,1] = eval(lbcs_fy[i])
        elif fun == 'ramp_impulse':
            if t<=1:
                load[id,0] = eval(lbcs_fx[i])
                load[id,1] = eval(lbcs_fy[i])
        elif fun == 'ramp':
            if t<=10:
                load[id,0] = (t/10)*eval(lbcs_fx[i])
                load[id,1] = (t/10)*eval(lbcs_fy[i])
            else:
                load[id,0] = eval(lbcs_fx[i])
                load[id,1] = eval(lbcs_fy[i])

        elif fun =='array':
            load[id,0] = lbcs_fx[i]
            load[id,1] = lbcs_fy[i]        

    
    swd = load[id,0]*mesh_obj.u[id,0] + load[id,1]*mesh_obj.u[id,1]            

    return np.sum(swd)


"""Function to compute work done by external forces for given time"""
def get_external_work(mesh_obj:mesh.mesh, time, surface = 'all'):
    dt = mesh_obj.solver['dt']
    skipsteps = mesh_obj.solver['skipsteps']
    total_nodes = len(mesh_obj.pos)

    # extract load boundary conditions
    lbcs_ids = mesh_obj.lbcs.ids
    lbcs_fx = mesh_obj.lbcs.fx
    lbcs_fy = mesh_obj.lbcs.fy
    lbcs_fun = mesh_obj.lbcs.fun

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
            crack.update_bond_state(mesh_obj, node_id=id, neighbor_id=neigh)             

    # read displacement data in buckets
    u = disp_file['u'][time_step]
    v = disp_file['v'][time_step]
    disp_file.close()

    # compute strain tensor
    mesh_obj.u[:,0] = u
    mesh_obj.u[:,1] = v

    load = np.zeros(shape=(total_nodes,2))

    for i, fun in enumerate(lbcs_fun):
        id = np.array(lbcs_ids[i])
        if fun == 'parser':
            load[id,0] = eval(lbcs_fx[i])
            load[id,1] = eval(lbcs_fy[i])
        elif fun == 'impulse' and t == 0:
            load[id,0] = eval(lbcs_fx[i])
            load[id,1] = eval(lbcs_fy[i])
        elif fun == 'ramp_impulse':
            if t<=1:
                load[id,0] = eval(lbcs_fx[i])
                load[id,1] = eval(lbcs_fy[i])
        elif fun == 'ramp':
            if t<=10:
                load[id,0] = (t/10)*eval(lbcs_fx[i])
                load[id,1] = (t/10)*eval(lbcs_fy[i])
            else:
                load[id,0] = eval(lbcs_fx[i])
                load[id,1] = eval(lbcs_fy[i])

        elif fun =='array':
            load[id,0] = lbcs_fx[i]
            load[id,1] = lbcs_fy[i]        

    
    swd = load[id,0]*mesh_obj.u[id,0] + load[id,1]*mesh_obj.u[id,1]            

    return np.sum(swd)

"""Function to compute initial residual stress field"""
def get_initial_residual_property(mesh_obj:mesh.mesh, property:str):
    if property == 'stress':
        stress = crack.compute_nodal_stress_tensor(mesh_obj)
        return stress
    elif property == 'strain':
        strain = crack.compute_nodal_strain_tensor(mesh_obj)
        return strain
    elif property == 'energy':
        strain = crack.compute_nodal_strain_tensor(mesh_obj)
        stress = crack.compute_nodal_stress_tensor(mesh_obj)
        energy = 0.5*(stress[:,0]*strain[:,0] + stress[:,1]*strain[:,1] + stress[:,2]*strain[:,2] + stress[:,3]*strain[:,3])
        return energy
    else:
        raise ValueError('Invalid property')
    
"""Function to estimate the bond force just before it breaks"""
def get_bond_force(mesh_obj:mesh.mesh, bond_ids: list, time:float):
    dt = mesh_obj.solver['dt']
    skipsteps = mesh_obj.solver['skipsteps']
    neighbors = mesh_obj.neighbors
    angles = mesh_obj.angles
    norm_stiff = mesh_obj.normal_stiffness
    tang_stiff = mesh_obj.tangential_stiffness
  
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
            crack.update_bond_state(mesh_obj, node_id=id, neighbor_id=neigh)             

    # read displacement data in buckets
    u = disp_file['u'][time_step]
    v = disp_file['v'][time_step]
    disp_file.close()

    # compute strain tensor
    mesh_obj.u[:,0] = u
    mesh_obj.u[:,1] = v

    # output variable
    bond_force = []
    for id1, id2 in bond_ids:
        idx = neighbors[id1].index(id2)
        ns = norm_stiff[id1][idx]; ts = tang_stiff[id1][idx]; aph = angles[id1][idx]
        ro = np.array([np.cos(np.pi*aph/180), np.sin(np.pi*aph/180)])
        uij = mesh_obj.u[id2] - mesh_obj.u[id1]
        un = np.dot(uij, ro); ut = uij - un*ro
        force = ns*un*ro + ts*ut
        bond_force.append(force)
                
    return bond_force


"""Function to estimate the residual bond force just before it breaks"""
def get_residual_bond_force(mesh_obj:mesh.mesh, bond_ids: list, time:float):
    dt = mesh_obj.solver['dt']
    skipsteps = mesh_obj.solver['skipsteps']
    total_nodes = len(mesh_obj.pos)
    neighbors = mesh_obj.neighbors
    angles = mesh_obj.angles
    norm_stiff = mesh_obj.normal_stiffness


    # time steps corresponding to the time intervals
    time_step = int((time/(dt*skipsteps)))    
    t = time_step*skipsteps*dt

    # compute plastic stress
    sxx, syy, sxy = solver.compute_plastic_stress(mesh_obj)

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
            crack.update_bond_state(mesh_obj, node_id=id, neighbor_id=neigh)             

    # compute work done by residual body forces
    bf = np.zeros((len(bond_ids),2))
    for i, [id1, id2] in enumerate(bond_ids):
        idx = neighbors[id1].index(id2)
        ns = norm_stiff[id1][idx]; aph = angles[id1][idx]
        if ns:
            ro = np.array([np.cos(np.pi*aph/180), np.sin(np.pi*aph/180)])
            sxx_ij = sxx[id2] - sxx[id1]
            sxy_ij = sxy[id2] - sxy[id1]
            syy_ij = syy[id2] - syy[id1]
            fx = (1/3)*(sxx_ij*ro[0] + sxy_ij*ro[1])
            fy = (1/3)*(sxy_ij*ro[0] + syy_ij*ro[1])
            bf[i] += np.array([fx,fy])
               

    return bf


"""Function to color bond based on property"""
# current property attributes can take - strain, stress, energy
def color_mesh_bonds(mesh_obj, property, time, **kwargs):
    # read optional keywords
    title = kwargs.get('title', 'field')
    cbarlim = kwargs.get('cbarlim', None)
    save = kwargs.get('save', False)
    xlim = kwargs.get('xlim', None)
    ylim = kwargs.get('ylim', None)
    save_path = kwargs.get('save_path', None)
    filename= kwargs.get('filename', 'field')
    

    """Function to get connectivity of network"""
    def get_network(mesh_obj, prop_value):
        edges = []
        kn = []

        for id in range(len(mesh_obj.pos)):
            for neigh, ns, aph in zip(mesh_obj.neighbors[id], mesh_obj.normal_stiffness[id], mesh_obj.angles[id]):
                if ns:
                    bond_value = 0.5*(prop_value[id] + prop_value[neigh])
                    
                    if (aph>=0) & (aph<=180):
                        edges.append([id, neigh])
                        if bond_value >= 0.02:
                            kn.append(bond_value)
                        else:
                            kn.append(0)    
        return edges, kn
    
    """Function to create line collection from the edges"""
    def get_line_collection(mesh_obj, prop_value):
        edges,kn = get_network(mesh_obj, prop_value)
        lines = np.zeros((len(edges), 2, 2))
        mesh_obj.pos = mesh_obj.pos
        for i, [id1, id2] in enumerate(edges):
            lines[i, 0,:] = mesh_obj.pos[id1]
            lines[i, 1,:] = mesh_obj.pos[id2]   
            
        return lines,kn  
    
    # get property
    if property == 'strain':
        prop_value = get_strain(mesh_obj, comp='max', time=time)
    elif property == 'stress':   
        prop_value = get_stress(mesh_obj, comp='max', time=time)

    # view mesh
    linecollection, kn = get_line_collection(mesh_obj, prop_value)

    if not cbarlim:
        ul = np.nanmax(prop_value)
        ll = np.nanmin(prop_value)
        cbarlim = [ll, ul]

    # plot network
    fig = plt.figure()
    ax = fig.add_subplot()
    ax_divider= make_axes_locatable(ax)
    # add an axes to the right of the main axes
    cax = ax_divider.append_axes("right", size="7%", pad="2%")

    lc = LineCollection(linecollection, linewidths=0.5, array = kn, cmap = 'jet')
    # lc.set_clim(cbarlim)
    ax.add_collection(lc)

    fig.colorbar(lc, cax=cax)
    ax.set_title(f'{title}', fontsize = 20)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    ax.set_xlabel(r'$x\rightarrow$')
    ax.set_ylabel(r'$y\rightarrow$')
    # ax.set_axis_off()
    ax.set_aspect('equal')
    # saving the plot
    if save:   
        fig.savefig(save_path + f'\\{filename}.png', bbox_inches = 'tight', dpi = 300)
        plt.close()    
    else:    
        plt.show()

       