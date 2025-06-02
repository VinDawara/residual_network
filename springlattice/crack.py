from . import mesh, bcs, solver
import numpy as np
import pickle
import h5py
import copy
import math
from scipy.special import erf
# from .nodal_stress import compute_nodal_stress, compute_nodal_strain

def breaking_parameters(mesh_obj, prop, threshold, **kwargs):
    """
    Set parameters for mesh cracking within the mesh object.

    Parameters:
    - mesh_obj (object): An object representing the mesh system.
    - prop (str): The property affecting the mesh cracking behavior.
    - threshold (float): The threshold value for the specified property.
    - **kwargs: Additional keyword arguments:
        - comp (str, optional): Required if 'prop' is 'stretch'. Specifies the component for stretching.

    Returns:
    - None

    This function configures the parameters related to mesh cracking within the given mesh object.
    'prop' defines the property influencing the cracking behavior, 'threshold' sets the threshold value
    for this property. If 'prop' is 'stretch', the 'comp' keyword argument is required to specify the
    stretching component. 
    """
    if not isinstance(mesh_obj,mesh.mesh):
        raise TypeError("mesh object must be of class mesh")
    
    if prop not in ['strain', 'stretch', 'stress']:
        raise NameError("Prop can take 'stretch' or 'strain")
    
    if not isinstance(threshold, (int,float)):
        raise TypeError("threshold must be either integer or float")
    
    mesh_obj.crack_param.update({'prop':prop, 'threshold':threshold})

    # temporary properties
    total_nodes = mesh_obj.nx*mesh_obj.ny
    # generate n random number between 0 and 1
    num = 0.002*np.random.rand(total_nodes)
    mesh_obj.crack_param.update({'noise':num})

    if prop == 'stretch':
        if 'comp' not in kwargs:
            raise ValueError("Keyword argument 'comp' is required when 'prop' is 'stretch'")
        mesh_obj.crack_param['comp'] = kwargs['comp']


"""Updated verlet integrator using matrix multiplication"""
def solve(mesh_obj:mesh.mesh, dt, endtime, **kwargs):
    """
    Solve the mesh system dynamics over time.

    Parameters:
    - mesh_obj (object): Mesh object containing system information.
    - dt (float): Time step for integration.
    - endtime (float): End time for simulation.
    - zeta (float, optional): Damping coefficient. Default is 0.
    - vectorfield (str, optional): Vector field visualization mode. Default is 'off'.
    - folder (str, optional): Folder name to store data. Default is None.
    - **kwargs: Additional keyword arguments.

    This function integrates the mesh system dynamics over time using Verlet integration.
    It imposes boundary conditions, computes load boundary conditions, updates the system state,
    and saves displacement data and deleted bonds during the simulation.
    """    
    # set keyward options
    zeta = kwargs.get('zeta', 0)
    vectorfield = kwargs.get('vectorfield', 'off')
    folder = kwargs.get('folder', "sl_crack")
    interval = kwargs.get('interval', False)
    save_ratio = kwargs.get('save_ratio', 0)    

    # creating directory for storing data
    mesh_obj.folder = mesh._create_directory(f"{folder}_{mesh_obj.ny}X{mesh_obj.nx}")

    mesh_obj.solver.update({'dt': dt, 'endtime': endtime, 'name': 'verlet','zeta':zeta})
    maxsteps = math.ceil(endtime/dt)
    skipsteps = math.ceil((save_ratio/100)*maxsteps) if save_ratio else 1
    mesh_obj.solver['skipsteps'] = skipsteps

    # save initial mesh
    mesh.save_mesh(mesh_obj)

    # initial mesh position
    pos = copy.deepcopy(mesh_obj.pos)
 
    # extract load boundary conditions
    lbcs_ids = copy.copy(mesh_obj.lbcs.ids)
    lbcs_fx = copy.copy(mesh_obj.lbcs.fx)
    lbcs_fy = copy.copy(mesh_obj.lbcs.fy)
    lbcs_fun = copy.copy(mesh_obj.lbcs.fun)

    # check residual stress is assigned
    if mesh_obj.plastic_strain:
        resi_x_list, resi_y_list = solver.get_residual_bondlist(mesh_obj)

    # create a matrix A
    A = solver.generate_matrix(mesh_obj)
    B = solver.generate_B_matrix(mesh_obj,resi_x_list = resi_x_list, resi_y_list = resi_y_list)

    # displacement variables for time step 't-1', 't' and 't+1', respectively
    u_prev, u_curr, u_next = (solver.flatten(mesh_obj.u) for _ in range(3))

    # read any supplemented displacement
    u_sup = bcs.dispbcs_supplementary(mesh_obj, u_curr)

    # impose boundary condition at t = 0
    u_curr = u_sup + bcs.impose_displacement(mesh_obj, u_curr, t=0)
    u_prev = copy.deepcopy(u_curr)

    # initial/current position of the nodes
    ro = solver.flatten(pos)
    r_curr = ro +  u_curr

    # Creating handler to file for displacement data
    total_nodes = len(pos)
    disp_file = h5py.File(mesh_obj.folder + '/disp.h5', 'w')
    dset_u = disp_file.create_dataset(
        name = 'u',
        shape=(0,total_nodes),
        maxshape = (None, total_nodes),
        dtype='float64',
        compression = 'gzip',
        compression_opts = 9
    )
    dset_v = disp_file.create_dataset(
        name = 'v',
        shape=(0,total_nodes),
        maxshape = (None, total_nodes),
        dtype='float64',
        compression = 'gzip',
        compression_opts = 9
    )

    # displacement bucket to write to file
    bucket = 0
    fill_steps = 0
    bucket_idx = 0
    remain_steps = int(maxsteps/skipsteps)
    bucket_size = min(1000,remain_steps)
    # bucket size to be stored in file
    U = np.zeros(shape=(bucket_size, total_nodes))
    V = np.zeros(shape=(bucket_size, total_nodes))

    # file handler for saving deleted bonds
    bonds_file = open(mesh_obj.folder + '/delbonds','wb')
    
    # integration begins
    for step in range(maxsteps):
        # time variable
        t = step*dt

        # velocity of the nodes
        v = (u_next - u_prev)/(2*dt)

        # impose load boundary condtions
        load = bcs.impose_loads(total_nodes, lbcs_ids, lbcs_fx, lbcs_fy, lbcs_fun, t)

        # time integration
        F = A@r_curr + B - load + zeta*v
        u_next= 2*u_curr - u_prev - F*dt**2

        # impose boundary conditions
        u_next = u_sup + bcs.impose_displacement(mesh_obj, u_next, t=t)

        mesh_obj.u = solver.reshape2vector(u_next)
        deleted_bonds = activate_breaking(mesh_obj)
        if deleted_bonds:
            A = solver.update_A(mesh_obj,A, deleted_bonds)
            B = solver.update_B(mesh_obj,B,deleted_bonds, resi_x_list = resi_x_list, resi_y_list = resi_y_list)
            # deleted bonds
            pickle.dump([t,deleted_bonds], bonds_file)

        # update node object
        if interval and step%int(interval/dt) == 0: 
            mesh_obj.pos = pos + mesh_obj.u
            mesh.mesh_plot(mesh_obj,filename = f"step_{step}.png", title = f'T = {np.round(t,3)}', vectorfield = vectorfield, save=True)
            mesh_obj.pos = pos
        
        u_prev = copy.deepcopy(u_curr)
        u_curr = copy.deepcopy(u_next)
        r_curr = ro + u_curr
        print('Time step = ',step, 'T = %0.4f' % t, 'Progress = %0.2f' % (100*step/maxsteps))

        # saving displacement data to disk
        if step%skipsteps == 0:
            # saving displacements fields
            u_shape = solver.reshape2vector(u_next)
            U[bucket_idx] = u_shape[:,0]
            V[bucket_idx] = u_shape[:,1] 
            bucket_idx += 1

            if bucket_idx == bucket_size: # if variable is full, empty bucket 
                dset_u.resize(dset_u.shape[0]+bucket_size, axis = 0)
                dset_u[-bucket_size:] = U
                dset_v.resize(dset_v.shape[0]+bucket_size, axis = 0)
                dset_v[-bucket_size:] = V
                bucket += 1
                fill_steps += bucket_size
                remain_steps += -bucket_size 
                bucket_idx = 0
                
                if  remain_steps < bucket_size:
                    bucket_size = remain_steps
                    U = np.zeros(shape = (bucket_size,total_nodes))
                    V = np.zeros(shape = (bucket_size,total_nodes))
                
    disp_file.close()
    bonds_file.close()
    print(f'Solver completed. Data saved to {mesh_obj.folder}')

        
"""Function to include bond breaking condition for dyanmic solver"""
def activate_breaking(mesh_obj):
    """
    Activate breaking of bonds in the mesh based on specified properties.

    Parameters:
    - mesh_obj (object): An object representing the mesh system with crack parameters.

    Returns:
    - deleted_bonds (list): List of deleted bond pairs. If no bonds are deleted, returns an empty list.

    This function activates the breaking of bonds within the mesh based on defined properties.
    It checks specific bond properties such as 'stretch' or 'strain' and their associated thresholds.
    Bonds exceeding the threshold are marked as deleted.
    """    
    prop = mesh_obj.crack_param['prop']
    
    if prop == 'stretch':
        comp = mesh_obj.crack_param['comp']
        threshold = mesh_obj.crack_param['threshold']
        deleted_bonds = check_stretched_bonds(mesh_obj, comp=comp, threshold=threshold) 

    elif prop == 'strain':
        threshold = mesh_obj.crack_param['threshold']
        deleted_bonds = check_strained_bonds(mesh_obj, threshold = threshold)  

    elif prop == 'stress':
        threshold = mesh_obj.crack_param['threshold']
        deleted_bonds = check_stressed_bonds(mesh_obj, threshold = threshold)   

    elif prop == 'energy':
        threshold = mesh_obj.crack_param['threshold']
        deleted_bonds = check_energetic_bonds(mesh_obj, threshold = threshold)      

    if deleted_bonds:
        for id, neigh in deleted_bonds:
            update_bond_state(mesh_obj, node_id=id, neighbor_id = neigh)

    return deleted_bonds          

"""Function to find critical bonds based on stretch"""
def check_stretched_bonds(mesh_obj,comp, threshold):
    """
    Check for stretched bonds in the mesh based on specified properties.

    Parameters:
    - mesh_obj (object): An object representing the mesh system.
    - comp (str): Component to calculate stretching ('normal', 'tangential', or 'abs').
    - threshold (float): Threshold value for bond stretching.

    Returns:
    - critical_bonds (list): List of bond pairs that exceed the threshold.

    This function checks for stretched bonds in the mesh based on the specified component and threshold.
    It iterates through all nodes and their neighbors, calculating bond stretching based on the given component.
    Bonds exceeding the threshold are marked as critical and returned as a list of bond pairs.
    """    
    critical_bonds = []       
    total_nodes = mesh_obj.nx*mesh_obj.ny

    # compute plastic stress
    sxx, syy, sxy = solver.compute_plastic_stress(mesh_obj)

    for id in range(total_nodes):
        for neigh, aph, ns, ts in zip(mesh_obj.neighbors[id],mesh_obj.angles[id], mesh_obj.normal_stiffness[id], mesh_obj.tangential_stiffness[id]):    
            if ns!=0:
                uij = mesh_obj.u[neigh] - mesh_obj.u[id]
                rij = np.array([np.cos(np.pi*aph/180), np.sin(np.pi*aph/180)])

                sxx_ij = sxx[neigh] - sxx[id]
                sxy_ij = sxy[neigh] - sxy[id]
                syy_ij = syy[neigh] - syy[id]
                fx = (1/3)*(sxx_ij*rij[0] + sxy_ij*rij[1])
                fy = (1/3)*(sxy_ij*rij[0] + syy_ij*rij[1])
            
                if comp == 'normal':
                    value = np.dot(uij,rij) - (fx*rij[0] + fy*rij[1])/ns

                elif comp == 'tangential':
                    tij = np.array([-np.sin(np.pi*aph/180), np.cos(np.pi*aph/180)])
                    value = np.abs(np.dot(uij,tij) - (-fx*rij[1] + fy*rij[0])/ts )
   
                elif comp == 'abs':
                    tij = np.array([-np.sin(np.pi*aph/180), np.cos(np.pi*aph/180)])
                    un = np.dot(uij,rij) - (fx*rij[0] + fy*rij[1])/ns
                    ut = np.dot(uij,tij) - (-fx*rij[1] + fy*rij[0])/ts
                    if un>0:
                        value = np.sqrt(un**2 + ut**2)
                    else:
                        value = -np.sqrt(un**2 + ut**2)

                elif comp =='any':
                    value1 = np.dot(uij,rij) - (fx*rij[0] + fy*rij[1])/ns
                    tij = np.array([-np.sin(np.pi*np.array(aph)/180), np.cos(np.pi*np.array(aph)/180)])
                    value2 = np.abs(np.dot(uij,tij) - (-fx*rij[1] + fy*rij[0])/ts )
                    value = max(value1,value2)

                else:
                    raise Exception('Wrongly assigned argument to \'comp\' keyword ')    

                if value >= threshold:
                    critical_bonds.append([id, neigh])               

    return critical_bonds


def check_strained_bonds(mesh_obj:mesh.mesh, threshold):
    """
    Check for strained bonds in the mesh based on a specified threshold.

    Parameters:
    - mesh_obj (object): An object representing the mesh system.
    - threshold (float): Threshold value for bond strain.

    Returns:
    - critical_bonds (list): List of bond pairs that exceed the threshold.

    This function checks for strained bonds in the mesh based on the provided strain tensor and threshold.
    It computes the nodal strain tensor, then iterates through nodes and their neighbors to calculate
    the bond strain. If the principal strain of a bond exceeds the given threshold, it marks it as critical.
    The function returns a list of bond pairs that exceed the threshold.
    """    
    strain = compute_nodal_strain_tensor(mesh_obj)
    # strain = compute_nodal_strain(mesh_obj, 0)
    # noise = mesh_obj.crack_param['noise']
    critical_bonds = []

    for node_id, node_neighbors in enumerate(mesh_obj.neighbors):
        for neigh, ns in zip(node_neighbors, mesh_obj.normal_stiffness[node_id]):
            if ns:
                bond_strain = 0.5 * (strain[node_id] + strain[neigh])
                # bond_noise = 0.5*(noise[node_id] + noise[neigh])
                principal_strain = principal_value(bond_strain)

                if principal_strain >= threshold:
                    critical_bonds.append([node_id, neigh])

    return critical_bonds


def check_stressed_bonds(mesh_obj:mesh.mesh, threshold):
   
    stress = compute_nodal_stress_tensor(mesh_obj)
    # stress = compute_nodal_stress(mesh_obj,0)
    noise = mesh_obj.crack_param['noise']
    critical_bonds = []

    for node_id, node_neighbors in enumerate(mesh_obj.neighbors):
        for neigh, ns in zip(node_neighbors, mesh_obj.normal_stiffness[node_id]):
            if ns:
                bond_strain = 0.5 * (stress[node_id] + stress[neigh])
                bond_noise = 0.5*(noise[node_id] + noise[neigh])
                principal_strain = principal_value(bond_strain)

                if principal_strain >= threshold+bond_noise:
                    critical_bonds.append([node_id, neigh])

    return critical_bonds

def check_energetic_bonds(mesh_obj:mesh.mesh, threshold):

    node_quantity, vol_strain = compute_nodal_strain_energy(mesh_obj)
    critical_bonds = []

    for node_id, node_neighbors in enumerate(mesh_obj.neighbors):
        for neigh, ns in zip(node_neighbors, mesh_obj.normal_stiffness[node_id]):
            if ns:
                quantity = 0.5 * (node_quantity[node_id] + node_quantity[neigh])
                vol = 0.5*(vol_strain[node_id] + vol_strain[neigh])
                if quantity >= threshold and vol>=0:
                    critical_bonds.append([node_id, neigh])

    return critical_bonds


"""Function to compute nodal strian tensor"""
def compute_nodal_strain_tensor(mesh_obj):
    """
    Compute the nodal strain tensor for a given mesh object.

    Parameters:
    - mesh_obj (object): An object representing the mesh system.

    Returns:
    - strain (numpy.ndarray): Nodal strain tensor of shape (total_nodes, 4).

    This function computes the nodal strain tensor for a given mesh object. It initializes an array 'strain' to store
    the computed strain values for each node. The computation is performed for triangular lattice nodes. It iterates
    through each node, calculates the strain contributions from its neighbors, and computes the total strain at each
    node based on the node's connectivity, angles, and normal stiffness.
    """    
    total_nodes = mesh_obj.nx*mesh_obj.ny
    strain = np.zeros(shape = (total_nodes,4))
    if mesh_obj.lattice == 'triangle':
        for id in range(0,total_nodes):
            si = 0
            for neigh, aph, ns in zip(mesh_obj.neighbors[id],mesh_obj.angles[id], mesh_obj.normal_stiffness[id]):
                uij = mesh_obj.u[neigh] - mesh_obj.u[id]
                rij = np.array([np.cos(np.pi*np.array(aph)/180), np.sin(np.pi*np.array(aph)/180)])
                si += (ns!=0)*(1/6)*(uij[:,np.newaxis]@rij[np.newaxis,:] + \
                rij[:,np.newaxis]@uij[np.newaxis,:])

            strain[id,:] = si.reshape(1,4)
        
    if bool(mesh_obj.plastic_strain):
        # read residual stress parser function (x,y)
        exx = mesh_obj.plastic_strain['exx']
        eyy = mesh_obj.plastic_strain['eyy']
        exy = mesh_obj.plastic_strain['exy']
   
        ep = np.column_stack((exx,exy, exy, eyy))
        strain += -ep

    return strain        

"""Function to compute nodal stress tensor"""
def compute_nodal_stress_tensor(mesh_obj):
    total_nodes = mesh_obj.nx*mesh_obj.ny
    nu = mesh_obj.bond_prop['poisson_ratio']
    stress = np.zeros(shape = (total_nodes,4))
    factor = 0.5*(0.874 + 0.162*nu)**2/(1 + nu)
    for id in range(0,total_nodes):
        si = 0
        for neigh, aph, ns in zip(mesh_obj.neighbors[id],mesh_obj.angles[id], mesh_obj.normal_stiffness[id]):
            uij = mesh_obj.u[neigh] - mesh_obj.u[id]
            rij = np.array([np.cos(np.pi*np.array(aph)/180), np.sin(np.pi*np.array(aph)/180)])
            si += (ns!=0)*(1/6)*(uij[:,np.newaxis]@rij[np.newaxis,:] + \
            rij[:,np.newaxis]@uij[np.newaxis,:])

        stress[id,0] = (1/((1+nu)*factor))*(si[0,0] + nu/(1-2*nu)*(si[0,0]+si[1,1]))
        stress[id,3] = (1/((1+nu)*factor))*(si[1,1] + nu/(1-2*nu)*(si[0,0]+si[1,1]))
        stress[id,1] = stress[id,2] = (1/((1+nu)*factor))*si[0,1]

    # if bool(mesh_obj.plastic_strain):
    #     stress += -compute_plastic_stress(mesh_obj)

    return stress


"""Function to compute nodal stress tensor"""
def compute_nodal_strain_energy(mesh_obj):
    total_nodes = mesh_obj.nx*mesh_obj.ny
    nu = mesh_obj.bond_prop['poisson_ratio']
    strain_energy = np.zeros(total_nodes)
    vol_strain = np.zeros_like(strain_energy)
    factor = 0.5*(0.874 + 0.162*nu)**2/(1 + nu)

    if bool(mesh_obj.plastic_strain):
        # read residual stress parser function (x,y)
        exx = mesh_obj.plastic_strain['exx']
        eyy = mesh_obj.plastic_strain['eyy']
        exy = mesh_obj.plastic_strain['exy']

        # residual stress values at each node
        x = mesh_obj.pos[:,0]; y = mesh_obj.pos[:,1]
        exx = eval(exx) if exx is not None else np.zeros_like(x)
        eyy = eval(eyy) if eyy is not None else np.zeros_like(y)
        exy = eval(exy) if exy is not None else np.zeros_like(x)
        
        ep = np.column_stack((exx,exy, exy, eyy))


    for id in range(0,total_nodes):
        si = 0
        for neigh, aph, ns in zip(mesh_obj.neighbors[id],mesh_obj.angles[id], mesh_obj.normal_stiffness[id]):
            uij = mesh_obj.u[neigh] - mesh_obj.u[id]
            rij = np.array([np.cos(np.pi*np.array(aph)/180), np.sin(np.pi*np.array(aph)/180)])
            si += (ns!=0)*(1/6)*(uij[:,np.newaxis]@rij[np.newaxis,:] + \
            rij[:,np.newaxis]@uij[np.newaxis,:])

        et = si.reshape(1,4) - ep[id]
        sxx = (1/((1+nu)*factor))*(et[0,0] + nu/(1-2*nu)*(et[0,0]+et[0,1]))
        syy = (1/((1+nu)*factor))*(et[0,1] + nu/(1-2*nu)*(et[0,0]+et[0,1]))
        sxy = (1/((1+nu)*factor))*et[0,1]

        strain_energy[id] = 0.5*sxx*et[0,0] + 0.5*syy*et[0,3] + sxy*et[0,1]
        vol_strain[id] = 0.5*(et[0,0] + et[0,3])

    return strain_energy, vol_strain

"""Function to compute principal strain (Not mesh object)"""
def principal_value(t, type = 'maximum'):
    """
    Compute the principal value of a tensor.

    Parameters:
    - t (list or numpy.ndarray): Input tensor of length 4 representing strain components.
    - type (str): Type of principal value to compute. Default is 'maximum'.

    Returns:
    - ps (float): Principal value of the tensor.

    This function computes the principal value of a tensor given as input, specifically designed for tensors
    representing strain components. The 'type' parameter determines the computation type. For 'maximum' type,
    it calculates the maximum principal value based on the input tensor components (t[0], t[1], t[3]).
    """    
    if type == 'maximum':
        ps = 0.5*(t[0] + t[3]) + math.sqrt((0.5*(t[0]-t[3]))**2 + \
            t[1]**2)
    return ps 


def compute_plastic_stress(mesh_obj:mesh.mesh):

    nu = mesh_obj.bond_prop['poisson_ratio']

    # read residual stress parser function (x,y)
    exx = mesh_obj.plastic_strain['exx']
    eyy = mesh_obj.plastic_strain['eyy']
    exy = mesh_obj.plastic_strain['exy']

    # # residual stress values at each node
    # x = mesh_obj.pos[:,0]; y = mesh_obj.pos[:,1]
    # exx = eval(exx) if exx is not None else np.zeros_like(x)
    # eyy = eval(eyy) if eyy is not None else np.zeros_like(y)
    # exy = eval(exy) if exy is not None else np.zeros_like(x)

    factor = 0.5*(0.874 + 0.162*nu)**2/(1 + nu)

    sxx = (1/((1+nu)*(1-2*nu)*factor))*(exx*(1-nu) + eyy*nu)
    syy = (1/((1+nu)*(1-2*nu)*factor))*(eyy*(1-nu) + exx*nu)
    sxy = 1/(2*(1+nu)*factor)*exy

    return np.column_stack((sxx,sxy, sxy, syy))


def update_bond_state(mesh_obj, node_id, neighbor_id, k=0):
    """
    Update bond states of the mesh for the specified bond.

    Parameters:
    - mesh_obj (object): Mesh object containing the bond states and properties.
    - node_id (int): Node ID of the bond.
    - neighbor_id (int): Neighbor node ID of the bond.
    - k (int, optional): New stiffness value to assign. Default is 0.

    This function updates the bond states (stiffness) in the mesh object for the specified bond.
    It updates the stiffness (both normal and tangential) for the given bond and its corresponding neighbor
    with the provided stiffness value 'k'.
    """
    id_idx = mesh_obj.neighbors[node_id].index(neighbor_id)
    neighbor_idx = mesh_obj.neighbors[neighbor_id].index(node_id)

    mesh_obj.normal_stiffness[node_id][id_idx] = k
    mesh_obj.tangential_stiffness[node_id][id_idx] = k
    mesh_obj.normal_stiffness[neighbor_id][neighbor_idx] = k
    mesh_obj.tangential_stiffness[neighbor_id][neighbor_idx] = k

def edge_crack(mesh_obj, crack_length, row=0, right=False):
    """
    Create an edge crack in the mesh.

    Parameters:
    - mesh_obj (object): Mesh object containing the geometry and connectivity.
    - crack_length (int): Length of the crack to be created.
    - row (int, optional): Row index where the crack starts or ends. Default is 0.
    - right (bool, optional): Indicates if the crack starts from the right side. Default is False.

    This function creates an edge crack in the mesh by deleting bonds based on the specified crack length,
    starting row index, and side (right or left). It identifies the crack node IDs and deletes bonds
    associated with those nodes to create the crack.
    """
    if not isinstance(mesh_obj, mesh.mesh):
        raise TypeError("mesh_obj must be of class mesh")
    
    if not isinstance(crack_length, int):
        raise TypeError("crack_length must be integer")

    if right:
        end_id = mesh_obj.right[row] + 1 if row else mesh_obj.right[math.ceil(0.5 * len(mesh_obj.right)) - 1] + 1
        start_id = end_id - crack_length
    else:
        start_id = mesh_obj.left[row] if row else mesh_obj.left[math.ceil(0.5 * len(mesh_obj.left)) - 1]
        end_id = start_id + crack_length

    crack_ids = range(start_id, end_id)
    for node_id in crack_ids:
        neighbors = [neighbor for neighbor in mesh_obj.neighbors[node_id] if neighbor > node_id + 1]
        mesh.delete_bonds(mesh_obj, node_id, neighbors=neighbors)

def central_crack(mesh_obj, crack_length, row=0, right=False):
    """
    Create an edge crack in the mesh.

    Parameters:
    - mesh_obj (object): Mesh object containing the geometry and connectivity.
    - crack_length (int): Length of the crack to be created.
    - row (int, optional): Row index where the crack starts or ends. Default is 0.
    - right (bool, optional): Indicates if the crack starts from the right side. Default is False.

    This function creates an edge crack in the mesh by deleting bonds based on the specified crack length,
    starting row index, and side (right or left). It identifies the crack node IDs and deletes bonds
    associated with those nodes to create the crack.
    """
    if not isinstance(mesh_obj, mesh.mesh):
        raise TypeError("mesh_obj must be of class mesh")
    
    if not isinstance(crack_length, int):
        raise TypeError("crack_length must be integer")

    nx = mesh_obj.nx
    start_id = mesh_obj.left[row]+int(0.5*nx)-crack_length 
    end_id = start_id + 2*crack_length-1

    crack_ids = range(start_id, end_id)
    for node_id in crack_ids:
        neighbors = [neighbor for neighbor in mesh_obj.neighbors[node_id] if neighbor > node_id + 1]
        mesh.delete_bonds(mesh_obj, node_id, neighbors=neighbors)

def edge_vertical_crack(mesh_obj, crack_length, column=0, return_surface_ids = False):
    """
    Create an edge crack in the mesh.

    Parameters:
    - mesh_obj (object): Mesh object containing the geometry and connectivity.
    - crack_length (int): Length of the crack to be created.
    - row (int, optional): Row index where the crack starts or ends. Default is 0.
    - right (bool, optional): Indicates if the crack starts from the right side. Default is False.

    This function creates an edge crack in the mesh by deleting bonds based on the specified crack length,
    starting row index, and side (right or left). It identifies the crack node IDs and deletes bonds
    associated with those nodes to create the crack.
    """
    if not isinstance(mesh_obj, mesh.mesh):
        raise TypeError("mesh_obj must be of class mesh")
    
    if not isinstance(crack_length, int):
        raise TypeError("crack_length must be integer")

    # variables suffix corresponding to node id 'i'
    nx = mesh_obj.nx; ny = mesh_obj.ny
    lex = lambda i,j: nx*i + j

    crack_ids = [lex(ny-1-i, column) for i in range(0,crack_length,2)]

    left_ids = []; right_ids = []
    for node_id in crack_ids:
        mesh.delete_bonds(mesh_obj, node_id, neighbors=mesh_obj.neighbors[node_id])
        idx = mesh_obj.angles[node_id].index(-60)
        neigh_id = mesh_obj.neighbors[node_id][idx]
        mesh.delete_bonds(mesh_obj, neigh_id, angles = [180])
        idx = mesh_obj.angles[node_id].index(-120)
        neigh_id = mesh_obj.neighbors[node_id][idx]
        mesh.delete_bonds(mesh_obj, neigh_id, angles = [0])
        if node_id != crack_ids[0]:
            idx = mesh_obj.angles[node_id].index(120); left_ids.append(mesh_obj.neighbors[node_id][idx])
            idx = mesh_obj.angles[node_id].index(60); right_ids.append(mesh_obj.neighbors[node_id][idx])

        idx = mesh_obj.angles[node_id].index(180); left_ids.append(mesh_obj.neighbors[node_id][idx])
        idx = mesh_obj.angles[node_id].index(0); right_ids.append(mesh_obj.neighbors[node_id][idx])
    
    if return_surface_ids:
        return left_ids, right_ids


