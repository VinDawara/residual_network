import numpy as np
import copy
from .mesh import mesh
from .solver import compute_plastic_stress
"""Function to create nodal stress"""     
def compute_nodal_stress(mesh:mesh,t):
    
    # total number of nodes
    total_nodes = mesh.nx * mesh.ny

    # extracting mesh info
    disp, pos, neighbors, norm_stiff, tang_stiff, angles = mesh.u, mesh.pos, mesh.neighbors, mesh.normal_stiffness, mesh.tangential_stiffness, mesh.angles
    
    # extract load boundary conditions
    lbcs_ids = copy.copy(mesh.lbcs.ids)
    lbcs_fx = copy.copy(mesh.lbcs.fx)
    lbcs_fy = copy.copy(mesh.lbcs.fy)
    lbcs_fun = copy.copy(mesh.lbcs.fun)

    # compute applied load 
    app_load = impose_loads(total_nodes, lbcs_ids, lbcs_fx, lbcs_fy, lbcs_fun, t)

    # find displaced nodes
    disp_nodes = find_displaced_nodes(mesh)

    # compute inelastic stress
    sxx, syy, sxy = compute_plastic_stress(mesh)

    # stress value
    stress = np.zeros((total_nodes, 4))

    wtg_yy = {'0': 0.5, '60': 1, '120': 1, '180': 0.5}


    for id in range(total_nodes):  # looping over the nodes
        aph = angles[id]        # neighboring bond angles
        id_disp = disp[id]      # node id displacement

        
        # syy node element info
        fyy = np.zeros(2)
        fb_yy = np.zeros(2)

        # sxx node element info
        fxx = np.zeros(2)
        fb_xx = np.zeros(2)

        # inertial force
        f_inertia = np.zeros(2)
        fb = np.zeros(2)

        for i, neigh_aph in enumerate(aph): # looping over each bond angle
            neigh_id, ns, ts = neighbors[id][i], norm_stiff[id][i], tang_stiff[id][i]
            neigh_disp = disp[neigh_id]
            ro = np.array([np.cos(np.radians(neigh_aph)), np.sin(np.radians(neigh_aph))])
            rf = pos[neigh_id] + neigh_disp - pos[id] - id_disp

            if ns:
                sxx_ij = sxx[neigh_id] - sxx[id]
                sxy_ij = sxy[neigh_id] - sxy[id]
                syy_ij = syy[neigh_id] - syy[id]
                fbx = (1/3)*(sxx_ij*ro[0] + sxy_ij*ro[1])
                fby = (1/3)*(sxy_ij*ro[0] + syy_ij*ro[1])
            else:
                fbx = 0
                fby = 0    
            
            fb += np.array([fbx, fby])            
            f_inertia += ((ns - ts) * np.dot(rf - ro, ro) * ro + ts * (rf - ro))

            if neigh_aph in [0, 60, 120, 180]:
                fyy += wtg_yy[str(neigh_aph)] * ((ns - ts) * np.dot(rf - ro, ro) * ro + ts * (rf - ro))
                fb_yy += wtg_yy[str(neigh_aph)] * np.array([fbx, fby])

            if neigh_aph in [0, 60, -60]:
                fxx += ((ns - ts) * np.dot(rf - ro, ro) * ro + ts * (rf - ro))
                fb_xx += np.array([fbx, fby])
     
        # Contribution from applied load and bond force
        f_inertia += - fb 

        fyy[0] = disp_nodes[id,0]*(f_inertia[0]+ 0.5*app_load[id,0]) - fyy[0] + fb_yy[0] 
        if app_load[id,1] == 0:     
            fyy[1] = disp_nodes[id,1]*f_inertia[1] - fyy[1] + fb_yy[1]
        else:
            fyy[1] = -app_load[id,1] if app_load[id,1] > 0 else app_load[id,1]

        fxx[1] = disp_nodes[id,1]*(f_inertia[1]+ 0.5*app_load[id,1]) - fxx[1] + fb_xx[1] 
        if app_load[id,0] == 0:     
            fxx[0] = disp_nodes[id,0]*f_inertia[0] - fxx[0] + fb_xx[0]
        else:
            fxx[0] = -app_load[id,0] if app_load[id,0] > 0 else app_load[id,0]


        # stress
        lx, ly = 1.15, 1
        nx, ny = -1, -1
        sx = nx*np.sqrt(3) * fxx / (2 * lx)
        sy = ny*np.sqrt(3) * fyy / (2 * ly)

        # # substracting body stress
        # rx = nx*np.sqrt(3) * fb_xx / (2 * lx)
        # ry = ny*np.sqrt(3) * fb_yy / (2 * ly)
        # sx -= rx; sy -= ry

        # # substracting inelastic stress
        sx[0] -= sxx[id]; sx[1] -= sxy[id]; sy[0] -= sxy[id]; sy[1] -= syy[id]
        stress[id, :2] = sx
        stress[id, 2:] = sy
        stress[id,1] = stress[id,2]

    return stress

"""Function to impose load boundary conditions for verlet integrator"""
def impose_loads(total_nodes, lbcs_ids, lbcs_fx, lbcs_fy, lbcs_fun, t):

    load = np.zeros(shape=(total_nodes,2))

    for i, fun in enumerate(lbcs_fun):
        id = np.array(lbcs_ids[i])
        if fun == 'parser':
            load[id,0] = eval(lbcs_fx[i])
            load[id,1] = eval(lbcs_fy[i])

    return load 

"""impose complementary boundary conditons for verlet integrator"""
def find_displaced_nodes(mesh_obj):  
    
    # read displacement boundary conditions from mesh object
    bcs = mesh_obj.bcs
    total_bcs = len(bcs)

    # total number of nodes
    total_nodes = mesh_obj.nx * mesh_obj.ny
    disp_nodes = np.ones(shape = (total_nodes,2), dtype=int) 
    for i in range(total_bcs):
        id = np.array(bcs[i]['ids'])
        comp = bcs[i]['comp']
        if comp == 'uc':
            disp_nodes[id,0] = 0
                
        elif comp == 'vc':
            disp_nodes[id,1] = 0  

        elif comp == 'hole':
            disp_nodes[id,0] = 0
            disp_nodes[id,1] = 0
                    
    return disp_nodes

"""Compute strain tensor"""
def compute_nodal_strain(mesh:mesh, t):
    
    # total number of nodes
    total_nodes = mesh.nx * mesh.ny

    # extracting mesh info
    disp, pos, neighbors, norm_stiff, tang_stiff, angles = mesh.u, mesh.pos, mesh.neighbors, mesh.normal_stiffness, mesh.tangential_stiffness, mesh.angles
    
    # extract load boundary conditions
    lbcs_ids = copy.copy(mesh.lbcs.ids)
    lbcs_fx = copy.copy(mesh.lbcs.fx)
    lbcs_fy = copy.copy(mesh.lbcs.fy)
    lbcs_fun = copy.copy(mesh.lbcs.fun)

    # compute applied load 
    app_load = impose_loads(total_nodes, lbcs_ids, lbcs_fx, lbcs_fy, lbcs_fun, t)

    # find displaced nodes
    disp_nodes = find_displaced_nodes(mesh)

    # compute inelastic stress
    sxx, syy, sxy = compute_plastic_stress(mesh)
    exx, eyy, exy = mesh.plastic_strain['exx'], mesh.plastic_strain['eyy'], mesh.plastic_strain['exy']

    # stress value
    strain = np.zeros((total_nodes, 4))
    nu = mesh.bond_prop['poisson_ratio']
    cr2 = (0.874 + 0.162*nu)**2

    # weight for syy element bonds
    wtg_yy = {'0': 0.5, '60': 1, '120': 1, '180': 0.5}

    for id in range(total_nodes):  # looping over the nodes
        aph = angles[id]        # neighboring bond angles
        id_disp = disp[id]      # node id displacement

        
        # syy node element info
        fyy = np.zeros(2)
        fb_yy = np.zeros(2)

        # sxx node element info
        fxx = np.zeros(2)
        fb_xx = np.zeros(2)

        # inertial force
        f_inertia = np.zeros(2)
        fb = np.zeros(2)

        for i, neigh_aph in enumerate(aph): # looping over each bond angle
            neigh_id, ns, ts = neighbors[id][i], norm_stiff[id][i], tang_stiff[id][i]
            neigh_disp = disp[neigh_id]
            ro = np.array([np.cos(np.radians(neigh_aph)), np.sin(np.radians(neigh_aph))])
            rf = pos[neigh_id] + neigh_disp - pos[id] - id_disp

            if ns:
                sxx_ij = sxx[neigh_id] - sxx[id]
                sxy_ij = sxy[neigh_id] - sxy[id]
                syy_ij = syy[neigh_id] - syy[id]
                fbx = (1/3)*(sxx_ij*ro[0] + sxy_ij*ro[1])
                fby = (1/3)*(sxy_ij*ro[0] + syy_ij*ro[1])
            else:
                fbx = 0
                fby = 0    
            
            fb += np.array([fbx, fby])            
            f_inertia += ((ns - ts) * np.dot(rf - ro, ro) * ro + ts * (rf - ro))

            if neigh_aph in [0, 60, 120, 180]:
                fyy += wtg_yy[str(neigh_aph)] * ((ns - ts) * np.dot(rf - ro, ro) * ro + ts * (rf - ro))
                fb_yy += wtg_yy[str(neigh_aph)] * np.array([fbx, fby])

            if neigh_aph in [0, 60, -60]:
                fxx += ((ns - ts) * np.dot(rf - ro, ro) * ro + ts * (rf - ro))
                fb_xx += np.array([fbx, fby])
     
        # Contribution from applied load and bond force
        f_inertia += - fb 

        fyy[0] = disp_nodes[id,0]*(f_inertia[0]+ 0.5*app_load[id,0]) - fyy[0] + fb_yy[0] 
        if app_load[id,1] == 0:     
            fyy[1] = disp_nodes[id,1]*f_inertia[1] - fyy[1] + fb_yy[1]
        else:
            fyy[1] = -app_load[id,1] if app_load[id,1] > 0 else app_load[id,1]

        fxx[1] = disp_nodes[id,1]*(f_inertia[1]+ 0.5*app_load[id,1]) - fxx[1] + fb_xx[1] 
        if app_load[id,0] == 0:     
            fxx[0] = disp_nodes[id,0]*f_inertia[0] - fxx[0] + fb_xx[0]
        else:
            fxx[0] = -app_load[id,0] if app_load[id,0] > 0 else app_load[id,0]


        # stress
        lx, ly = 1.15, 1
        nx, ny = -1, -1
        sx = nx*np.sqrt(3) * fxx / (2 * lx)
        sy = ny*np.sqrt(3) * fyy / (2 * ly)

        # compute total strain
        exxt = 0.5*(1-2*nu)*cr2*((1-nu)*sx[0] - nu*sy[1])
        eyyt = 0.5*(1-2*nu)*cr2*((1-nu)*sy[1] - nu*sx[0])
        exyt = cr2*(1+nu)*sy[0]

        # substracting inelastic strain
        strain[id, 0] = exxt - exx[id]
        strain[id, 1] = exyt - exy[id]
        strain[id, 2] = exyt - exy[id]
        strain[id, 3] = eyyt - eyy[id]

    return strain