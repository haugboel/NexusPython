import sys, os, multiprocessing
import numpy as np
from ..main import dataclass
from ..path_config import config

max_threads = multiprocessing.cpu_count()
os.environ['OMP_NUM_THREADS'] = str(max_threads)
# Now import Numba and configure the threads
from numba import  set_num_threads 
# Set the number of threads before any parallel operation
try: set_num_threads(max_threads)
except: set_num_threads(8)

sys.path.insert(0, config["user_osyris_path"])
import osyris

# This function does not take vectors. 
# This means if you want to extract the entirety of the B-field either extract |B| or bx, by, and bz individually 
# You must assign the wanted variables to the data class (except 'd' and 'P')
def osyris2Dslab(self, variables, 
                 data_name = 'data1', 
                 view = 200, 
                 height = None, 
                 dz = None, 
                 center = None, 
                 resolution = 400, 
                 viewpoint = 'face-on', 
                 weights = [None], 
                 verbose = 1,
                 use_trans_pos=False):

    #### For storing data ####
    try: self.osyris_ivs[data_name] = {}
    except: self.osyris_ivs = {data_name: {}}
    
    if height == None: height = view
    selection_radius = (np.sqrt((0.5 * height)**2 + (0.5*view)**2) * 2) / self.code2au
    mask = self.dist < selection_radius

    ds = self.amr['ds'][mask]
    cartcoor = self.rel_xyz[:,mask]
    if use_trans_pos:
        cartcoor = self.trans_xyz[:,mask]

    values = {ivs: [] for ivs in variables}
    for i, ivs in enumerate(variables):
        if (ivs == np.array(['d', 'P'])).any():
            values[ivs] = self.mhd[ivs][mask]
        else:
            values[ivs] = getattr(self, ivs)[mask]
    
    weights_dict = {ivs: [] for ivs in variables}
    for i, ivs in enumerate(variables):
        w = weights[i]
        if w != None:
            if w == 'mass': weights_dict[ivs] = self.m[mask]
            if w == 'volume': weights_dict[ivs] = (self.amr['ds']**3)[mask]
        else: 
            weights_dict[ivs] = np.ones(mask.sum())


    DS = osyris.Dataset(nout = None)
     # overwrite units
    DS.meta['unit_l'] = self.l_cgs
    DS.meta['unit_t'] = self.t_cgs
    DS.meta['unit_d'] = self.d_cgs
    DS.set_units()
    DS.meta["ndim"] = 3

    #### Defining viewpoint coordinate system #####
    try: self.new_x
    except: self.calc_trans_xyz(verbose = verbose)
    try: self.L
    except: self.recalc_L() 

    if (viewpoint == np.array(['x', 'y', 'z'])).any():
        to_view = viewpoint
    # Example of viewpoint dictionary:
    # viewpoint = {'view_vector:':np.array([1,0,0]), 'new_x': np.array([0,1,0]), 'new_y': np.array([0,0,1])}
    elif type(viewpoint) == dict:
        dir_vecs = {}
        dir_vecs['pos_u'] = osyris.Vector(*viewpoint['new_x'])
        dir_vecs['pos_v'] = osyris.Vector(*viewpoint['new_y'])
        dir_vecs['normal'] = osyris.Vector(*viewpoint['view_vector'])
        to_view = dir_vecs
    else:
        dir_vecs = {}
        dir_vecs['pos_u'] = osyris.Vector(*self.new_x)
        dir_vecs['pos_v'] = osyris.Vector(*self.new_y)
        dir_vecs['normal'] = osyris.Vector(*self.L)
    if viewpoint == 'face-on':
        to_view = dir_vecs
    elif viewpoint == 'edge-on':
        dir_vecs2 = dir_vecs.copy()
        dir_vecs2['normal'] = dir_vecs['pos_u']
        dir_vecs2['pos_u'] = dir_vecs['pos_v']
        dir_vecs2['pos_v'] = dir_vecs['normal']
        to_view = dir_vecs2
   
    view *= osyris.units('au')
    height *= osyris.units('au')
    if dz == None: dz = 0.1 * view
    else: dz *= osyris.units('au')
    
    if not isinstance(center, np.ndarray): center = osyris.Vector(x=0,y=0,z=0,unit='au')
    else: center = osyris.Vector(*center, unit='au')
    
    
    DS['amr'] = osyris.Datagroup()
    DS['amr']['dx'] = osyris.Array(ds*self.l_cgs, unit='cm')
    DS['amr']['position'] = osyris.Vector(*(cartcoor * self.l_cgs), unit='cm')
    
    DS['hydro'] = osyris.Datagroup()
    #Looping over the scalar variables set for extraction
    for i, ivs in enumerate(variables):
        DS['hydro'][ivs] = osyris.Array(weights_dict[ivs] * values[ivs], unit = 'dimensionless')

        res = osyris.map({"data": DS['hydro'][ivs], "norm": "log"}, dx=view, dy = height, dz = dz, 
                         origin=center, resolution=resolution, direction=to_view, plot=False, operation = "sum", verbose=verbose)
                
        if type(weights[i]) == str: 
            DS['hydro']['w'] = osyris.Array(weights_dict[ivs], unit = 'dimensionless')
            res_weight = osyris.map({"data": DS['hydro']['w'], "norm": "log"}, dx = view, dy = height, dz = dz, 
                             origin=center, resolution=resolution, direction=to_view, plot=False, operation = "sum", verbose=verbose)
            final_weights = res_weight.layers[0]['data']
            
        else: final_weights = np.ones_like(res.layers[0]['data'])

        self.osyris_ivs[data_name][ivs] = res.layers[0]['data'] / final_weights
    
        
dataclass.osyris2Dslab = osyris2Dslab