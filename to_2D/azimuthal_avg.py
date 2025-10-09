import numpy as np
import warnings
from ..main import dataclass, _fill_2Dhist



def azimuthal_avg(self, 
                  variables,
                  weights,
                  r_out = 50,
                  r_in = 0,
                  n_theta = 45,
                  n_r = 50,
                  verbose = 1,
                  return_grid = True):
    
    grid_dict = {} 
    grid_dict['theta'] = np.linspace(-np.pi / 2, np.pi / 2, n_theta)
    grid_dict['r'] = np.linspace(0, r_out, n_r)

    if verbose > 0:
        max_length = max(len(name) for name in variables)
        for w, v in zip(weights, variables):
            print(f'Variable: {v:<{max_length}} - weight: {w}')

    values_dict = {ivs: [] for ivs in variables}
    weights_dict = values_dict.copy()
    
    radius =  r_out / self.code2au;
    mask = (abs(self.trans_xyz) < radius*1.5).all(axis=0)

    data_theta = np.arctan2(self.cyl_z[mask], self.cyl_R[mask])
    data_R = self.dist[mask]

    for i, ivs in enumerate(variables):
        if (ivs == np.array(['d', 'P'])).any():
            values_dict[ivs] = self.mhd[ivs][mask]
        else:
            values_dict[ivs] = getattr(self, ivs)[mask]
    
    for i, ivs in enumerate(variables):
        w = weights[i]
        if w != None:
            if w == 'mass': weights_dict[ivs] = self.m[mask]
            if w == 'volume': weights_dict[ivs] = (self.amr['ds']**3)[mask]
        else: 
            weights_dict[ivs] = np.ones(mask.sum())

    def get_bins(grid_2D):
        x, y = grid_2D
        x_bins = x[:-1] + np.diff(x) / 2
        y_bins = y[:-1] + np.diff(y) / 2
        return x_bins, y_bins

    theta_grid, r_grid =[grid_dict[comp] for comp in grid_dict.keys()]
    theta_bins, r_bins = get_bins((theta_grid, r_grid))

    # Full theta ensures that the range goes from -π/2 to π/2
    full_theta = np.insert(theta_bins, [0, len(theta_bins)], [-np.pi/2, np.pi/2])
    
    coor = (theta_bins, r_bins) 
    new_coor = (full_theta, r_bins)
    
    results_dict = {}
    for iv in values_dict.keys():
        hist_value,_,_ = np.histogram2d(data_theta, data_R* self.code2au, bins = (theta_grid, r_grid), weights =  weights_dict[iv] * values_dict[iv])
        hist_weight, _, _ = np.histogram2d(data_theta, data_R * self.code2au, bins = (theta_grid, r_grid), weights = weights_dict[iv])  
        
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            results_dict[iv] = _fill_2Dhist(hist_value / hist_weight, coor, new_coor, periodic_x = False)

    if return_grid: 
        final_grid = {}
        final_grid['theta'] = full_theta
        final_grid['r'] = r_bins

        return results_dict, final_grid
    else:
        results_dict

dataclass.azimuthal_avg = azimuthal_avg