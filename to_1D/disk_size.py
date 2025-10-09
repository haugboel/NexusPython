import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d
from scipy.interpolate import interp1d
from astropy.constants import G, M_sun
import astropy.units as u
from ..main import dataclass

def calc_disksize(self, 
                  use_fitted_H = False,
                  h = 20, 
                  r_in = 10, 
                  r_out = 500, 
                  n_bins = 200, 
                  a = 0.8, 
                  plot = True, 
                  smooth_pct = 0.1, 
                  verbose = 1):
    
    try: self.cyl_z
    except: self.recalc_L()
    
    if use_fitted_H:
        try: self.r_bins
        except: 
            self.fit_HΣ(r_in = r_in, 
                        r_out = r_out, 
                        n_bins = n_bins, 
                        plot = False, 
                        verbose=verbose)
            
    if use_fitted_H:
        rad_bins = self.r_bins
        r_plot = rad_bins[:-1] + 0.5 * np.diff(rad_bins)
        H_func = interp1d(r_plot, self.H_1D[:,0], fill_value='extrapolate')

        mask_r = (self.cyl_R > rad_bins.min()) & (self.cyl_R < rad_bins.max())
        mask_h = abs(self.cyl_z[mask_r]) < 3 * H_func(self.cyl_R[mask_r])
        mask = np.zeros_like(mask_r, dtype = 'bool')
        mask[mask_r] = mask_h
    
    else:
        h, r_in, r_out = np.array([h, r_in, r_out]) / self.code2au
        rad_bins = np.logspace(np.log10(r_in), np.log10(r_out), n_bins)
        r_plot = rad_bins[:-1] + 0.5 * np.diff(rad_bins)    
        mask = (abs(self.cyl_z) <  h) & (self.cyl_R < r_out)
    
    vφ = np.sum(self.vrel[:,mask] * self.e_phi[:,mask], axis=0)
    m = self.m[mask]
    R = self.cyl_R[mask]

    h_mass, _ = np.histogram(R, bins = rad_bins, weights =  m)
    h_vφ, _ = np.histogram(R, bins = rad_bins, weights =  vφ * m)
    h_vφ2, _ = np.histogram(R, bins = rad_bins, weights =  vφ**2 * m)

    vφ_1D = (h_vφ/h_mass) 
    vφ2 = (h_vφ2/h_mass) 
    σvφ_1D = np.sqrt(vφ2 - vφ_1D**2) 
    self.vφ_1D = np.stack((vφ_1D, σvφ_1D), axis = 1)
    self.r_plot_disksize = r_plot

    ####### Include self-gravity from the disk #######
    origo_bins = np.insert(rad_bins, 0, 0)[:-1]
    annulus_mass, _ = np.histogram(np.linalg.norm(self.rel_xyz, axis = 0), bins = origo_bins, weights=self.mhd['m'])
    accumulated_mass = np.cumsum(annulus_mass)

    self.kep_vel = (((G * ((self.sink_mass + accumulated_mass)  * self.m_cgs) * u.g) / (r_plot * self.code2au * u.au))**0.5).to('cm/s').value

    keplerian_disk = True
    self.disksize = {
        'Disksize convergence': keplerian_disk,
        'Velocity drop': np.full((2), np.nan),
        'Rayleigh': np.full((2), np.nan),
        'Combined': np.full((2), np.nan)}
    
    filter_size = int(len(r_plot) * smooth_pct)

    ####__________________METHOD: DROP IN VELOCITY FROM 80% OF KEPLERIAN_______________#####
    ratio = self.vφ_1D[:,0] / (self.kep_vel / self.v_cgs)
    ratio_sigma = self.vφ_1D[:,1] / (self.kep_vel / self.v_cgs)
    orbitvel_ratio_mean = uniform_filter1d(ratio, size = filter_size)
    orbitvel_ratio_sigma = uniform_filter1d(ratio_sigma, size = filter_size)

    indencies_above = np.array(orbitvel_ratio_mean > a, dtype = 'int')

    if sum(indencies_above) == 0 or (indencies_above == 1).all(): 
        keplerian_disk = False
    segment_starts = np.where(np.diff(indencies_above, prepend=0) == 1)[0]
    segment_ends = np.where(np.diff(indencies_above, append=0) == -1)[0]
    if segment_ends.size == 0: keplerian_disk = False
    else:
        largest_segment_end = np.argmax(r_plot[segment_ends] - r_plot[segment_starts])
        gradient = np.gradient(orbitvel_ratio_mean, r_plot)[segment_ends[largest_segment_end]]
        if gradient > 0 or np.isnan(gradient): keplerian_disk = False
        else:  
            r_index = segment_ends[largest_segment_end]
            self.disksize['Velocity drop'][0] = r_plot[r_index] * self.code2au
            self.disksize['Velocity drop'][1] = abs(np.gradient(orbitvel_ratio_mean, r_plot)[r_index]**(-1) *  orbitvel_ratio_sigma[r_index]) * self.code2au

    ####__________________METHOD: RAYLEIGH DISCRIMINANT < 0_______________#####

    R = r_plot 
    Ω = self.vφ_1D[:,0] / R
    σ_Ω = self.vφ_1D[:,1] / R

    dR_RΩ = uniform_filter1d(2 * Ω * R * np.gradient(Ω,R), size = filter_size) # Smooth out the differential to remove oscillation
    prefactor = 4 *  Ω**2  
    kappa2 = uniform_filter1d(prefactor + dR_RΩ, size = filter_size)
    filter_1dev = uniform_filter1d(np.gradient(Ω, R), size=filter_size)

    filter_2dev = uniform_filter1d(np.gradient(filter_1dev, R), size=filter_size)
    sigma_kappa2 = uniform_filter1d(abs((8 * Ω  
                                        + 2 * R * filter_1dev 
                                        + 2 * R * Ω * filter_2dev * filter_1dev**-1) * σ_Ω), size = filter_size)

    if np.isnan(kappa2).all(): keplerian_disk = False
    indencies_below = np.array(kappa2 < 0, dtype = 'int')
    if sum(indencies_below) == 0 or (indencies_below == 1).all(): keplerian_disk = False
    gradient = np.gradient(kappa2, r_plot)
    ints2grad = int(smooth_pct * len(r_plot)) // 2
    for j, r_au in enumerate(R):
        running_gradient = gradient[j - ints2grad: j + ints2grad]
        if kappa2[j] < 0 and running_gradient.mean() < 0 and ~np.isnan(gradient[j]):
            self.disksize['Rayleigh'][0] = r_au * self.code2au
            self.disksize['Rayleigh'][1] = abs(running_gradient.mean()**(-1) *  sigma_kappa2[j]) * self.code2au
            break

    mean = ((self.disksize['Velocity drop'][0] / self.disksize['Velocity drop'][1]**2 
        + self.disksize['Rayleigh'][0] / self.disksize['Rayleigh'][1]**2) 
        / (self.disksize['Velocity drop'][1]**-2 + self.disksize['Rayleigh'][1]**-2))
    mean_sigma = (1 / (self.disksize['Velocity drop'][1]**-2 + self.disksize['Rayleigh'][1]**-2))**0.5

    self.disksize['Combined'] = [mean, mean_sigma]
    self.disksize['Disksize convergence'] = keplerian_disk

    if verbose > 0: 
        print('Disk size from 2 methods [au]:')
        for key, value in self.disksize.items():
            print(f"{key}: {value}")

    if plot:
        fig, axs = plt.subplots(1, 2, figsize = (20,6),gridspec_kw={'width_ratios': [2, 1.5]})


        axs[0].loglog(r_plot * self.code2au, self.kep_vel, label = 'Keplerian Orbital Velocity', color = 'black')
        axs[0].loglog(r_plot * self.code2au, self.vφ_1D[:,0]* self.v_cgs , label = 'Azimuthal velocity v$_\phi$', c = 'blue')
        axs[0].fill_between(r_plot * self.code2au, (self.vφ_1D[:,0]- self.vφ_1D[:,1]) * self.v_cgs, (self.vφ_1D[:,0]+ self.vφ_1D[:,1])* self.v_cgs, alpha = 0.5, label = '$\pm1\sigma_{\phi}$')

        axs[0].set(xlabel = 'Distance from sink [au]', ylabel = 'Orbital speed [cm/s]')

        axs[0].legend(frameon = False)
        axs[1].semilogx(r_plot * self.code2au, orbitvel_ratio_mean, label = 'v$_\phi$/v$_K$ ratio', color = 'black', lw = 0.8)
        axs[1].fill_between(r_plot * self.code2au, orbitvel_ratio_mean - orbitvel_ratio_sigma, orbitvel_ratio_mean + orbitvel_ratio_sigma, alpha = 0.5, color = 'grey', label = '$\pm1\sigma_{v_\phi/v_K}$')
        axs[1].axhline(a, color = 'red', ls = '--', label = f'a = {a}')
        axs[1].axhline(1, color = 'black', ls = '-', alpha = 0.7)
        axs[1].set(xlabel = 'Distance from sink [au]', ylim = (0.5, 1.1))
        axs[1].legend(frameon = False)
    
    
dataclass.calc_disksize = calc_disksize


###### The following function is only for already extracted data #######
###### All should be given in code units, and vφ_1D should be a (N, 2) array, i.e. containing variability

def calc_disksize_postprocessing(self, 
                  vφ_1D,
                  r_plot,
                  kep_vel,                
                  a = 0.8,
                  smooth_pct = 0.1, 
                  verbose = 1):
    
    keplerian_disk = True
    disksize = {
        'Disksize convergence': keplerian_disk,
        'Velocity drop': np.full((2), np.nan),
        'Rayleigh': np.full((2), np.nan),
        'Combined': np.full((2), np.nan)}
    
    filter_size = int(len(r_plot) * smooth_pct)

    ####__________________METHOD: DROP IN VELOCITY FROM 80% OF KEPLERIAN_______________#####
    ratio = vφ_1D[:,0] / (kep_vel )
    ratio_sigma = vφ_1D[:,1] / (kep_vel)
    orbitvel_ratio_mean = uniform_filter1d(ratio, size = filter_size)
    orbitvel_ratio_sigma = uniform_filter1d(ratio_sigma, size = filter_size)

    indencies_above = np.array(orbitvel_ratio_mean > a, dtype = 'int')

    if sum(indencies_above) == 0 or (indencies_above == 1).all(): 
        keplerian_disk = False
    segment_starts = np.where(np.diff(indencies_above, prepend=0) == 1)[0]
    segment_ends = np.where(np.diff(indencies_above, append=0) == -1)[0]
    if segment_ends.size == 0: keplerian_disk = False
    else:
        largest_segment_end = np.argmax(r_plot[segment_ends] - r_plot[segment_starts])
        gradient = np.gradient(orbitvel_ratio_mean, r_plot)[segment_ends[largest_segment_end]]
        if gradient > 0 or np.isnan(gradient): keplerian_disk = False
        else:  
            r_index = segment_ends[largest_segment_end]
            disksize['Velocity drop'][0] = r_plot[r_index] * self.code2au
            disksize['Velocity drop'][1] = abs(np.gradient(orbitvel_ratio_mean, r_plot)[r_index]**(-1) *  orbitvel_ratio_sigma[r_index]) * self.code2au

    ####__________________METHOD: RAYLEIGH DISCRIMINANT < 0_______________#####

    R = r_plot 
    Ω = vφ_1D[:,0] / R
    σ_Ω = vφ_1D[:,1] / R

    dR_RΩ = uniform_filter1d(2 * Ω * R * np.gradient(Ω,R), size = filter_size) # Smooth out the differential to remove oscillation
    prefactor = 4 *  Ω**2  
    kappa2 = uniform_filter1d(prefactor + dR_RΩ, size = filter_size)
    filter_1dev = uniform_filter1d(np.gradient(Ω, R), size=filter_size)

    filter_2dev = uniform_filter1d(np.gradient(filter_1dev, R), size=filter_size)
    sigma_kappa2 = uniform_filter1d(abs((8 * Ω  
                                        + 2 * R * filter_1dev 
                                        + 2 * R * Ω * filter_2dev * filter_1dev**-1) * σ_Ω), size = filter_size)

    if np.isnan(kappa2).all(): keplerian_disk = False
    indencies_below = np.array(kappa2 < 0, dtype = 'int')
    if sum(indencies_below) == 0 or (indencies_below == 1).all(): keplerian_disk = False
    gradient = np.gradient(kappa2, r_plot)
    ints2grad = int(smooth_pct * len(r_plot)) // 2
    for j, r_au in enumerate(R):
        running_gradient = gradient[j - ints2grad: j + ints2grad]
        if kappa2[j] < 0 and running_gradient.mean() < 0 and ~np.isnan(gradient[j]):
            disksize['Rayleigh'][0] = r_au * self.code2au
            disksize['Rayleigh'][1] = abs(running_gradient.mean()**(-1) *  sigma_kappa2[j]) * self.code2au
            break

    # Weigthed mean and uncertainty:
    mean = ((disksize['Velocity drop'][0] / disksize['Velocity drop'][1]**2 
        + disksize['Rayleigh'][0] / disksize['Rayleigh'][1]**2) 
        / (disksize['Velocity drop'][1]**-2 + disksize['Rayleigh'][1]**-2))
    mean_sigma = (1 / (disksize['Velocity drop'][1]**-2 + disksize['Rayleigh'][1]**-2))**0.5

    disksize['Combined'] = [mean, mean_sigma]
    disksize['Disksize convergence'] = keplerian_disk
    
    return disksize

dataclass.calc_disksize_postprocessing = calc_disksize_postprocessing