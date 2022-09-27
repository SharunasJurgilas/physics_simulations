'''
Script for propagating a Gaussian laser beam trhough a sequence of lenses.
Some useful methods to find final waist, plot optical system and the pro-
pagation of a laser beam through it.
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelmin

plt.rcParams['axes.linewidth'] = 1.5
font = {'family' : 'DejaVu Sans',
        'weight' : 'normal',
        'size'   : 12}
plt.rc('font',**font)

class GaussianBeam:
    '''
    Define a Gaussian Laser beam (TEM 00) by its waist, radius of curvature and the wavelength.
    The waists and radii of curvature are stored in two seperate lists (class attributes).
    To propagate a Gaussian beam it is useful to define the complex beam parameter, q.
    q's are also stored in a list as a class attribute.
    
    Parameters
    ----------
    waist: define initial waist size (position is alway going to be at z=0). All waists
    are stored as a list.
    R: define initial radius of curvature . All radii of curvature
    are stored as a list.
    Lambda: wavelength of the laser beam.
    '''
    
    def __init__(self, waist, Lambda, R=0.0):
        self.waist = [waist]
        self.R = [R]
        self.Lambda = Lambda
        
        self.q = [self.get_q()]        
        
    def get_q(self):
        '''
        Calculate the complex beam paramater using the radius of curavture, waist size and
        wavelength of the laser beam.
        '''
        return 1 / complex(self.R[-1], - self.Lambda / (3.14 * self.waist[-1] ** 2))
    
    def append_q(self, q_new):
        self.q.append(q_new)
        
    def append_waist(self):
        '''
        Find waist of the beam from the complex beam parameter.
        Append it to the list of waists.
        '''
        self.waist.append(np.sqrt(- self.Lambda / (3.14 * (1 / self.q[-1]).imag)))
        
    def append_R(self):
        '''
        Find radius of curvature of the beam from the complex beam parameter.
        Append it to the list of R.
        '''
        self.R.append((1 / self.q[-1]).real)
        
    def overwrite_last_q(self, q_new):
        self.q[-1] = q_new
        
    def overwrite_last_w(self):
        self.waist[-1] = np.sqrt(- self.Lambda / (3.14 * (1 / self.q[-1]).imag))
        
    def overwrite_last_R(self):
        self.R[-1] = (1 / self.q[-1]).real
        
        
class ThinLens:
    '''
    Thin lens class to define a lens. Propagate method computes the new complex beam
    parameter after the beam was refracted by this lens.
    
    Parameters
    ----------
    f: focal length of the lens.
    '''
    
    def __init__(self, f):
        self.f = f
        
    def propagate(self, beam):
        '''
        Method to compute new q and update lists of q, w, R.
        
        Parameters
        ----------
        beam: GaussianBeam object.
        '''
        q = beam.q[-1]
        q_new = self.f * q / (self.f - q)
        beam.overwrite_last_q(q_new)
        beam.overwrite_last_w()
        beam.overwrite_last_R()
        
        
class FreeSpace:
    '''
    Free space class to propagate beam trhough free space. Propagate method computes the new complex beam
    parameter.
    
    Parameters
    ----------
    d: distance of free propagation.
    '''
    def __init__(self, d):
        self.d = d
        
    def propagate(self, beam):
        '''
        Method to compute new q and update lists of q, w, R.
        
        Parameters
        ----------
        beam: GaussianBeam object.
        '''
        q = beam.q[-1]
        ds = np.arange(0, self.d, 0.1)
        for d in ds:
            q_new = q + d
            beam.append_q(q_new)
            beam.append_waist()
            beam.append_R()
        
            
class OpticalSystemAnalysis:
    '''
    Class to analyse propagation of a Gaussian laser beam trhough an optical system.
    Parameters
    ----------
    system: list of optical elements: free space propagators or thin lenses.
    zoom_z: zoom in along position axis in plots of beam path.
    zoom_z: zoom in along beam waist axis in plots of beam path.
    '''
    
    def __init__(self, 
                 system=[], 
                 zoom_z=[], 
                 zoom_w=[]):
        self.system = system
        self.zoom_z = zoom_z
        self.zoom_w = zoom_w
        
    def propagate_through_system(self, beam):
        '''
        Propagates input Gaussian laser beam through optical system.
        '''
        for element in self.system:
            element.propagate(beam)
        return self
        
    def show_propagation(self, beam):
        '''
        Plots entire optical system and the propagation of the input laser beam.
        '''
        w = np.array(beam.waist)[1:]
        prop_length = self.get_total_distance()
        zs = np.arange(0, prop_length, prop_length / w.shape[0])
        fig, ax = plt.subplots(1, 1, figsize=(20,6))
        ax.plot(zs, w, color='maroon', linewidth=2)
        ax.plot(zs, -w, color='maroon', linewidth=2)
        ax.set_xlim([zs[0], zs[-1]])
        ax.set_ylim([-1.3 * w.max(), 1.3 * w.max()])
        if len(self.zoom_z) != 0 or len(self.zoom_w) != 0:
            ax.set_xlim(self.zoom_z)
            ax.set_ylim(self.zoom_w)
        ax.set_xlabel('z (mm)')
        ax.set_ylabel('waist (mm)')
        
        lens_positions, types = self.get_lens_positions()
        arrows_styles = {'pos':'<->', 'neg':']-['}
        for z, t in zip(lens_positions, types):
            ax.annotate("", xy=(z, -1.2 * w.max()), xytext=(z, 1.2 * w.max()),
            arrowprops=dict(arrowstyle=arrows_styles[t], lw=3))    
        ax.grid()
        
    def get_total_distance(self):
        '''
        Computes total distance of propagation and returns it.
        '''
        d_tot = 0
        for element in self.system:
            if isinstance(element, FreeSpace):
                d_tot += element.d
        return d_tot
    
    def get_lens_positions(self):
        '''
        Finds positions of lenses in the optical system.
        '''
        zs = []
        types = []
        for e in self.system:
            if isinstance(e, FreeSpace):
                zs.append(e.d)
        for e in self.system:
            if isinstance(e, ThinLens):
                if e.f < 0:
                    types.append('neg')
                else:
                    types.append('pos')
        zs_l = [sum(zs[:i]) for i in range(1, len(zs))]
        return zs_l, types
    
    def get_waist_at_(self, z_array):
        '''
        Get the waist size at positions specified in z_array.
        Parameters
        ----------
        z_array: numpy array of positions where the beam waist should be evaluated.
        '''
        w = np.array(beam.waist)[1:]
        prop_length = self.get_total_distance()
        zs = np.arange(0, prop_length, prop_length / w.shape[0])
        ws = []
        for z in z_array:
            i = (np.abs(zs - z)).argmin()
            ws.append(w[i])
        return np.array(ws)
    
    def get_minimum_waist_distance_from_last_lens(self):
        '''
        Finds the position of the beam waist relative to the last lens.
        Returns this position and the size of the list (tuple).
        '''
        zl, types = self.get_lens_positions()
        prop_length = self.get_total_distance()
        w = np.array(beam.waist)[1:]
        zs = np.arange(0, prop_length, prop_length / w.shape[0])
        indices = argrelmin(w)
        i = indices[0][-1]
        return zs[i] - zl[-1], w[i]