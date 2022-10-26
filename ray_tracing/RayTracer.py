"""
Created on Fri Feb 16 08:14:21 2020

@author: Sarunas Jurgilas (CCM, 2020)

Written as part of the object oriented programming course for physics undergrads at Imperial College. Adapted to do some useful tasks related to my own research.

Optical ray tracing through spherical and planar surfaces. Some of the tasks this code is useful for (i) looking at optical aberrations in optical systems; (ii) finding the light collection efficiency in imaging systems; (iii) simulating image formation to, for example, understand how misalignments in optical elements in an experiment affect the observed images.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


class Ray:
    
    '''
    Ray class tracks ray intercepts and propagation directions.
    
    Parameters
    ----------
    p0: ray intercept
    k0: ray unit vector, which indicates propagation direction
    fullHistory: if set to True will keep track of all ray intercepts. If only the last intercept is relevant, set to False (default)
    '''
    
    def __init__(self, p0, k0, fullHistory = False):
        if type(p0) != type([]) or type(k0) != type([]):
            raise TypeError('p0 and k0 must be 3-element lists')
        self.p0 = np.array(p0, dtype='float64')
        self.k0 = np.array(k0, dtype='float64')
        self.p0List = [p0]
        self.k0List = [k0]
        self.fullHistory = fullHistory
        
    def updatePosition(self,p): #Update position and direction; update position and direction lists
        self.p0 = p
        if self.fullHistory:
            self.p0List.append(p)
        
    def updateDirection(self,k):
        self.k0 = k
        if self.fullHistory:
            self.k0List.append(k)


class SphericalSurface:
    
    '''
    Refracts ray through spherical surface: first finds intercept of the ray on the surface, then using Snell's Law finds the direction
    of propgataion of the refracted ray.
    
    Parameters
    ----------
    position: position of surface on the optical axis
    radius: radius of curvature of spherical surface
    aperture: diameter of aperture
    n1: index of refraction in the first medium
    n1: index of refraction in the second medium
    '''
    
    def __init__(self, position, radius, aperture, n1, n2):
        if radius == 0:
            raise Exception('Use FlatSurface class for a zero radius of curvature')
        self.position = float(position)
        self.radius = float(radius)
        self.aperture = float(aperture)
        self.n1 = float(n1)
        self.n2 = float(n2)
        self.o = np.array([0, 0, self.position + self.radius],dtype='float64') #vector pointing to the origin of spherical srface
        
    def intercept(self, ray):
        '''
        Computes the intercept of an incident ray. Takes in ray object.
        Deals with rays moving away from surface; rays which hit the aperture.
        '''
        p = ray.p0
        k = ray.k0
        r =  p - self.o
        rdotk = r.dot(k)
        discriminant = rdotk ** 2 - (np.sum(r ** 2) - self.radius**2)
        
        if discriminant < 0:
            return None
        
        l = -rdotk + np.sign(k[2]) * np.sqrt(discriminant)
        
        if self.radius > 0:
            l = -rdotk - np.sign(k[2]) * np.sqrt(discriminant)
        
        if l < 0: #Ray moving away from lens
            return None
        
        intercept = p + l * k
        
        if np.sqrt(intercept[0] ** 2 + intercept[1] ** 2) > self.aperture:#Aperturing
            return None
        
        ray.updatePosition(intercept)
        return intercept
    
    def refract(self, ray):
        '''
        Use Snell's Law in vector form. Return the unit vector representing the direction of propagation of the refracted ray.
        '''
        p = ray.p0
        k = ray.k0
        norm = (p - self.o) / np.linalg.norm((p - self.o)) #unit normal vector
        nRatio = self.n1 / self.n2
        cosi = -k.dot(norm) #cosine of angle between unit normal and incident ray
        
        if cosi < 0:
            cosi = -cosi
            norm = -norm
            
        d = 1 - nRatio ** 2 * (1 - cosi ** 2)
        
        if d <= 0: #total internal reflection
            return None
        
        k_new = nRatio * k + (nRatio * cosi - np.sqrt(d)) * norm #direction of refracted ray
        
        k_new_norm = k_new / np.linalg.norm(k_new)
        
        ray.updateDirection(k_new_norm)
        return k_new_norm
    
    def surfaceContour(self):
        '''
        Returns the surface coordinates. Useful for graphical representation of the optical systemt.
        '''
        alpha = np.arcsin(self.aperture / np.abs(self.radius))
        angles = np.arange(np.pi - alpha, np.pi + alpha + 2 * alpha / 100, 2 * alpha / 100)
        if self.radius < 0:
            angles = np.arange(np.pi * 2 - alpha + 2 * alpha / 100, np.pi * 2 + alpha, 2 * alpha / 100)
        z = self.o[2] + np.abs(self.radius) * np.cos(angles)
        y = np.abs(self.radius) * np.sin(angles)
        return z, y


class FlatSurface:
    
    '''
    Dose all the same things as SphericalSurface class but now for a flat surface.
    
    Parameters
    ----------
    position: position of surface on the optical axis
    radius: radius of curvature of spherical surface
    aperture: diameter of aperture
    n1: index of refraction in the first medium
    n1: index of refraction in the second medium
    '''
    
    def __init__(self, position, aperture, n1, n2):
        self.position = float(position)
        self.aperture = float(aperture)
        self.n1 = n1
        self.n2 = n2
        
    def intercept(self, ray):
        p = ray.p0
        k = ray.k0
        point = np.array([0., 0., self.position])
        normal = np.array([0., 0., 1.])
        l = -1 * normal.dot( p - point ) / k.dot(normal)
        
        if l < 0.: #going away from surface
            return None
        
        intercept = p + (l * k)
        
        if np.sqrt(intercept[0] ** 2 + intercept[1] ** 2) > self.aperture: #hits aperture
            return None
        
        ray.updatePosition(intercept)
        return intercept
    
    def refract(self, ray):
        p = ray.p0
        k = ray.k0
        norm = np.array([0,0,-1]) #unit normal vector
        nRatio = self.n1 / self.n2
        cosi = -k.dot(norm) #cosine of angle between unit normal and incident ray
        
        if cosi < 0:
            cosi = -cosi
            norm = -norm
            
        d = 1 - nRatio ** 2 * (1 - cosi ** 2)
        
        if d <= 0: #total internal reflection
            return None
        
        k_new = nRatio * k + (nRatio * cosi - np.sqrt(d)) * norm #direction of refracted ray
        
        k_new_norm = k_new / np.linalg.norm(k_new)
        
        ray.updateDirection(k_new_norm)
        return k_new        
    
    def surfaceContour(self):
        z = [self.position, self.position]
        y = [-self.aperture, self.aperture]
        return z, y
        

class ImagePlane:
    
    '''
    Finds final intercepts in image plane.
    
    
    Parameters
    ----------
    position: position of image plane on optical axis
    size: size of image plane (detector size)
    '''
    
    def __init__(self, position, size):
        self.position = position
        self.size = size
        self.intercepts = []
        
    def intercept(self, ray):
        p = ray.p0
        k = ray.k0
        point = np.array([0., 0., self.position])
        normal = np.array([0., 0., 1.])
        l = -1 * normal.dot( p - point ) / k.dot(normal)
        
        if l < 0.: #going away from surface
            return None
        
        intercept = p + (l * k)
        
        if np.abs(intercept[0]) > np.abs(self.size) or np.abs(intercept[1]) > np.abs(self.size):
            return None
        
        ray.updatePosition(intercept)
        self.intercepts.append(intercept)
        return intercept
    
    def surfaceContour(self):
        z = [self.position, self.position]
        y = [-self.size, self.size]
        return z, y
    

class RayPropagator():
    
    '''
    Class to propagate rays through a user defined optical system.
    
    Parameters
    ----------
    rays: ray objects as a list with defined intercept and direction
    surfaces: surface objects which describe the optcial system as a list.
    
    Keeps tracks of rays which miss the optical systems or are apertured.
    '''
    
    def __init__(self, rays, surfaces):
        self.rays = rays
        self.surfaces = surfaces
        self.missedRaysCounter = 0
        self.ray_bundle = []
    
    def propagate(self):
        '''
        Propagate ray bundle through the optical system.
        Update the ray object with ray intercetp and direction unit vector.
        '''
        for ray in self.rays:
            for surface in self.surfaces:
                
                intercept = surface.intercept(ray)
                if intercept is None: #move to next surface if ray misses surface.
                    self.missedRaysCounter = self.missedRaysCounter + 1
                    break
                    
                if isinstance(surface, ImagePlane): #check if surface is image plane.
                    break
                    
                direction = surface.refract(ray) #refract ray.
                
                if direction is None: #move to next surface if total internatl reflection occurs.
                    self.missedRaysCounter = self.missedRaysCounter + 1
                    break
        
        self.ray_bundle.append(ray.p0List)
