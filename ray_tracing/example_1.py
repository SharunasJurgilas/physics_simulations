"""
Created on Fri Feb 22 19:01:20 2020

@author: Sarunas Jurgilas (CCM, 2020)
"""

from RayTracer import *
from utility import *

plt.rcParams['axes.linewidth'] = 2.5
font = {'family' : 'DejaVu Sans',
        'weight' : 'normal',
        'size'   : 18}
plt.rc('font',**font)

#Build optical system consisting of two lenses. This is the most basic infinite-conjugate microscope
#with a tube lens for imaging.
rays = [Ray(p0=[0,0,0], k0=[0,0.15,1] / np.sqrt(0.2 ** 2 + 1), fullHistory = True), 
       Ray(p0=[0,0,0], k0=[0,0.0,1], fullHistory = True),
       Ray(p0=[0,0,0], k0=[0,-0.15,1] / np.sqrt(0.2 ** 2 + 1), fullHistory = True)]

surfaces = [FlatSurface(position=40.0, aperture=6.3, n1=1.0, n2=1.0),
            FlatSurface(position=50, aperture=25.4, n1=1.0, n2=1.5),
            SphericalSurface(position=66.3, radius=-30.9, aperture=25.4, n1=1.5, n2=1.0),
            SphericalSurface(position=161.3, radius=14.3, aperture=14.3, n1=1.0, n2=1.5),
            FlatSurface(position=176.7, aperture=14.3, n1=1.5, n2=1.0),
            FlatSurface(position=180.0, aperture=6.3, n1=1.0, n2=1.0),
            ImagePlane(position=192, size=4)]

tracer = RayPropagator(rays, surfaces)

tracer.propagate()

fig, ax = plt.subplots(1,1,figsize=(16,10))
for ray in rays:
    ax.plot(np.array(ray.p0List).flatten()[2::3], np.array(ray.p0List).flatten()[1::3], color='firebrick')    
ax.plot(surfaces[0].surfaceContour()[0], surfaces[0].surfaceContour()[1], color='black')
ax.plot(surfaces[1].surfaceContour()[0], surfaces[1].surfaceContour()[1], color='black')
ax.plot(surfaces[2].surfaceContour()[0], surfaces[2].surfaceContour()[1], color='black')
ax.plot(surfaces[3].surfaceContour()[0], surfaces[3].surfaceContour()[1], color='black')
ax.plot(surfaces[4].surfaceContour()[0], surfaces[4].surfaceContour()[1], color='black')
ax.plot(surfaces[5].surfaceContour()[0], surfaces[5].surfaceContour()[1], color='blue')
ax.set_xlabel('z (mm)')
ax.set_ylabel('y (mm)')
plt.show()


# Here I look at how the image of atoms trapped in a single beam dipole trap will look like. The atoms
# cloud has a non-trivial tilt relative to the imaging system. In addidition, the spatial extent of the cloud
# is comparable to the depth of focus of the imaging system, so it is interesting to see how this will
# effect the formed image.
# I include a parameter 'd' which represents the displacement of the cloud center away from the object focal
# plane of the imaging system.
d = 0
surfaces = [FlatSurface(position=50 + d, aperture=25.4, n1=1.0, n2=1.5111),
            SphericalSurface(position=66.3 + d, radius=-30.9, aperture=25.4, n1=1.5111, n2=1.0),
            SphericalSurface(position=161.3 + d, radius=14.3, aperture=14.3, n1=1.0, n2=1.5225),
            FlatSurface(position=176.7 + d, aperture=14.3, n1=1.5225, n2=1.0),
            FlatSurface(position=180.0 + d, aperture=6.3, n1=1.0, n2=1.0),
            ImagePlane(position=194.6 + d, size=4)]

ps = randomPositionFromGaussianCloud(1000, 0.005, 8.0, np.pi * 21 / 180, -np.pi * 30 / 180)

for pos in ps:

    ks = uniforms3DSphereSampleGaussian(2000)
    rays = [Ray(p0=pos, k0=direction, fullHistory = False) for direction in ks]

    tracer = RayPropagator(rays, surfaces)
    tracer.propagate()

    
# Plot the formed image:
plt.scatter(np.array(surfaces[-1].intercepts)[:,0], np.array(surfaces[-1].intercepts)[:,1], s=0.05)