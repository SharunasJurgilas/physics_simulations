import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


class Particle:
        """
        Particle class which stores the complete phase-space trajecotry of a particle. Includes methods
        to update and retrieve trajectories.
        
        Parameters
        ----------
        x, y, z: spatial coordinates of a particle.
        v_x, v_y, v_z: components of the 3D velocity vector of a particle
        m: mass of the particle
        """
    
    def __init__(self,
                 x=0, 
                 y=0,
                 z=0,
                 v_x=0,
                 v_y=0,
                 v_z=0,
                 m=1):
        self.x = x
        self.y = y
        self.z = z
        self.v_x = v_x
        self.v_y = v_y
        self.v_z = v_z
        self.m = m
        self.trajectory_x = np.array([])
        self.trajectory_y = np.array([])
        self.trajectory_z = np.array([])
        self.trajectory_v_x = np.array([])
        self.trajectory_v_y = np.array([])
        self.trajectory_v_z = np.array([])
        
    def getTrajectory_r(self):
        return np.array(self.trajectory_x), np.array(self.trajectory_y), np.array(self.trajectory_z)
        
    def getTrajectory_v(self):
        return np.array(self.trajectory_v_x), np.array(self.trajectory_v_y), np.array(self.trajectory_v_z)
    
    def update_trajectory_r(self, r):
        self.trajectory_x = np.array(r[0])
        self.trajectory_y = np.array(r[1])
        self.trajectory_z = np.array(r[2])
        self.x = self.trajectory_x[-1]
        self.y = self.trajectory_y[-1]
        self.z = self.trajectory_z[-1]
        return self
        
    def update_trajectory_v(self, v):
        self.trajectory_v_x = np.array(v[0])
        self.trajectory_v_y = np.array(v[1])
        self.trajectory_v_z = np.array(v[2])
        self.v_x = self.trajectory_v_x[-1]
        self.v_y = self.trajectory_v_y[-1]
        self.v_z = self.trajectory_v_z[-1]
        return self
    
    
class Gas:
        """
        Gas class. Use to generate a collection of particles based on thermodynamic parameters of the gas.
        
        Parameters
        ----------
        sigma_x, sigma_y, sigma_z: size of the particle cloud along the 3 orthogonal directions. Positions
        are sampled from Gaussian distributions.
        x_c, y_c, z_c: position of the center of the cloud.
        Tx, Ty, Tz: temperature of the particle cloud along the 3 orthogonal directions. This is used to
        generate a velocity distribution of the gas of particles.
        n: the number of particles in the gas.
        mass: mass of the particle in atomic mass units.
        
        When class is instantiated, the attribute self.gas is automatically set to the collection of particles
        which have their position and velocity values sampled from the defined macroscopic distributions.
        """
    
    kb = 1.38 * 10 ** (-23) # Boltman's constant
    amu = 1.66 * 10 ** (-27) # atomic mass unit to kg.
    
    def __init__(self, sigma_x, sigma_y, sigma_z, 
                 x_c, y_c, z_c, 
                 Tx, Ty, Tz, 
                 n, 
                 trap, 
                 mass=59):
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y
        self.sigma_z = sigma_z
        self.x_c = x_c
        self.y_c = y_c
        self.z_c = z_c
        self.Tx = Tx
        self.Ty = Ty
        self.Tz = Tz
        self.n = n
        self.trap = trap
        self.mass = mass
        self.gas = self.generate_initial_distributions()
        
    def generate_initial_distributions(self):
        """
        Generates initial distributions (position and velocity) based on the define thermodynamic parameters.
        Returns a list of particle objects-> the gas.
        """
        sigmaV_x = np.sqrt(Gas.kb * self.Tx / (self.mass * Gas.amu))
        sigmaV_y = np.sqrt(Gas.kb * self.Ty / (self.mass * Gas.amu))
        sigmaV_z = np.sqrt(Gas.kb * self.Tz / (self.mass * Gas.amu))
        
        x0Dist = np.random.normal(self.x_c, self.sigma_x, self.n)
        y0Dist = np.random.normal(self.y_c, self.sigma_y, self.n)
        z0Dist = np.random.normal(self.z_c, self.sigma_z, self.n)
        vx0Dist = np.random.normal(0, sigmaV_x, self.n)
        vy0Dist = np.random.normal(0, sigmaV_y, self.n)
        vz0Dist = np.random.normal(0, sigmaV_z, self.n)
        
        particles = [Particle(x0, y0, z0, vx0, vy0, vz0) for x0, y0, z0, vx0, vy0, vz0 in zip(x0Dist, y0Dist, z0Dist,
                                                                                              vx0Dist, vy0Dist, vz0Dist)]
        
        return particles
    
    def unravel_trajetories(self):
        """
        Extracts position and velocity trajectories of each particle at each instant of time.
        Returns numpy arrays of size (number of particles, number of time steps) corresponding
        to x, y, z positions and vx, vy, vz velocity components.
        """
        xs, ys, zs = [], [], []
        vxs, vys, vzs = [], [], []

        for p in self.gas:
            xs.append(p.trajectory_x)
            ys.append(p.trajectory_y)
            zs.append(p.trajectory_z)
            vxs.append(p.trajectory_v_x)
            vys.append(p.trajectory_v_y)
            vzs.append(p.trajectory_v_z)
        
        return np.array(xs), np.array(ys), np.array(zs), np.array(vxs), np.array(vys), np.array(vzs)
    
    def get_macro_parameters(self):
        """
        Computes the macroscopic paramaters of the cloud (size and temperature) from the positions
        and velocities of individual particles.
        Returns numpy arrays of size (time steps, ) corresponding to the size of the cloud along
        x, y, z and temperatures along the three orthogonal directions.
        """
        xs, ys, zs, vxs, vys, vzs = self.unravel_trajetories()
        
        sigma_x = np.array(xs).std(axis=0)
        sigma_y = np.array(ys).std(axis=0)
        sigma_z = np.array(zs).std(axis=0)
        T_x = vxs.std(axis=0) ** 2 * self.mass * Gas.amu / Gas.kb
        T_y = vys.std(axis=0) ** 2 * self.mass * Gas.amu / Gas.kb
        T_z = vzs.std(axis=0) ** 2 * self.mass * Gas.amu / Gas.kb
        
        return sigma_x, sigma_y, sigma_z, T_x, T_y, T_z
    
    
class Integrator:
    """
    Integrator class which uses the odeint method from the Scipy package to perform
    numerical integration of coupled differential equations.
    ----------
    Parameters:
    t0: start time of integration.
    t0: end time of integration. Choose wisely, depending on the timescale of the 
    dynamics inside the trap.
    steps: number of steps in integration.
    """
    
    def __init__(self, t0 = 0.0, t1 = 0.5, steps = 1000):
        self.t0 = t0
        self.t1 = t1
        self.steps = steps
    
    def integrate(self, particle, trap):
        """
        Integrator. Takes in a particle object and the trap (this is a function that returns
        the first and second order derivatives of position with respect to time, as a list).
        Updates the trajectories of each particle with the solutions found by integrating
        the equations of motion.
        ----------
        Parameters:
        particle: particle object with defined initial phase space parameters.
        trap: a function that returns the first and second order derivatives of 
        position with respect to time, as a list
        """
        
        t = np.linspace(self.t0, self.t1, self.steps)
        
        x0 = particle.x
        y0 = particle.y
        z0 = particle.z
        vx0 = particle.v_x
        vy0 = particle.v_y
        vz0 = particle.v_z
        
        sol1 = odeint(trap, [x0, y0, z0, vx0, vy0, vz0], t)
        
        particle.update_trajectory_r([sol1[:, 0], 
                                     sol1[:, 1], 
                                     sol1[:, 2]])
        particle.update_trajectory_v([sol1[:, 3],
                                     sol1[:, 4],
                                     sol1[:, 5]])
        return self
        
    def evolve_gas(self, gas):
        """
        Evolve the gas of particles moving inside the trap. Uses the integrate method to
        numerically integrate the equations of motion. Loops through the list of particles
        inside the gas to find the corresponding trajectories in phase-space.
        """
        
        trap = gas.trap
        
        particles = gas.gas
        
        for p in particles:
            self.integrate(p, trap)  
            
        return self

    
class Visualizer:
    """
    A collection of methods used to visualize results of the numerical integration.
    """
    
    def __init__(self, gas, integrator):
        self.gas = gas
        self.integrator = integrator
        self.macro = self.get_macro_params()
        
    def get_times(self):
        t0 = self.integrator.t0
        t1 = self.integrator.t1
        n_steps = self.integrator.steps
        return np.arange(t0, t1, (t1 - t0) / n_steps)
    
    def get_macro_params(self):
        sigma_x, sigma_y, sigma_z, T_x, T_y, T_z = self.gas.get_macro_parameters()
        return sigma_x, sigma_y, sigma_z, T_x, T_y, T_z
    
    def plot_size(self):
        ts = self.get_times()
        fig, ax = plt.subplots(1, 3, figsize=(20,4))
        ax[0].plot(ts, self.macro[0] * 1e3, color='purple', linewidth=2.0)
        ax[0].set_xlim([ts[0], ts[-1]])
        ax[0].set_xlabel('Time (s)')
        ax[0].set_ylabel(r'$\sigma_{x}$ (mm)')
        ax[1].plot(ts, self.macro[1] * 1e3, color='purple', linewidth=2.0)
        ax[1].set_xlim([ts[0], ts[-1]])
        ax[1].set_ylabel(r'$\sigma_{y}$ (mm)')
        ax[1].set_xlabel('Time (s)')
        ax[2].plot(ts, self.macro[2] * 1e3, color='purple', linewidth=2.0)
        ax[2].set_xlim([ts[0], ts[-1]])
        ax[2].set_xlabel('Time (s)')
        ax[2].set_ylabel(r'$\sigma_{z}$ (mm)')
    
    def plot_temperature(self):
        ts = self.get_times()
        fig, ax = plt.subplots(1, 3, figsize=(20,4))
        ax[0].plot(ts, self.macro[3] * 1e6, color='teal', linewidth=2.0)
        ax[0].set_xlim([ts[0], ts[-1]])
        ax[0].set_xlabel('Time (s)')
        ax[0].set_ylabel(r'$T_{x}$ ($\mu$K)')
        ax[1].plot(ts, self.macro[4] * 1e6, color='teal', linewidth=2.0)
        ax[1].set_xlim([ts[0], ts[-1]])
        ax[1].set_xlabel('Time (s)')
        ax[1].set_ylabel(r'$T_{y}$ ($\mu$K)')
        ax[2].plot(ts, self.macro[5] * 1e6, color='teal', linewidth=2.0)
        ax[2].set_xlim([ts[0], ts[-1]])
        ax[2].set_xlabel('Time (s)')
        ax[2].set_ylabel(r'$T_{z}$ ($\mu$K)')