from gaussian_LB_tracer import *

# Propagate Gaussian beam with waist size of 0.4 mm, 1.8 m away from first lens. 
# First lens is f=175 mm, second lens f=175 mm, last lens f=175 mm. Can
# vary the distance between lens to investigate how misalignments change the final
# spot size, waist position, etc...

beam = GaussianBeam(waist=0.4, Lambda=1064 * 1e-6, R=0.0)
system = [FreeSpace(1800), 
          ThinLens(175),
          FreeSpace(367),
          ThinLens(175),
          FreeSpace(500),
          ThinLens(175),
          FreeSpace(200)]
optical_system = OpticalSystemAnalysis(system=system)
os = optical_system.propagate_through_system(beam)
os.show_propagation(beam)