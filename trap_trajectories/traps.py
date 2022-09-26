mm = (9.27 * 10 ** (-24)) / (59 * 1.66 * 10 ** (-27))
b = 0.3

# Equation of motion for a particle in a magentic quadrupole trap. Gravity included.
def mqt(S, t):
    
    x = S[0]
    y = S[1]
    z = S[2]
    vx = S[3]
    vy = S[4]
    vz = S[5]
    
    dx_dt = vx
    dy_dt = vy
    dz_dt = vz
    dvx_dt = (-mm * b * x) / (4.0 * (x ** 2.0 / 4.0 + y ** 2.0 / 4.0 + z ** 2.0) ** 0.5)
    dvy_dt = (-mm * b * y) / (4.0 * (x ** 2.0 / 4.0 + y ** 2.0 / 4.0 + z ** 2.0) ** 0.5)
    dvz_dt = (-mm * b * z) / ((x ** 2.0 / 4.0 + y ** 2.0 / 4.0 + z ** 2.0) ** 0.5) - 9.8
    
    return [dx_dt, dy_dt, dz_dt, dvx_dt, dvy_dt, dvz_dt]


P = 5.0
w0 = 37 * 1e-6
Lambda = 1064 * 1e-9
epsilon0 = 8.85 * 1e-12
c = 3 * 1e8
theta = 0.0
alpha = 1.13 * 1e-38 / 3.76 # The divider makes CaF polarizability. Bare number is that of Rb.
m = 59 * 1.66 * 10 ** (-27)
theta = theta * np.pi / 180

# Equation of motion for a particle in an optical dipole trap.
def odt(S, t):
    
    x = S[0]
    y = S[1]
    z = S[2]
    vx = S[3]
    vy = S[4]
    vz = S[5]
    
    xx = np.cos(theta) * x + np.sin(theta) * y
    yy = -np.sin(theta) * x + np.cos(theta) * y
    zz = z
    
    vxx = np.cos(theta) * vx + np.sin(theta) * vy
    vyy = -np.sin(theta) * vx + np.cos(theta) * vy
    vzz = vz
    
    I0 = 2 * P / (np.pi * w0 ** 2)
    u0 = alpha * I0 / (2 * epsilon0 * c)
    zr = np.pi * w0 ** 2 / Lambda
    w = w0 * np.sqrt(1 + (xx / zr) ** 2)
    
    dx_dt = vxx
    dy_dt = vyy
    dz_dt = vzz
    dvx_dt = -(2 / m) * np.exp(-2 * (yy ** 2 + zz ** 2) * zr ** 2 / (w0 ** 2 * (xx ** 2 + zr ** 2))) * u0 * xx * zr ** 2 * (-2 * (yy ** 2 + zz ** 2) * zr ** 2 + w0 ** 2 * (xx ** 2 + zr ** 2)) / (w0 ** 2 * (xx ** 2 + zr ** 2) ** 3)
    dvy_dt = -(4 / m) * np.exp(-2 * (yy ** 2 + zz ** 2) * zr ** 2 / (w0 ** 2 * (xx ** 2 + zr ** 2))) * u0 * yy / (w0  + w0 * (xx / zr) ** 2) ** 2
    dvz_dt = -(4 / m) * np.exp(-2 * (yy ** 2 + zz ** 2) * zr ** 2 / (w0 ** 2 * (xx ** 2 + zr ** 2))) * u0 * zz / (w0  + w0 * (xx / zr) ** 2) ** 2

    return [dx_dt, dy_dt, dz_dt, dvx_dt, dvy_dt, dvz_dt]