import numpy as np

# Define units, with GeV as base unit
GeV = 10**6
eV = 10**-9*GeV
KeV = 10**-6*GeV
MeV = 10**-3*GeV
TeV = 10**3*GeV

Sec = (1/(6.582119*10**-16))/eV
Kmps = 3.3356*10**-6
Centimeter = 5.0677*10**13/GeV
Meter = 100*Centimeter
Km = 10**5*Centimeter
Kilogram = 5.6085*10**35*eV
Day = 86400*Sec
Year = 365*Day
KgDay = Kilogram*Day
amu = 1.66053892*10**-27*Kilogram
Mpc = 3.086*10**24*Centimeter
joule = Kilogram*Meter**2/Sec**2
erg = 1e-7*joule
Angstrom = 1e-10*Meter

# Particle and astrophysics parameters
M_s = 1.99*10**30*(Kilogram)

# Some conversions
kpc = 1.e-3*Mpc
pc = 1e-3*kpc
asctorad = np.pi/648000.
radtoasc = 648000./np.pi

# Constants
GN = 6.67e-11*Meter**3/Kilogram/Sec**2
h = 0.7
H_0 = 100*h*(Kmps/Mpc)
rho_c = 3*H_0**2/(8*np.pi*GN)
rho_m = 1.2201511280342523e-23
delta_c = 1.686
Rsun = 8.*kpc
R200_MW = 200.*kpc
M_MW = 1.1e12 * M_s

# To get velocity of Earth
vE0 = 29.79
omega = 2*np.pi/365.25
e = 0.016722
e1 = np.array([.9940, .1095, .003116])
e2 = np.array([-.05173, .4945, -.8677])
e1 /= np.linalg.norm(e1)
e2 /= np.linalg.norm(e2)
lambdap = np.deg2rad(281.93)
t1 = 79.3266