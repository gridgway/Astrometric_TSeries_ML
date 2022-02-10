""" Tools that are useful for lensing calculations

We choose km, s and M_sol as our system of units. 


"""

import numpy as np

# Convenient variables
pi   = np.pi
norm = np.linalg.norm
mult = np.matmul
dot  = np.dot

# Units
from astropy import units as u

# Constants
from astropy import constants as const

# Random number generation
import scipy.stats as stat

#####################################
# Conversion Factors                #
#####################################

AU = u.AU.to(u.km)
"""Astronomical Unit in km"""

mu_as = 2*pi/360/60**2 / 1e6
"""Numerical value of micro arc-seconds in radians"""

#####################################
# Constants                         #
#####################################

c = const.c.to(u.km/u.s)
"""Speed of light in km s\ :sup:`-1`\ ."""

G_N = const.G.to(u.km**3 / u.M_sun / u.s**2)
"""Newton's Gravitational Constant in km\ :sup:`3` M_\odot\ :sup:`-1` s\ :sup:`-2`\ ."""

v_esc = 550 * u.km/u.s
"""Milky Way escape velocity in km s\ :sup:`-1`\ ."""

R_mw = (10*u.kpc).to(u.km)
"""Size of the baryonic component of the Milky Way in km."""


#####################################
# Functions                         #
#####################################

def random_perp(vec):
    """ Pick a random direction normal to vec """

    n = vec/norm(vec)

    # arbitrarily pick a direction normal to n
    n_perp1 = np.cross(n, [0,0,1])

    # if n is [0,0,1], incidentally
    if norm(n_perp1) == 0:
        n_perp1 = np.cross(n, [0,1,0])

    # complete the orthonormal basis
    n_perp2 = np.cross(n, n_perp1)

    # Random angle in the n_perp1, n_perp2 plane
    phi = np.random.uniform(0, 2*pi)

    return n_perp1 * np.cos(phi) + n_perp2 * np.sin(phi)



def sample(method='normal', N=1, scale=1.0, 
    inv_cdf=None, # if method == 'custom_sphere'
    observer=None # if method == 'relative'
    ):
    """ Multi-purpose sampling function

    methods
    -------
    'normal' 
        sample 3D normal distribution (mu=0, sigma=scale) N times
    'sphere_surface' 
        uniformly sample the surface of a unit sphere (pick a random direction)
    'uniform_sphere'
        uniformly sample the interior of a unit sphere
    'custom_sphere'
        pick a random direction, then pick a random radius according to custom cdf
    'relative'
        sample relative coordinates of lens and source wrt observer in three steps:
        (i)   pick a random direction, randomly place lens in that direction (uniform with scale[0])
        (ii)  randomly place source past the lens in that direction (uniformly with scale[1])
        (iii) randomly offset source from observer-lens line, uniformly sampling impact parameter with scale[2]
        randomly sample lens and source velocities with gaussian, sigma given by v_esc of Milky Way


    Parameters
    ----------
    N : int
        number of points to be sampled
    scale : float
        scale of the coordinates
    inv_cdf : function
        if method == 'custom_sphere', sample the radius using the inverse cdf, inv_cdf
    observer : Observer object
        used if method == 'relative'

    Returns
    -------
    ndarray
    """

    methods = ['normal', 'sphere_surface', 'uniform_sphere', 'custom_sphere', 'relative']

    if method == 'normal':
        """ N by 3 Normal distributed points with \mu=zero, \sigma=scale  """

        if type(scale) == u.quantity.Quantity:

            return np.squeeze(
                np.random.normal(scale=scale.value, size=(N,3))
            ) * scale.unit

        else:

            return np.squeeze(np.random.normal(scale=scale, size=(N,3)))

    elif 'sphere' in method:
        """ N by 3 points distributed spherically symmetrically  
        
            if method=='sphere_surface': 
                randomly pick points on the surface of unit sphere

            if method=='uniform_sphere': 
                unifomly pick points interior to unit sphere

            if method=='custom_sphere': 
                spherically symmetric distribution, but radius has custom cdf
        """

        mu  = np.random.uniform(-1, 1, N)
        phi = np.random.uniform(0, 2*pi, N)

        x = np.cos(phi) * np.sqrt(1-mu**2)
        y = np.sin(phi) * np.sqrt(1-mu**2)
        z = mu

        if method != 'sphere_surface':

            if method == 'uniform_sphere':
                r = np.random.uniform(size=N)**(1/3.) * scale

            elif method == 'custom_sphere':
                r = inv_cdf(np.random.uniform(size=N)) * scale

            x,y,z = r*x, r*y, r*z

        if N==1:
            return np.transpose([x[0],y[0],z[0]])

        else:
            return np.transpose([x,y,z]) * x.unit

    elif method == 'relative':
        """ 
            Lens sampled spherically symmetrically wrt observer with scale scale[0]
            Source sampled past the lens, 
                along the lens-observer axis with scale[1], and impact parameter with scale scale[2]
            Both sets of velocities normally distributed with \sigma=v_esc
        """

        # Setting the distance scale for D_l, D_s, b 
        # (Distance to lens, to source, and impact parameter)
        if type(scale) not in [list, np.ndarray]:
            scale = np.array([1., 1., 1.]) * scale

        # Pick a random direction
        n = sample('sphere_surface')

        # Place the lens uniformly along that direction, starting from the observer
        x_l = observer.x + n * np.random.uniform() * scale[0]
        
        # Place the source beyond the lens, uniformly along that direction
        x_s = x_l + n * np.random.uniform() * scale[1]

        # Give it a random impact parameter
        x_s += random_perp(n) * np.random.uniform() * scale[2]

        return (
            Lens(x=x_l, v=sample('normal') * v_esc), 
            Source(x=x_s, v=sample('normal') * v_esc)
        )

    else:

        raise TypeError('choose method in', methods)


#####################################
# Parent Classes                    #
#####################################

# Following ``Introduction to Gravitational Lensing: With Python Examples'' by Massimo Meneghetti

### Single points ###

class Point(object):
    """ Point class

        Serves as the parent class of Lens, Source, and Observer
    """

    def __init__(self, x=None, v=None):
        """ Initialize a point.

            !!! Should I buy into SkyCoord? 
            !!! Should I add in Flux for a Source?

            Parameters
            ----------
            x : 1D ndarray
                3D position of object
            v : 1D ndarray
                3D velocity of object
        """

        self.x = x
        self.v = v

    def copy(self):
        return Point(x=self.x, v=self.v)

    def time_evolve(self, dt):
        """ evolve the position of the point

            !!! If we know the galactic potential, we should also evolve v.

            Parameters
            ----------
            dt : float
                time increment
        """

        self.x += self.v*dt


class Points(object):
    """ Collection of points
    """

    def __init__(self, x=None, v=None):
        """ Initialize a collection of points

            Parameters
            ----------
            x : 2D ndarray
                3D position of lens
            v : 2D ndarray
                3D velocity of lens
        """

        if (x is None):

            if v is None:
                self.N = None
            else:
                self.N = v.shape[0]

        else:

            self.N = x.shape[0]


        self.x = x
        self.v = v


    #!!! If I were smarter, I could make a Lenses make a Lens, while Sources makes a Source
    def __setitem__(self, ind, point):
          self.x[ind] = point.x
          self.v[ind] = point.v

    def __getitem__(self, ind):
        return Point(self.x[ind], self.v[ind])


    def time_evolve(self, dt):
        """ evolve the position of the lenses

            !!! If we know the galactic potential, we should also evolve v. 
            !!! dv/dt = - d \Phi / dr

            Parameters
            ----------
            dt : float
                time increment
        """

        if type(dt) != u.quantity.Quantity:
            dt = dt * u.s

        if norm(self.x) == 0:
            self.x = self.v*dt

        else:
            self.x += self.v*dt


#####################################
# Derived Classes                   #
#####################################


class Lens(Point):
    """ Point lens class
    """

    def __init__(self, x=None, v=None, M=None):
        """ Initialize a point Lens.

            Parameters
            ----------
            M : float
                Mass of lens
        """

        self.M = M
        super().__init__(x,v)

    def copy(self):
        return Lens(x=self.x, v=self.v, M=self.M)

class Lenses(Points):
    """ Collection of point lenses
    """
    #pass

    def __init__(self, x=None, v=None, M=None):
        self.M = M
        super().__init__(x,v)

    def __setitem__(self, ind, lens):
          self.x[ind] = lens.x
          self.v[ind] = lens.v
          self.M      = lens.M

    def __getitem__(self, ind):
        return Lens(self.x[ind], self.v[ind], self.M)


class Source(Point):
    """ Point Source class
    """

    def __init__(self, x=None, v=None, F=None):
        """ Initialize a point Lens.

            Parameters
            ----------
            F : float
                Flux of source
        """

        self.F = F
        super().__init__(x,v)

    def copy(self):
        return Source(x=self.x, v=self.v, F=self.F)

class Sources(Points):
    """ Collection of point sources
    """

    def __init__(self, x=None, v=None, F=None):
        self.F = F
        super().__init__(x,v)
    
    def __setitem__(self, ind, source):
          self.x[ind] = source.x
          self.v[ind] = source.v
          self.F      = source.F

    def __getitem__(self, ind):
        return Source(self.x[ind], self.v[ind], self.F)


class Observer(Point):
    """ Observer Class
    """

    def __init__(self, x=None, v=None):
        """ Initialize an observer

            Default to the origin (position and velocity {0,0,0})
        """

        if x is None:
            x = np.array([0,0,0]) * u.km
        if v is None:
            v = np.array([0,0,0]) * u.km/u.s
        
        super().__init__(x,v)

    def copy(self):
        return Observer(x=self.x, v=self.v)

    def observe(self, source, lens=None, method='fully_approximate', 
        N=1, dt=None, cross_check=False):
        """ observe a source, possibly deflected by a lens

            z/r = cos\theta
            x/r = cos\phi sin\theta
            y/r = sin\phi sin\theta

            \theta = arccos z/r
            \phi = arctan y/x
        """

        if N==1:

            # Ultimately, observed angular position on the sky
            theta,phi = 0,0

            ## Observe the angular position of a source without deflection
            if lens is None:
                x,y,z = source.x
                r = norm(source.x)
                phi = np.arctan(y/x)
                if x<0: 
                    phi += pi * u.rad
                theta = np.arccos(z/r)

                return np.array([theta.value, phi.value]) * u.rad

            ## Otherwise, include the deflection due to the lens

            # Distance from observer to lens
            Dl = norm(lens.x-self.x)

            # direction of line passing through the lens and observer
            zhat = (lens.x-self.x)/Dl

            # Distance from observer to source along z axis
            Ds = dot(zhat, source.x)

            # Perpendicular position, angle, and direction of source relative to z axis
            eta = source.x - Ds * zhat
            beta = np.arctan(norm(eta)/Ds)
            theta_hat = eta/norm(eta)

            if cross_check:
                print( (eta**2 + Ds**2)/norm(source.x)**2 )

            # Distance between source and lens along z axis
            Dls = Ds - Dl

            # Einstein radius
            thetaE = (np.sqrt(4 * G_N * lens.M/c**2 * Dls/Ds/Dl)).decompose() * u.rad

            if method == 'fully_approximate': 
                """ Formula from 1804.01991 (Ken+)
                    - all angles assumed small
                    - leading order in 4GM/bc^2
                    - convenient impact parameter (|b| = |eta| = |xi|)
                """

                # Assume the impact parameter of the source is the 
                # distance that appears in the lensing equation
                b = eta

                dTheta = (
                    -(1-Dl/Ds) * 4*G_N*lens.M/c**2/norm(b)
                ).decompose()  / mu_as * 1e-3 * u.mas
                # dTheta = -thetaE**2 * Dl/Ds / beta  / mu_as * 1e-3 * u.mas
                # dx = - Dl/Ds / y  / mu_as * 1e-3 * u.mas

                # Where the source appears
                image = Source(x = source.x + Ds * np.sin(dTheta) * theta_hat)

                # Angles of the image
                return self.observe(image)

            if method == 'quadratic': 
                """ Meneghetti
                    - all angles assumed small
                    - leading order in 4GM/bc^2
                    - less convenient impact parameter (|eta| != |xi|)
                    - but xi is assumed to be approximately the distance of closest approach
                """

                # Everything in units of Einstein radius
                yhalf = (beta/thetaE/2).value

                # print(yhalf)

                # The two images: x-plus and x-minus
                # x = theta/thetaE
                xp, xm = yhalf + np.sqrt(1 + yhalf**2) * np.array([1,-1])

                # print(xp * thetaE)

                image_p, image_m = [
                    Source(x = source.x + Ds * np.sin(x * thetaE) * theta_hat) 
                for x in [xp,xm] ]

                return [
                    self.observe(image_p), 
                    self.observe(image_m)
                ][1]

        elif method == 'Virbhadra_Ellis':
            """Formula from https://arxiv.org/pdf/astro-ph/9904193v2.pdf"""

            pass


        # Multiple observations
        else:

            angle_unit = u.mas

            # Initial observation
            theta_phi0 = self.observe(source, lens, method)
            theta_phi0TMP = self.observe(source)

            # print(theta_phi0, theta_phi0TMP)

            # Make N-1 more observations. Observe
            data = []
            dataTMP = []
            for i in np.arange(N):
    
                # Deviations from initial position
                data.append((self.observe(source, lens, method) - theta_phi0).to(angle_unit))

                dataTMP.append((self.observe(source) - theta_phi0TMP).to(angle_unit))

                self.time_evolve(dt)
                lens.time_evolve(dt)
                source.time_evolve(dt)

            # Bring the observer back to its original position
            self.time_evolve(-dt * N)

            return np.array(data) * angle_unit, np.array(dataTMP) * angle_unit


    # def dTheta(self, lens, source, method='fully_approximate', test = False):
    #     """ Compute deflection angle seen by observer of source due to lens
        
    #         Parameters
    #         ----------

    #         lens : Lens
    #             Lens that is deflecting light
    #         source : Source
    #             Source of light
        
    #     """
        # pass

