import numpy as np

# Constants
from units import GN
# Conversions
from units import asctorad, Year, radtoasc

from profiles import Profiles

class AstrometricSim:
    def __init__(self, theta_x_lims=[-1.6, 1.6], theta_y_lims=[-0.9, 0.9]):
        """
        Class to create animations of astrometric weak lensing. For demo and not science!
        :param theta_x_lims: x-axis coordinate limits [x_min, x_max] in arcsecs
        :param theta_y_lims: y-axis coordinate limits [y_min, y_max] in arcsecs
        """

        self.theta_x_lims = theta_x_lims
        self.theta_y_lims = theta_y_lims

        # Area of sky region
        self.roi_area = (self.theta_x_lims[1] - self.theta_x_lims[0]) * \
                        (self.theta_y_lims[1] - self.theta_y_lims[0])

    def animation(self, pos_l, M_l, R_l, v_l, D_l,
                  n_sources_fix=True,
                  # Source properties
                  std_mu_s=1e-6,
                  n_dens=20, source_pos="random"
                  ):
        """
        :param pos_l: tuple of lens positions, format [[x_1, y_1], [x_2, y_2]...]
        :param M_l: tuple of lens masses
        :param R_l: tuple of lens sizes (Gaussian lens)
        :param v_l: tuple of lens velocities
        :param D_l: tuple of distances of lenses
        :param n_dens: density of sources (per arcsecs^2); default 20
        :param source_pos: must be one of ["uniform", "random"]; default "random"
        """

        # Get total number of sources in sky region

        if n_sources_fix:
            self.n_total = np.floor(n_dens * self.roi_area).astype(np.int32)
        else:
            self.n_total = np.random.poisson(n_dens * self.roi_area)

        # Source velocities
        self.mu_s_intrinsic = np.random.normal(
            loc=0, scale=std_mu_s, size=(self.n_total, 2)
        )

        # Set source positions

        # Random positions + custom if specified
        if source_pos == "random":

            # Initial source property array
            self.sources = np.zeros(
                self.n_total,
                dtype=[("theta_x", float, 1),
                       ("theta_y", float, 1),
                       ("theta_x_0", float, 1),
                       ("theta_y_0", float, 1),
                       ("mu", float, 1)]
            )

            self.sources["theta_x_0"] = np.array(
                list(np.random.uniform(*self.theta_x_lims, self.n_total)))
            self.sources["theta_y_0"] = np.array(
                list(np.random.uniform(*self.theta_x_lims, self.n_total)))

        # Uniform grid of sources
        elif source_pos == "uniform":
            xy_ratio = (self.theta_y_lims[1] - self.theta_y_lims[0]) / \
                (self.theta_x_lims[1] - self.theta_x_lims[0])
            x_pos = np.linspace(self.theta_x_lims[0], self.theta_x_lims[1], np.round(
                np.sqrt(self.n_total / xy_ratio)))
            y_pos = np.linspace(self.theta_y_lims[0], self.theta_y_lims[1], np.round(
                np.sqrt(self.n_total * xy_ratio)))

            self.n_total = len(np.meshgrid(x_pos, y_pos)[0].flatten())

            # Initialize source property array
            self.sources = np.zeros(self.n_total, dtype=[("theta_x", float, 1),
                                                         ("theta_y", float, 1),
                                                         ("theta_x_0", float, 1),
                                                         ("theta_y_0", float, 1)])

            self.sources["theta_x_0"] = np.meshgrid(x_pos, y_pos)[0].flatten()
            self.sources["theta_y_0"] = np.meshgrid(x_pos, y_pos)[1].flatten()

        assert len(pos_l) == len(v_l) == len(M_l) == len(R_l) == len(D_l), \
            "Lens property arrays must be the same size!"

        # Infer number of lenses
        self.n_lens = len(pos_l)

        # Initialize lens property array
        self.lenses = np.zeros(self.n_lens, dtype=[("theta_x", float, 1),
                                                   ("theta_y", float, 1),
                                                   ("M_0", float, 1),
                                                   ("R_0", float, 1),
                                                   ("D", float, 1),
                                                   ("v_x", float, 1),
                                                   ("v_y", float, 1)])

        # Set initial source positions
        self.sources["theta_x"] = self.sources["theta_x_0"]
        self.sources["theta_y"] = self.sources["theta_y_0"]

        # Set initial lens positions...
        self.lenses["theta_x"] = np.array(pos_l)[:, 0]
        self.lenses["theta_y"] = np.array(pos_l)[:, 1]

        # ... and lens properties
        self.lenses["v_x"] = np.array(v_l)[:, 0]
        self.lenses["v_y"] = np.array(v_l)[:, 1]
        self.lenses["M_0"] = np.array(M_l)
        self.lenses["R_0"] = np.array(R_l)
        self.lenses["D"] = np.array(D_l)

        theta_s = np.zeros((self.n_total, 2))

        # Deflection and proper motion vectors
        for i_lens in range(self.n_lens):
            b_ary = np.transpose(
                [self.sources["theta_x"] - self.lenses["theta_x"][i_lens],
                 self.sources["theta_y"] - self.lenses["theta_y"][i_lens]]
            ) * asctorad

            vel_l = np.array([self.lenses["v_x"][i_lens],
                              self.lenses["v_y"][i_lens]])

            for i_source in range(self.n_total):

                theta_s[i_source] += self.theta(
                    b_ary[i_source], self.lenses["R_0"][i_lens],
                    self.lenses["M_0"][i_lens], self.lenses["D"][i_lens])

        # New source positions including deflection
        self.sources["theta_x"] = self.sources["theta_x_0"] + theta_s[:, 0]
        self.sources["theta_y"] = self.sources["theta_y_0"] + theta_s[:, 1]

    def propagate(self, dt):
        """ Update lens and star positions"""

        theta_s = np.zeros((self.n_total, 2))

        for i_lens in range(self.n_lens):

            b_ary = np.transpose(
                [self.sources["theta_x"] - self.lenses["theta_x"][i_lens],
                 self.sources["theta_y"] - self.lenses["theta_y"][i_lens]]
            ) * asctorad

            vel_l = np.array([self.lenses["v_x"][i_lens],
                              self.lenses["v_y"][i_lens]])

            for i_source in range(self.n_total):

                theta_s[i_source] += self.theta(
                    b_ary[i_source], self.lenses["R_0"][i_lens],
                    self.lenses["M_0"][i_lens], self.lenses["D"][i_lens])

            mu_l = (vel_l / self.lenses["D"][i_lens]) / (Year ** -1) * radtoasc

            self.lenses["theta_x"][i_lens] = self.lenses["theta_x"][i_lens] + mu_l[0] * dt
            self.lenses["theta_y"][i_lens] = self.lenses["theta_y"][i_lens] + mu_l[1] * dt

        # Update intrinsic source position
        self.sources["theta_x_0"] += self.mu_s_intrinsic[:, 0] * dt
        self.sources["theta_y_0"] += self.mu_s_intrinsic[:, 1] * dt

        self.sources["theta_x"] = self.sources["theta_x_0"] + theta_s[:, 0]
        self.sources["theta_y"] = self.sources["theta_y_0"] + theta_s[:, 1]

    @classmethod
    def mu(self, beta_vec, v_ang_vec, R_0, M_0, d_lens):
        """ Get lens-induced proper motion vector
        """

        # Convert angular to physical impact parameter
        b_vec = d_lens * np.array(beta_vec)
        # Convert angular to physical velocity
        v_vec = d_lens * np.array(v_ang_vec)
        b = np.linalg.norm(b_vec)  # Impact parameter
        M, dMdb, _ = Profiles.MdMdb_Gauss(b, R_0, M_0)
        b_unit_vec = b_vec / b  # Convert angular to physical impact parameter
        b_dot_v = np.dot(b_unit_vec, v_vec)
        factor = (dMdb / b * b_unit_vec * b_dot_v
                  + M / b ** 2 * (v_vec - 2 * b_unit_vec * b_dot_v))

        return -factor * 4 * GN / (asctorad / Year)  # Convert to as/yr

    @classmethod
    def theta(self, beta_vec, R_0, M_0, d_lens):
        """ Get lens-induced deflection vector
        """

        # Convert angular to physical impact parameter
        b_vec = d_lens * np.array(beta_vec)
        b = np.linalg.norm(b_vec)  # Impact parameter
        M, _, _ = Profiles.MdMdb_Gauss(b, R_0, M_0)
        b_unit_vec = b_vec / b  # Convert angular to physical impact parameter

        return 4 * GN * M / b * b_unit_vec * radtoasc  # Convert to as
