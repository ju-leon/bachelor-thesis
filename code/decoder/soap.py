import numpy as np
from scipy.special import gamma
from scipy.linalg import sqrtm, inv
from scipy.special import sph_harm


class SOAPDecoder():

    def cart2sph(self, x, y, z):
        r = np.sqrt(x**2 + y**2 + z**2)
        phi = np.arctan2(np.sqrt(x**2 + y**2), z)
        theta = np.arctan2(y, x)
        return phi, theta, r

    def get_basis_gto(self):
        """Used to calculate the alpha and beta prefactors for the gto-radial
        basis.

        Args:
            rcut(float): Radial cutoff.
            nmax(int): Number of gto radial bases.

        Returns:
            (np.ndarray, np.ndarray): The alpha and beta prefactors for all bases
            up to a fixed size of l=10.
        """

        rcut = self.rcut
        nmax = self.nmax

        # These are the values for where the different basis functions should decay
        # to: evenly space between 1 angstrom and rcut.
        a = np.linspace(1, rcut, nmax)
        threshold = 1e-3  # This is the fixed gaussian decay threshold

        alphas_full = np.zeros((10, nmax))
        betas_full = np.zeros((10, nmax, nmax))

        for l in range(0, 10):
            # The alphas are calculated so that the GTOs will decay to the set
            # threshold value at their respective cutoffs
            alphas = -np.log(threshold/np.power(a, l))/a**2

            # Calculate the overlap matrix
            m = np.zeros((alphas.shape[0], alphas.shape[0]))
            m[:, :] = alphas
            m = m + m.transpose()
            S = 0.5*gamma(l + 3.0/2.0)*m**(-l-3.0/2.0)

            # Get the beta factors that orthonormalize the set with Löwdin
            # orthonormalization
            betas = sqrtm(inv(S))

            # If the result is complex, the calculation is currently halted.
            if (betas.dtype == np.complex128):
                raise ValueError(
                    "Could not calculate normalization factors for the radial "
                    "basis in the domain of real numbers. Lowering the number of "
                    "radial basis functions (nmax) or increasing the radial "
                    "cutoff (rcut) is advised."
                )

            alphas_full[l, :] = alphas
            betas_full[l, :, :] = betas

        return alphas_full, betas_full

    def get_basis_poly(rcut, nmax):
        """Used to calculate discrete vectors for the polynomial basis functions.

        Args:
            rcut(float): Radial cutoff.
            nmax(int): Number of polynomial radial bases.

        Returns:
            (np.ndarray, np.ndarray): Tuple containing the evaluation points in
            radial direction as the first item, and the corresponding
            orthonormalized polynomial radial basis set as the second item.
        """
        # Calculate the overlap of the different polynomial functions in a
        # matrix S. These overlaps defined through the dot product over the
        # radial coordinate are analytically calculable: Integrate[(rc - r)^(a
        # + 2) (rc - r)^(b + 2) r^2, {r, 0, rc}]. Then the weights B that make
        # the basis orthonormal are given by B=S^{-1/2}
        S = np.zeros((nmax, nmax), dtype=np.float64)
        for i in range(1, nmax+1):
            for j in range(1, nmax+1):
                S[i-1, j-1] = (2*(rcut)**(7+i+j))/((5+i+j)*(6+i+j)*(7+i+j))

        # Get the beta factors that orthonormalize the set with Löwdin
        # orthonormalization
        betas = sqrtm(np.linalg.inv(S))

        # If the result is complex, the calculation is currently halted.
        if (betas.dtype == np.complex128):
            raise ValueError(
                "Could not calculate normalization factors for the radial "
                "basis in the domain of real numbers. Lowering the number of "
                "radial basis functions (nmax) or increasing the radial "
                "cutoff (rcut) is advised."
            )

        # The radial basis is integrated in a very specific nonlinearly spaced
        # grid given by rx
        x = np.zeros(100)
        x[0] = -0.999713726773441234
        x[1] = -0.998491950639595818
        x[2] = -0.996295134733125149
        x[3] = -0.99312493703744346
        x[4] = -0.98898439524299175
        x[5] = -0.98387754070605702
        x[6] = -0.97780935848691829
        x[7] = -0.97078577576370633
        x[8] = -0.962813654255815527
        x[9] = -0.95390078292549174
        x[10] = -0.94405587013625598
        x[11] = -0.933288535043079546
        x[12] = -0.921609298145333953
        x[13] = -0.90902957098252969
        x[14] = -0.895561644970726987
        x[15] = -0.881218679385018416
        x[16] = -0.86601468849716462
        x[17] = -0.849964527879591284
        x[18] = -0.833083879888400824
        x[19] = -0.815389238339176254
        x[20] = -0.79689789239031448
        x[21] = -0.77762790964949548
        x[22] = -0.757598118519707176
        x[23] = -0.736828089802020706
        x[24] = -0.715338117573056447
        x[25] = -0.69314919935580197
        x[26] = -0.670283015603141016
        x[27] = -0.64676190851412928
        x[28] = -0.622608860203707772
        x[29] = -0.59784747024717872
        x[30] = -0.57250193262138119
        x[31] = -0.546597012065094168
        x[32] = -0.520158019881763057
        x[33] = -0.493210789208190934
        x[34] = -0.465781649773358042
        x[35] = -0.437897402172031513
        x[36] = -0.409585291678301543
        x[37] = -0.380872981624629957
        x[38] = -0.351788526372421721
        x[39] = -0.322360343900529152
        x[40] = -0.292617188038471965
        x[41] = -0.26258812037150348
        x[42] = -0.23230248184497397
        x[43] = -0.201789864095735997
        x[44] = -0.171080080538603275
        x[45] = -0.140203137236113973
        x[46] = -0.109189203580061115
        x[47] = -0.0780685828134366367
        x[48] = -0.046871682421591632
        x[49] = -0.015628984421543083
        x[50] = 0.0156289844215430829
        x[51] = 0.046871682421591632
        x[52] = 0.078068582813436637
        x[53] = 0.109189203580061115
        x[54] = 0.140203137236113973
        x[55] = 0.171080080538603275
        x[56] = 0.201789864095735997
        x[57] = 0.23230248184497397
        x[58] = 0.262588120371503479
        x[59] = 0.292617188038471965
        x[60] = 0.322360343900529152
        x[61] = 0.351788526372421721
        x[62] = 0.380872981624629957
        x[63] = 0.409585291678301543
        x[64] = 0.437897402172031513
        x[65] = 0.465781649773358042
        x[66] = 0.49321078920819093
        x[67] = 0.520158019881763057
        x[68] = 0.546597012065094168
        x[69] = 0.572501932621381191
        x[70] = 0.59784747024717872
        x[71] = 0.622608860203707772
        x[72] = 0.64676190851412928
        x[73] = 0.670283015603141016
        x[74] = 0.693149199355801966
        x[75] = 0.715338117573056447
        x[76] = 0.736828089802020706
        x[77] = 0.75759811851970718
        x[78] = 0.77762790964949548
        x[79] = 0.79689789239031448
        x[80] = 0.81538923833917625
        x[81] = 0.833083879888400824
        x[82] = 0.849964527879591284
        x[83] = 0.866014688497164623
        x[84] = 0.881218679385018416
        x[85] = 0.89556164497072699
        x[86] = 0.90902957098252969
        x[87] = 0.921609298145333953
        x[88] = 0.933288535043079546
        x[89] = 0.94405587013625598
        x[90] = 0.953900782925491743
        x[91] = 0.96281365425581553
        x[92] = 0.970785775763706332
        x[93] = 0.977809358486918289
        x[94] = 0.983877540706057016
        x[95] = 0.98898439524299175
        x[96] = 0.99312493703744346
        x[97] = 0.99629513473312515
        x[98] = 0.998491950639595818
        x[99] = 0.99971372677344123

        rx = rcut*0.5*(x + 1)

        # Calculate the value of the orthonormalized polynomial basis at the rx
        # values
        fs = np.zeros([nmax, len(x)])
        for n in range(1, nmax+1):
            fs[n-1, :] = (rcut-np.clip(rx, 0, rcut))**(n+2)

        gss = np.dot(betas, fs)

        return rx, gss

    def __init__(self, rcut, nmax, lmax, rbf="gto", species="None", periodic=False):
        self.species = species
        self.periodic = periodic
        self.rcut = rcut
        self.nmax = nmax
        self.lmax = lmax
        self.rbf = rbf

        if rbf == "gto":
            self.alphas, self.betas = self.get_basis_gto()

    def get_c(self, features, center, Z, n, l, m):
        """
        Returns the specified c coefficent from the coefficent listas specified in the Dscribe library
        """
        NsTs100 = self.nmax*len(self.species)*100
        offset = l*l + m
        coeffs_for_species = NsTs100*center + \
            (self.lmax*self.lmax + 2*self.lmax) * self.nmax + self.nmax

        index = NsTs100*center + coeffs_for_species * Z + offset*self.nmax + n

        return features[index]

    def radial_basis(self, r, n, l):
        sum = 0
        for n2 in range(0, self.nmax):
            sum += self.betas[l][n][n2] * \
                (r**l) * np.exp(-1 * self.alphas[l][n] * r * r)

        return sum

    def spherical_harm(self, theta, phi, l, m):
        return sph_harm(m, l, theta, phi).real

    def density(self, features, Z, x, y, z):
        phi, theta, r = self.cart2sph(x, y, z)
        sum_real = 0

        for n in range(self.nmax):
            for l in range(self.lmax+1):
                for m in range(0, 2*l+1):
                    # TODO: Adapt for multiple centers(outer loop?)
                    c = self.get_c(features, 0, Z, n, l, m) * self.radial_basis(r,
                                                                                n, l) * self.spherical_harm(theta, phi, l, m)
                    # if np.isnan(c):
                    #    c = 0
                    c = np.nan_to_num(c)

                    sum_real += c

        return sum_real
