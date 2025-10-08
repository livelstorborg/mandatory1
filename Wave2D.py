import numpy as np
import sympy as sp
import scipy.sparse as sparse

x, y, t = sp.symbols('x,y,t')

class Wave2D:
    def create_mesh(self, N, sparse=False):
        """Create 2D mesh and store in self.xij and self.yij"""
        self.N = N
        self.h = 1.0 / N  
        x = np.linspace(0, 1, N + 1)
        y = np.linspace(0, 1, N + 1)
        self.xij, self.yij = np.meshgrid(x, y, indexing='ij', sparse=sparse)

    def D2(self, N):
        """Second derivative matrix with Dirichlet BC"""
        D = sparse.diags([1, -2, 1], [-1, 0, 1], (N+1, N+1), 'lil')
        D[0, :4] = 2, -5, 4, -1
        D[-1, -4:] = -1, 4, -5, 2
        return D

    @property
    def w(self):
        """Return the dispersion coefficient"""
        k_x = self.mx * sp.pi
        k_y = self.my * sp.pi
        return self.c * sp.sqrt(k_x**2 + k_y**2)

    def ue(self, mx, my):
        """Return exact standing wave"""
        return sp.sin(mx * sp.pi * x) * sp.sin(my * sp.pi * y) * sp.cos(self.w * t)

    def initialize(self, N, mx, my):
        r"""Initialize the solution at $U^{n}$ and $U^{n-1}$

        Parameters
        ----------
        N : int
            The number of uniform intervals in each direction
        mx, my : int
            Parameters for the standing wave
        """
        self.create_mesh(N)
        self.mx = mx
        self.my = my

        D = self.D2(N) / self.h**2

        Unm1 = sp.lambdify((x, y, t), self.ue(mx, my))(self.xij, self.yij, 0)
        Un = Unm1 + 0.5 * (self.c * self.dt)**2 * (D @ Unm1 + Unm1 @ D.T)

        return Unm1, Un

    @property
    def dt(self):
        """Return time step"""
        return self.cfl * self.h / self.c

    def l2_error(self, u, t0):
        """Return l2-error norm

        Parameters
        ----------
        u : array
            The solution mesh function
        t0 : number
            The time of the comparison
        """
        ue = sp.lambdify((x, y, t), self.ue(self.mx, self.my))(self.xij, self.yij, t0)
        error = u - ue
        l2 = np.sqrt(self.h**2 * np.sum(error**2))
        return l2

    def apply_bcs(self):
        """Dirichlet BCs"""
        self.Unp1[0, :] = 0
        self.Unp1[-1, :] = 0
        self.Unp1[:, 0] = 0
        self.Unp1[:, -1] = 0

    def __call__(self, N, Nt, cfl=0.5, c=1.0, mx=3, my=3, store_data=-1):
        """Solve the wave equation

        Parameters
        ----------
        N : int
            The number of uniform intervals in each direction
        Nt : int
            Number of time steps
        cfl : number
            The CFL number
        c : number
            The wave speed
        mx, my : int
            Parameters for the standing wave
        store_data : int
            Store the solution every store_data time step
            Note that if store_data is -1 then you should return the l2-error
            instead of data for plotting. This is used in `convergence_rates`.

        Returns
        -------
        If store_data > 0, then return a dictionary with key, value = timestep, solution
        If store_data == -1, then return the two-tuple (h, l2-error)
        """
        self.c = c
        self.cfl = cfl
        self.mx = mx
        self.my = my

        # Initialize solution (this sets self.h)
        Unm1, Un = self.initialize(N, mx, my)
        Unp1 = np.zeros((N + 1, N + 1))
        D = self.D2(N) / self.h**2

        # Use a local dt variable to keep total time fixed
        dt = self.dt        # initial dt from CFL and h
        T = Nt * dt         # total simulation time
        dt = T / Nt         # scaled dt (same total time for all grids)

        if store_data > 0:
            plotdata = {0: Unm1.copy()}
        else:
            errors = [self.l2_error(Unm1, 0)]

        for n in range(1, Nt):
            Unp1[:] = 2*Un - Unm1 + (c * dt)**2 * (D @ Un + Un @ D.T)
            self.Unp1 = Unp1
            self.apply_bcs()

            # Swap solutions
            Unm1[:], Un[:] = Un, Unp1

            if store_data > 0 and n % store_data == 0:
                plotdata[n] = Un.copy()
            elif store_data <= 0:
                errors.append(self.l2_error(Un, n*dt))

        if store_data > 0:
            return plotdata
        else:
            return self.h, np.array(errors)

    def convergence_rates(self, m=4, cfl=0.1, Nt=10, mx=3, my=3):
        """Compute convergence rates for a range of discretizations

        Parameters
        ----------
        m : int
            The number of discretizations to use
        cfl : number
            The CFL number
        Nt : int
            The number of time steps to take
        mx, my : int
            Parameters for the standing wave

        Returns
        -------
        3-tuple of arrays. The arrays represent:
            0: the orders
            1: the l2-errors
            2: the mesh sizes
        """
        E = []
        h = []
        N0 = 8
        for m in range(m):
            dx, err = self(N0, Nt, cfl=cfl, mx=mx, my=my, store_data=-1)
            E.append(err[-1])
            h.append(dx)
            N0 *= 2
            Nt *= 2
        r = [np.log(E[i-1]/E[i])/np.log(h[i-1]/h[i]) for i in range(1, m+1, 1)]
        return r, np.array(E), np.array(h)


class Wave2D_Neumann(Wave2D):
    def D2(self, N):
        """Second derivative with Neumann BCs"""
        D = sparse.diags([1, -2, 1], [-1, 0, 1], (N + 1, N + 1), format='lil')
        D[0, 0:3] = [-2, 2, 0]
        D[-1, -3:] = [0, 2, -2]
        return D

    def ue(self, mx, my):
        """Exact solution for Neumann case"""
        kx = mx * sp.pi
        ky = my * sp.pi
        return sp.cos(kx * x) * sp.cos(ky * y) * sp.cos(self.w * t)

    def apply_bcs(self):
        """Neumann BCs"""
        self.Unp1[0, :] = self.Unp1[1, :]
        self.Unp1[-1, :] = self.Unp1[-2, :]
        self.Unp1[:, 0] = self.Unp1[:, 1]
        self.Unp1[:, -1] = self.Unp1[:, -2]




def test_convergence_wave2d():
    sol = Wave2D()
    r, E, h = sol.convergence_rates(mx=2, my=3)
    assert abs(r[-1]-2) < 1e-2


def test_convergence_wave2d_neumann():
    solN = Wave2D_Neumann()
    r, E, h = solN.convergence_rates(mx=2, my=3)
    assert abs(r[-1] - 2) < 0.05


def test_exact_wave2d():
    N = 40
    mx = my = 1
    Nt = 100
    C = 1 / np.sqrt(2)
    c = 1.0

    for wave_class in [Wave2D, Wave2D_Neumann]:
        wave = wave_class()
        h, errors = wave(N, Nt, cfl=C, c=c, mx=mx, my=my, store_data=-1)
        assert errors[-1] < 1e-12, f"{wave_class.__name__} L2 error too large: {errors[-1]}"







