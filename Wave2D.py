import numpy as np
import sympy as sp
import scipy.sparse as sparse
import matplotlib.pyplot as plt
from matplotlib import cm

x, y, t = sp.symbols('x,y,t')


class Wave2D:
    
    def create_mesh(self, N, sparse=False):
        """Create 2D mesh and store in self.xij and self.yij"""
        self.N = N
        self.h = 1.0 / N
        self.x = np.linspace(0, 1, N + 1)
        self.y = np.linspace(0, 1, N + 1)
        self.xij, self.yij = np.meshgrid(self.x, self.y, indexing='ij')
    
    def D2(self, N):
        """Return second order differentiation matrix"""
        D2 = sparse.diags([1, -2, 1], [-1, 0, 1], (N + 1, N + 1), 'lil')
        D2[0, :4] = [2, -5, 4, -1]
        D2[-1, -4:] = [-1, 4, -5, 2]
        D2 /= self.h**2
        return D2
    
    @property
    def w(self):
        """Return the dispersion coefficient"""
        return self.c * sp.pi * sp.sqrt(self.mx**2 + self.my**2)
    
    def ue(self, mx, my):
        """Return the exact standing wave"""
        return sp.sin(mx*sp.pi*x)*sp.sin(my*sp.pi*y)*sp.cos(self.w*t)
    
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
        self.mx, self.my = mx, my
        ue_func = sp.lambdify((x, y, t), self.ue(mx, my), "numpy")
        U0 = ue_func(self.xij, self.yij, 0)
        Um1 = ue_func(self.xij, self.yij, -self.dt)
        return U0, Um1
    
    @property
    def dt(self):
        """Return the time step"""
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
        ue_func = sp.lambdify((x, y, t), self.ue(self.mx, self.my), "numpy")
        u_exact = ue_func(self.xij, self.yij, t0)
        error = u - u_exact
        return np.sqrt(self.h**2 * np.sum(error**2))
    
    def apply_bcs(self):
        """Apply boundary conditions"""
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
        U0, Um1 = self.initialize(N, mx, my)
        
        D2 = self.D2(N)
        I = sparse.identity(N + 1)
        L = sparse.kron(D2, I) + sparse.kron(I, D2)
        C2 = (c * self.dt)**2
        
        u_n = U0.copy().ravel()
        u_nm1 = Um1.copy().ravel()
        
        if store_data > 0:
            data = {0: U0.copy()}
        else:
            errors = []
        
        for n in range(Nt):
            u_np1 = 2*u_n - u_nm1 + C2 * L.dot(u_n)
            self.Unp1 = u_np1.reshape((N + 1, N + 1))
            self.apply_bcs()
            
            u_nm1 = u_n
            u_n = self.Unp1.ravel()
            
            if store_data > 0:
                if (n + 1) % store_data == 0:
                    data[n + 1] = self.Unp1.copy()
            else:
                errors.append(self.l2_error(self.Unp1, (n + 1)*self.dt))
        
        if store_data > 0:
            return data
        else:
            return self.h, np.array(errors)
    
    def convergence_rates(self, m=5, cfl=0.1, Nt=10, mx=3, my=3):
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
        """Return second order differentiation matrix"""
        D2 = sparse.diags([1, -2, 1], [-1, 0, 1], (N + 1, N + 1), 'lil')

        D2[0, 0:3] = [-2, 2, 0]
        D2[-1, -3:] = [0, 2, -2]
        D2 /= self.h**2
        return D2
    
    def ue(self, mx, my):
        """Return the exact standing wave"""
        return sp.cos(mx*sp.pi*x)*sp.cos(my*sp.pi*y)*sp.cos(self.w*t)
    
    def apply_bcs(self):
        """Apply boundary conditions"""
        pass  # Neumann BCs are handled by D2 matrix


def test_convergence_wave2d():
    sol = Wave2D()
    r, E, h = sol.convergence_rates(mx=2, my=3)
    assert abs(r[-1]-2) < 1e-2

def test_convergence_wave2d_neumann():
    solN = Wave2D_Neumann()
    r, E, h = solN.convergence_rates(mx=2, my=3)
    assert abs(r[-1]-2) < 0.05

def test_exact_wave2d():
    mx = my = 3
    cfl = 1/np.sqrt(2)
    N = 20
    Nt = 50
    
    # Dirichlet problem
    sol = Wave2D()
    h, errors = sol(N=N, Nt=Nt, cfl=cfl, c=1.0, mx=mx, my=my, store_data=-1)
    max_error_dirichlet = np.max(errors)
    assert max_error_dirichlet < 1e-12

    # Neumann problem
    solN = Wave2D_Neumann()
    h, errors = solN(N=N, Nt=Nt, cfl=cfl, c=1.0, mx=mx, my=my, store_data=-1)
    max_error_neumann = np.max(errors)
    assert max_error_neumann < 1e-12