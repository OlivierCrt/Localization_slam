import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

def data_simulation(plot_p=0):
    """
    Simulation of a random experiment: robot motion, measurement collection, etc.
    
    Parameters:
    ----------
    plot_p : int
        If 1, plots an animation, otherwise does nothing
        
    Returns:
    -------
    N : int
        Number of time samples
    T : ndarray
        Vector of time samples (1xN)
    M : int
        Number of landmarks
    Z : ndarray
        2MxN (2D) array of the outcomes of the measurement random process
        (NaN entries correspond to unperceived landmarks; time is along the 2nd dimension)
    arrayH : ndarray
        2Mx(2+2M)xN (3D) array of observation matrices
        (NaN entries correspond to unperceived landmarks; time is along the 3rd dimension)
    arrayR : ndarray
        2Mx2MxN (3D) array of measurement noise covariance matrices
        (NaN entries correspond to unperceived landmarks; time is along the 3rd dimension)
    in_fov : ndarray
        2MxN (2D) array of landmark visibility indexes
        (NaN indexes correspond to unperceived landmarks; time is along the 2nd dimension)
    F : ndarray
        Matrix involved in the prior dynamics of the full state vector
    B : ndarray
        Vector involved in the prior dynamics of the full state vector
    CC : ndarray
        Vector involved in the prior dynamics of the full state vector
    Hfull : ndarray
        2Mx(2+2M) observation matrix if all landmarks were in the sensor fov
    mX0 : ndarray
        (2+2M)x1 expectation vector of the initial state vector (at time 0)
    PX0 : ndarray
        (2+2M)x(2+2M) covariance matrix of the initial state vector (at time 0)
    Qw : ndarray
        (2+2M)x(2+2M) covariance matrix of the (stationary) dynamics noise
    Rv : ndarray
        2Mx2M covariance matrix of the (stationary) measurement noise if all landmarks were in the sensor fov
    X : ndarray
        (2+2M)xN (2D) array of the outcomes of the hidden state random process
        (time is along the 2nd dimension)
    """
    
    N = 50
    deltaT = 1
    T = np.arange(0, N) * deltaT
    w = np.pi/4
    
    M = 5  # Number of landmarks
    mX0robot = np.array([[0], [0]])
    mX0landmarks = np.array([[-2], [4], [-4], [2], [-4], [-2], [-2], [-4], [4], [-2]])
    PX0robot = np.zeros((2, 2))
    PX0landmarks = 0.2 * np.eye(2*M)
    PX0 = np.block([[PX0robot, np.zeros((2, 2*M))], 
                    [np.zeros((2*M, 2)), PX0landmarks]])
    
    cholPX0 = np.block([[np.zeros((2, 2)), np.zeros((2, 2*M))], 
                        [np.zeros((2*M, 2)), np.linalg.cholesky(PX0landmarks)]])
    
    Qwrobot = 0.2**2 * np.eye(2)
    Qwlandmarks = (1e-10)**2 * np.eye(2*M)
    Rv = 0.15**2 * np.eye(2*M)
    
    mX0 = np.vstack((mX0robot, mX0landmarks))
    Qw = np.block([[Qwrobot, np.zeros((2, 2*M))], 
                   [np.zeros((2*M, 2)), Qwlandmarks]])
    
    Frobot = np.array([[np.cos(w*deltaT), -np.sin(w*deltaT)], 
                        [np.sin(w*deltaT), np.cos(w*deltaT)]])
    
    CC = np.array([[-1.5], [0]])  # Center of ideal circle
    Brobot = (np.eye(2) - Frobot) @ CC
    
    # Create block diagonal matrix with Frobot once and eye(2) M times
    F = np.block([[Frobot, np.zeros((2, 2*M))],
                  [np.zeros((2*M, 2)), np.eye(2*M)]])
    
    B = np.vstack((Brobot, np.zeros((2*M, 1))))
    
    # Create Hfull matrix
    neg_eye2 = -np.eye(2)
    Hfull = np.block([np.kron(np.ones((M, 1)), neg_eye2), np.eye(2*M)])
    
    arrayH = np.full((2*M, 2+2*M, N), np.nan)
    arrayR = np.full((2*M, 2*M, N), np.nan)
    X = np.full((2+2*M, N), np.nan)
    Z = np.full((2*M, N), np.nan)
    in_fov = np.full((2*M, N), np.nan)
    
    # Noise processes and State vector at initial time 0
    W = np.linalg.cholesky(Qw) @ np.random.randn(2+2*M, N)
    V = np.linalg.cholesky(Rv) @ np.random.randn(2*M, N)
    X[:, 0] = mX0.flatten() + (cholPX0.T @ np.random.randn(2+2*M, 1)).flatten()
    Z[:, 0] = np.full(2*M, np.nan)
    
    # Time instants 1:(N-1)
    for k in range(1, N):
        X[:, k] = F @ X[:, k-1] + B.flatten() + W[:, k-1]  # noisy state
        Z[:, k] = Hfull @ X[:, k]  # noise-free measurement if everything is visible
        
        mat_Z = Z[:, k].reshape(2, M, order='F')
        mat_OR = np.kron(np.ones((1, M)), (X[0:2, k] - CC.flatten()).reshape(2, 1))
        orth_vec = np.array([-(X[1, k] - CC[1, 0]), X[0, k] - CC[0, 0]])
        mat_OR_orth = np.kron(np.ones((1, M)), orth_vec.reshape(2, 1))
        
        # Calculate dot products
        dot_product1 = np.sum(mat_Z * mat_OR.reshape(2, M), axis=0)
        dot_product2 = np.sum(mat_Z * mat_OR_orth.reshape(2, M), axis=0)
        
        # Create visibility indicator
        visibility = np.logical_and(dot_product1 >= 0, dot_product2 >= 0).astype(float)
        in_fov[:, k] = np.kron(visibility, np.ones(2))
        
        # Normalize in_fov (equivalent to dividing by itself in MATLAB)
        mask = in_fov[:, k] != 0
        in_fov[mask, k] = in_fov[mask, k] / in_fov[mask, k]
        
        # Apply visibility to measurements
        Z[:, k] = in_fov[:, k] * (Z[:, k] + V[:, k])
        
        # Update observation matrices
        arrayH[:, :, k] = np.outer(in_fov[:, k], np.ones(2+2*M)) * Hfull
        arrayR[:, :, k] = np.outer(in_fov[:, k], in_fov[:, k]) * Rv
    
    if plot_p == 1:
        fig = plt.figure(figsize=(10, 10))
        
        for k in range(N):
            plt.clf()
            plt.plot(CC[0, 0], CC[1, 0], 'k*')
            plt.grid(True)
            plt.axis('equal')
            plt.xlim(-5, 5)
            plt.ylim(-5, 5)
            plt.title('GROUND TRUTH (STATE SPACE)')
            
            plt.plot(X[0, k], X[1, k], 'o', markeredgecolor='none', markerfacecolor='b')
            plt.plot(X[0, max(k-15, 0):k+1], X[1, max(k-15, 0):k+1], '--b')
            
            # Create visibility domain
            visib_domain = np.array([
                [X[0, k], X[0, k]+10*(X[0, k]-CC[0, 0]), X[0, k]-100*(X[1, k]-CC[1, 0])],
                [X[1, k], X[1, k]+10*(X[1, k]-CC[1, 0]), X[1, k]+100*(X[0, k]-CC[0, 0])]
            ])
            poly = Polygon(visib_domain.T, facecolor='c', edgecolor='none', alpha=0.1)
            plt.gca().add_patch(poly)
            
            for m in range(M):
                if not np.isnan(in_fov[2*m, k]):
                    color = 'g'
                    plt.plot([-5, 5], [X[1, k], X[1, k]], '--', color=[0.5, 0.5, 0.5], linewidth=0.5)
                    plt.plot([X[0, k], X[0, k]], [-5, 5], '--', color=[0.5, 0.5, 0.5], linewidth=0.5)
                    plt.plot([X[2*m+2, k], X[2*m+2, k]], [X[1, k], X[2*m+3, k]], ':m', linewidth=0.5)
                    plt.text(X[2*m+2, k], X[1, k]-0.2, f'(u^1-u^R)_{{{m+1}}}')
                    plt.plot([X[2*m+2, k], X[0, k]], [X[2*m+3, k], X[2*m+3, k]], ':m', linewidth=0.5)
                    plt.text(X[0, k]-0.8, X[2*m+3, k], f'(v^1-v^R)_{{{m+1}}}')
                else:
                    color = 'r'
                
                plt.plot(X[2*m+2, k], X[2*m+3, k], 's', markeredgecolor='k', markerfacecolor=color)
                plt.text(X[2*m+2, k]+0.3, X[2*m+3, k], f'Amer {m+1}')
            
            plt.draw()
            plt.pause(deltaT)
    
    return N, T, M, Z, arrayH, arrayR, in_fov, F, B, CC, Hfull, mX0, PX0, Qw, Rv, X