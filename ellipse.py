import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

def ellipse(mx, Px, color):
    """
    Draws an ellipse containing 99% of the realizations of a 2D Gaussian 
    random variable with mean mx and covariance Px.
    
    Parameters:
    ----------
    mx : ndarray
        Mean vector (2x1)
    Px : ndarray
        Covariance matrix (2x2)
    color : str
        Color code for the ellipse ('b', 'g', 'r', 'm', etc.)
        
    Returns:
    -------
    h : matplotlib.patches.Polygon
        Handle to the created ellipse patch
    """
    
    k = 9.21
    # sqrt(k)=1:39.4%, sqrt(k)=2:86.5%, sqrt(k)=2.447:95%, sqrt(k)=3:98.9%
    
    # Make sure Px is symmetric
    Px = (Px + Px.T) / 2
    
    # Eigenvalue decomposition
    D, V = np.linalg.eig(Px)
    
    # Create points for the ellipse
    t = np.arange(0, 2*np.pi + 0.1, 0.1)
    
    xx = np.zeros((2, len(t)))
    xx[0, :] = np.sqrt(k * D[0]) * np.cos(t)
    xx[1, :] = np.sqrt(k * D[1]) * np.sin(t)
    
    # Transform points
    X = np.zeros_like(xx)
    for i in range(len(t)):
        X[:, i] = V @ xx[:, i] + mx
    
    # Create and return the patch
    h = plt.fill(X[0, :], X[1, :], color, alpha=0.2, edgecolor='none')
    
    return h