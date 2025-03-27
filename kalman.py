import numpy as np
import matplotlib.pyplot as plt
from data_simulation import data_simulation
from ellipse import ellipse

def kalman(N, T, M, Z, arrayH, arrayR, F, B, CC, mX0, PX0, Qw, X):
    """    
    Param:
    ----------
    N : int
        Nombre d'échantillons temporels
    T : ndarray
        Vecteur des échantillons temporels (1xN)
    M : int
        Nombre de points de repère (amers)
    Z : ndarray
        Mesures 2MxN - la colonne l+1 contient l'observation z_l à l'instant l
    arrayH : ndarray
        Matrices d'observation 2Mx(2+2M)xN
    arrayR : ndarray
        Matrices de covariance du bruit de mesure 2Mx2MxN
    F : ndarray
        Matrice  blkdiag
    B : ndarray
        Vecteur  dynamique a priori
    CC : ndarray
        Centre du cercle idéal
    mX0 : ndarray
        Vecteur d'espérance de l'état initial (à l'instant 0)
    PX0 : ndarray
        Matrice de covariance de l'état initial (à l'instant 0)
    Qw : ndarray
        Matrice de covariance du bruit de dynamique
    X : ndarray
        État réel pour comparaison (2+2M)xN
        
    Return:
    -------
    Xpred : ndarray
        Prédictions d'état (x_{l|l-1}) (2+2M)xN
    Ppred : ndarray
        Matrices de covariance des  prédictions (P_{l|l-1}) (2+2M)x(2+2M)xN
    Xest : ndarray
        Estimations d'état (x_{l|l}) (2+2M)xN
    Pest : ndarray
        Matrices de covariance des estimation (P_{l|l}) (2+2M)x(2+2M)xN
    Gain : ndarray
        Gains  (K_l) (2+2M)x(2*M)xN
    """

    # taille
    n = mX0.shape[0]


    # Initialisation
    Xpred = np.full((n, N), np.nan)      # x{l|l-1}
    Ppred = np.full((n, n, N), np.nan)   # P{l|l-1}
    Xest = np.full((n, N), np.nan)       # x{l|l}
    Pest = np.full((n, n, N), np.nan)    # P{l|l}
    Gain = np.full((n, 2*M, N), np.nan)  # K_l

    Xest[:, 0] = mX0.flatten()
    Pest[:, :, 0] = PX0


    for l in range(1, N):

        # Prédiction 
        Xpred[:, l] = F @ Xest[:, l-1] + B.flatten()
        Ppred[:, :, l] = F @ Pest[:, :, l-1] @ F.T + Qw

        # récup H et R pour le l
        H = arrayH[:, :, l]
        R = arrayR[:, :, l]
        indices_valides = ~np.isnan(Z[:, l])

        if np.any(indices_valides):
            H_valid = H[indices_valides, :]

            z_valid = Z[indices_valides, l]
            R_valid = R[np.ix_(indices_valides, indices_valides)] #sous-matrice de la matrice de covariance du bruit de mesure R, uniquement les lignes et colonnes correspondant aux observations valides.

            truc = R_valid + H_valid @ Ppred[:, :, l] @ H_valid.T



            K_valid = Ppred[:, :, l] @ H_valid.T @ np.linalg.pinv(truc) # gain


            K_full = np.zeros((n, 2*M))
            K_full[:, indices_valides] = K_valid
            Gain[:, :, l] = K_full



            # mise a jour

            nouv = z_valid - H_valid @ Xpred[:, l]
            Xest[:, l] = Xpred[:, l] + K_valid @ nouv


            Pest[:, :, l] = Ppred[:, :, l] - K_valid @ H_valid @ Ppred[:, :, l]

        else:
            # Si aucune mesure valide, l'estim =  prédiction
            Xest[:, l] = Xpred[:, l]
            Pest[:, :, l] = Ppred[:, :, l]

    return Xpred, Ppred, Xest, Pest, Gain





