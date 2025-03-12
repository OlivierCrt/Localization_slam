# 1. Matrices de base

- **\( I \)** : C'est la **matrice identité**, une matrice carrée avec des 1 sur la diagonale et des 0 ailleurs. Elle joue un rôle neutre dans les multiplications matricielles.  
- **\( O \)** : C'est la **matrice nulle**, une matrice remplie de 0.  
- **\( \text{blkdiag}(A_1, \dots, A_L) \)** : C'est une **matrice diagonale par blocs**, qui place les matrices \( A_1, \dots, A_L \) sur la diagonale d'une grande matrice et remplit le reste avec des 0.  
- **\( M' \) (ou \( M^\top \))** : Transposée d'une matrice \( M \), c'est-à-dire qu'on échange les lignes et les colonnes.  

# 2. Variables aléatoires et distribution normale  

- \( x \sim \mathcal{N}(\bar{x}, P) \) signifie que \( x \) suit une **distribution normale multivariée** avec :  
  - \( \bar{x} \) : moyenne de \( x \)  
  - \( P \) : matrice de covariance, qui décrit la dispersion de \( x \)  
- La **fonction de densité de probabilité** associée est notée :  
  \[
  p_x(x) = \mathcal{N}(x; \bar{x}, P)
  \]

# 3. État du système  

On définit :  
- **\( r = (u_R, v_R)' \)** : position du robot.  
- **\( m_1, \dots, m_M \)** : positions des **M** landmarks (repères fixes).  
- **\( x \)** : le **vecteur d'état caché**, qui regroupe \( r \) et les \( m_i \), soit :  

  \[
  x = (r', m_1', \dots, m_M')'
  \]

  avec \( x \in \mathbb{R}^{2+2M} \).  

À chaque instant \( k \), on a les valeurs :  

\[
x_k = (r_k', m_{1,k}', \dots, m_{M,k}')'
\]

# 4. Évolution du robot  

Le robot se déplace sur un arc de cercle centré en \( C \), ce qui implique :  

\[
r_{k+1} - c = F_R (r_k - c)
\]

avec  

\[
F_R =
\begin{bmatrix}
\frac{\sqrt{2}}{2} & -\frac{\sqrt{2}}{2} \\
\frac{\sqrt{2}}{2} & \frac{\sqrt{2}}{2}
\end{bmatrix}
\]

Cette matrice \( F_R \) représente une **rotation de 45°**.  

Les landmarks, eux, **ne bougent pas** :  

\[
\forall m, \quad m_{m,k+1} = m_{m,k}
\]

# 5. Modélisation avec bruit  

Dans la réalité, il y a toujours du bruit (glissements, erreurs de mesure...). L'évolution devient donc **stochastique** :  

\[
x_{k+1} - b = F(x_k - b) + w_k
\]

avec :  
- \( F = \text{blkdiag}(F_R, I, ..., I) \), matrice dynamique de transition.  
- \( w_k \sim \mathcal{N}(0, Q_w) \), un bruit blanc gaussien avec covariance \( Q_w \).  
- \( b \) : un vecteur constant qui dépend du centre de rotation \( c \).  

# 6. Mesures et observations  

À chaque instant \( k \), le robot capte des mesures \( z_k \) qui dépendent de la position des landmarks visibles :  

\[
z_k = H_k x_k + v_k
\]

avec :  
- \( H_k \) : matrice d'observation.  
- \( v_k \sim \mathcal{N}(0, R) \) : bruit de mesure.  

Les mesures sont **aléatoires** et leur **taille varie** selon les landmarks visibles.  
