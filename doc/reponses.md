# Réponses

## 1
Rotation de pi/4 :

\[
F^R = R(\theta) = \begin{pmatrix}
\cos(\theta) & -\sin(\theta) \\
\sin(\theta) & \cos(\theta)
\end{pmatrix}
\]

## 2


Hk est une matrice composé de -1 sur la premiere colonne ( pour retirer la pos du robot position relative), le reste est 0 ou 1, 1 à la colone correspondant à l amer observé.

## 3

### 3.1
On voit : 
- Champs de vision
- Amers
- Trajectoire qui suit les 45* à chaque frame
- Centre du cercle de traj

### 3.2 

T : (temps)
X : (mesures bruitées)
X : (Etat caché à estimer, coord réelles, vitesse, coo des amers)

### 3.3

mX0 :  Le vecteur d 'état initial. Position et vitesse initiale estimés et pos estimées des amers.
Po (PX0) : C est la matrice de covariance du vecteur d'état initial. Incertitude
Qw : C est la matrice de cov du modele dyn du robot
Rv covariance bruit de mesure

### 3.4