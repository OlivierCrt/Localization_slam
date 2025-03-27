import numpy as np
import matplotlib.pyplot as plt
import time
from matplotlib.patches import Polygon

from data_simulation import data_simulation
from ellipse import ellipse
from kalman import kalman

def run_test(plot_simulation=True, run_animation=True):
    """
    Run a complete test of the simulation and Kalman filter
    
    Parameters:
    ----------
    plot_simulation : bool
        Whether to plot the initial simulation
    run_animation : bool
        Whether to run the animation of results
    """
    start_time = time.time()
    
    # Generate simulation data
    print("Running data simulation...")
    N, T, M, Z, arrayH, arrayR, in_fov, F, B, CC, Hfull, mX0, PX0, Qw, Rv, X = data_simulation(plot_p=1 if plot_simulation else 0)
    sim_time = time.time() - start_time
    print(f"Simulation completed in {sim_time:.2f} seconds with {N} time steps and {M} landmarks.")
    
    # Run Kalman filter
    print("Running Kalman filter...")
    filter_start = time.time()
    Xpred, Ppred, Xest, Pest, Gain = kalman(N, T, M, Z, arrayH, arrayR, F, B, CC, mX0, PX0, Qw, X)
    filter_time = time.time() - filter_start
    print(f"Kalman filter completed in {filter_time:.2f} seconds.")
    
    # erreurs
    robot_position_erreur = np.sqrt(np.sum((X[0:2, :] - Xest[0:2, :])**2, axis=0))
    amers_erreur = np.zeros((M, N))




    for m in range(M):
        amers_erreur[m, :] = np.sqrt(np.sum((X[2*m+2:2*m+4, :] - Xest[2*m+2:2*m+4, :])**2, axis=0))
    
    # Print error statistics
    print("\n Stats erreur de pos:")
    print(f"Erreur de pos finale: {robot_position_erreur[-1]:.4f}")
    print(f"moyenne erreur de pos: {np.mean(robot_position_erreur):.4f}")
    print(f"erreur max: {np.max(robot_position_erreur):.4f}")
    print("\nstats erreurs amers:")
    for m in range(M):
        print(f"Amer {m+1} - Final: {amers_erreur[m, -1]:.4f}, " + 
              f"moy: {np.mean(amers_erreur[m, :]):.4f}, " +
              f"Max: {np.max(amers_erreur[m, :]):.4f}")
    
    plt.figure(figsize=(12, 10))
    plt.title("Robot et amers traj")
    
    # Plot ground truth robot trajectory
    plt.plot(X[0, :], X[1, :], 'b-', label='Trajectoire réelle du robot')
    
    # Plot estimated robot trajectory
    plt.plot(Xest[0, :], Xest[1, :], 'r--', label='Trajectoire estimée')
    
    # Plot center of circle
    plt.plot(CC[0, 0], CC[1, 0], 'k*', markersize=10, label='Centre du cercle')
    
    # Plot initial and final positions with larger markers
    plt.plot(X[0, 0], X[1, 0], 'bo', markersize=8, label=' départ réel')
    plt.plot(X[0, -1], X[1, -1], 'bx', markersize=8, label='fin réelle')
    plt.plot(Xest[0, 0], Xest[1, 0], 'ro', markersize=8, label='Est. depart')
    plt.plot(Xest[0, -1], Xest[1, -1], 'rx', markersize=8, label='Est. fin')
    
    # Plot
    for m in range(M):
        plt.plot(X[2*m+2, -1], X[2*m+3, -1], 'gs', label=f'Vrai amer {m+1}' if m==0 else "")
        plt.plot(Xest[2*m+2, -1], Xest[2*m+3, -1], 'mo', label=f'Est. amer {m+1}' if m==0 else "")
        
        # ellispse
        ellipse(Xest[2*m+2:2*m+4, -1], Pest[2*m+2:2*m+4, 2*m+2:2*m+4, -1], 'm')
    
    # ellispe finale
    ellipse(Xest[0:2, -1], Pest[0:2, 0:2, -1], 'r')
    
    plt.grid(True)
    plt.axis('equal')
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    
    plt.tight_layout()
    plt.savefig("kalman_filter_result.png", dpi=300)
    
    # Plot error stats
    plt.figure(figsize=(15, 10))
    
    # erreurs
    plt.subplot(2, 1, 1)
    plt.plot(T, robot_position_erreur, 'b-')
    plt.grid(True)
    plt.xlabel('Time')
    plt.ylabel('Position Error')
    plt.title('Erreur de pos')
    
    plt.subplot(2, 1, 2)
    for m in range(M):
        plt.plot(T, amers_erreur[m, :], label=f'amer {m+1}')
    
    plt.grid(True)
    plt.xlabel('Temps')
    plt.ylabel('Erreur de pos')
    plt.title('Erreur amers estimation')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("erreurs_stats.png", dpi=300)
    
    # Run animation if requested
    if run_animation:
        animate_results(N, T, M, Z, X, Xest, Pest, CC, in_fov)
    
    plt.show()
    
    return N, T, M, Z, X, Xpred, Ppred, Xest, Pest, Gain

def animate_results(N, T, M, Z, X, Xest, Pest, CC, in_fov):
    
    print("animation...")
    
    fig = plt.figure(figsize=(12, 10))
    plt.title("Kalman Animation")
    
    frames = min(N, 50)
    step = max(1, N // frames)
    
    for k in range(0, N, step):
        plt.clf()
        
        plt.plot(CC[0, 0], CC[1, 0], 'k*', markersize=10, label='centre cercle')
        
        plt.plot(X[0, :k+1], X[1, :k+1], 'b-', label='vrai traj')
        plt.plot(X[0, k], X[1, k], 'bo', markersize=6)
        
        plt.plot(Xest[0, :k+1], Xest[1, :k+1], 'r--', label='traj estim')
        plt.plot(Xest[0, k], Xest[1, k], 'ro', markersize=6)
        
        ellipse(Xest[0:2, k], Pest[0:2, 0:2, k], 'r')
        
        for m in range(M):
            plt.plot(X[2*m+2, k], X[2*m+3, k], 'gs', markersize=6, 
                     label=f'vrai amer {m+1}' if k==0 and m==0 else "")
            
            plt.plot(Xest[2*m+2, k], Xest[2*m+3, k], 'mo', markersize=6,
                    label=f'Est. amer {m+1}' if k==0 and m==0 else "")
            
            ellipse(Xest[2*m+2:2*m+4, k], Pest[2*m+2:2*m+4, 2*m+2:2*m+4, k], 'm')
            
            if k > 0 and not np.isnan(in_fov[2*m, k]):
                plt.plot([X[0, k], X[2*m+2, k]], [X[1, k], X[2*m+3, k]], 'g:', alpha=0.5,
                         label='Visibility' if k==0 and m==0 else "")
        
        if k > 0:
            visib_domain = np.array([
                [X[0, k], X[0, k]+10*(X[0, k]-CC[0, 0]), X[0, k]-100*(X[1, k]-CC[1, 0])],
                [X[1, k], X[1, k]+10*(X[1, k]-CC[1, 0]), X[1, k]+100*(X[0, k]-CC[0, 0])]
            ])
            poly = Polygon(visib_domain.T, facecolor='c', edgecolor='none', alpha=0.1)
            plt.gca().add_patch(poly)
            if k == step:
                plt.plot([], [], 'c-', alpha=0.3, label='Visibility cone')
        
        plt.grid(True)
        plt.axis('equal')
        plt.xlim(-5, 5)
        plt.ylim(-5, 5)
        if k == 0:
            plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.title(f"Step {k+1}/{N} (Temps: {T[k]:.1f})")
        
        plt.draw()
        plt.pause(0.05)
    
    plt.savefig("animation.png", dpi=300)

if __name__ == "__main__":
    run_test(plot_simulation=True, run_animation=True)