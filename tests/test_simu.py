import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from data_simulation import data_simulation
from ellipse import ellipse
from kalman import kalman

def main():
    """
    Main function to run simulation, apply Kalman filter and visualize results
    """
    # Generate simulation data
    print("Running data simulation...")
    N, T, M, Z, arrayH, arrayR, in_fov, F, B, CC, Hfull, mX0, PX0, Qw, Rv, X = data_simulation(plot_p=0)
    print(f"Simulation completed with {N} time steps and {M} landmarks.")
    
    # Run Kalman filter
    print("Running Kalman filter...")
    Xpred, Ppred, Xest, Pest, Gain = kalman(N, T, M, Z, arrayH, arrayR, F, B, CC, mX0, PX0, Qw, X)
    print("Kalman filter completed.")
    
    # Visualization of results
    plot_results(N, T, M, Z, X, Xpred, Ppred, Xest, Pest, CC, in_fov)
    
    # Additional analysis plots
    plot_position_error(T, X, Xest)
    plot_landmark_error(T, X, Xest, M)
    
    return N, T, M, Z, X, Xpred, Ppred, Xest, Pest, Gain

def plot_results(N, T, M, Z, X, Xpred, Ppred, Xest, Pest, CC, in_fov):
    """
    Visualize the Kalman filter results
    """
    # Create figure for trajectory and landmarks
    plt.figure(figsize=(12, 10))
    plt.title("Robot and Landmark Trajectories")
    
    # Plot ground truth robot trajectory
    plt.plot(X[0, :], X[1, :], 'b-', label='True robot trajectory')
    
    # Plot estimated robot trajectory
    plt.plot(Xest[0, :], Xest[1, :], 'r--', label='Estimated robot trajectory')
    
    # Plot center of circle
    plt.plot(CC[0, 0], CC[1, 0], 'k*', markersize=10, label='Circle center')
    
    # Plot true landmarks
    for m in range(M):
        plt.plot(X[2*m+2, -1], X[2*m+3, -1], 'gs', label=f'True landmark {m+1}' if m==0 else "")
    
    # Plot estimated landmarks
    for m in range(M):
        plt.plot(Xest[2*m+2, -1], Xest[2*m+3, -1], 'mo', label=f'Estimated landmark {m+1}' if m==0 else "")
    
    # Draw uncertainty ellipses for the final position
    ellipse(Xest[0:2, -1], Pest[0:2, 0:2, -1], 'r')
    
    # Draw uncertainty ellipses for landmarks
    for m in range(M):
        ellipse(Xest[2*m+2:2*m+4, -1], Pest[2*m+2:2*m+4, 2*m+2:2*m+4, -1], 'm')
    
    plt.grid(True)
    plt.axis('equal')
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    plt.legend()
    
    # Create animation of the process
    create_animation(N, T, M, Z, X, Xest, Pest, CC, in_fov)
    
    plt.show()

def create_animation(N, T, M, Z, X, Xest, Pest, CC, in_fov):
    """
    Create an animation of the Kalman filter process
    """
    fig = plt.figure(figsize=(12, 10))
    plt.title("Kalman Filter Animation")
    
    skip_frames = max(1, N // 20)  # Show only some frames for faster visualization
    
    for k in range(0, N, skip_frames):
        plt.clf()
        
        # Plot center of circle
        plt.plot(CC[0, 0], CC[1, 0], 'k*', markersize=10)
        
        # Plot ground truth robot trajectory
        plt.plot(X[0, :k+1], X[1, :k+1], 'b-', label='True trajectory')
        plt.plot(X[0, k], X[1, k], 'bo', markersize=6)
        
        # Plot estimated robot trajectory
        plt.plot(Xest[0, :k+1], Xest[1, :k+1], 'r--', label='Estimated trajectory')
        plt.plot(Xest[0, k], Xest[1, k], 'ro', markersize=6)
        
        # Draw uncertainty ellipse for the robot
        ellipse(Xest[0:2, k], Pest[0:2, 0:2, k], 'r')
        
        # Plot landmarks
        for m in range(M):
            # True landmarks
            plt.plot(X[2*m+2, k], X[2*m+3, k], 'gs', markersize=6)
            
            # Estimated landmarks
            plt.plot(Xest[2*m+2, k], Xest[2*m+3, k], 'mo', markersize=6)
            
            # Draw uncertainty ellipses for landmarks
            ellipse(Xest[2*m+2:2*m+4, k], Pest[2*m+2:2*m+4, 2*m+2:2*m+4, k], 'm')
            
            # Show visibility
            if k > 0 and not np.isnan(in_fov[2*m, k]):
                # Connect robot to visible landmarks
                plt.plot([Xest[0, k], Xest[2*m+2, k]], [Xest[1, k], Xest[2*m+3, k]], 'g:', alpha=0.5)
        
        # Create visibility domain
        if k > 0:
            visib_domain = np.array([
                [X[0, k], X[0, k]+10*(X[0, k]-CC[0, 0]), X[0, k]-100*(X[1, k]-CC[1, 0])],
                [X[1, k], X[1, k]+10*(X[1, k]-CC[1, 0]), X[1, k]+100*(X[0, k]-CC[0, 0])]
            ])
            poly = Polygon(visib_domain.T, facecolor='c', edgecolor='none', alpha=0.1)
            plt.gca().add_patch(poly)
        
        plt.grid(True)
        plt.axis('equal')
        plt.xlim(-5, 5)
        plt.ylim(-5, 5)
        plt.legend()
        plt.title(f"Step {k} / {N-1}")
        
        plt.draw()
        plt.pause(0.1)
    
    plt.show()

def plot_position_error(T, X, Xest):
    """
    Plot position error over time
    """
    plt.figure(figsize=(10, 6))
    
    # Calculate position error
    pos_error = np.sqrt((X[0, :] - Xest[0, :])**2 + (X[1, :] - Xest[1, :])**2)
    
    plt.plot(T, pos_error, 'b-')
    plt.grid(True)
    plt.xlabel('Time')
    plt.ylabel('Position Error (Euclidean distance)')
    plt.title('Robot Position Estimation Error')
    
    plt.show()

def plot_landmark_error(T, X, Xest, M):
    """
    Plot landmark position errors over time
    """
    plt.figure(figsize=(10, 6))
    
    for m in range(M):
        # Calculate landmark position error
        landmark_error = np.sqrt((X[2*m+2, :] - Xest[2*m+2, :])**2 + 
                                 (X[2*m+3, :] - Xest[2*m+3, :])**2)
        
        plt.plot(T, landmark_error, label=f'Landmark {m+1}')
    
    plt.grid(True)
    plt.xlabel('Time')
    plt.ylabel('Landmark Position Error (Euclidean distance)')
    plt.title('Landmark Position Estimation Error')
    plt.legend()
    
    plt.show()

if __name__ == "__main__":
    main()