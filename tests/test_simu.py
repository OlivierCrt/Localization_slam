import numpy as np
import matplotlib.pyplot as plt
from data_simulation import data_simulation
from ellipse import ellipse

def test():
    # Test data_simulation with animation
    print("Running data simulation...")
    N, T, M, Z, arrayH, arrayR, in_fov, F, B, CC, Hfull, mX0, PX0, Qw, Rv, X = data_simulation(plot_p=1)
    print(f"Simulation completed with {N} time steps and {M} landmarks.")
    
    # Test ellipse function with some data from the simulation
    plt.figure(figsize=(10, 10))
    plt.title("Test of ellipse function")
    
    # Get the robot position at the last time step
    robot_pos = X[0:2, -1]
    
    # Create a simple covariance matrix
    cov = np.array([[0.2, 0.05], [0.05, 0.3]])
    
    # Draw the ellipse
    h = ellipse(robot_pos, cov, 'b')
    
    # Plot the robot position
    plt.plot(robot_pos[0], robot_pos[1], 'ro')
    
    # Add labels and grid
    plt.grid(True)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.axis('equal')
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    
    plt.show()
    print("Ellipse test completed.")
    print( Hfull)
    print (Z)
    print("T\n\n",T)
    print ("X:\n\n",X)
    
if __name__ == "__main__":
    test()