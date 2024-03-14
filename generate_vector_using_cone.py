import numpy as np
import matplotlib.pyplot as plt

def random_vector_on_unit_sphere():
    """
    Generate a random 3D unit vector.
    """
    phi = np.random.uniform(0, 2 * np.pi)
    costheta = np.random.uniform(-1, 1)
    
    theta = np.arccos(costheta)
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = costheta

    return np.array([x, y, z])

def is_within_cone(vector, origin, angle):
    """
    Check if a vector is within a cone defined by an angle from an origin vector.
    """
    cos_angle = np.cos(np.radians(angle))
    
    # Taking the dot product of normalized vectors gives the cosine of the angle between them
    dot_product = np.dot(vector, origin)
    
    return dot_product > cos_angle

def generate_vectors_in_cone(origin, angle, num_samples):
    """
    Generate a set of vectors within a cone defined by an angle from an origin vector.
    """
    samples = []
    
    while len(samples) < num_samples:
        sample = random_vector_on_unit_sphere()
        if is_within_cone(sample, origin, angle):
            samples.append(sample)

    return np.array(samples)



def plot_vectors_in_3d(vectors, origin):
    """
    Plot vectors in 3D using matplotlib.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot each vector from the origin
    for v in vectors:
        ax.quiver(0, 0, 0, v[0], v[1], v[2], length=1, arrow_length_ratio=0.1)
    
    # Setting the limits
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    plt.show()