import numpy as np
import matplotlib.pyplot as plt

def random_vectors_on_unit_sphere(num_samples):
    """
    Generate `num_samples` random 3D unit vectors.
    """
    phi = np.random.uniform(0, 2 * np.pi, num_samples)
    costheta = np.random.uniform(-1, 1, num_samples)
    
    theta = np.arccos(costheta)
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = costheta

    return np.vstack([x, y, z]).T

def is_within_cone_batch(vectors, origin, angle):
    """
    Check if a set of vectors are within a cone defined by an angle from an origin vector.
    """
    cos_angle = np.cos(np.radians(angle))
    
    # Compute dot products for a batch of vectors
    dot_products = np.einsum('ij,j->i', vectors, origin)
    
    return dot_products > cos_angle

def generate_vectors_in_cone(origin, angle, num_samples):
    """
    Generate a set of vectors within a cone defined by an angle from an origin vector.
    """
    # An overestimate for the number of samples to generate at once
    estimate_samples = num_samples * 5
    
    vectors = random_vectors_on_unit_sphere(estimate_samples)
    mask = is_within_cone_batch(vectors, origin, angle)
    samples_in_cone = vectors[mask]
    
    # In rare cases where we don't get enough vectors within the cone, we can call the function recursively
    while len(samples_in_cone) < num_samples:
        additional_samples = generate_vectors_in_cone(origin, angle, num_samples - len(samples_in_cone))
        samples_in_cone = np.vstack([samples_in_cone, additional_samples])

    return samples_in_cone[:num_samples]


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