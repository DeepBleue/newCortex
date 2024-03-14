import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def random_vectors_on_unit_sphere(num_samples):
    phi = np.random.uniform(0, 2 * np.pi, num_samples)
    costheta = np.random.uniform(-1, 1, num_samples)
    
    theta = np.arccos(costheta)
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = costheta

    return np.vstack([x, y, z]).T

def is_within_cone_batch(vectors, origins, angle):
    """
    Check if a set of vectors are within cones defined by angles from origin vectors.
    Returns a mask of shape (len(vectors), len(origins)).
    """
    cos_angle = np.cos(np.radians(angle))
    # Use broadcasting to compute the dot product between vectors and each origin
    dot_products = np.einsum('ik,jk->ij', vectors, origins)
    return dot_products > cos_angle

def generate_vectors_in_cone(origins, angle, num_samples):
    num_origins = origins.shape[0]
    estimate_samples = num_samples * 5
    
    vectors = random_vectors_on_unit_sphere(estimate_samples)
    mask = is_within_cone_batch(vectors, origins, angle)
    
    samples_in_cones = []
    for origin_idx in tqdm(range(num_origins)):
        origin_mask = mask[:, origin_idx]
        samples_for_origin = vectors[origin_mask][:num_samples]
        
        # If not enough samples, recursively call to gather more
        while samples_for_origin.shape[0] < num_samples:
            additional_samples = generate_vectors_in_cone(np.array([origins[origin_idx]]), angle, num_samples - samples_for_origin.shape[0])
            samples_for_origin = np.vstack([samples_for_origin, additional_samples[0][:num_samples - samples_for_origin.shape[0]]])
        
        samples_in_cones.append(samples_for_origin)

    return samples_in_cones


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

# # Usage:
# origins = np.array([[0, 0, 1], [0, 1, 0]])  # 2 origin vectors as an example
# samples = generate_vectors_in_cone(origins, 15, 50)  # Generate 1000 samples for each origin vector
# plot_vectors_in_3d(samples[0],np.array([0,0,0]))
