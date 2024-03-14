import cupy as cp

def random_vector_on_unit_sphere():
    """
    Generate a random 3D unit vector using CuPy.
    """
    phi = cp.random.uniform(0, 2 * cp.pi)
    costheta = cp.random.uniform(-1, 1)
    
    theta = cp.arccos(costheta)
    x = cp.sin(theta) * cp.cos(phi)
    y = cp.sin(theta) * cp.sin(phi)
    z = costheta

    return cp.array([x, y, z])

def is_within_cone(vector, origin, angle):
    """
    Check if a vector is within a cone defined by an angle from an origin vector using CuPy.
    """
    cos_angle = cp.cos(cp.radians(angle))
    
    # Taking the dot product of normalized vectors gives the cosine of the angle between them
    dot_product = cp.dot(vector, origin)
    
    return dot_product > cos_angle

def generate_vectors_in_cone(origin, angle, num_samples):
    """
    Generate a set of vectors within a cone defined by an angle from an origin vector using CuPy.
    """
    samples = []
    origin = cp.array(origin)
    
    while len(samples) < num_samples:
        sample = random_vector_on_unit_sphere()
        if is_within_cone(sample, origin, angle):
            samples.append(sample)  # Get the array from the GPU to CPU for appending to the list

    return cp.array(samples)  # Return the samples as a CuPy array

# ... [previous functions: random_vector_on_unit_sphere, is_within_cone, generate_vectors_in_cone]
