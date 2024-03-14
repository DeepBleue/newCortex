import numpy as np
from tqdm import tqdm
import trimesh
from make_vtk import write_lines_to_vtk,write_points_to_vtk
from vtk_revise import read_vtk,write_vtk
from generate_vector_using_cone import generate_vectors_in_cone



white = read_vtk('./data/SUBJ_001_MR_BL/surf/lh.CortexODE.white.vtk')
pial = read_vtk('./data/SUBJ_001_MR_BL/surf/lh.CortexODE.pial.vtk')

white_vertices = white['vertices']
white_faces = white['faces'][:,1:]

pial_vertices = pial['vertices']
pial_faces = pial['faces'][:,1:]

# Now, process the results
pairs = []
thicknesses = []
not_intersect = []


# Load your target mesh
origin_mesh = trimesh.Trimesh(vertices=white_vertices, faces=white_faces)
target_mesh = trimesh.Trimesh(vertices=pial_vertices, faces=pial_faces)

# origin vertex normals
origin_directions = origin_mesh.vertex_normals

# Pre-compute all the cone directions for all origins
all_origins = []
all_directions = []

for idx in tqdm(range(len(white_vertices))):
    origin_pos = white_vertices[idx]
    origin_cone_directions = generate_vectors_in_cone(origin_directions[idx], 15, 50)
    all_origins.extend([origin_pos] * len(origin_cone_directions))
    all_directions.extend(origin_cone_directions)

# Find ray-mesh intersections for all origins and directions at once
locations, index_ray, index_tri = target_mesh.ray.intersects_location(ray_origins=all_origins, ray_directions=all_directions)



# 'idx' is a factor to know which original vertex we are looking at. 
# This will change every 50 iterations (since there are 50 samples for each vertex).
for idx in tqdm(range(0, len(all_origins), 50)):
    subset_locations = locations[idx: idx+50]
    subset_origin = all_origins[idx]
    
    distances = [np.linalg.norm(loc - subset_origin) for loc in subset_locations]

    if distances:
        min_distance_index = np.argmin(distances)
        closest_location = subset_locations[min_distance_index]
        closest_distance = distances[min_distance_index]
        
        pairs.append((subset_origin, closest_location))
        thicknesses.append(closest_distance)
    else:
        thicknesses.append(float('nan'))
        not_intersect.append(subset_origin)
        
        
thicknesses = np.array(thicknesses)
white['new_thickness'] = thicknesses
write_vtk(white,'./cone_data/subject1_lh_white.vtk')
write_lines_to_vtk(pairs, f"./cone_data/pair_line.vtk")
write_points_to_vtk(not_intersect,f"./cone_data/not_intersect_point.vtk",color=(255,0,0))
