import numpy as np
from tqdm import tqdm
import trimesh
from queue import PriorityQueue
import matplotlib.pyplot as plt
from make_vtk import write_lines_to_vtk,write_points_to_vtk
from vtk_revise import read_vtk,write_vtk
from faster_generate_cone import generate_vectors_in_cone
# import cupy as cp

white = read_vtk('./AllCortexData/GS_CortexODE/SUBJ_001_MR_BL/surf/lh.CortexODE.white.vtk')
pial = read_vtk('./AllCortexData/GS_CortexODE/SUBJ_001_MR_BL/surf/lh.CortexODE.pial.vtk')

white_vertices = white['vertices']
white_faces = white['faces'][:,1:]

pial_vertices = pial['vertices']
pial_faces = pial['faces'][:,1:]


import trimesh
import numpy as np

# Load your target mesh
origin_mesh = trimesh.Trimesh(vertices=white_vertices, faces=white_faces)
target_mesh = trimesh.Trimesh(vertices=pial_vertices, faces=pial_faces)

# origin vertex normals
origin_directions = origin_mesh.vertex_normals.copy()
white2pial_dir = pial_vertices - white_vertices
dir_sign = (np.sum(white2pial_dir*origin_directions,axis=1) < 0)
origin_directions[dir_sign] *= -1
    

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

# Now, process the results
pairs = []
thicknesses = []
not_intersect = []

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
