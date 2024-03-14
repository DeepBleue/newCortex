import numpy as np
from tqdm import tqdm
import trimesh
import matplotlib.pyplot as plt
from make_vtk import write_lines_to_vtk,write_points_to_vtk
from vtk_revise import read_vtk,write_vtk
from generate_interpolate_vector import generate_multiple_vectors
from util import normalize_points
import pyautogui



def get_thick_norm_and_ori(file_name,white,pial,separate_num):
    white_vertices = white['vertices']
    white_faces = white['faces'][:,1:]

    pial_vertices = pial['vertices']
    pial_faces = pial['faces'][:,1:]

    threshold = max(white['thickness'])

    origin_mesh = trimesh.Trimesh(vertices=white_vertices, faces=white_faces)
    target_mesh = trimesh.Trimesh(vertices=pial_vertices, faces=pial_faces)

    origin_directions = origin_mesh.vertex_normals.copy()
    white2pial_dir = normalize_points(pial_vertices - white_vertices)

    dir_sign = (np.sum(white2pial_dir*origin_directions,axis=1) < 0)
    origin_directions[dir_sign] *= -1


    all_origins = []

    print('--- Generate Multiple Vectors ---')

    all_directions = generate_multiple_vectors(origin_directions,white2pial_dir , separate_num)
    for idx in range(len(white_vertices)):
        origin_pos = white_vertices[idx]
        all_origins.extend([origin_pos] * separate_num)

    print('--- Search intersect location ---')
    locations, index_ray, index_tri = target_mesh.ray.intersects_location(
        ray_origins=all_origins, 
        ray_directions=all_directions)
    
    pairs = []
    thicknesses = []
    not_intersect = []
    not_intersect_idx = []
    too_long_pair = []

    index_groups = {}

    print('--- Data setting ---')

    for i, ray_idx in tqdm(enumerate(index_ray), total=len(index_ray)):
        if ray_idx not in index_groups:
            index_groups[ray_idx] = []
        index_groups[ray_idx].append(i)



    print('--- Calculating ---')
    
    for idx in tqdm(range(0, len(all_origins), separate_num)):

        real_idx_2 = []
        for i in range(idx, idx+separate_num):
            real_idx_2.extend(index_groups.get(i, []))
            
        subset_locations = locations[real_idx_2]

        subset_origin = all_origins[idx]
        
        distances = [np.linalg.norm(loc - subset_origin) for loc in subset_locations]

        if distances:
            min_distance_index = np.argmin(distances)
            closest_location = subset_locations[min_distance_index]
            closest_distance = distances[min_distance_index]
            
            
            if closest_distance > threshold:
                too_long_pair.append((subset_origin, closest_location))
                not_intersect.append(subset_origin)
                real_idx = int(idx/separate_num)
                not_intersect_idx.append(real_idx)
                white2pial_dist = np.linalg.norm(white_vertices[real_idx]-pial_vertices[real_idx])
                pairs.append((white_vertices[real_idx], pial_vertices[real_idx]))
                thicknesses.append(white2pial_dist)

                
            else : 
                pairs.append((subset_origin, closest_location))
                thicknesses.append(closest_distance)
        else:
            not_intersect.append(subset_origin)
            not_intersect_idx.append(int(idx/50))
            
            real_idx = int(idx/50)
            not_intersect_idx.append(real_idx)
            white2pial_dist = np.linalg.norm(white_vertices[real_idx]-pial_vertices[real_idx])
            pairs.append((white_vertices[real_idx], pial_vertices[real_idx]))
            thicknesses.append(white2pial_dist)

    nan_count = np.isnan(thicknesses).sum()
    thicknesses = np.array(thicknesses)
    white['new_thickness'] = thicknesses
    not_inter_ori_ray = []
    for i in not_intersect_idx : 
        not_inter_ori_ray.append((white_vertices[i],pial_vertices[i]))


    print(f'num Nan : {nan_count}')
    print(f'not intersect : {len(not_intersect)}')
    print(f'Max thickness : {max(thicknesses)}')
    print(f'min thickness : {min(thicknesses)}')

    write_vtk(white,f'./test/' + file_name)
    write_lines_to_vtk(pairs, f"./test/pair_line_"+file_name+".vtk")
    write_points_to_vtk(not_intersect,f"./test/not_intersect_point_"+file_name+".vtk")
    write_lines_to_vtk(too_long_pair, f"./test/long_line_"+file_name+".vtk")
    write_lines_to_vtk(not_inter_ori_ray, f"./test/not_inter_ori_ray"+file_name+".vtk")
    pyautogui.press('enter')




