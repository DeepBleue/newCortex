import trimesh
import numpy as np
from tqdm import tqdm
from vtk_revise import write_vtk
# (in_dic, file, vertices='vertices'):

def get_thickness(subject_num,left_or_right,white,pial,threshold):
    white_vertices = white['vertices']
    white_faces = white['faces'][:,1:]

    pial_vertices = pial['vertices']
    pial_faces = pial['faces'][:,1:]
    
    white2pial_dir = white_vertices - pial_vertices

    # Create the meshes
    origin_mesh = trimesh.Trimesh(vertices=white_vertices, faces=white_faces)
    target_mesh = trimesh.Trimesh(vertices=pial_vertices, faces=pial_faces)

    # Create intersectors for both meshes
    origin_intersector = trimesh.ray.ray_pyembree.RayMeshIntersector(origin_mesh)
    target_intersector = trimesh.ray.ray_pyembree.RayMeshIntersector(target_mesh)

    # For each vertex in the origin mesh, shoot a ray in the direction of its normal
    origins = origin_mesh.vertices
    directions = origin_mesh.vertex_normals

    # Get self - intersections   # trimesh는 스스로 겹치는 것은 취급을 안한다. 
    self_locations, self_index_ray, self_index_tri = origin_intersector.intersects_location(
        origins,
        directions, 
        multiple_hits=True)

    # Get intersections
    locations, index_ray, index_tri = target_intersector.intersects_location(
        origins,
        directions, 
        multiple_hits=True)

    thickness = []
    self_intersect_idx = []
    not_intersect_idx = []

    self_intersect_pos = self_locations
    self_origin_pos = origins[self_index_ray]
    self_directions_vector = directions[self_index_ray]

    self_distances = np.sqrt(np.sum((self_intersect_pos - self_origin_pos)**2,axis=1)) 


    intersect_pos = locations                    # ( 530738 , 3 ) 
    origin_pos = origins[index_ray]              # ( 530738 , 3 ) 
    directions_vector = directions[index_ray]    # ( 530738 , 3 ) 

    distances = np.sqrt(np.sum((intersect_pos - origin_pos)**2,axis=1))   # ( 530738 ,  ) 

    pair = []
    too_long = []
    too_long_pair = []
    
    print(f'--- Subject number {subject_num} , {left_or_right} brain get thickness ( threshold : {threshold}) ---')

    for i in tqdm(range(len(origins))):

        if i in index_ray : 

            if i in self_index_ray: 
                self_idx = np.where(self_index_ray == i)[0]
                self_temp_dist = self_distances[self_idx]
                self_smallest_dist= np.min(self_temp_dist)

                idx = np.where(index_ray == i)[0]
                temp_dist = distances[idx]
                smallest_dist= np.min(temp_dist)
                smallest_dist_idx = np.argmin(temp_dist)
                real_idx = idx[smallest_dist_idx]

            
                if self_smallest_dist < smallest_dist : 
                    thickness.append(np.nan)
                    self_intersect_idx.append(i)

                elif smallest_dist > threshold: 
                    thickness.append(np.nan)
                    too_long.append(i)
                    too_long_pair.append((origin_pos[real_idx],intersect_pos[real_idx]))

                else : 
                    thickness.append(smallest_dist)
                    pair.append((origin_pos[real_idx],intersect_pos[real_idx]))

            else : 
                idx = np.where(index_ray == i)[0]
                temp_dist = distances[idx]
                smallest_dist= np.min(temp_dist)
                smallest_dist_idx = np.argmin(temp_dist)
                real_idx = idx[smallest_dist_idx]

                if smallest_dist > threshold: 
                    thickness.append(np.nan)
                    too_long.append(i)
                    too_long_pair.append((origin_pos[real_idx],intersect_pos[real_idx]))

                else : 
                    thickness.append(smallest_dist)
                    pair.append((origin_pos[real_idx],intersect_pos[real_idx]))

        else : 
            thickness.append(np.nan)
            not_intersect_idx.append(i)

    thickness = np.array(thickness)
    white['new_thickness'] = thickness.astype(np.float64)
    
    file_name = './vtk_file/subj_' + str(subject_num) + f'_{left_or_right}_white.vtk'
    
    write_vtk(white,file_name)
    
