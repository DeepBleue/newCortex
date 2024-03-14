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
    
    

    # Load the origin and target white_faces
    origin_mesh = trimesh.Trimesh(vertices=white_vertices, faces=white_faces)
    target_mesh = trimesh.Trimesh(vertices=pial_vertices, faces=pial_faces)

    # Create intersectors for both meshes
    origin_intersector = trimesh.ray.ray_pyembree.RayMeshIntersector(origin_mesh)
    target_intersector = trimesh.ray.ray_pyembree.RayMeshIntersector(target_mesh)

    # For each vertex in the origin mesh, shoot a ray in the direction of its normal
    origins = origin_mesh.vertices
    directions = origin_mesh.vertex_normals.copy()
    white2pial_dir = pial_vertices - white_vertices
    dir_sign = (np.sum(white2pial_dir*directions,axis=1) < 0)
    directions[dir_sign] *= -1
    

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
    pair = []

    self_intersect_idx = []
    not_intersect_idx = []

    too_long = []
    too_long_pair = []



    self_intersect_pos = self_locations                   # ( 530738 , 3 ) 
    self_origin_pos = origins[self_index_ray]             # ( 530738 , 3 ) 
    self_directions_vector = directions[self_index_ray]   # ( 530738 , 3 ) 

    self_distances = np.sqrt(np.sum((self_intersect_pos - self_origin_pos)**2,axis=1)) 


    intersect_pos = locations                    # ( 530738 , 3 ) 
    origin_pos = origins[index_ray]              # ( 530738 , 3 ) 
    directions_vector = directions[index_ray]    # ( 530738 , 3 ) 


    distances = np.sqrt(np.sum((intersect_pos - origin_pos)**2,axis=1))   # ( 530738 ,  ) 

    print(f'--- Subject number {subject_num} , {left_or_right} brain get thickness ( threshold : {threshold}) ---')

    for i in tqdm(range(len(origins))):
    

        if i in index_ray :     # white에서 pial intersect 있을 때 
            white2pial_d = white2pial_dir[i]
            
            idx = np.where(index_ray == i)[0]                           # 530738개 중에 i를 가지는 index 
            temp_dist = distances[idx]                                  # i로 이루어진 거리들
            temp_direction_vetors = directions_vector[idx]              # i로 이루어진 방향들
            real_sign = np.dot(temp_direction_vetors,white2pial_d) > 0  # 기존의 방향과 우리가 구한 방향이 같은 방향이라면 양수 아니면 음수 ( 기존 방향 : white 2 pial )
            positive_count = np.sum(real_sign)                          # 맞는 방향으로 intersect하는 친구들의 갯수 
            
            
            if positive_count == 0:                                     # 만일 맞는 방향으로 intersect하는 친구가 없다면 continue
                thickness.append(float('nan'))
                not_intersect_idx.append(i)
                continue 
        
            else :                                                      # 맞는 방향으로 통과한다면, 그중 가장 작은 친구 선택.
                smallest_index = np.nonzero(real_sign)[0][np.argmin(temp_dist[real_sign])]
                smallest_dist = temp_dist[smallest_index]
                real_idx = idx[smallest_index]
            

            if i in self_index_ray:                                                    # self intersect 할때
                
                self_idx = np.where(self_index_ray == i)[0]
                self_temp_dist = self_distances[self_idx]
                temp_self_direction_vetors = self_directions_vector[self_idx]
                self_sign_mask = np.dot(temp_self_direction_vetors,white2pial_d) > 0     # self intersect 하는 친구가 맞는 방향이라면
                
                self_positive_count = np.sum(self_sign_mask)     # self intersect하는 친구들 갯수
                
                if self_positive_count == 0 :        # 방향이 맞지 않아 self intersect하는 친구들이 없다면 thickness 저장.
                
                    if smallest_dist > threshold: 
                        thickness.append(float('nan'))
                        too_long.append(i)
                        too_long_pair.append((origin_pos[real_idx],intersect_pos[real_idx]))
                        
                    else : 
                        thickness.append(smallest_dist)
                        pair.append((origin_pos[real_idx],intersect_pos[real_idx]))
                    
                else : # self intersect하는 친구들이 있다면 
                    self_mask_index = np.nonzero(self_sign_mask)[0][np.argmin(self_temp_dist[self_sign_mask])]
                    self_smallest_dist= self_temp_dist[self_mask_index]
                

                    # if self_smallest_dist < smallest_dist : 
                    #     thickness.append(float('nan'))
                    #     self_intersect_idx.append(i)
                
                    if  smallest_dist > threshold: 
                        thickness.append(float('nan'))
                        too_long.append(i)
                        too_long_pair.append((origin_pos[real_idx],intersect_pos[real_idx]))

                    else : 
                        thickness.append(smallest_dist)
                        pair.append((origin_pos[real_idx],intersect_pos[real_idx]))

                
            else : # 애초에 self intersect 하는 친구가 없다면,
                
                if smallest_dist > threshold: 
                    thickness.append(float('nan'))
                    too_long.append(i)
                    too_long_pair.append((origin_pos[real_idx],intersect_pos[real_idx]))
                else : 
                    thickness.append(smallest_dist)
                    pair.append((origin_pos[real_idx],intersect_pos[real_idx]))
                
        else : # 아무것도 intersect하지 않을 경우.
            thickness.append(float('nan'))
            not_intersect_idx.append(i)

    thickness = np.array(thickness)
    white['new_thickness'] = thickness
    print(thickness)
    
    file_name = './vtk_file/subj_' + str(subject_num) + f'_{left_or_right}_white.vtk'
    
    write_vtk(white,file_name)
    
