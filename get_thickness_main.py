import os
from tqdm import tqdm
import re
from vtk_revise import read_vtk
from get_thickness3 import get_thickness



rootDir = './AllCortexData'
threshold =  7


for dirName, subdirList, fileList in os.walk(rootDir):
    if 'BL' and 'surf' in dirName and 'FU' not in dirName:
        read_dir_name = dirName.replace("\\", "/")
        subject_num = int(re.findall(r'\d+', read_dir_name)[0])

        # if subject_num == 11 :
        #     break
        

        
        for fname in fileList:
            if 'vtk' in fname:
                if 'lh' in fname : 
                    if 'white' in fname : 
                        full_file_name_lh_white = read_dir_name + '/' + fname
                        # print(full_file_name_lh_white)
                        
                    elif 'pial' in fname : 
                        full_file_name_lh_pial = read_dir_name + '/' + fname
                        # print(full_file_name_lh_pial)
                        
                    
                    
                elif 'rh' in fname : 
                    if 'white' in fname : 
                        full_file_name_rh_white = read_dir_name + '/' + fname
                        # print(full_file_name_rh_white)
                    elif 'pial' in fname : 
                        full_file_name_rh_pial = read_dir_name + '/' + fname
                        # print(full_file_name_rh_pial)
        
        lh_white = read_vtk(full_file_name_lh_white) 
        lh_pial = read_vtk(full_file_name_lh_pial)  
        
        get_thickness(subject_num,'lh',lh_white,lh_pial,threshold)
        
        rh_white = read_vtk(full_file_name_rh_white)  
        rh_pial = read_vtk(full_file_name_rh_pial)  

        get_thickness(subject_num,'rh',rh_white,rh_pial,threshold)
