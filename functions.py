import os
import subprocess

def print_files_in_folder(folder_path):
    nii_files = []
    try:
        if os.path.isdir(folder_path):
            for root, dirs, files in os.walk(folder_path):
                for file in files:
                    f = os.path.join(root, file)
                    if file.endswith('.nii'):
                        print(f)
                        nii_files.append(f)
        else:
            print(f"{folder_path} is not a directory.")
    except Exception as e:
        print(f"An error occurred: {e}")
    return nii_files

def run_recon_all_on_files(nii_files, subject_dir):
    try:
        source_cmd = 'source $FREESURFER_HOME/SetUpFreeSurfer.sh && '
        
        for nii_file in nii_files:
            subject_name = os.path.basename(nii_file).replace('.nii', '')
            recon_all_cmd = f"recon-all -i {nii_file} -s {subject_name} -all -sd {subject_dir}"
            full_cmd = source_cmd + recon_all_cmd
            
            print(f"Running: {full_cmd}")
            result = subprocess.run(full_cmd, shell=True, check=True, executable='/bin/bash', stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            print(result.stdout.decode())
    except subprocess.CalledProcessError as e:
        print(f"Error running recon-all: {e.stderr.decode()}")
