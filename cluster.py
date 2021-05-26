import subprocess

def submit_slurm_job(slurm_folder,field,rank,job_command,hours):
    job_file_name = '%s/%s_rank_%d.slurm'%(slurm_folder,field,rank)
    job_file = open(job_file_name,'w')
    job_file.write('#!/bin/bash\n')
    job_file.write('#SBATCH -N 1\n')
    job_file.write('#SBATCH --cpus-per-task=30\n')
    job_file.write('#SBATCH --mem=256GB\n')
    job_file.write('#SBATCH -t %d:00:00\n'%hours)
    job_file.write('#SBATCH --output=%s/%s_rank_%d_logs.txt\n'%(slurm_folder,field,rank))
    job_file.write('#SBATCH --error=%s/%s_rank_%d_logs.txt\n'%(slurm_folder,field,rank))
    job_file.write('#SBATCH --mail-type=FAIL\n')
    job_file.write('#SBATCH --mail-user=weit@cs.princeton.edu\n')
    
    job_file.write('export OMP_NUM_THREADS=16\n')
    job_file.write('source ~/.bashrc\n')
    job_file.write('conda deactivate\n')
    job_file.write('conda activate qenv\n')
    job_file.write('module load gurobi\n')
    job_file.write('module load intel\n')
    job_file.write('cd /n/fs/weit-proj/circuit_cutting\n')
    job_file.write('%s'%job_command)

    job_file.close()
    subprocess.run(['rm','%s/%s_rank_%d_logs.txt'%(slurm_folder,field,rank)])
    subprocess.run(['chmod','755','%s'%job_file_name])
    subprocess.run(['sbatch','%s'%job_file_name])