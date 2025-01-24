from functions.F_plotting2 import *
from IPython.display import clear_output
import pickle
clear_output(wait=True)

info_file=os.path.join('./data/acq_data_parameters_cgg.pkl')  # 80 sources
info_file=os.path.join('./data/acq_data_parameters_cgg_correct.pkl')
print(info_file)
with open(info_file,'rb') as input:
    acq_data=pickle.load(input)
log_dict=acq_data['log_dict']
log_loc=log_dict['loc']
log=log_dict['data']
idx = int(log_loc / 25)
vh = log_loc * np.ones_like(log)/1000

########################   parameters
launch_jobs  =0
record_tasks =1
run_fwi_flag =0
parallel_processing=1
flag_plotting=1
calculation_spacing=25
#################   FWI simulation parameters
pars={'data_mode':'cnn_16'}
pars.update({'current_data_type':'record_cnn_data'} )
#################   velocity model generator parameters
pars.update({'dx': calculation_spacing, 'dz': calculation_spacing,
    'out_shape':[496,150],  # grid size of generated model
    'dsrc':200, # source spacing
    'taper_shift':0,    #shift of the water taper
    'extend_model_x':False,
    'last_source_position':'nx',
    'computation_platform':'workstation',
    'gen_mode':'generator1',     # 'vlad' or 'oleg' option
    'initial_velocity_models_source':'generator',
    'root_denise':'',
    'data_gen_mode':'pseudo_field'
    })
################# seismic data filtering parameters
corner_frequency=5  # boundary of low frequencies
pars.update({'full_band':False})    # high-pass filter for seismic data 
pars.update({'corner_frequency':corner_frequency} )
pars.update({'delete_low_freqs':True} )
################# DENISE forward modelling parameters
pars.update({
        'NNODES': 1,'NPROCX':2,'NPROCY':1,      # small
        'ncores':2,'HOURS':24
        })

res_folder='./data/gradients/'+pars['data_mode']
os.makedirs(res_folder,exist_ok=True)
processing_batch_size=10
sample_list=list(np.arange(0,2,1))
batches=[sample_list[x:x+processing_batch_size]   for x in range(0, len(sample_list), processing_batch_size)]
list_of_sbatch_files=[]

clear_output()
list_of_sbatch_files=[]
for batch in batches:
    for number in batch:
        velocity_model_name=number
        directory='model_'+str(number)
        results_folder = os.path.join(res_folder,directory)
        filename_model = os.path.join(results_folder,directory+'.hdf5')
        os.makedirs(results_folder,exist_ok=True)
        ########################   create folders and files to launch denise
        generate_data_for_batch_of_jobs(velocity_generator,denise_fwi,pars['gen_mode'],res_folder,calculation_spacing,pars,number)
        denise_fwi(filename_model,results_folder,os.getcwd(),calculation_spacing=calculation_spacing,pars=pars,mode='generate_task_files')
        os.system('export GEN='+os.path.join(results_folder,'fld'))
        # os.system('export DENISE=./DENISE-Black-Edition\n')
        os.system('export DENISE=./data_generation/DENISE-Black-Edition\n')
        os.system(f"mpirun -np {pars['ncores']} $DENISE/bin/denise $GEN/seis_forward.inp $GEN/seis_fwi.inp\n")
        # os.system(f"mpirun -np {pars['ncores']} $DENISE/bin/denise $GEN/seis_inversion.inp $GEN/seis_fwi.inp\n")
        ss=1
        ########################   create field data processing file
        # if pars['computation_platform']=='workstation':
        #     str1 = '#!/bin/bash\n'
        #     str1=str1+'export DENISE=./DENISE-Black-Edition\n'
        #     str1=str1+f"source ~/.bashrc\n"
        #     str1=str1+f"conda activate lw\n"
        #     str1=str1+f"which python\n"
        #     str1=str1+'export GEN='+os.path.join(results_folder,'fld')+'\n'
        #     #######
        #     str1 = str1 + f"python {os.path.join(results_folder,'data_generation_script.py')}\n"
        #     str1=str1+f"mpirun -np {pars['ncores']} $DENISE/bin/denise $GEN/seis_forward.inp $GEN/seis_fwi.inp\n"
        #     str1 = str1 + f"python {os.path.join(results_folder,'field_data_processing_script.py')}\n"
        #     str1=str1+f"mpirun -np {pars['ncores']} $DENISE/bin/denise $GEN/seis_inversion.inp $GEN/seis_fwi.inp\n"
        #     str1 = str1 + f"python {os.path.join(results_folder,'post_processing_script.py')}\n"
print('!!!!!!!!halas!!!!!!!!')