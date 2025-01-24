from functions.F_plotting2 import *
from IPython.display import clear_output
import pickle
clear_output(wait=True)
info_file=os.path.join('./data/acq_data_parameters_cgg_correct.pkl')
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
################# DENISE forward modelling parameters
pars.update({
        'NNODES': 1,'NPROCX':2,'NPROCY':1,      # small
        'ncores':20,'HOURS':24
        })
################# seismic data filtering parameters
pars.update({'full_band':False})    # high-pass filter for seismic data 
pars.update({'corner_frequency':4} )
pars.update({'delete_low_freqs':True} )

res_folder='./data/gradients/'+pars['data_mode']
os.makedirs(res_folder,exist_ok=True)
processing_batch_size=10
sample_list=list(np.arange(0,2,1))
batches=[sample_list[x:x+processing_batch_size]   for x in range(0, len(sample_list), processing_batch_size)]
list_of_sbatch_files=[]
for batch in batches:
    for number in batch:
        velocity_model_name=number
        directory='model_'+str(number)
        results_folder = os.path.join(res_folder,directory)
        filename_model = os.path.join(results_folder,directory+'.hdf5')
        os.makedirs(results_folder,exist_ok=True)
        ########################
        # generated_model,initial_model,water_taper=velocity_generator('generator1',dh=25,out_shape=pars['out_shape'])
        # generated_model,initial_model,water_taper=velocity_generator('test','model__Marmousi',dh=calculation_spacing,out_shape=pars['out_shape'])
        # generated_model,initial_model,water_taper=velocity_generator('test','model__Overthrust',dh=calculation_spacing,out_shape=pars['out_shape'])
        # generated_model,initial_model,water_taper=velocity_generator('test','model__Seam',dh=calculation_spacing,out_shape=pars['out_shape'])
        # generated_model,initial_model,water_taper=velocity_generator('test','model__Marmousi_linear_initial',dh=calculation_spacing,out_shape=pars['out_shape'])
        # generated_model,initial_model,water_taper=velocity_generator('test','model__Overthrust_linear_initial',dh=calculation_spacing,out_shape=pars['out_shape'])
        # generated_model,initial_model,water_taper=velocity_generator('test','model__Seam',dh=calculation_spacing,out_shape=pars['out_shape'])
        ########################   create folders and files to launch denise
        # generate_data_for_batch_of_jobs(generator_,denise_fwi,'test',res_folder,calculation_spacing,pars,'_Seam2')
        generate_data_for_batch_of_jobs(velocity_generator,denise_fwi,pars['gen_mode'],res_folder,calculation_spacing,pars,number)
        denise_fwi(filename_model,results_folder,os.getcwd(),calculation_spacing=calculation_spacing,pars=pars,mode='generate_task_files')
        ########################   create field_data_processing file
        # denise_folder_process('crop_zero_freqs',os.path.join(results_folder))
        # denise_folder_process('plot',os.path.join(results_folder))
        # exit()
        field_data_processing = imports
        field_data_processing = field_data_processing + f"denise_folder_process('crop_zero_freqs','{os.path.join(results_folder)}',pars={pars})\n"
        field_data_processing_script_name = os.path.join(res_folder, directory, 'field_data_processing_script.py')
        f = open(field_data_processing_script_name, 'w')
        f.write(field_data_processing)
        f.close()

generated_model,initial_model,water_taper=generator_to_use(gen_mode,model_name,dh=calculation_spacing,out_shape=pars['out_shape'])
