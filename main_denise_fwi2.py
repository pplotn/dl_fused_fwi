from genericpath import exists
import sys
import os
import subprocess
import multiprocessing
import shutil
from F_utils import *
from F_plotting import *
from F_fwi import *
import fnmatch
from glob import glob
import numpy as np
import functions.pyapi_denise as api
import numpy as np
# import pyapi_denise as api
from pathlib import Path
import time
from functools import partial
sys.path.append('./for_pasha/utils')
from utils import shared as sd
from utils import shared as sd
from utils import loaders as ld
from utils import vis
from F_plotting2 import *
#################   recording log file
log_save_const=F_calculate_log_number('./logs','log','')
log_path='./logs/log'+str(log_save_const)
os.makedirs(log_path,exist_ok=True)
f = open(os.path.join(log_path,'log'+str(log_save_const)),'w')
sys.stdout = Tee(sys.stdout,f)
print('Writing log to '+log_path)
# fwi_data_path='/lustre2/project/k1404/pavel/DENISE-Black-Edition'
fwi_data_path='/home/plotnips/Dropbox/Log_extrapolation/scripts/DENISE-Black-Edition-master'
#################
info_file=os.path.join('./for_pasha/acq_data_parameters_cgg.pkl')  # 80 sources
# info_file=os.path.join('./','for_pasha','acq_data_parameters_cgg_wider.pkl')  # 95 sources
with open(info_file,'rb') as input:
    acq_data=pickle.load(input)
log_dict=acq_data['log_dict']
log_loc=log_dict['loc']
log=log_dict['data']
idx = int(log_loc / 25)
vh = log_loc * np.ones_like(log)/1000
################# plot initial models from different stages
flag_plot_initial_models=0
flag_plot_fwi_results=0
flag_plot_fwi_results4=1
################# plot initial models from different stages
if flag_plot_initial_models==1:
    # marm=os.path.join('./fwi/cgg_real_data/fwi_56_strategy_l5multi_cnn_13_special_weight_236_0_model__Overthrust_1d_lin_300_f_z_2')
    # over=os.path.join('./fwi/cgg_real_data/fwi_56_strategy_l5_236_0_st1_1')
    # './fwi/multi/multi_cnn_13_special_weight_236_0/model__Marmousi_1d_lin_300_f_z/stage1'
    # './fwi/multi/multi_cnn_13_special_weight_236_0/model__Overthrust_1d_lin_300_f_z/stage2'

    marm=os.path.join(fwi_data_path,'./fwi/multi/multi_cnn_13_special_weight_236_0/model__Marmousi_1d_lin_300_f_z')
    over=os.path.join(fwi_data_path,'./fwi/multi/multi_cnn_13_special_weight_236_0/model__Overthrust_1d_lin_300_f_z')
    cgg=os.path.join(fwi_data_path,'./fwi/multi/multi_cnn_13_special_weight_236_2/model__cgg_lin_vp_long_300_f_z')

    # alternative function for plotting below=plot_init_models_from_seq_cnn_application
    plot_3_init_models_in_column(marm,log_path,last_stage=1)
    plot_3_init_models_in_column(over,log_path,last_stage=3) 

    # alternative function for plotting below=plot_init_models_from_seq_cnn_application_cgg
    plot_3_init_models_in_column(cgg,log_path,last_stage=4)
#################   cgg visualization
if flag_plot_fwi_results==1:
    save_path='/home/plotnips/Dropbox/Apps/Overleaf/draft_Geophysics_FWI_init_model_prediction/paper_geophysics/Fig/log1402'
    save_path='./pictures_geophysics'
    results_root=os.path.join(fwi_data_path,'./fwi/cgg_real_data')      #cgg results
    # results_root=os.path.join(fwi_data_path,'./fwi')      #for marm over
    # results_root='./mtl_low-work'
    paths_2=next(os.walk(os.path.join(results_root)))[1]
    paths_=[]
    for p_ in paths_2:
        paths_.append(os.path.join(results_root,p_))
    paths_=sorted(paths_,key=os.path.getmtime)     #2nd approach
    ##################   Vyborka11,  21.12.21    
    # simulation_folders=paths_[-7:]
    simulation_folders=paths_[-50:]
    #################
    paths_not_to_process = fnmatch.filter(simulation_folders,'*intermediate_smoothing*')+fnmatch.filter(simulation_folders,'*gen3*')+fnmatch.filter(simulation_folders,'*.pkl*')    #+fnmatch.filter(simulation_folders,'*cnn_*')
    simulation_folders = list(set(simulation_folders)-set(paths_not_to_process))
    simulation_folders = list(set(simulation_folders)-set(['./fwi/cgg_real_data','./fwi/all_folders_result_pictures']))
    simulation_folders=sorted(simulation_folders,key=os.path.getmtime)
    ####################################    cgg, final plotting for the paper!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    simulation_folders=[
                        'fwi_60_strategy_fullband_l5_236_2_st0_1',      #95 sources
                        'fwi_60_strategy_fullband_l5_del_low_gen__5hz_0',      #80 sources
                        'fwi_56_strategy_l5_236_2_st0_3',       #95 sources
                        'fwi_56_strategy_l5_del_low_gen__5hz_3',       #80 sources
                        'fwi_56_strategy_l5multi_cnn_13_special_weight_236_2_model__cgg_lin_vp_long_300_f_z_1',       #80 sources
                        ]
    #########
    # simulation_folders=['fwi_56_strategy_l5multi_cnn_13_special_weight_675_4_model___cgg_lin_vp_long_300_f_z_stage3'
    #     # ,'fwi_56_strategy_l5multi_cnn_13_special_weight_236_1_model___cgg_lin_vp_long_300_f_z_stage1'
    #     'fwi_56_strategy_l5multi_cnn_13_special_weight_147_7_model___cgg_lin_vp_long_300_f_z_stage4',
    #     'fwi_56_strategy_l5_multi_cnn_13_special_weight_236_2_model__cgg_lin_vp_long_300_f_z_stage1_1']
    ################# First submission of the paper 27.05.22
    # simulation_folders=[
    #                     'fwi_56_strategy_l5_del_low_gen__5hz_3',    # cgg 1d model
    #                     'fwi_56_strategy_l5multi_cnn_13_special_weight_236_2_model__cgg_lin_vp_long_300_f_z_1', #cgg predicted previous chosen result problem with logic prediction not so accurate as 1d
    #                     'fwi_56_strategy_l5multi_cnn_13_special_weight_147_7_model___cgg_lin_vp_long_300_f_z_stage4',
    #                     'fwi_56_strategy_l5multi_cnn_13_special_weight_675_4_model___cgg_lin_vp_long_300_f_z_stage3',
    #                     ]
    simulation_folders=[
                        'fwi_60_strategy_fullband_l5_236_2_st0_1',
                        'fwi_60_strategy_fullband_l5_del_low_gen__5hz_0',      #80 sources
                        'fwi_56_strategy_l5_236_2_st0_3',       #95 sources
                        'fwi_56_strategy_l5_del_low_gen__5hz_3',    # cgg 1d model
                        'fwi_56_strategy_l5multi_cnn_13_special_weight_236_2_model__cgg_lin_vp_long_300_f_z_1', #cgg predicted previous chosen result problem with logic prediction not so accurate as 1d
                        ]
    #################
    # # wch=os.listdir(results_root)
    # # wch=sorted(wch)
    # # simulation_folders=[wch[155]]+wch[159:162]+wch[225:235]+wch[225:235]+wch[289:(289+43)]
    # wch=os.listdir(results_root)
    # wch=sorted(wch)
    # wch2=fnmatch.filter(wch,'*cgg_lin_vp_long_300*')
    # wch3=fnmatch.filter(wch2,'*236*')
    # simulation_folders=wch3lkl
    ####################################   marm over folders
    # THINK ABOUT IT !!!!
    # 'fwi_56_strategy_l5_del_low_weight_147_5hz_0'
    # simulation_folders=['ws_fwi_56_strategy_l5_5hz_7']
    # simulation_folders=['ws_fwi_56_strategy_l5_5hz_8']
    ####################################
    save_path='./pictures_geophysics'
    # save_path=log_path
    single_thread=1
    if results_root==os.path.join(fwi_data_path,'./fwi'):
        if single_thread==1:
            for simulation_folder in simulation_folders:
                fwi_results_visualization(log_path,save_path,results_root,simulation_folder)
        else:
            pool = multiprocessing.Pool(multiprocessing.cpu_count()-3)
            temp = partial(fwi_results_visualization,log_path,save_path,results_root)
            result = pool.map(func=temp, iterable=simulation_folders)
            pool.close()
            pool.join()
    elif results_root==os.path.join(fwi_data_path,'./fwi/cgg_real_data'):
        if single_thread==1:
            for simulation_folder in simulation_folders:
                print(simulation_folder)
                # fwi_results_cgg_visualization2(log_path,save_path,results_root,simulation_folder)
                # fwi_results_cgg_visualization2_2(log_path,log_path,results_root,simulation_folder)
                # fwi_results_cgg_visualization2_3(log_path,log_path,results_root,simulation_folder)
                fwi_results_cgg_visualization2_4(log_path,save_path,results_root,simulation_folder)
                ss=1
        else:
            pool = multiprocessing.Pool(multiprocessing.cpu_count()-3)
            temp = partial(fwi_results_cgg_visualization2_2,log_path,log_path,results_root)
            result = pool.map(func=temp, iterable=simulation_folders)
            pool.close()
            pool.join()
################# plot marmousi overthrust
if flag_plot_fwi_results4==1:
    # # save_path='/home/plotnips/Dropbox/Apps/Overleaf/2nd_Edition_Artificially_Intelligent_Earth_Exploration_workshop_30_11_21/Pictures'
    save_path='./pictures_geophysics'
    # log_path=save_path
    ################################    1d, cnn-target model data root
    root1=os.path.join(fwi_data_path,'./fwi','ws_fwi_56_strategy_l5_5hz_7')
    ################################    CNN-predicted model data root
    # data on shaheen, current draft condition
    root2=os.path.join(fwi_data_path,'./fwi/cgg_real_data/fwi_56_strategy_l5multi_cnn_13_special_weight_236_0_model__Overthrust_1d_lin_300_f_z_2')
    root3=os.path.join(fwi_data_path,'./fwi/cgg_real_data/fwi_56_strategy_l5_236_0_st1_1')
    # fwi from stage 1
    # root2=os.path.join(fwi_data_path,'./fwi/cgg_real_data/fwi_56_strategy_l5multi_cnn_13_special_weight_236_0_model__Overthrust_1d_lin_300_f_z_1') # bad,same as _2
    # root2=os.path.join(fwi_data_path,'./fwi/cgg_real_data/fwi_56_strategy_l5multi_cnn_13_special_weight_236_0_model__Overthrust_1d_lin_300_f_z_stage3') # BAD
    root2=os.path.join(fwi_data_path,'./fwi/cgg_real_data/')
    # root2=os.path.join(fwi_data_path,'./fwi/cgg_real_data/')
    ################################    final results
    dir2=os.listdir(root2)
    dir3=os.listdir(root3)
    paths=[
        os.path.join(root1,'Overthrust_1d_lin_300_f_z'),    #row 0
        os.path.join(root2,dir2[0]),    #row 1
        os.path.join(root1,'Overthrust_1d_lin_300_true'),
        ]
    file_name='Overthrust_final.png'
    comparison_initial_models_with_fwi_misfits_9_letters(paths,file_name,log_path=log_path,save_path=save_path)
    ################################
    paths=[
        os.path.join(root1,'Marmousi_1d_lin_300_f_z'),    #row 0
        os.path.join(root3,dir3[0]),    #row 1
        os.path.join(root1,'Marmousi_1d_lin_300_true'),
        ]
    file_name='Marmousi_final.png'
    # comparison_initial_models_with_fwi_misfits_9_letters(paths,file_name,log_path=log_path,save_path=save_path)
    exit()
#################
def flag_reprocess_dataset():
    # files_path=fnmatch.filter(glob(dataset_path+'/*'),'*.npz'); 
    # path_=sorted(files_path)
    old_dataset_folder='./datasets/gen3_marine_pseudofield_data_cnn_13_test_gradhor_start_above_water_09_09_standardization_300'
    old_dataset_folder='./datasets/gen3_marine_pseudofield_data_cnn_13_test_gradhor_start_above_water_09_09_standardization'
    new_dataset_folder='./datasets/gen3_marine_pseudofield_data_cnn_13_standardization_300_0812'
    source_file_list=os.listdir(old_dataset_folder)
    path_test=fnmatch.filter(source_file_list,'*__*')
    source_file_list=list(set(source_file_list)-set(path_test))
    os.makedirs(new_dataset_folder,exist_ok=True)
    api._cmd(f"ls {new_dataset_folder} | wc -l")
    for file_ in source_file_list:
        remake_taper_save_file(new_dataset_folder,old_dataset_folder,file_)
    #     pars={'source_file':file_,'new_dataset_folder':new_dataset_folder,'scaling_constants_dict':scaling_constants_dict}
    ss=1
    exit()
    pool = multiprocessing.Pool(multiprocessing.cpu_count()-3)
    temp = partial(remake_taper_save_file,new_dataset_folder,old_dataset_folder)
    result = pool.map(func=temp, iterable=source_file_list)
    pool.close()
    pool.join()
def flag_process_dataset():
    flag_copy_picture_folders=0
    # parameters
    flag_single_thread_processing=0
    mode = 'check_save_clean'
    mode = 'check_save'
    # mode='none'
    # mode='check'
    # mode='clean_'
    # mode='check_clean_'
    mode='plot'
    # mode='real_data_plotting_session'
    # mode='optimizing_space_plot'
    # mode='copy_pictures'
    # mode='crop_zero_freqs'
    # mode='optimizing_space_plot'
    # mode='check_plot'
    # mode='optimizing_space_'
    # mode='save'
    # mode='delete_pictures_'
    # mode='clean_empty_folders'  # USE CAREFULLY, WITH NO RUNNING JOBS!!! DELETE folders without results, with only .py files. 
    ##################
    results_root = './fwi'
    # results_root = '/home/plotnips/Dropbox/Log_extrapolation/scripts/DENISE-Black-Edition-master/fwi'
    paths_ = sorted(Path(results_root).iterdir(), key=os.path.getmtime)
    paths = []
    for item in paths_:     paths.append(item._str)
    paths_not_to_process = fnmatch.filter(paths, '*intermediate_smoothing*')+fnmatch.filter(paths,'*strategy_12*')+fnmatch.filter(paths, '*strategy_13*')+fnmatch.filter(paths, '*gen3*')
    paths_not_to_process = fnmatch.filter(paths, '*intermediate_smoothing*')+fnmatch.filter(paths, '*gen3*')
    paths_extra_processing=fnmatch.filter(paths,'*intermediate_smoothing*')+fnmatch.filter(paths, '*gen3_7*')
    simulation_folders = paths
    simulation_folders=fnmatch.filter(paths, '*fwi_18_strategy_14*')
    # simulation_folders = list( set(simulation_folders)-set(paths_not_to_process) )
    simulation_folders=['gen3_marine_data_cnn_fwi_strategy_13_nx_1200','gen3_marine_data_cnn_fwi_strategy_13_nx_672']
    simulation_folders=['gen3_marine_data_cnn_fwi_strategy_13_nx_672']
    simulation_folders=['gen3_marine_pseudofield_data_cnn_fwi_strategy_13_pseudo_field_nx_500']
    # simulation_folders=['gen3_marine_data_cnn_fwi_strategy_13_nx_1200']
    ##########################  find paths for processing
    simulation_folders=[
        # 'gen3_marine_pseudofield_data_cnn_fwi_strategy_13_dsrc_325_pseudo_field_nx_500',
        # 'gen3_marine_pseudofield_data_cnn_fwi_strategy_13_pseudo_field_nx_500',
        # 'gen3_marine_data_cnn_fwi_strategy_13_nx_1200',
        # 'gen3_marine_pseudofield_data_cnn_fwi_strategy_13_dsrc_200_pseudo_field_nx_500',
        # 'gen3_marine_pseudofield_data_cnn_fwi_strategy_13_dsrc_200_pseudo_field_nx_496'
        # 'gen3_marine_pseudofield_data_cnn_fwi_strategy_13_dsrc_200_pseudo_field_nx_496_test3',
        # 'gen3_marine_pseudofield_data_cnn_fwi_strategy_13_dsrc_200_pseudo_field_nx_496_random_trends_1st_attempt',
        # 'gen3_marine_pseudofield_data_cnn_fwi_strategy_13_dsrc_200_pseudo_field_nx_496',
        # 'gen3_marine_pseudofield_data_cnn_fwi_strategy_13_dsrc_200_pseudo_field_nx_496_test8_new'
        # 'ws_fwi_19_strategy_zero_freq_no_cropping_7'
        # 'ws_fwi_19_strategy_zero_freq_no_cropping_15'
        # 'ws_fwi_31_strategy_5',
        # 'ws_fwi_31_strategy_6',
        # 'ws_fwi_31_strategy_7',
        # 'ws_fwi_33_strategy_0',
        # 'ws_fwi_33_strategy_1',
        # 'ws_fwi_33_strategy_2',
        # 'ws_fwi_33_strategy_3',
        # 'ws_fwi_33_strategy_4',
        # 'ws_fwi_33_strategy_5',
        # 'ws_fwi_34_strategy_2',
        # 'ws_fwi_35_strategy_4',
        # 'ws_cnn_13_special_1/model__Marmousi_cnn',
        # 'ws_cnn_13_special_1/model__Marmousi_f_z',
        # 'ws_cnn_13_special_1/model__Overthrust_cnn',
        # 'ws_cnn_13_special_1/model__Overthrust_f_z',
        # 'ws_cnn_13_special_1/model__Marmousi_linear_initial_cnn',
        # 'ws_cnn_13_special_1/model__Marmousi_linear_initial_f_z',
        # 'ws_cnn_13_special_1/model__Overthrust_linear_initial_cnn',
        # 'ws_cnn_13_special_1/model__Overthrust_linear_initial_f_z',
        # 'ws_cnn_13_special_1/model__Seam2_cnn',
        # 'ws_cnn_13_special_1/model__Seam2_f_z',
        # 'ws_fwi_32_strategy_0'
        # 'gen3_marine_pseudofield_data_cnn_fwi_strategy_13_dsrc_200_pseudo_field_nx_496_test_no_gradhor'
        'gen3_marine_pseudofield_data_cnn_13_test_gradhor_start_above_water',
        # 'gen3_marine_data_cnn_fwi_strategy_13_nx_1200'
        ]
    # simulation_folders=[
    #     'cgg_real_data/cnn_13_12',
    #     'cgg_real_data/cnn_14_0',
    #     'cgg_real_data/fwi_34_strategy_0',
    #     'cgg_real_data/fwi_35_strategy_0',
    #     'cgg_real_data/fwi_36_strategy_0',
    #     'cgg_real_data/cnn_13_13',
    #     'cgg_real_data/cnn_14_1',
    #     'cgg_real_data/fwi_37_strategy_0',
    #     'cgg_real_data/fwi_38_strategy_oleg_0',
    #     'cgg_real_data/fwi_37_strategy_1',
    #     'cgg_real_data/fwi_38_strategy_oleg_1',
    #     'cgg_real_data/fwi_37_strategy_2',
    #     'cgg_real_data/fwi_38_strategy_oleg_2',
    #     'cgg_real_data/fwi_38_strategy_oleg_3',
    #     'cgg_real_data/fwi_37_strategy_3',
    #     'cgg_real_data/fwi_37_strategy_4',
    #     'cgg_real_data/fwi_38_strategy_oleg_4',
    #     'cgg_real_data/fwi_38_strategy_oleg_5',
    #     'cgg_real_data/fwi_37_strategy_5'
    #     ]
    # simulation_folders=[
    #     # 'gen3_marine_data_cnn_fwi_strategy_13_nx_672',
    #     'gen3_marine_data_cnn_14'
    #     ]
    # simulation_folders=simulation_folders+paths_extra_processing
    
    # simulation_folders=fnmatch.filter(paths,'*generator1*')+fnmatch.filter(paths,'*generator3*')+fnmatch.filter(paths,'*outputs_pseudo_elastic*')+fnmatch.filter(paths,'*ws_fwi_18*')
    # simulation_folders = list( set(simulation_folders)-set(fnmatch.filter(paths,'*ws_fwi_18_strategy_14*')) -set(fnmatch.filter(paths,'*ws_fwi_18_strategy_13*')) )
    # simulation_folders=fnmatch.filter(paths, '*ws_fwi_20_strategy_oleg_zero_freq_no_cropping*')+fnmatch.filter(paths, '*ws_fwi_19_strategy_oleg_zero_freq_no_cropping*')+fnmatch.filter(paths, '*ws_fwi_18_strategy_17_weak_spatfilter*')
    # simulation_folders=fnmatch.filter(paths, '*ws_fwi_18_strategy*')
    # simulation_folders=fnmatch.filter(paths, '*gen3_marine_pseudofield_data_cnn_fwi_strategy_13_dsrc_200_pseudo_field_nx_496*')
    # simulation_folders =['gen3_7_30_hz_data_cnn_fwi_strategy_11_OBN']
    # simulation_folders=fnmatch.filter(paths, '*gen3_marine_data_cnn_fwi_strategy*')
    # simulation_folders=fnmatch.filter(paths, '*gen3_marine_pseudofield_data_cnn_fwi_strategy_13_dsrc_*')    
    # simulation_folders=fnmatch.filter(paths, '*gen3_wddamp_standard*')    
    ##########################  process last folders
    # simulation_folders = paths[-2:]
    # simulation_folders =['ws_fwi_19_strategy_zero_freq_no_cropping_2']
    # simulation_folders=['ws_fwi_20_strategy_oleg_zero_freq_no_cropping_6']
    ##########################  delete extra paths!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # for ii in range( len(simulation_folders) ):
    #     if '/' in simulation_folders[ii]:
    #         simulation_folders[ii]=simulation_folders[ii].split('/')[-1]
    ##########################
    # simulation_folders=['gen3_marine_pseudofield_data_cnn_fwi_strategy_13_dsrc_200_pseudo_field_nx_500']
    # simulation_folders=['ws_fwi_18_strategy_14_18','ws_fwi_18_strategy_14_19','ws_fwi_18_strategy_14_20','ws_fwi_18_strategy_14_21']
    # simulation_folders=['gen3_marine_pseudofield_data_cnn_fwi_strategy_13_dsrc_325_pseudo_field_nx_500']
    # simulation_folders=fnmatch.filter(paths,'*gen3*')
    # simulation_folders=paths[0:292]
    simulation_folders=['ws_fwi_54_strategy_fullband_l2_3']
    ##########################
    # for simulation in simulation_folders:
    #     api._cmd(f"ls {os.path.join('datasets',simulation)} | wc -l\n")
    #     # api._cmd(f"rm -r {os.path.join('./datasets',simulation)} \n")
    #     # api._cmd(f"rm {os.path.join(simulation,'*.png')} \n")
    # exit()
    ##########################
    # results_root = './for_pasha'
    # simulation_folders=['./'];    
    # mode='plot'
    ##########################
    for simulation in simulation_folders:
        if mode=='save':
            pars={'scaling_range':'-11'}
            pars={'scaling_range':'standardization'}
            with open(os.path.join('./fwi','dataset_to_create_09_09.pkl'),'rb') as input:
                data_dict=pickle.load(input)
            # dataset_path = os.path.join('./datasets', simulation)
            # dataset_path=os.path.join('./datasets', simulation+'_better_scaling')
            # dataset_path = os.path.join('./datasets', simulation+'_better_scaling')
            # dataset_path = os.path.join('./datasets', simulation+'_2_')
            # dataset_path = os.path.join('./datasets', simulation+'_3_21_08')
            dataset_path=os.path.join('./datasets', simulation+'_09_09_'+pars['scaling_range']+'_300')
        else:   dataset_path = os.path.join('./datasets','test_code_folder')
        os.makedirs(dataset_path, exist_ok=True)
        # simulation = str(simulation).split('/')[-1] ### !!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        res_folder = os.path.join(results_root, simulation)
        ########################## find directories in the folder
        tmp=os.listdir(res_folder)
        # print('os.listdir(res_folder)=',os.listdir(res_folder))
        dirs=[]
        for dir_ in tmp:
            path_=os.path.join(res_folder,dir_)
            if os.path.isdir(path_):
                dirs.append(dir_)
        ########################## process specific files in res_folder. pick up the file list from different directory
        # tmp=os.listdir(os.path.join('./datasets','gen3_7_30_hz_data_cnn18_sm100_bad_tapers_cropped'))
        # dirs=[]
        # for dir_ in tmp:
        #     dirs.append(os.path.join(dir_[0:-4]))
        # sort files
        dirs=sorted(dirs)
        dirs.reverse()
        dirs = sorted(dirs)
        # with open(os.path.join('./fwi','dataset_to_create.pkl'),'rb') as input:
        # with open(os.path.join('./fwi','dataset_to_create_new.pkl'),'rb') as input:
        with open(os.path.join('./fwi','dataset_to_create_09_09.pkl'),'rb') as input:
            data_dict=pickle.load(input)
        dirs=data_dict['new_dataset_file_list']
        ##########################
        # # dirs=['model_25','model_27']
        # # dirs=['model_6006']
        # # dirs=dirs[-260:]
        test_dirs = fnmatch.filter(dirs, '*__*')
        dirs = list(set(dirs)-set(test_dirs))
        ##########################
        # # if simulation=='generator1':
        # dirs_i_want=[]
        # for i in range(len(dirs)):
        #     pts=dirs[i].split('model_');
        #     s=pts[1]
        #     if RepresentsInt(s):
        #         if int(s)>+19443:        #  8000
        #             dirs_i_want.append('model_'+s)
        # dirs=dirs_i_want
        # dirs2=dirs
        # Verify it works
        # bubble_sort2(dirs2)
        # print(dirs2)
        ##########################      
        if mode=='optimizing_space_':
            dirs_list=dirs[0:]
        else:
            dirs_list=test_dirs+dirs[0:]
        # dirs_list = test_dirs+dirs[0:2]
        # dirs_list=['model_2754']
        # dirs_list=test_dirs+dirs
        # dirs_list=['stage0','stage1','stage2','stage3','stage4','stage5','stage6']
        # dirs_list=dirs_list[0:120]
        # dirs_list=test_dirs
        # dirs_list=['model__Marmousi']
        # dirs_list=['model__Overthrust']
        if flag_copy_picture_folders==1:    #    and mode=='plot'
            os.makedirs(os.path.join(results_root,'all_folders_result_pictures'),exist_ok=True)
            for dir_ in dirs_list:
                tmp=res_folder.split('/')[-1]+'_'+dir_
                api._cmd(f"scp -r {os.path.join(res_folder,dir_,'pictures')} {os.path.join(results_root,'all_folders_result_pictures',tmp)}")
            exit()
        ############################
        if mode=='plot':    pars=None
        if flag_single_thread_processing == 1:
            for dir_ in dirs_list:
                denise_folder_process(mode,os.path.join(res_folder,dir_),save_path=dataset_path,pars=pars)
        else:
            path_list = []
            for dir_ in dirs_list:
                path_list.append(os.path.join(res_folder, dir_))
            pool = multiprocessing.Pool(multiprocessing.cpu_count()-3)
            temp = partial(denise_folder_process, mode, save_path=dataset_path,pars=pars)
            result = pool.map(func=temp, iterable=path_list)
            pool.close()
            pool.join()
        ############################
        if mode=='plot':
            os.makedirs(os.path.join(results_root,'all_folders_result_pictures'),exist_ok=True)
            for dir_ in dirs_list:
                tmp=res_folder.split('/')[-1]+'_'+dir_
                api._cmd(f"scp -r {os.path.join(res_folder,dir_,'pictures')} {os.path.join(results_root,'all_folders_result_pictures',tmp)}")
        ############################
    exit()
def copy_from_to(src_dst):
    print(f"scp -r { src_dst[0] } { src_dst[1] }")
    api._cmd(f"scp -r { src_dst[0] } { src_dst[1] }")
    return None
def flag_process_dataset2():
    # parameters
    flag_copy_picture_folders=0
    flag_single_thread_processing=0
    mode='plot'
    # mode='real_data_plotting_session'
    # mode='optimizing_space_plot'
    # mode='copy_pictures'
    # mode='crop_zero_freqs'
    # mode='optimizing_space_plot'
    # mode='optimizing_space'
    mode='multi_cnn_plotting'
    # mode='check_plot'
    # mode='optimizing_space_'
    # mode='delete_pictures_optimizing_space_'
    # mode='save'
    # mode='delete'
    # mode='delete_certain_folder'
    # mode='delete_pictures_'
    # mode='delete_empty_folders'
    # mode='delete_empty_folders2'
    # mode='clean_everything_except_denise_data_pkl'
    ##################
    if mode=='save':        
        pars={'scaling_range':'standardization'}    # -11
        pars.update({'target_smoothing_diameter':300})
        # pars.update({'target_smoothing_diameter':100})
    else:   pars=None
    results_root = './fwi'
    results_root = './fwi/cgg_real_data'
    results_root = './fwi/multi'
    # results_root='./mtl_low-work'
    # paths_ = sorted(Path(results_root).iterdir(), key=os.path.getmtime)   #1st approach
    paths_2=next(os.walk(os.path.join(results_root)))[1]
    paths_=[]
    for p_ in paths_2:
        paths_.append(os.path.join(results_root,p_))
    paths_=sorted(paths_,key=os.path.getmtime)     #2nd approach
    paths = []
    for item in paths_:
        # paths.append(item._str.split('/')[-1])
        paths.append(item.split('/')[-1])
    # paths.reverse()
    simulation_folders=paths[0:]
    ##################################
    # simulation_folders=simulation_folders[179::]
    ##################   Vyborka8,  16.12.21    
    # simulation_folders=simulation_folders[-40:]
    # simulation_folders=simulation_folders[22:]
    ##################   Vyborka8,  18.12.21    
    # simulation_folders=simulation_folders[-80:]
    # simulation_folders=simulation_folders[38:]
    ##################   Vyborka9,  18.12.21    
    # simulation_folders=simulation_folders[-40:]
    # simulation_folders=simulation_folders[23:]
    ##################   Vyborka9,  18.12.21    
    # simulation_folders=simulation_folders[-10:]
    # simulation_folders=simulation_folders[23:]
    ##################   Vyborka10,  18.12.21    
    # simulation_folders=simulation_folders[-60:]
    # simulation_folders=simulation_folders[49:]
    ##################   Vyborka10,  22.12.21    
    simulation_folders=simulation_folders[-60:]
    simulation_folders=simulation_folders[54:]
    ##################################
    # simulation_folders=simulation_folders[88:]
    # simulation_folders=simulation_folders[0:124]
    # simulation_folders=simulation_folders[0:561]
    # simulation_folders=simulation_folders[-100:]
    # simulation_folders=simulation_folders[80::]
    # ss=1
    # simulation_folders = list(set(simulation_folders)-set(['cgg_real_data']))
    simulation_folders = list(set(simulation_folders)-set(['all_folders_result_pictures']))
    ##################################
    if results_root== './fwi':
        # paths_not_to_process = fnmatch.filter(paths, '*intermediate_smoothing*')+fnmatch.filter(
        #     paths, '*strategy_12*')+fnmatch.filter(paths, '*strategy_13*')+fnmatch.filter(paths, '*gen3*')
        paths_not_to_process = fnmatch.filter(paths,'*intermediate_smoothing*')+fnmatch.filter(paths, '*gen3*')+fnmatch.filter(paths,'*.pkl*')
        simulation_folders = paths
        simulation_folders = list(set(simulation_folders)-set(paths_not_to_process))
        # simulation_folders = fnmatch.filter(paths, '*intermediate_smoothing*')
        # simulation_folders = simulation_folders[12:]
    ##################################
    # simulation_folders = ['gen3_marine_pseudofield_data_cnn_13_test_gradhor_start_above_water']
    # simulation_folders = ['ws_fwi_54_strategy_fullband_l2_0','ws_fwi_54_strategy_fullband_l5_0']
    # # simulation_folders = ['record_cgg_data_cnn_13_for_real_data']
    # # simulation_folders=['multi_cnn_13_weight_1500_3']
    # # # simulation_folders=simulation_folders[0:9]
    # # simulation_folders=simulation_folders[-100:]
    # # # simulation_folders=['weights_845']
    # # # simulation_folders=fnmatch.filter(paths,'*gen3*')
    # # # simulation_folders=paths[0:292]
    ##################################
    # # simulation_folders=['ws_fwi_54_strategy_fullband_l2_3']
    # # simulation_folders=[
    # #                     'ws_fwi_54_strategy_fullband_l2_0',
    # #                     'ws_fwi_54_strategy_fullband_l5_0',
    # #                     'ws_fwi_55_strategy_fullband_l2_1',
    # #                     'ws_fwi_55_strategy_fullband_l5_1',
    # #                     ]
    # # simulation_folders=['cnn_13_special_test_new_2_chosen','cnn_13_special_test_new_1','cnn_13_test_new_5']
    # # simulation_folders=['cnn_13_test_new_5','cnn_15_test_new_1','cnn_14_test_new_1']
    # simulation_folders=['cnn_13_test_new_5_1d_lin_models']
    # # simulation_folders=['gen3_marine_data_cnn_16_dl_12']
    # # simulation_folders=[
    # #                     'ws_fwi_55_strategy_l2_0',
    # #                     'ws_fwi_55_strategy_l5_0',
    # #                     ]
    # # simulation_folders=['fwi_55_strategy_fullband_l5_weight_574_0']
    # # simulation_folders=[
    # #                     'ws_fwi_55_strategy_fullband_l2_7',
    # #                     'ws_fwi_55_strategy_fullband_l5_2',
    # #                     'ws_fwi_55_strategy_fullband_l5_3',
    # #                     ]
    # # simulation_folders=[
    # #                     'fwi_55_strategy_fullband_l2',
    # #                     'fwi_55_strategy_fullband_l2_weight_574_0',
    # #                     'fwi_55_strategy_fullband_l5_weight_574_0',
    # #                     'fwi_55_strategy_fullband_l5',
    # #                     ]
    # simulation_folders=[
    #                     'ws_fwi_56_strategy_l2_5hz_3',
    #                     'ws_fwi_56_strategy_l5_5hz_3',
    #                     'ws_fwi_56_strategy_l2_5hz_4',
    #                     'ws_fwi_56_strategy_l5_5hz_4'
    #                     ]
    ##################################  multi-cnn results plotting
    # simulation_folders=[
    #                     'multi_cnn_13_special_weight_147_14',
    #                     'multi_cnn_13_special_weight_675_5',
    #                     ]
    # simulation_folders=[
    #                     # 'multi_cnn_13_special_weight_245_4',
    #                     'multi_cnn_13_special_weight_355_0',
    #                     ]
    ##################################
    # simulation_folders=['fwi_56_strategy_special_l2_del_low_gen__5hz_0']
    # simulation_folders=['gen3_marine_data_cnn_16_dl_12']
    # simulation_folders=['gen3_marine_data_cnn_16_dl_13']
    # simulation_folders=['gen3_marine_data_cnn_13__test_samples_2nd_run']
    # simulation_folders=['multi_cnn_13_special_weight_147_6']
    # simulation_folders=['ws_fwi_56_strategy_l5_5hz_7']
    simulation_folders=['multi_cnn_13_special_weight_236_2']
    ##################################
    # simulation_folders=[
    #                     'fwi_56_strategy_l2_del_low_gen__5hz_1',
    #                     'fwi_56_strategy_l2_del_low_weight_675_5hz_0',
    #                     'fwi_56_strategy_l2_del_low_weight_147_5hz_0',
    #                     ]
    ##################################
    # construct list of denise folders to process
    denise_folders_list = []
    for simulation in simulation_folders:
        # api._cmd(f"rm -r {os.path.join(results_root, simulation)}\n")
        # continue
        if '.pkl' in simulation:
            continue
        # if 'gen3' in simulation:
        #     continue
        dataset_path=os.path.join('./datasets',simulation)
        if mode=='save':
            dataset_path=os.path.join('./datasets',simulation)
        # simulation = str(simulation).split('/')[-1]
        models_list=next(os.walk(os.path.join(results_root,simulation)))[1]
        models_list = sorted(models_list)
        models_list.reverse()
        test_dirs = fnmatch.filter(models_list, '*__*')
        models_list2 = list(set(models_list)-set(test_dirs))
        models_list2 = test_dirs+models_list2
        # models_list2 = test_dirs
        for model_ in models_list2:
            f_list_=next(os.walk(os.path.join(results_root, simulation, model_)))[1]
            if 'stage0' in f_list_:
                stage_list = fnmatch.filter(f_list_, '*stage*')
                stage_list = sorted(stage_list)
                for stage in stage_list:
                    denise_folders_list.append(os.path.join(results_root, simulation, model_, stage))
            else:
                denise_folders_list.append(os.path.join(results_root, simulation, model_))
            denise_folders_list.append(os.path.join(results_root, simulation, model_))
    # exit()
    #######################################
    if flag_copy_picture_folders==1:    #    and mode=='plot'
        os.makedirs(os.path.join(results_root,'all_folders_result_pictures'),exist_ok=True)
        from_to=[];
        for dir_ in denise_folders_list:
            tmp=dir_.split('/')[-2]+'_'+dir_.split('/')[-1]           
            from_to.append( [os.path.join(dir_,'pictures') , os.path.join('./fwi','all_folders_result_pictures',tmp) ] )
        flag_single_thread_processing=0
        if flag_single_thread_processing == 1:
            for src_dst in from_to:
                copy_from_to(src_dst)
        else:
            pool = multiprocessing.Pool(multiprocessing.cpu_count()-3)
            temp = partial(copy_from_to)
            result = pool.map(func=temp, iterable=from_to)
            pool.close()
            pool.join()
        exit()
    #######################################
    # exit()
    #######################################     performing processing
    if flag_single_thread_processing == 1:
        for dir_ in denise_folders_list:
            denise_folder_process(mode, dir_, save_path=dataset_path,pars=pars)
    else:
        pool = multiprocessing.Pool(multiprocessing.cpu_count()-3)
        temp = partial(denise_folder_process, mode, save_path=dataset_path,pars=pars)
        result = pool.map(func=temp, iterable=denise_folders_list)
        pool.close()
        pool.join()
    #######################################
    # if mode=='plot' or mode=='real_data_plotting_session':    #    and mode=='plot' 
    #     os.makedirs(os.path.join(results_root,'all_folders_result_pictures'),exist_ok=True)
    #     from_to=[];
    #     for dir_ in denise_folders_list:
    #         tmp=dir_.split('/')[-2]+'_'+dir_.split('/')[-1]
    #         from_to.append( [os.path.join(dir_,'pictures') , os.path.join('./fwi','all_folders_result_pictures',tmp) ] )
    #     flag_single_thread_processing=0
    #     if flag_single_thread_processing == 1:
    #         for src_dst in from_to:
    #             copy_from_to(src_dst)
    #     else:
    #         pool = multiprocessing.Pool(multiprocessing.cpu_count()-3)
    #         temp = partial(copy_from_to)
    #         result = pool.map(func=temp, iterable=from_to)
    #         pool.close()
    #         pool.join()
    #     exit()
corner_frequency=5

def flag_run_full_fwi_on_testing_models3():
    """ FWI on synthetic velocity models
        development started on 07.07.21 
        apriori I know the water thickness and ocean bottom thickness """
    run_fwi_flag = 0
    flag_submitjob_or_exec_locally = 1
    calculation_spacing = 25
    vp1 = 1500; vp2 = 4500
    # prediction_path
    # prediction_path='/ibex/scratch/plotnips/intel/MWE/Keras-GAN/pix2pix/logs/log'+str(number)
    # prediction_path='/home/plotnips/Dropbox/Log_extrapolation/scripts/DENISE-Black-Edition-master/predictions/log'+str(number)
    prediction_path = './predictions/predictions_1169'  # 1150
    prediction_path = './predictions/predictions_1189'  
    prediction_path = './predictions/predictions_1192'  
    prediction_path = './predictions/predictions_1202'  
    prediction_path = './predictions/predictions_1204'
    prediction_path = './predictions/predictions_1206'
    prediction_path = './predictions/predictions_1207'
    prediction_path = './predictions/predictions_1227'
    # prediction_path = './predictions/predictions_1230'
    # prediction_path = './predictions/predictions_1235'
    # prediction_path = './predictions/predictions_1239'
    prediction_path = './predictions/predictions_1251'
    prediction_path = './predictions/predictions_1273'
    prediction_path = './predictions/predictions_1294'
    prediction_path = './predictions/predictions_27'
    prediction_path = './predictions/predictions_76'
    prediction_path = './predictions/predictions_1351'
    prediction_path = './predictions/predictions_1400'
    prediction_path = './predictions/predictions_1401'
    prediction_path = './predictions/predictions_1500'
    prediction_path = './predictions/predictions_549'   #smoothing 300
    prediction_path = './predictions/predictions_574'   #smoothing 300 100
    prediction_path = './predictions/predictions_653_sm_100'   #smoothing 100
    prediction_path = './predictions/predictions_75'    #   smoothing 300
    prediction_path = './predictions/predictions_141'    #   smoothing 300
    prediction_path = './predictions/predictions_147'   #these weights used for Marm, Over (300m) log65
    prediction_path = './predictions/predictions_253'   #these weights used for Marm, Over (300m) noise level=0.3
    prediction_path = './predictions/predictions_269'   #these weights used for Marm, Over (300m) noise level=0.0 from 221
    prediction_path = './predictions/predictions_237'   #Marm, Over (300m). GIW=237. noise level=0.0 from 221
    prediction_path = './predictions/predictions_236'   #Marm, Over (300m).  noise level=0.0 from 
    # prediction_path = './predictions/predictions_244'   #Marm, Over (300m).  noise level=0.0 from 
    prediction_path=os.path.join(fwi_data_path,prediction_path)
    file_list=os.listdir(prediction_path)
    prediction_number=prediction_path.split('/')[-1]
    prediction_number=prediction_number.split('predictions_')[-1]
    # choose models to invert
    models = ['Overthrust', 'Marmousi', 'Seam', 'Seam2']
    models = [ 'Marmousi','Overthrust','Seam2']
    # models = [ 'Marmousi','Overthrust','lin_vp_long','Seam2']
    # models=['Overthrust','9774','Seam','Seam2']
    models = ['Overthrust','Marmousi']
    # models = ['Overthrust_linear_initial', 'Marmousi_linear_initial', 'Seam', 'Seam2']
    # models = ['Overthrust','Marmousi','Overthrust_linear_initial', 'Marmousi_linear_initial']
    # models = ['Overthrust','Marmousi','Overthrust_linear_initial', 'Marmousi_linear_initial']
    # models = ['Overthrust_1d_lin_300', 'Marmousi_1d_lin_300']
    # models = ['Overthrust_linear_initial']
    models = ['Overthrust','Marmousi',
            'Overthrust_linear_initial','Marmousi_linear_initial',
            'Overthrust_1d_lin_300', 'Marmousi_1d_lin_300']
    # models = ['Marmousi_1d_lin_300','Overthrust']   # best prediction results in terms of difference between cnn-predicted and 1D model
    ###
    #############################################################################
    # models = ['Overthrust','Marmousi_1d_lin_300']  #   best choice of initial models 1
    #############################################################################
    models = ['Marmousi_1d_lin_300','Overthrust_1d_lin_300']  #   best choice of initial models 2. The most spectacular results.
    #############################################################################   try everything
    # models = ['Overthrust_linear_initial_300','Overthrust_1d_lin_300']  #   best choice of initial models 2
    models = [  'Marmousi_300','Marmousi_1d_lin_300','Marmousi_linear_initial_300',
    'Overthrust_300','Overthrust_1d_lin_300','Overthrust_linear_initial_300']
    #############################################################################
    # choose inversion strategy
    pars = {'data_mode': 'cnn1'}  # generating cnn data startegy 1
    pars = {'data_mode': 'fwi_18_strategy'}
    pars = {'data_mode': 'fwi_18_strategy_2'}
    # pars={'data_mode':'fwi_18_strategy_3'}
    pars = {'data_mode': 'fwi_18_strategy_5'}
    # pars={'data_mode':'fwi_18_strategy_6'}
    pars = {'data_mode': 'fwi_18_strategy_7'}
    # pars={'data_mode':'fwi_18_strategy_8'}
    # pars={'data_mode':'fwi_18_strategy_9'}
    pars = {'data_mode': 'fwi_18_strategy_10'}
    pars = {'data_mode': 'fwi_18_strategy_11'}
    pars = {'data_mode': 'fwi_18_strategy_12'}
    pars = {'data_mode': 'fwi_18_strategy_13'}
    pars = {'data_mode': 'fwi_18_strategy_14'}
    pars = {'data_mode': 'fwi_18_strategy_15'}
    pars = {'data_mode': 'fwi_18_strategy_16'}
    pars = {'data_mode': 'fwi_18_strategy_17_weak_spatfilter'}
    pars = {'data_mode': 'fwi_18_strategy_18_no_spatfilter'}
    pars = {'data_mode': 'fwi_19_strategy_zero_freq_no_cropping'}
    pars = {'data_mode': 'fwi_21_strategy'}
    pars = {'data_mode': 'fwi_21_strategy_no_filtering'}
    # pars = {'data_mode': 'fwi_22_strategy'}
    # pars = {'data_mode': 'fwi_23_strategy'}
    # pars = {'data_mode': 'fwi_24_strategy'}
    #pars = {'data_mode': 'fwi_25_cyclic_strategy'}
    pars = {'data_mode': 'fwi_20_strategy_oleg_zero_freq_no_cropping'}
    pars = {'data_mode': 'fwi_26_strategy'}
    pars = {'data_mode': 'fwi_27_strategy'}
    pars = {'data_mode': 'fwi_28_strategy_lnorm5'}
    pars = {'data_mode': 'fwi_28_strategy_lnorm6'}
    pars = {'data_mode': 'fwi_28_strategy_lnorm7'}
    pars = {'data_mode': 'fwi_29_strategy'}
    pars = {'data_mode': 'fwi_30_strategy'}
    pars = {'data_mode': 'fwi_31_strategy'}
    pars = {'data_mode': 'fwi_33_strategy'}
    # pars = {'data_mode': 'fwi_34_strategy'}
    pars = {'data_mode': 'fwi_35_strategy_l2'}
    pars = {'data_mode': 'fwi_35_strategy_l5'}
    # pars={'data_mode':'fwi_37_strategy'}
    # # pars={'data_mode':'fwi_38_strategy_oleg'}
    # # pars={'data_mode':'fwi_35_strategy'}
    # # pars={'data_mode':'fwi_36_strategy'}
    # pars={'data_mode':'fwi_37_lnorm_5_mine'}
    # # pars={'data_mode':'fwi_38_lnorm_5_oleg'}
    # pars={'data_mode':'fwi_39_lnorm_2_mine'}
    # pars={'data_mode':'fwi_40_lnorm_2_oleg'}
    # pars = {'data_mode': 'fwi_41_strategy'}      # offset windowing
    # pars = {'data_mode': 'fwi_42_strategy'}      # time windowing
    # pars = {'data_mode': 'fwi_43_strategy'}      # time windowing+offset windowing
    # pars = {'data_mode': 'fwi_44_strategy_l2'}      # time windowing+offset windowing+l2
    # pars = {'data_mode': 'fwi_45_strategy_l2'}      # time windowing+offset windowing+l2
    # pars = {'data_mode': 'fwi_45_strategy_l5'}      # time windowing+offset windowing+l2
    # pars = {'data_mode': 'fwi_46_strategy_l2'}      # time windowing+offset windowing+l2
    # pars = {'data_mode': 'fwi_46_strategy_l5'}      # time windowing+offset windowing+l2
    # pars = {'data_mode': 'fwi_47_strategy_l2'}      # time windowing+offset windowing+l2
    # pars = {'data_mode': 'fwi_47_strategy_l5'}      # time windowing+offset windowing+l2
    # pars = {'data_mode': 'fwi_32_strategy'}
    pars = {'data_mode': 'fwi_48_strategy_l2'}
    pars = {'data_mode': 'fwi_48_strategy_l5'}
    pars = {'data_mode': 'fwi_49_strategy_l2'}
    pars = {'data_mode': 'fwi_49_strategy_l2'}
    pars = {'data_mode': 'fwi_49_strategy_l5'}
    pars = {'data_mode': 'fwi_50_strategy_l2'}
    # pars = {'data_mode': 'fwi_50_strategy_l5'}
    # pars = {'data_mode': 'fwi_51_strategy_l2'}
    # pars = {'data_mode': 'fwi_51_strategy_l5'}
    # pars = {'data_mode': 'fwi_52_strategy_l2'}
    # pars = {'data_mode': 'fwi_52_strategy_l5'}
    # pars = {'data_mode': 'fwi_53_strategy_oleg_l2'}
    # pars = {'data_mode': 'fwi_53_strategy_oleg_l5'}
    pars = {'data_mode': 'fwi_54_strategy_fullband_l2'}
    # pars = {'data_mode': 'fwi_54_strategy_fullband_l5'}
    pars = {'data_mode': 'fwi_55_strategy_fullband_l2'}
    pars = {'data_mode': 'fwi_55_strategy_fullband_l5'}
    # pars = {'data_mode': 'fwi_55_strategy_l2'}
    # pars = {'data_mode': 'fwi_55_strategy_l5'}
    pars = {'data_mode': 'fwi_56_strategy_l2'}
    pars = {'data_mode': 'fwi_56_strategy_l5'}
    # pars = {'data_mode': 'fwi_56_strategy_special_l2'}
    # pars = {'data_mode': 'fwi_56_strategy_special_l5'}

    # pars = {'data_mode': 'fwi_57_strategy_l2'}
    # pars = {'data_mode': 'fwi_57_strategy_l5'}
    # pars = {'data_mode': 'fwi_58_strategy_l2'}
    # pars = {'data_mode': 'fwi_58_strategy_l5'}
    # pars = {'data_mode': 'fwi_59_strategy_l2'}
    # pars = {'data_mode': 'fwi_59_strategy_l5'}
    # pars = {'data_mode': 'fwi_60_strategy_l2'}
    # pars = {'data_mode': 'fwi_60_strategy_l5'}
    # choose initial models for fwi
    data_types = ['cnn', 'f_z', 'true','multi_cnn']
    # data_types = ['f_z','true']
    data_types=['cnn','f_z','true']
    # data_types=['cnn','f_z']
    # data_types=['cnn','true']
    # data_types = ['multi_cnn']
    # data_types = ['f_z']
    # data_types = ['true']
    # data_types = ['true_sharp']
    # data_types = ['cnn']
    # data_types = ['cnn_smooth_300']
    ####################### parallelization parameters
    pars.update({
        # 'NNODES': 40,'NPROCX':8,'NPROCY':2,      # big
        # 'NNODES': 20,'NPROCX':4,'NPROCY':2,      # medium
        # 'NNODES': 10,'NPROCX':4,'NPROCY':1,      # small 2
        'NNODES': 5,'NPROCX':2,'NPROCY':1,      # small
        # 'NNODES': 3,'NPROCX':1,'NPROCY':1,      # small
        'HOURS':24
        } )
    ####################### other parameters
    pars.update({'dx': calculation_spacing,'dz':calculation_spacing, 
        'out_shape':[496,150] ,'dsrc':200,
        # 'out_shape':[496,150] ,'dsrc':1200,
        'data_gen_mode':'pseudo_field','taper_shift':0,
        'extend_model_x':False,
        'last_source_position':'nx',
        } )
    if 'fullband' in pars['data_mode']:     
        pars.update({'full_band':True} )
    else:           
        pars.update({'full_band':False,
                     'corner_frequency':corner_frequency,
                    #  'delete_low_freqs':False})
                     'delete_low_freqs':True})
    # pars.update({'full_band':False})
    print(pars)
    #######################
    pars.update({'gen_mode':'synthetic'})
    list_of_sbatch_files=[]
    # if 'cnn' in pars['data_mode']:
    #     data_types = ['f_z']
    # create folder for inversion
    special_string = 'spacing_'+str(calculation_spacing)
    # res_folder='./results/individual_scaling_iter'+str(ITERATIONS)+'_freq_'+str(Freq)+'_source_spacing_'+str(source_spacing)+'_weights_'+str(number)+special_string
    # res_folder='./fwi/weights_'+str(number)
    # res_folder='./fwi/ws_'+str(number)+'_0'
    res_folder = './fwi/ws_'+pars['data_mode']+'_'+str(pars['corner_frequency'])+'hz'+'_0'
    print('res_folder', res_folder)
    val = 0
    while os.path.exists(res_folder):
        val = val+1
        res_folder2 = ''
        for tmp in res_folder.split('_')[0:-1]:
            res_folder2 = res_folder2+tmp+'_'
        res_folder = res_folder2+str(val)
        print('res_folder', res_folder)
    os.makedirs(res_folder, exist_ok=True)
    ###############
    print(pars)
    for data_type in data_types[0::]:
        pars.update({'current_data_type':data_type} )
        for model_ in models[0::]:
            # create folder for fwi run
            if data_type=='cnn' or data_type=='multi_cnn' or 'cnn' in data_type:
                case_name = model_+'_'+data_type+'_w_'+prediction_number
            else:
                case_name = model_+'_'+data_type
            results_folder = os.path.join(res_folder, case_name)
            os.makedirs(results_folder, exist_ok=True)
            os.makedirs(os.path.join(results_folder, 'fld'), exist_ok=True)
            # find corresponding file with initial models data
            files_path = fnmatch.filter(os.listdir(prediction_path), '*'+model_+'_weights*')
            # print(os.listdir(prediction_path));print(files_path);   exit()
            files_path2 = fnmatch.filter(files_path, '*.npz')
            NAME = files_path2[-1];  print(NAME)           
            # unpack data from file with predictions
            with open(os.path.join(prediction_path, NAME), 'rb') as f:
                data = np.load(f)
                ideal_initial_model = data['ideal_init_model']
                initial_model_1d = data['models_init']
                MODELS = data['models']                
                dx = data['dx']
                dz = data['dz']
                # water_taper=data['water_taper']
                water_taper=data['water_taper']
                # input_data=data['input_data'],
                # output_data=data['output_data'],
                # predicted_update=data['predicted_update'],
                # models_init=data['models_init'],
                # models=data['models'],
                predicted_initial_model=data['predicted_initial_model'],
                # ideal_init_model=data['ideal_init_model'],
                # fwi_result=data['fwi_result'],
                data.close()
            predicted_initial_model=predicted_initial_model[0]
            # predicted_update=predicted_update[0]
            # input_data=input_data[0]
            # output_data=output_data[0]

            # Plot_image_(MODELS.T, Show_flag=0, Save_flag=1, Title='MODELS_'+case_name, Save_pictures_path=res_folder, c_lim=[vp1, vp2])
            # Plot_image_(initial_model_1d.T, Show_flag=0, Save_flag=1, Title='initial_model_1d_'+case_name, Save_pictures_path=res_folder, c_lim=[vp1, vp2])
            # Plot_image_(ideal_initial_model.T, Show_flag=0, Save_flag=1, Title='ideal_initial_model_'+case_name, Save_pictures_path=res_folder, c_lim=[vp1, vp2])
            # Plot_image_(predicted_initial_model.T, Show_flag=0, Save_flag=1, Title='predicted_initial_model_'+case_name, Save_pictures_path=res_folder, c_lim=[vp1, vp2])
            # Plot_image_(predicted_update.T, Show_flag=0, Save_flag=1, Title='predicted_update_'+case_name, Save_pictures_path=res_folder)
            # Plot_image_(input_data.T, Show_flag=0, Save_flag=1, Title='input_data_'+case_name, Save_pictures_path=res_folder)
            # Plot_image_(output_data.T, Show_flag=0, Save_flag=1, Title='output_data_'+case_name, Save_pictures_path=res_folder)
            # Plot_image(water_taper.T, Show_flag=0, Save_flag=1, Title='water_taper_'+case_name, Save_pictures_path=res_folder)
            a=1
            # choose, what initial model to use
            if data_type == 'f_z':
                MODELS_INIT_FINAL = initial_model_1d
                # if model_=='Marmousi':
                #     data_source_file=os.path.join('./fwi','ws_cnn_13_special_1','model__'+model_+'_'+data_type,'stage6','denise_data.pkl')
                # elif model_=='Overthrust':
                #     data_source_file=os.path.join('./fwi','ws_cnn_13_special_1','model__'+model_+'_'+data_type,'stage6','denise_data.pkl')
                # with open(data_source_file,'rb') as input:
                #     d=pickle.load(input)
                # MODELS_INIT_FINAL=np.fliplr(d.model_init.vp.T)
            elif data_type == 'true':
                MODELS_INIT_FINAL =ideal_initial_model
            elif data_type == 'true_sharp':
                MODELS_INIT_FINAL =MODELS
            elif data_type == 'cnn':
                MODELS_INIT_FINAL = predicted_initial_model
            elif data_type =='cnn_smooth_300':
                predicted_initial_model2=F_smooth(predicted_initial_model,sigma_val=int(300/pars['dx']))
                predicted_initial_model2[water_taper==0]=1500
                Plot_image_(predicted_initial_model2.T,Show_flag=0, Save_flag=1, Title='predicted_initial_model2', Save_pictures_path=res_folder)
                MODELS_INIT_FINAL=predicted_initial_model2
            elif data_type == 'multi_cnn':
                ####################################
                synthetic_models_path='./fwi/multi/multi_cnn_15_weight_1351_7'
                synthetic_models_path='./fwi/multi/multi_cnn_13_special_weight_1351_20'
                cgg_models_path='./fwi/multi/multi_cnn_15_weight_1351_8'
                cgg_models_path='./fwi/multi/multi_cnn_13_special_weight_236_2'
                if model_=='Marmousi':  stage_number='stage3'
                if model_=='Overthrust':  stage_number='stage7'
                if model_=='Overthrust_linear_initial':  stage_number='stage1'
                if model_=='Marmousi_linear_initial':  stage_number='stage1'
                if model_=='Seam':  stage_number='stage1'
                if model_=='Seam2':  stage_number='stage1'
                if model_=='lin_vp_long':  
                    stage_number='stage1'
                    stage_number='stage3'
                if model_=='cgg_tomo_long2':  stage_number='stage1'
                ####################################
                # synthetic_models_path='./fwi/multi/multi_cnn_15_weight_1351_7'
                # cgg_models_path='./fwi/multi/multi_cnn_15_weight_1351_8'
                # if model_=='Marmousi':  stage_number='stage1'
                # if model_=='Overthrust':  stage_number='stage1'
                # if model_=='Overthrust_linear_initial':  stage_number='stage1'
                # if model_=='Marmousi_linear_initial':  stage_number='stage1'
                # if model_=='Seam':  stage_number='stage1'
                # if model_=='Seam2':  stage_number='stage1'
                # if model_=='lin_vp_long':  stage_number='stage1'
                # if model_=='cgg_tomo_long2':  stage_number='stage1'
                ####################################
                if model_=='Marmousi' or model_=='Overthrust' or model_=='Overthrust_linear_initial' or model_=='Marmousi_linear_initial' or model_=='Seam' or model_=='Seam2':
                    models_path=synthetic_models_path
                elif model_=='lin_vp_long' or model_=='cgg_tomo_long2':
                    models_path=cgg_models_path
                else:
                    print('exit!');exit()
                models_path=os.path.join(fwi_data_path,models_path)
                os.path.exists(models_path)
                fld_list=next(os.walk(models_path))[1]
                for fld in fld_list:
                    if 'model__'+model_+'_f_z' in fld:
                        models_path=os.path.join(models_path,fld,stage_number,'velocity_models_file.hdf5')
                MODELS=load_file(models_path,'models')
                MODELS_INIT_FINAL=load_file(models_path,'models_init')
                water_taper=load_file(models_path,'water_taper')
                ####################################
            # Plot results
            Plot_image_(MODELS_INIT_FINAL.T, Show_flag=0, Save_flag=1,Title='init_cropped_R2'+numstr3(F_r2(MODELS_INIT_FINAL,MODELS))+'_'+case_name, Save_pictures_path=res_folder, c_lim=[vp1,vp2])
            Plot_image_(MODELS.T,     Show_flag=0, Save_flag=1, Title='true_cropped_'+case_name, Save_pictures_path=res_folder, c_lim=[vp1,vp2])
            Plot_image_(water_taper.T,     Show_flag=0, Save_flag=1, Title='water_taper', Save_pictures_path=res_folder)
            # create files to start FWI
            velocity_models_file = os.path.join(results_folder, case_name+'.hdf5')
            f = h5py.File(velocity_models_file, 'w')
            f.create_dataset('models', data=MODELS)
            f.create_dataset('models_init', data=MODELS_INIT_FINAL)
            f.create_dataset('water_taper', data=water_taper)
            f.create_dataset('dx', data=dx)
            f.create_dataset('dz', data=dz)
            f.close()
            # generate python imports for python files
            imports = 'import sys,os\n'
            imports = imports+f"sys.path.append(os.getcwd())\n"
            imports = imports+'from F_utils import *\n'
            imports = imports+'from F_plotting import *\n'
            imports = imports+'from F_fwi import *\n'
            imports = imports+'import fnmatch\n'
            imports = imports+'from glob import glob\n'
            imports = imports+'import numpy as np\n'
            imports = imports+'import pyapi_denise_pavel as api\n'
            # create folders and files to launch denise

            denise_fwi(velocity_models_file,results_folder,os.getcwd(),calculation_spacing=calculation_spacing, mode='generate_task_files', pars=pars)
            
            data_generation = imports+f"denise_fwi('{velocity_models_file}','{results_folder}','{os.getcwd()}',calculation_spacing={calculation_spacing},pars={pars},mode='generate_task_files')\n"
            data_generation_script_name = os.path.join(results_folder, 'data_generation_script.py')
            f = open(data_generation_script_name, 'w')
            f.write(data_generation)
            f.close()
            # create field_data_processing file
            field_data_processing = imports
            if pars['full_band']==False:
                field_data_processing = field_data_processing + f"denise_folder_process('crop_zero_freqs','{results_folder}',pars={pars})\n"
            # field_data_processing = field_data_processing + f"denise_folder_process('plot','{results_folder}')\n"
            field_data_processing_script_name = os.path.join(results_folder, 'field_data_processing_script.py')
            f = open(field_data_processing_script_name, 'w')
            f.write(field_data_processing)
            f.close()
            # create post processing file
            post_processing = imports
            # post_processing = post_processing+f"denise_folder_process('plot','{results_folder}')\n"
            post_processing = post_processing+f"denise_folder_process('optimizing_space_','{os.path.join(results_folder)}')\n"
            post_processing = post_processing+f"denise_folder_process('plot','{results_folder}')\n"
            if data_type == 'cnn':
                post_processing = post_processing+f"denise_folder_process('plot_with_logs','{results_folder}')\n"
            post_processing_script_name = os.path.join(results_folder, 'post_processing_script.py')
            f = open(post_processing_script_name, 'w')
            f.write(post_processing)
            f.close()
            # run_fwi
            if flag_submitjob_or_exec_locally == 1:
                # construct submission script
                # folder to save job reports
                os.makedirs('./jobs', exist_ok=True)
                str1 = '#!/bin/bash\n'
                str1 = str1+'#SBATCH -N '+str(pars['NNODES'])+'\n'
                str1 = str1+'#SBATCH --partition=workq\n'
                # str1=str1+'#SBATCH --partition=debug\n'
                if 'cnn' in pars['data_mode']:
                    str1 = str1+'#SBATCH -t 4:00:00\n'
                else:
                    str1=str1+'#SBATCH -t '+str(pars['HOURS'])+':00:00\n'
                str1 = str1+'#SBATCH --account=k1404\n'  # k1404
                # str1 = str1+'#SBATCH --account=k1394\n'
                str1 = str1 + '#SBATCH --job-name='+'_'+case_name+'\n'
                str1 = str1 + '#SBATCH -o ' + os.path.join(results_folder, '%J'+res_folder.split('/')[-1]+'_'+case_name) + '.out\n'
                str1 = str1 + '#SBATCH -e ' + os.path.join(results_folder, '%J'+res_folder.split('/')[-1]+'_'+case_name) + '.err\n'
                # str1 = str1 +'salloc --nodes=4 --ntasks-per-node=32 --ntasks-per-socket=16 --cpus-per-task=1 --time=0:30:00 --account=k1404 --partition=debug\n'
                str1 = str1+'export DENISE=/project/k1404/pavel/DENISE-Black-Edition\n'
                str1=str1+f"module swap PrgEnv-gnu PrgEnv-intel\n"   
                str1=str1+f"module swap PrgEnv-cray PrgEnv-intel\n"   
                str1=str1+f"module load madagascar\n"
                str1=str1+f"module list\n"
                str1 = str1+f"source /project/k1404/pavel/DENISE-Black-Edition/denise_env/bin/activate\n"
                str1 = str1+f"which python\n"
                str1 = str1+'export GEN='+os.path.join(results_folder,'fld')+'\n'
                str1 = str1 + f"srun -n 1 python {os.path.join(results_folder,'data_generation_script.py')}\n"
                str1 = str1+'srun --hint=nomultithread --ntasks-per-node=32 --ntasks-per-socket=16 --cpus-per-task=1 $DENISE/bin/denise $GEN/seis_forward.inp $GEN/seis_fwi.inp\n'
                str1 = str1 + f"srun -n 1 python {os.path.join(results_folder,'field_data_processing_script.py')}\n"
                str1 = str1+'srun --hint=nomultithread --ntasks-per-node=32 --ntasks-per-socket=16 --cpus-per-task=1 $DENISE/bin/denise $GEN/seis_inversion.inp $GEN/seis_fwi.inp\n'
                str1 = str1 + f"srun -n 1 python {os.path.join(results_folder,'post_processing_script.py')}\n"
                batch_file_name = os.path.join(results_folder, 'fwi.sh')
                list_of_sbatch_files.append(batch_file_name)
                f = open(batch_file_name, 'w')
                f.write(str1)
                f.close()
                print(str1)
    #########   make bash file, which submits sbatch files from login node
    str1 = '#!/bin/bash\n'
    for sbatch_file in list_of_sbatch_files:
        # str1 = str1+'echo '+sbatch_file+'\n'
        str1 = str1+'sbatch '+sbatch_file+'\n'

    batch_file_name = os.path.join('run_full_fwi.sh')
    f = open(batch_file_name, 'w')
    f.write(str1)
    f.close()
    print('bash '+batch_file_name)
    print('FULL-BAND=',pars['full_band'])

def flag_record_cnn_data2():
    """ 
    run FWI on real data (scenarios 4,5)
    or record data for cnn (scenarios 1,2,3)
    """
    ########################   parameters
    launch_jobs = 0
    record_tasks = 1
    run_fwi_flag = 0
    parallel_processing = 1
    flag_plotting = 1
    calculation_spacing = 25
    pars={'data_mode':'cnn_16'}
    pars = {'data_mode': 'fwi_56_strategy_l2'}
    pars = {'data_mode': 'fwi_56_strategy_l5'}
    ########    paper results
    # pars = {'data_mode': 'fwi_60_strategy_fullband_l5'}
    # pars = {'data_mode': 'fwi_56_strategy_l5'}
    ########
    if 'fullband' in pars['data_mode']:     
        pars.update({'full_band':True} )
    else:           
        pars.update({'full_band':False})
    pars.update({'dx': calculation_spacing, 'dz': calculation_spacing,
        'out_shape':[496,150] ,'dsrc':200,
        # 'out_shape':[816,150] ,'dsrc':200,
        'data_gen_mode':'pseudo_field','taper_shift':0,
        'extend_model_x':False,
        'last_source_position':'nx',
        'HOURS':24
        } )
    pars.update({'current_data_type':'record_cnn_data'} )
    pars.update({'computation_platform':'workstation'} )
    # pars.update({'computation_platform':'ibex'} )
    ######################## cores shaheen
    pars.update({
        # 'NNODES': 40,'NPROCX':8,'NPROCY':2,      # big
        # 'NNODES': 10,'NPROCX':4,'NPROCY':1,      # small
        'NNODES': 5,'NPROCX':2,'NPROCY':1,      # small
        # 'NNODES': 2,'NPROCX':1,'NPROCY':1,      # small
        'ncores':40
        } )
    ######################## cores workstation
    pars.update({
        'NNODES': 1,'NPROCX':2,'NPROCY':1,      # small
        'ncores':20
        } )
    ########################    choose program scenario
    scenario=1  #record dataset train models from generator, synthetic data
    scenario=2  #record dataset test models, synthetic data
    scenario=3  #record test cgg models, field data
    scenario=4  #run fwi after predicted cgg models or 1d models, field data
    scenario=5  #run fwi on sequentially predicted initial model
    ########################
    if scenario==1:#######################   record dataset train models from generator
        pars.update({'corner_frequency':4} )
        pars.update({'delete_low_freqs':True} )
        processing_batch_size = 10
        sample_list=list(np.arange(0,6000,1))
        sample_list=list(np.arange(12100,18000,1))
        sample_list=list(np.arange(20000,25000,1))
        sample_list=list(np.arange(25000,28000,1))
        # sample_list=list(3+np.arange(0,45,1))

        # processing_batch_size = 2
        # sample_list=list(np.arange(0,4,1))

        pars.update({'extend_model_x':False} )
        res_folder_ = './fwi/gen3_marine_data_'
        # res_folder=res_folder+pars['data_mode']+'_dsrc_'+str(pars['dsrc'])+'_'+pars['data_gen_mode']+'_nx_'+str( pars['out_shape'][0] )+'_test9_gradt1_16_min5'
        # res_folder=res_folder+pars['data_mode']+'_dsrc_'+str(pars['dsrc'])+'_'+pars['data_gen_mode']+'_nx_'+str( pars['out_shape'][0] )+'_test9_gradt1_16'
        # res_folder=res_folder+pars['data_mode']+'_dsrc_'+str(pars['dsrc'])+'_'+pars['data_gen_mode']+'_nx_'+str( pars['out_shape'][0] )+'_test_no_gradhor_sure'
        res_folder=res_folder_+pars['data_mode']+'_dl_dataset'
        gen_mode = 'vlad'
        gen_mode = 'oleg'
        initial_velocity_models_source='generator'
    elif scenario==2:#######################   record benchmark dataset test models
        pars.update({'full_band':False})
        pars.update({'corner_frequency':corner_frequency} )
        pars.update({'delete_low_freqs':True} )
        pars.update({'taper_shift':0} )
        gen_mode='test'
        processing_batch_size=1
        pars.update({'extend_model_x':False} )
        sample_list=['_Marmousi','_Overthrust','_Seam','_Seam2']
        # sample_list=['_Overthrust','_Seam','_Seam2']
        # sample_list=['_Seam','_Seam2']
        # sample_list=['_Overthrust','_Seam','_Seam2']
        # sample_list=['_Seam2']
        # sample_list=['_Marmousi','_Seam2']
        # sample_list=['_Marmousi']
        sample_list=['_Overthrust']
        # sample_list=['_Overthrust','_Marmousi']
        sample_list=['_Marmousi','_Overthrust']
        sample_list=['_Overthrust','_Marmousi',
            '_Overthrust_linear_initial','_Marmousi_linear_initial',
            '_Overthrust_1d_lin', '_Marmousi_1d_lin']
        initial_velocity_models_source='generator'  # pickup initial model from basic initial cgg model
        # res_folder='./fwi/'+pars['data_mode']+'_test_new_models'
        res_folder='./fwi/'+pars['data_mode']+'_record_testing_models'
    elif scenario==3:########################   record cgg models CNN data, field
        pars.update({'full_band':False})
        pars.update({'corner_frequency':corner_frequency} )
        pars.update({'taper_shift':0} )
        pars.update({'delete_low_freqs':True} )
        gen_mode='test_real_data'
        sample_list=['_cgg_tomo_long1','_cgg_tomo_long2','_cgg_lin_vp_long']
        # sample_list=['_cgg_tomo_long2','_cgg_lin_vp_long']
        # sample_list=['_cgg_tomo_long1']
        # sample_list=['_cgg_tomo_long2']
        # sample_list=['_cgg_lin_vp_long']
        processing_batch_size=1
        pars.update({'extend_model_x':False} )
        res_folder='./fwi/record_cgg_data_'+pars['data_mode']
        initial_velocity_models_source='generator'  # pickup initial model from basic initial cgg model
    elif scenario==4:########################   run fwi after predicted cgg models or 1d models, field data
        pars.update({'taper_shift':0} )
        pars.update({'corner_frequency':corner_frequency} )
        # pars.update({'delete_low_freqs':False} )
        pars.update({'delete_low_freqs':True} )
        sample_list=['_cgg_tomo_long1','_cgg_tomo_long2','_cgg_lin_vp_long']
        sample_list=['_cgg_lin_vp_long_300']
        gen_mode='test_real_data'
        processing_batch_size=1
        prediction_path = './predictions/predictions_1296'
        prediction_path = './predictions/predictions_27'
        prediction_path = './predictions/predictions_1351'
        prediction_path = './predictions/predictions_1400'
        prediction_path = './predictions/predictions_1401'
        prediction_path = './predictions/predictions_1500'
        prediction_path = './predictions/predictions_1500'
        # prediction_path = './predictions/predictions_574'
        prediction_path = './predictions/predictions_675'   #weights corresponding to 1497(100m), used for CGG
        prediction_path = './predictions/predictions_147'   #these weights used for Marm, Over (300m)
        prediction_path = './predictions/predictions_253'   #these weights used for Marm, Over (300m)
        pars.update({'prediction_path':prediction_path})
        prediction_number=prediction_path.split('/')[-1]
        prediction_number=prediction_number.split('predictions_')[-1]
        initial_velocity_models_source='generator'  # pickup initial model from basic initial cgg model
        # initial_velocity_models_source='cnn_prediction'   # generator, cnn_prediction, pickup initial model from folder prediction path 
        if initial_velocity_models_source=='cnn_prediction':
            res_folder=os.path.join('./fwi','cgg_real_data',pars['data_mode']+'_weight_'+str(prediction_number)+'_'+str(pars['corner_frequency'])+'hz'+'_0')
            if pars['delete_low_freqs']==True:
                res_folder=os.path.join('./fwi','cgg_real_data',pars['data_mode']+'_del_low_weight_'+str(prediction_number)+'_'+str(pars['corner_frequency'])+'hz'+'_0')
        elif initial_velocity_models_source=='generator':
            res_folder=os.path.join('./fwi','cgg_real_data',pars['data_mode']+'_gen_'+'_'+str(pars['corner_frequency'])+'hz'+'_0')
            if pars['delete_low_freqs']==True:
                res_folder=os.path.join('./fwi','cgg_real_data',pars['data_mode']+'_del_low_gen_'+'_'+str(pars['corner_frequency'])+'hz'+'_0')
    elif scenario==5:########################   run fwi on sequentially predicted initial model
        pars.update({'taper_shift':0} )
        pars.update({'corner_frequency':corner_frequency} )
        # pars.update({'delete_low_freqs':False} )
        pars.update({'delete_low_freqs':True} )
        processing_batch_size=1
        synthetic_data_flag=1
        initial_velocity_models_source='multi_cnn'
        if synthetic_data_flag==1:
            ####################    marm, over fwi
            # models = [  'Marmousi_300','Marmousi_1d_lin_300','Marmousi_linear_initial_300',
            #     'Overthrust_300','Overthrust_1d_lin_300','Overthrust_linear_initial_300']
            gen_mode='synthetic_data'
            prediction_path='./fwi/multi/multi_cnn_13_special_weight_235_0/model__Marmousi_1d_lin_300_f_z/stage2'
            prediction_path='./fwi/multi/multi_cnn_13_special_weight_235_0/model__Overthrust_1d_lin_300_f_z/stage3'
            # prediction_path='./fwi/multi/multi_cnn_13_special_weight_237_0/model__Marmousi_1d_lin_300_f_z/stage0'
            # prediction_path='./fwi/multi/multi_cnn_13_special_weight_237_0/model__Overthrust_1d_lin_300_f_z/stage0'

            prediction_path='./fwi/multi/multi_cnn_13_special_weight_236_6/model__Marmousi_1d_lin_300_f_z/stage2'
            prediction_path='./fwi/multi/multi_cnn_13_special_weight_236_6/model__Overthrust_1d_lin_300_f_z/stage3'
            
            prediction_path='./fwi/multi/multi_cnn_13_special_weight_244_6/model__Marmousi_1d_lin_300_f_z/stage3'
            prediction_path='./fwi/multi/multi_cnn_13_special_weight_244_6/model__Overthrust_1d_lin_300_f_z/stage2'            
            #####   Marmousi_1d_lin_300 model
            prediction_path='./fwi/multi/multi_cnn_13_special_weight_236_6/model__Marmousi_1d_lin_300_f_z/stage0'
            prediction_path='./fwi/multi/multi_cnn_13_special_weight_236_6/model__Marmousi_1d_lin_300_f_z/stage1'
            prediction_path='./fwi/multi/multi_cnn_13_special_weight_236_6/model__Marmousi_1d_lin_300_f_z/stage2'
            #####   Marmousi_1d_lin_300 model, better choice
            prediction_path='./fwi/multi/multi_cnn_13_special_weight_236_7/model__Marmousi_1d_lin_300_f_z/stage1'
            prediction_path='./fwi/multi/multi_cnn_13_special_weight_236_7/model__Marmousi_1d_lin_300_f_z/stage2'
            prediction_path='./fwi/multi/multi_cnn_13_special_weight_236_0/model__Marmousi_300_f_z/stage1'
            prediction_path='./fwi/multi/multi_cnn_13_special_weight_236_0/model__Marmousi_1d_lin_300_f_z/stage1'
            prediction_path='./fwi/multi/multi_cnn_13_special_weight_236_0/model__Marmousi_1d_lin_300_f_z/stage2'
            prediction_path='./fwi/multi/multi_cnn_13_special_weight_236_0/model__Marmousi_linear_initial_300_f_z/stage1'
            prediction_path='./fwi/multi/multi_cnn_13_special_weight_236_0/model__Marmousi_linear_initial_300_f_z/stage2'
            prediction_path='./fwi/multi/multi_cnn_13_special_weight_236_8/model__Marmousi_300_f_z/stage2'
            prediction_path='./fwi/multi/multi_cnn_13_special_weight_236_8/model__Marmousi_linear_initial_300_f_z/stage3'
            prediction_path='./fwi/multi/multi_cnn_13_special_weight_236_8/model__Marmousi_linear_initial_300_f_z/stage2'
            prediction_path='./fwi/multi/multi_cnn_13_special_weight_236_8/model__Marmousi_linear_initial_300_f_z/stage1'
            #####   Marmousi_300 model
            # prediction_path='./fwi/multi/multi_cnn_13_special_weight_236_3/model__Marmousi_300_f_z/stage0'
            # prediction_path='./fwi/multi/multi_cnn_13_special_weight_236_3/model__Marmousi_300_f_z/stage1'
            #####   Marmousi_linear_initial_300 model
            # prediction_path='./fwi/multi/multi_cnn_13_special_weight_236_3/model__Marmousi_linear_initial_300_f_z/stage0'
            # prediction_path='./fwi/multi/multi_cnn_13_special_weight_236_3/model__Marmousi_linear_initial_300_f_z/stage1'
            #####   Overthrust_1d_lin_300 model . Chosen FWI result
            # prediction_path='./fwi/multi/multi_cnn_13_special_weight_236_0/model__Overthrust_1d_lin_300_f_z/stage3'
            ####################    chosen initial models for final results
            prediction_path='./fwi/multi/multi_cnn_13_special_weight_236_0/model__Marmousi_1d_lin_300_f_z/stage1'
            prediction_path='./fwi/multi/multi_cnn_13_special_weight_236_0/model__Overthrust_1d_lin_300_f_z/stage1'
            prediction_path='/home/plotnips/Dropbox/Log_extrapolation/scripts/DENISE-Black-Edition-master/fwi/multi/multi_cnn_13_special_weight_236_0/model__Overthrust_1d_lin_300_f_z/stage2'
            ####################
        else:
            ####################    cgg fwi
            gen_mode='test_real_data'
            prediction_path = './predictions/predictions_675'   #weights corresponding to 1497(100m), used for CGG
            prediction_path = './predictions/predictions_147'   #these weights used for Marm, Over (300m)
            # prediction_path='./fwi/multi/multi_cnn_13_special_weight_147_7/model___cgg_lin_vp_long_300_f_z/stage4'
            prediction_path='./fwi/multi/multi_cnn_13_special_weight_675_4/model___cgg_lin_vp_long_300_f_z/stage3'

            # prediction_path='./fwi/multi/multi_cnn_13_special_weight_235_1/model__cgg_lin_vp_long_300_f_z/stage1'
            #### prediction_path='./fwi/multi/multi_cnn_13_special_weight_236_1/model__cgg_lin_vp_long_300_f_z/stage2'
            prediction_path='./fwi/multi/multi_cnn_13_special_weight_236_2/model__cgg_lin_vp_long_300_f_z/stage1'
            prediction_path='./fwi/multi/multi_cnn_13_special_weight_236_2/model__cgg_lin_vp_long_300_f_z/stage2'
            prediction_path='./fwi/multi/multi_cnn_13_special_weight_236_2/model__cgg_lin_vp_long_300_f_z/stage4'
            prediction_path='./fwi/multi/multi_cnn_13_special_weight_236_10/model__cgg_lin_vp_long_300_f_z/stage1'
            prediction_path='./fwi/multi/multi_cnn_13_special_weight_236_10/model__cgg_lin_vp_long_300_f_z/stage2'
            prediction_path='./fwi/multi/multi_cnn_13_special_weight_236_1/model__cgg_lin_vp_long_300_f_z/stage1'
            #   path for apriori initial model
            # prediction_path='./fwi/multi/multi_cnn_13_special_weight_236_2/model__cgg_lin_vp_long_300_f_z/stage0'
            #   recalculation with new data
            prediction_path='./fwi/multi/multi_cnn_13_special_weight_236_1/model__cgg_lin_vp_long_300_f_z/stage1'
            prediction_path='./fwi/multi/multi_cnn_13_special_weight_236_2/model__cgg_lin_vp_long_300_f_z/stage1'
            prediction_path='./fwi/multi/multi_cnn_13_special_weight_236_2/model__cgg_lin_vp_long_300_f_z/stage2'
            prediction_path='./fwi/multi/multi_cnn_13_special_weight_236_2/model__cgg_lin_vp_long_300_f_z/stage4'

            prediction_path='./fwi/multi/multi_cnn_13_special_weight_236_2/model__cgg_lin_vp_long_300_f_z/stage4'
            prediction_path='./fwi/multi/multi_cnn_13_special_weight_236_11/model__cgg_lin_vp_long_300_f_z/stage1'
            #   new hope
            prediction_path='./fwi/multi_cnn_13_special_weight_236_2/model__cgg_lin_vp_long_300_f_z/stage5'
            prediction_path='./fwi/multi_cnn_13_special_weight_236_2/model__cgg_lin_vp_long_300_f_z/stage7'
            prediction_path='./fwi/multi_cnn_13_special_weight_236_2/model__cgg_lin_vp_long_300_f_z/stage13'
            #   after prediction research, 11/06/22
            prediction_path='./fwi/multi/multi_cnn_13_special_weight_236_2/model__cgg_lin_vp_long_300_f_z/stage3'
        ####################
        sample_list=[prediction_path.split('/')[-2].split('model__')[-1]]
        pars.update({'prediction_path':prediction_path})
        ####################    Naming approach # 1
        # stage_number=( prediction_path.split('/')[-1].split('stage')[-1] )
        # weight_number=prediction_path.split('/')[-3].split('weight_')[1]
        # res_folder=os.path.join('./fwi','cgg_real_data',pars['data_mode']+'_'+weight_number+'_st'+stage_number+'_0')
        ####################    Naming approach # 2
        ### prediction_number=prediction_number.split('predictions_')[-1]
        listToStr = '_'.join([str(elem) for elem in prediction_path.split('/')[-3:]])
        # res_folder=os.path.join(fwi_data_path,'./fwi','cgg_real_data',pars['data_mode']+'_'+listToStr+'_0')
        res_folder=os.path.join('./fwi','cgg_real_data',pars['data_mode']+'_'+listToStr+'_0')
    ####################    Append number to res folder name
    # res_folder='./fwi/cgg_real_data/fwi_56_strategy_l5_236_test_0'
    ########################
    val = 0
    while os.path.exists(res_folder):
        val = val+1
        res_folder2 = ''
        for tmp in res_folder.split('_')[0:-1]:
            res_folder2 = res_folder2+tmp+'_'
        res_folder = res_folder2+str(val)
        print('res_folder', res_folder)
    ########################   record to specific res_folder
    if scenario==1:
        res_folder='./fwi/gen3_marine_data_cnn_16_dl_12'
        res_folder='./fwi/gen3_marine_data_cnn_16_dl_13'
    if scenario==2 or scenario==3:  # cgg data
        res_folder='./fwi/gen3_marine_data_'+pars['data_mode']+'_'+'_test_samples'
        res_folder='./fwi/gen3_marine_data_'+pars['data_mode']+'_'+'_test_samples_2nd_run'
    ########################   job parameters
    pars.update({'initial_velocity_models_source':initial_velocity_models_source} )
    pars.update({'gen_mode':gen_mode})
    batches = [sample_list[x:x+processing_batch_size]   for x in range(0, len(sample_list), processing_batch_size)]
    # batches=batches[0:1]
    # batches=batches[1:]
    res_folders=[res_folder]
    list_of_sbatch_files=[]
    for res_folder in res_folders:
        print('Created folder with results called, ',res_folder)
        if gen_mode!='test_real_data':
            ########################   record task files
            list_of_sbatch_files=[]
            for batch in batches:
                for number in batch:
                    velocity_model_name=number
                    directory = 'model_'+str(number)
                    results_folder = os.path.join(res_folder, directory)
                    filename_model = os.path.join(results_folder, directory+'.hdf5')
                    os.makedirs(results_folder, exist_ok=True)
                    # denise_folder_process('crop_zero_freqs',os.path.join(results_folder))
                    # denise_folder_process('plot',os.path.join(results_folder))
                    # exit()
                    ########################   generate python imports for python files
                    imports = 'import sys,os\n'
                    imports = imports+f"sys.path.append(os.getcwd())\n"
                    imports = imports+'from F_utils import *\n'
                    imports = imports+'from F_plotting import *\n'
                    imports = imports+'from F_fwi import *\n'
                    imports = imports+'import fnmatch\n'
                    imports = imports+'from glob import glob\n'
                    imports = imports+'import numpy as np\n'
                    imports = imports+'import pyapi_denise_pavel as api\n'
                    ########################   create folders and files to launch denise
                    # generate_data_for_batch_of_jobs(generator_2,denise_fwi,gen_mode,res_folder,calculation_spacing,pars,number)
                    ########################   generate_data_for_batch_of_jobs(generator_,denise_fwi,'test',res_folder,calculation_spacing,pars,'_Seam2')
                    pars.update({'current_gen_mode':gen_mode} )
                    data_generation = imports
                    data_generation = data_generation+f"generate_data_for_batch_of_jobs(generator_2,denise_fwi,'{gen_mode}','{res_folder}',{calculation_spacing},{pars},'{number}')\n"
                    # denise_fwi(filename_model,results_folder,os.getcwd(),calculation_spacing=calculation_spacing,pars=pars,mode='generate_task_files')
                    # exit()
                    data_generation = data_generation+f"denise_fwi('{filename_model}','{results_folder}',os.getcwd(),calculation_spacing={calculation_spacing},pars={pars},mode='generate_task_files')\n"
                    data_generation_script_name = os.path.join(res_folder, directory, 'data_generation_script.py')
                    f = open(data_generation_script_name, 'w')
                    f.write(data_generation)
                    f.close()
                    ########################   create field_data_processing file
                    field_data_processing = imports
                    field_data_processing = field_data_processing + f"denise_folder_process('crop_zero_freqs','{os.path.join(results_folder)}',pars={pars})\n"
                    field_data_processing_script_name = os.path.join(res_folder, directory, 'field_data_processing_script.py')
                    f = open(field_data_processing_script_name, 'w')
                    f.write(field_data_processing)
                    f.close()
                    ########################   create post processing file
                    if gen_mode == 'test':
                        post_processing = imports
                        # # post_processing = post_processing+f"denise_folder_process('plot','{os.path.join(results_folder)}')\n"
                        # if str(number)!='_Overthrust':
                        #     post_processing=post_processing+f"denise_folder_process('optimizing_space_','{os.path.join(results_folder)}')\n"
                        post_processing = post_processing+f"denise_folder_process('plot','{os.path.join(results_folder)}')\n"
                        post_processing=post_processing+f"denise_folder_process('optimizing_space_','{os.path.join(results_folder)}')\n"
                        # post_processing = post_processing+f"denise_folder_process('plot','{os.path.join(results_folder)}')\n"
                    else:
                        post_processing = imports
                        post_processing = post_processing+f"denise_folder_process('plot','{os.path.join(results_folder)}')\n"
                        # post_processing = post_processing + f"denise_folder_process('optimizing_space_','{os.path.join(results_folder)}')\n"
                        # post_processing = post_processing + f"denise_folder_process('clean_everything_except_denise_data_pkl','{os.path.join(results_folder)}')\n"
                        # post_processing = post_processing + f"denise_folder_process('plot','{os.path.join(results_folder)}')\n"
                    post_processing_script_name = os.path.join(res_folder, directory, 'post_processing_script.py')
                    f = open(post_processing_script_name, 'w')
                    f.write(post_processing)
                    f.close()
                    ###########     construct batch file
                    if pars['computation_platform']=='workstation':
                        str1 = '#!/bin/bash\n'
                        str1=str1+'export DENISE=./DENISE-Black-Edition\n'
                        str1=str1+f"source ~/.bashrc\n"
                        str1=str1+f"conda activate lw\n"
                        str1=str1+f"which python\n"
                        str1=str1+'export GEN='+os.path.join(results_folder,'fld')+'\n'
                        #######
                        str1 = str1 + f"python {os.path.join(results_folder,'data_generation_script.py')}\n"
                        str1=str1+f"mpirun -np {pars['ncores']} $DENISE/bin/denise $GEN/seis_forward.inp $GEN/seis_fwi.inp\n"
                        str1 = str1 + f"python {os.path.join(results_folder,'field_data_processing_script.py')}\n"
                        str1=str1+f"mpirun -np {pars['ncores']} $DENISE/bin/denise $GEN/seis_inversion.inp $GEN/seis_fwi.inp\n"
                        str1 = str1 + f"python {os.path.join(results_folder,'post_processing_script.py')}\n"
                    elif pars['computation_platform']=='shaheen':
                        # batch_file_name=launch_batch_of_jobs(job, batch, res_folder,run_fwi_flag=run_fwi_flag)
                        os.makedirs('./jobs', exist_ok=True)
                        str1 = '#!/bin/bash\n'
                        str1 = str1+'#SBATCH -N '+str(pars['NNODES'])+'\n'
                        str1 = str1+'#SBATCH --partition=workq\n'
                        # str1=str1+'#SBATCH --partition=debug\n'
                        if 'cnn' in pars['data_mode']:
                            str1 = str1+'#SBATCH -t 4:00:00\n'
                        else:
                            str1=str1+'#SBATCH -t '+str(pars['HOURS'])+':00:00\n'
                        str1 = str1+'#SBATCH --account=k1404\n'  # k1404
                        # str1 = str1+'#SBATCH --account=k1394\n'
                        case_name=results_folder.split('/')[-1]
                        str1 = str1 + '#SBATCH --job-name='+'_'+case_name+'\n'
                        str1 = str1 + '#SBATCH -o ' + os.path.join(results_folder, '%J'+res_folder.split('/')[-1]+'_'+case_name) + '.out\n'
                        str1 = str1 + '#SBATCH -e ' + os.path.join(results_folder, '%J'+res_folder.split('/')[-1]+'_'+case_name) + '.err\n'
                        # str1 = str1 +'salloc --nodes=4 --ntasks-per-node=32 --ntasks-per-socket=16 --cpus-per-task=1 --time=0:30:00 --account=k1404 --partition=debug\n'
                        str1 = str1+'export DENISE=/project/k1404/pavel/DENISE-Black-Edition\n'
                        str1=str1+f"module swap PrgEnv-gnu PrgEnv-intel\n"   
                        str1=str1+f"module swap PrgEnv-cray PrgEnv-intel\n"   
                        str1=str1+f"module load madagascar\n"
                        str1=str1+f"module list\n"
                        str1 = str1+f"source /project/k1404/pavel/DENISE-Black-Edition/denise_env/bin/activate\n"
                        str1 = str1+f"which python\n"
                        str1 = str1+'export GEN='+os.path.join(results_folder,'fld')+'\n'
                        str1 = str1 + f"srun -n 1 python {os.path.join(results_folder,'data_generation_script.py')}\n"
                        str1 = str1+'srun --hint=nomultithread --ntasks-per-node=32 --ntasks-per-socket=16 --cpus-per-task=1 $DENISE/bin/denise $GEN/seis_forward.inp $GEN/seis_fwi.inp\n'
                        str1 = str1 + f"srun -n 1 python {os.path.join(results_folder,'field_data_processing_script.py')}\n"
                        str1 = str1+'srun --hint=nomultithread --ntasks-per-node=32 --ntasks-per-socket=16 --cpus-per-task=1 $DENISE/bin/denise $GEN/seis_inversion.inp $GEN/seis_fwi.inp\n'
                        str1 = str1 + f"srun -n 1 python {os.path.join(results_folder,'post_processing_script.py')}\n"
                    ###########
                    batch_file_name = os.path.join(results_folder, 'fwi.sh')
                    list_of_sbatch_files.append(batch_file_name)
                    f = open(batch_file_name, 'w')
                    f.write(str1)
                    f.close()
                    print(str1)
        else:
            pars.update({'current_gen_mode':gen_mode} )
            ########################   record task files
            for batch in batches:
                for number in batch:
                    velocity_model_name=number
                    directory = 'model_'+str(number)
                    # res_folder='./fwi/cgg_real_data/cnn_14_1'
                    # directory='model__cgg_lin_vp_long'
                    # res_folder='./fwi/cgg_real_data/fwi_36_strategy_0'
                    # directory='model__cgg_lin_vp_long'
                    results_folder = os.path.join(res_folder, directory)
                    # if scenario!=5:
                    # else:   
                    #     results_folder = res_folder
                    os.makedirs(results_folder,exist_ok=True)
                    filename_model = os.path.join(results_folder,directory+'.hdf5')
                    ########################   generate python imports for python files
                    imports = 'import sys,os\n'
                    imports = imports+f"sys.path.append(os.getcwd())\n"
                    imports = imports+'from F_utils import *\n'
                    imports = imports+'from F_plotting import *\n'
                    imports = imports+'from F_fwi import *\n'
                    imports = imports+'import fnmatch\n'
                    imports = imports+'from glob import glob\n'
                    imports = imports+'import numpy as np\n'
                    imports = imports+'import pyapi_denise_pavel as api\n'
                    ########################   create file for FWI. create data generation file
                    data_generation = imports
                    data_generation = data_generation+f"generate_data_for_batch_of_jobs(generator_2,denise_fwi,'{gen_mode}','{res_folder}',{calculation_spacing},{pars},'{number}')\n"
                    data_generation = data_generation+f"denise_fwi('{filename_model}','{results_folder}',os.getcwd(),calculation_spacing={calculation_spacing},pars={pars},mode='generate_task_files')\n"
                    generate_data_for_batch_of_jobs(generator_2,denise_fwi,gen_mode,res_folder,calculation_spacing,pars,number)
                    # denise_fwi(filename_model,results_folder,os.getcwd(),calculation_spacing=calculation_spacing,pars=pars,mode='generate_task_files')
                    # denise_folder_process('next_fdmd',results_folder)
                    # denise_folder_process('real_data_plotting_session',results_folder)
                    # exit()
                    ########################   copy real data (filtered or not) to the folder. old variant
                    # su_field_path1 = os.path.join('./for_pasha','su_field_high_pass_5_hz/')
                    # su_field_path2 = os.path.join('./for_pasha','su_field/')
                    # if pars['full_band']==True:  
                    #     root_from=su_field_path2
                    # else:
                    #     root_from=su_field_path1
                    # # copy_su_from_to(su_field_path1,os.path.join(results_folder,'fld','su'))
                    # data_generation = data_generation+f"copy_su_from_to('{root_from}','{os.path.join(results_folder,'fld','su')}')\n"
                    ########################   copy not filtered real data to the folder and then filter it
                    su_field_path1 = os.path.join('./for_pasha','su_field_high_pass_5_hz/')
                    su_field_path2 = os.path.join('./for_pasha','su_field/')
                    if pars['full_band']==True:  
                        data_generation = data_generation+f"copy_su_from_to('{su_field_path2}','{os.path.join(results_folder,'fld','su')}')\n"
                    else:
                        data_generation = data_generation+f"copy_su_from_to('{su_field_path1}','{os.path.join(results_folder,'fld','su')}')\n"
                        # data_generation = data_generation+f"denise_folder_process('crop_zero_freqs','{results_folder}',pars={pars})\n"
                    ########################
                    data_generation_script = os.path.join(results_folder, 'data_generation_script.py')
                    f = open(data_generation_script, 'w')
                    f.write(data_generation)
                    f.close()
                    ########################   create post processing file
                    post_processing = imports
                    post_processing = post_processing + f"denise_folder_process('plot','{os.path.join(results_folder)}')\n"
                    # post_processing = post_processing + f"denise_folder_process('optimizing_space_','{os.path.join(results_folder)}')\n"
                    # denise_folder_process('next_fdmd',os.path.join(results_folder))
                    post_processing = post_processing + f"denise_folder_process('next_fdmd','{os.path.join(results_folder)}')\n"
                    post_processing_script = os.path.join(results_folder, 'post_processing_script.py')
                    f = open(post_processing_script,'w')
                    f.write(post_processing)
                    f.close()
                    ########################   create file for plotting forward modelled data from inversion result
                    post_processing2 = imports
                    # denise_folder_process('real_data_plotting_session',results_folder)
                    post_processing2 = post_processing2 + f"denise_folder_process('real_data_plotting_session','{os.path.join(results_folder)}')\n"
                    post_processing_script2 = os.path.join(results_folder,'post_processing_script2.py')
                    f = open(post_processing_script2, 'w')
                    f.write(post_processing2)
                    f.close()
                    ########################   create file for plotting forward modelled data from inversion result
                    post_processing3 = imports
                    # post_processing3 = post_processing3 + f"denise_folder_process('optimizing_space_','{os.path.join(results_folder)}')\n"
                    post_processing_script3 = os.path.join(results_folder,'post_processing_script3.py')
                    f = open(post_processing_script3, 'w')
                    f.write(post_processing3)
                    f.close()
                    ########################   create sbatch file
                    if pars['computation_platform']=='workstation':
                        str1 = '#!/bin/bash\n'
                        str1=str1+'export DENISE=./DENISE-Black-Edition\n'
                        str1=str1+f"source ~/.bashrc\n"
                        str1=str1+f"conda activate lw\n"
                        str1=str1+f"which python\n"
                        str1=str1+'export GEN='+os.path.join(results_folder,'fld')+'\n'
                        
                        str1=str1+f"python {data_generation_script}\n"
                        str1=str1+f"mpirun -np {pars['ncores']} $DENISE/bin/denise $GEN/seis_inversion.inp $GEN/seis_fwi.inp\n"

                        str1=str1+f"python {post_processing_script}\n"
                        str1=str1+f"mpirun -np {pars['ncores']} $DENISE/bin/denise $GEN/seis_forward_next_fdmd.inp $GEN/seis_fwi.inp\n"
                        str1=str1+f"mpirun -np {pars['ncores']} $DENISE/bin/denise $GEN/seis_forward_next_fdmd_from_init_model.inp $GEN/seis_fwi.inp\n"
                        str1=str1+f"python {post_processing_script2}\n"
                        tmp=results_folder.split('/')[-2]+'_'+results_folder.split('/')[-1]
                        str1=str1+f"scp -r {os.path.join(results_folder,'pictures')} {os.path.join('./fwi','all_folders_result_pictures',tmp)}\n"
                        str1=str1+f"python {post_processing_script3}\n"
                    elif pars['computation_platform']=='shaheen':
                        job = '#!/bin/bash\n'
                        job = job+'#SBATCH -N '+str(pars['NNODES'])+'\n'
                        job = job+'#SBATCH --partition=workq\n' # debug
                        # job=job+'#SBATCH -t 4:00:00\n'
                        job=job+'#SBATCH -t 24:00:00\n'
                        job=job+'#SBATCH --account=k1404\n'
                        # batch_file_name=launch_batch_of_jobs(job, batch, res_folder,run_fwi_flag=run_fwi_flag)
                        ##############
                        str1 = job + '#SBATCH --job-name='+'_'+directory+'\n'
                        str1 = str1 + '#SBATCH -o ' + os.path.join(results_folder,'%J_'+res_folder.split('/')[-1]+'_'+directory) + '.out\n'
                        str1 = str1 + '#SBATCH -e ' + os.path.join(results_folder,'%J_'+res_folder.split('/')[-1]+'_'+directory) + '.err\n'
                        str1=str1+'export DENISE=/project/k1404/pavel/DENISE-Black-Edition\n'
                        # str1=str1+f"source ~/.bashrc\n"
                        str1=str1+f"module swap PrgEnv-gnu PrgEnv-intel\n"   
                        str1=str1+f"module swap PrgEnv-cray PrgEnv-intel\n"   
                        str1=str1+f"module load madagascar\n"
                        str1=str1+f"module list\n"
                        str1=str1+f"source /project/k1404/pavel/DENISE-Black-Edition/denise_env/bin/activate\n"
                        str1=str1+f"which python\n"
                        str1=str1+'export GEN='+os.path.join(results_folder,'fld')+'\n'
                        
                        str1=str1+f"srun -n 1 python {data_generation_script}\n"
                        str1=str1+'srun --hint=nomultithread --ntasks-per-node=32 --ntasks-per-socket=16 --cpus-per-task=1 $DENISE/bin/denise $GEN/seis_inversion.inp $GEN/seis_fwi.inp\n'
                        str1=str1+f"srun -n 1 python {post_processing_script}\n"
                        str1=str1+'srun --hint=nomultithread --ntasks-per-node=32 --ntasks-per-socket=16 --cpus-per-task=1 $DENISE/bin/denise $GEN/seis_forward_next_fdmd.inp $GEN/seis_fwi.inp\n'
                        str1=str1+'srun --hint=nomultithread --ntasks-per-node=32 --ntasks-per-socket=16 --cpus-per-task=1 $DENISE/bin/denise $GEN/seis_forward_next_fdmd_from_init_model.inp $GEN/seis_fwi.inp\n'
                        str1=str1+f"srun -n 1 python {post_processing_script2}\n"
                        tmp=results_folder.split('/')[-2]+'_'+results_folder.split('/')[-1]
                        str1=str1+f"scp -r {os.path.join(results_folder,'pictures')} {os.path.join('./fwi','all_folders_result_pictures',tmp)}\n"
                        str1=str1+f"srun -n 1 python {post_processing_script3}\n"
                    ########################
                    batch_file_name=os.path.join(results_folder,'fwi.sh')
                    f = open(batch_file_name,'w');  
                    f.write(str1);
                    f.close()
                    os.chmod(batch_file_name, 0o777)
                    # if run_fwi_flag==1:
                    #     os.system('%s' % 'sbatch ' + batch_file_name)
                    list_of_sbatch_files.append(batch_file_name)
                    ########################
    ####################################   make bash file, which submits sbatch files from login node
    str1 = '#!/bin/bash\n'
    for sbatch_file in list_of_sbatch_files:
        if pars['computation_platform']=='workstation':
            str1 = str1+'bash '+sbatch_file+'\n'
        else:
            str1 = str1+'sbatch '+sbatch_file+'\n'
    batch_file_name = os.path.join('run_cnn_data_generation.sh')
    f = open(batch_file_name, 'w+')
    f.write(str1)
    f.close()
    os.chmod(batch_file_name, 0o777)
    print('bash '+batch_file_name)
    
def flag_run_sequential_cnn_inference():
    """ development started on 22.08.21 
        10 iterations-->cnn-->10 iterations-->cnn-->10 iterations-->cnn--> """
    ########################################################################################################################
    run_fwi_flag = 1
    flag_submitjob_or_exec_locally=1
    calculation_spacing = 25;   vp1 = 1500; vp2 = 4500
    ##############################  
    prediction_path = './predictions/predictions_1273'
    prediction_path = './predictions/predictions_1351'
    prediction_path = './predictions/predictions_1500'
    prediction_path = './predictions/predictions_675'   #weights corresponding to 1497(100m), used for CGG
    # prediction_path = './predictions/predictions_147'   #these weights used for Marm, Over (300m)
    prediction_path = './predictions/predictions_253'   #these weights used for Marm, Over (300m)

    # prediction_path = './predictions/predictions_234'
    prediction_path = './predictions/predictions_355' #to run
    prediction_path = './predictions/predictions_236' #to run
    # prediction_path = './predictions/predictions_237' #to run

    # prediction_path = './predictions/predictions_242'
    # prediction_path = './predictions/predictions_243'
    # prediction_path = './predictions/predictions_244' #to run
    # prediction_path = './predictions/predictions_245' #to run

    prediction_number=prediction_path.split('/')[-1]
    prediction_number=prediction_number.split('predictions_')[-1]
    ##############################
    pars = {'data_mode': 'cnn_13'}  # misfit=5 (GCN)
    # pars={'data_mode':'cnn_13_short_version'}
    pars={'data_mode':'cnn_13_special'}
    # pars={'data_mode':'cnn_14'}
    # pars={'data_mode':'cnn_15'}
    pars.update({'dx': calculation_spacing, 'dz': calculation_spacing,
        'out_shape':[496,150] ,'dsrc':200,
        'data_gen_mode':'pseudo_field','taper_shift':0,
        'last_source_position':'nx',
        } )
    pars.update({'corner_frequency':corner_frequency} )
    pars.update({'full_band':False})
    pars.update({'delete_low_freqs':True} )
    pars.update({'current_data_type':'record_cnn_data'} )
    pars.update({'prediction_path':prediction_path})
    pars.update({
        # 'NNODES': 40,'NPROCX':8,'NPROCY':2,      # big
        'NNODES': 10,'NPROCX':4,'NPROCY':1,      # small
        # 'NNODES': 5,'NPROCX':2,'NPROCY':1,      # small
        # 'NNODES': 2,'NPROCX':1,'NPROCY':1,      # small
        } )
    EXTEND_velocity_models=False
    ############################## apply CNN sequentially on synthetic velocity models.
    #############################################################################
    synthetic_models=1
    if synthetic_models==1:
        # models = [
        #         'Marmousi_1d_lin_300','Overthrust_1d_lin_300', 
        #         # 'Overthrust_300','Marmousi_300',
        #         # 'Overthrust_linear_initial_300','Marmousi_linear_initial_300',
        #         ]
        models=['Marmousi_1d_lin_300','Marmousi_300','Marmousi_linear_initial_300','Overthrust_1d_lin_300']
        # models=['Marmousi_300','Marmousi_linear_initial_300']
        # models = ['Marmousi_1d_lin_300','Overthrust_1d_lin_300']  #   best choice of initial models 2. The most spectacular results.
        # models = ['Overthrust', 'Marmousi']
        # models = ['Overthrust', 'Marmousi','Seam2']
        # models = [ 'Marmousi','Overthrust','Overthrust_linear_initial','Marmousi_linear_initial','Seam2']
        # models=['Overthrust']
        # data_types=['cnn','f_z']
        data_types=['f_z']
        # data_types=['cnn']
        gen_mode='synthetic_data'
        pars.update({'gen_mode':gen_mode})
        pars.update({'extend_model_x':False} )
    else:
        #############################  apply CNN sequentially on real data
        # gen_mode='run_fwi_from_predicted_model'
        gen_mode='test_real_data'
        pars.update({'gen_mode':gen_mode})
        pars.update({'extend_model_x':False} )
        initial_velocity_models_source='cnn_prediction'   # generator, cnn_prediction
        pars.update({'initial_velocity_models_source':initial_velocity_models_source,'prediction_path':prediction_path} )
        data_types=['cnn','f_z'];
        data_types=['f_z'];
        # models=['cgg_tomo_long2','cgg_lin_vp_long']
        models=['cgg_lin_vp_long_300']
        # models=['_cgg_tomo_long1_300','_cgg_tomo_long2_300','_cgg_lin_vp_long_300']
    ##############################
    special_string = 'spacing_'+str(calculation_spacing)
    res_folder = './fwi/ws_'+pars['data_mode']+'_0'
    # res_folder=os.path.join('./fwi','cgg_real_data','multi_'+pars['data_mode']+'_weight_'+str(prediction_number)+'_0')
    res_folder=os.path.join('./fwi','multi','multi_'+pars['data_mode']+'_weight_'+str(prediction_number)+'_0')
    # res_folder = './fwi/ws_'+str(number)+'_'+pars['data_mode']+'_multi_cnn'+'_0'
    print('res_folder', res_folder)
    val = 0
    while os.path.exists(res_folder):
        val = val+1
        res_folder2 = ''
        for tmp in res_folder.split('_')[0:-1]:
            res_folder2 = res_folder2+tmp+'_'
        res_folder = res_folder2+str(val)
        print('res_folder', res_folder)
    os.makedirs(res_folder, exist_ok=True)
    ##############################  declare number of CNN inferences
    n_cnn=30
    ##############################  debugging part of the code
    # parent_results_folder='./fwi/ws_cnn_13_short_version_9/model__Overthrust'
    # results_folder = os.path.join(parent_results_folder, 'stage1')
    # create_velocity_model_file(results_folder,stage=1,pars=pars)
    # create_velocity_model_file(results_folder,stage=2,pars=pars)
    # res_folder=os.path.join('./fwi','cgg_real_data','multi_cnn_13_special_weight_1296_11','model__cgg_lin_vp_long_cnn','stage1')
    # create_velocity_model_file(res_folder,stage=1,pars=pars)
    # exit()
    ########################################################################################################################
    list_of_sbatch_files=[]
    for data_type in data_types[0::]:
        for model_ in models[0::]:
            # create folder for fwi run
            case_name = 'model__'+model_+'_'+data_type
            parent_results_folder = os.path.join(res_folder,case_name)
            os.makedirs(parent_results_folder, exist_ok=True)
            os.makedirs(os.path.join(parent_results_folder, 'fld'), exist_ok=True)
            # find corresponding file with initial models data
            NAME = fnmatch.filter(os.listdir(prediction_path), '*'+model_+'_weights*')[-1]
            # unpack data from file with predictions
            with open(os.path.join(prediction_path, NAME), 'rb') as f:
                data = np.load(f)
                ideal_initial_model = data['ideal_init_model']
                initial_model_1d = data['models_init']
                predicted_initial_model = data['predicted_initial_model']
                MODELS = data['models']
                water_taper=data['water_taper']
                dx = data['dx']
                dz = data['dz']
                data.close()
            # choose, what initial model to use
            if data_type == 'f_z':
                MODELS_INIT_FINAL = initial_model_1d
            elif data_type == 'true':
                MODELS_INIT_FINAL =ideal_initial_model
            elif data_type == 'cnn':
                MODELS_INIT_FINAL = predicted_initial_model
            if EXTEND_velocity_models==True:
                MODELS=(extend(MODELS.T,0,320)).T
                MODELS_INIT_FINAL=(extend(MODELS_INIT_FINAL.T,0,320)).T
                water_taper=(extend(water_taper.T,0,320)).T
            # Plot results
            Plot_image_(MODELS_INIT_FINAL.T, Show_flag=0, Save_flag=1, Title='MODELS_INIT_FINAL_'+case_name,Save_pictures_path=res_folder,c_lim=[vp1,vp2])
            Plot_image_(MODELS.T,     Show_flag=0, Save_flag=1, Title='true_'+case_name, Save_pictures_path=res_folder, c_lim=[vp1, vp2])
            Plot_image_(water_taper.T,     Show_flag=0, Save_flag=1, Title='water_taper_'+case_name, Save_pictures_path=res_folder)
            # Record true and initial velocity models to file to read them after during FWI
            velocity_models_file = os.path.join(parent_results_folder,'velocity_models_file.hdf5')
            f = h5py.File(velocity_models_file, 'w')
            f.create_dataset('models', data=MODELS)
            f.create_dataset('models_init', data=MODELS_INIT_FINAL)
            f.create_dataset('water_taper', data=water_taper)
            f.create_dataset('dx', data=pars['dx'])
            f.create_dataset('dz', data=pars['dz'])
            f.close()
            # generate python imports for python files
            imports = 'import sys,os\n'
            imports = imports+f"sys.path.append(os.getcwd())\n"
            imports = imports+'from F_utils import *\n'
            imports = imports+'from F_plotting import *\n'
            imports = imports+'from F_fwi import *\n'
            imports = imports+'import fnmatch\n'
            imports = imports+'from glob import glob\n'
            imports = imports+'import numpy as np\n'
            imports = imports+'import pyapi_denise_pavel as api\n'
            # run_fwi
            if flag_submitjob_or_exec_locally == 1:
                # construct submission script
                # folder to save job reports
                os.makedirs('./jobs', exist_ok=True)
                str1 = '#!/bin/bash\n'
                str1 = str1+'#SBATCH -N '+str(pars['NNODES'])+'\n'
                str1 = str1+'#SBATCH --partition=workq\n'
                str1 = str1+'#SBATCH -t 24:00:00\n'
                # str1=str1+'#SBATCH -t 12:00:00\n'
                # str1=str1+'#SBATCH -t 0:30:00\n'
                str1 = str1+'#SBATCH --account=k1404\n'  # k1404
                str1 = str1 + '#SBATCH --job-name='+'_'+case_name+'\n'
                str1 = str1 + '#SBATCH -o ' + os.path.join(parent_results_folder, '%J'+ res_folder.split('/')[-1]+'_'+case_name) + '.out\n'
                str1 = str1 + '#SBATCH -e ' + os.path.join(parent_results_folder, '%J' +res_folder.split('/')[-1]+'_'+case_name) + '.err\n'
                ############################################
                for i in range(n_cnn):
                    results_folder = os.path.join(parent_results_folder,'stage'+str(i))
                    os.makedirs(results_folder, exist_ok=True)
                    ######################
                    str1 = str1+'export DENISE=/project/k1404/pavel/DENISE-Black-Edition\n'
                    ############################################
                    str1 = str1+'export GEN=' +os.path.join(results_folder, 'fld')+'\n'
                    str1 = str1+f"source /project/k1404/pavel/DENISE-Black-Edition/denise_env/bin/activate\n"
                    str1 = str1+f"which python\n"
                    str1 = str1+f"pwd\n"
                    # create folders and files to launch denise
                    data_generation=imports
                    # create velocity model file to start FWI
                    
                    # results_folder='./fwi/multi/multi_cnn_13_weight_1500_1/model__Overthrust_cnn/stage1'
                    # create_velocity_model_file(results_folder,stage=1,pars=pars)
                    # create_velocity_model_file(results_folder,stage=i,pars=pars)

                    data_generation = data_generation + f"create_velocity_model_file('{results_folder}',stage={i},pars={pars})\n"
                    if i>0:
                        post_processing = post_processing+f"denise_folder_process('optimizing_space_','{ os.path.join(parent_results_folder,'stage'+str(i-1)) }')\n"
                    velocity_models_file = os.path.join(results_folder, 'velocity_models_file.hdf5')
                    ############################################    data generation script .py
                    # denise_fwi(velocity_models_file,results_folder,os.getcwd(),calculation_spacing=calculation_spacing,mode='generate_task_files',pars=pars)
                    data_generation = data_generation+f"denise_fwi('{velocity_models_file}','{results_folder}','{os.getcwd()}',calculation_spacing={calculation_spacing},pars={pars},mode='generate_task_files')\n"
                    if gen_mode=='test_real_data':
                        data_generation = data_generation+f"copy_su_from_to('{os.path.join('./for_pasha','su_field_high_pass_5_hz/')}','{os.path.join(results_folder,'fld','su')}')\n"
                    data_generation_script_name = os.path.join(results_folder, 'data_generation_script.py')
                    f = open(data_generation_script_name,'w')
                    f.write(data_generation)
                    f.close()
                    str1 = str1+f"srun -n 1 python {os.path.join(results_folder,'data_generation_script.py')}\n"
                    ############################################    create field_data_processing file
                    field_data_processing = imports
                    if gen_mode=='test_real_data':
                        field_data_processing = field_data_processing+f"denise_folder_process('real_data_plotting_session','{results_folder}')\n"
                    else:
                        field_data_processing = field_data_processing+f"denise_folder_process('crop_zero_freqs','{results_folder}',pars={pars})\n"
                        field_data_processing = field_data_processing +f"denise_folder_process('plot','{results_folder}')\n"
                    field_data_processing_script_name = os.path.join(results_folder, 'field_data_processing_script.py')
                    f = open(field_data_processing_script_name, 'w')
                    f.write(field_data_processing)
                    f.close()
                    ############################################    run inversion
                    if gen_mode!='test_real_data':
                        str1 = str1+'srun --hint=nomultithread --ntasks-per-node=32 --ntasks-per-socket=16 --cpus-per-task=1 $DENISE/bin/denise $GEN/seis_forward.inp $GEN/seis_fwi.inp\n'
                        str1 = str1+f"srun -n 1 python {os.path.join(results_folder,'field_data_processing_script.py')}\n"
                    str1 = str1+'srun --hint=nomultithread --ntasks-per-node=32 --ntasks-per-socket=16 --cpus-per-task=1 $DENISE/bin/denise $GEN/seis_inversion.inp $GEN/seis_fwi.inp\n'
                    ############################################ create post processing file
                    post_processing = imports
                    post_processing = post_processing +f"denise_folder_process('plot','{results_folder}')\n"
                    post_processing = post_processing +f"denise_folder_process('real_data_plotting_session','{results_folder}')\n"
                    post_processing = post_processing +f"denise_folder_process('multi_cnn_plotting','{results_folder}')\n"
                    # if i==(n_cnn-1):
                    #     post_processing = post_processing+f"denise_folder_process('optimizing_space_','{os.path.join(results_folder)}')\n"
                    post_processing_script_name = os.path.join(results_folder, 'post_processing_script.py')
                    f = open(post_processing_script_name, 'w');f.write(post_processing);f.close()
                    str1 = str1+f"srun -n 1 python {os.path.join(results_folder,'post_processing_script.py')}\n"
                ############################################    optimize space for all stages.  deprecated
                post_processing2 = imports
                for i in range(n_cnn):
                    results_folder = os.path.join(parent_results_folder,'stage'+str(i))
                    post_processing2=post_processing2+f"denise_folder_process('optimizing_space_','{os.path.join(results_folder)}')\n"
                post_processing_script_name = os.path.join(results_folder, 'post_processing_script2.py')
                f = open(post_processing_script_name,'w');f.write(post_processing2);f.close()
                str1 = str1+f"srun -n 1 python {os.path.join(results_folder,'post_processing_script2.py')}\n"
                ############################################
                # sbatch the script
                batch_file_name = os.path.join(parent_results_folder, 'fwi.sh')
                f = open(batch_file_name, 'w')
                f.write(str1)
                f.close()
                list_of_sbatch_files.append(batch_file_name)
                # if run_fwi_flag == 1:
                #     os.system('%s' % 'sbatch ' + batch_file_name)
                print('create '+batch_file_name)
            a = 1
    ####################################   make bash file, which submits sbatch files from login node
    str1 = '#!/bin/bash\n'
    for sbatch_file in list_of_sbatch_files:
        str1 = str1+'sbatch '+sbatch_file+'\n'
    batch_file_name = os.path.join('multi_run_cnn_data_generation.sh')
    f = open(batch_file_name, 'w')
    f.write(str1)
    f.close()
    print('bash '+batch_file_name)
#######################################################################################################################################################################################################
# flag_reprocess_dataset()
# flag_run_full_fwi_on_testing_models3()
# flag_record_cnn_data2()
# flag_process_dataset2()
# flag_run_sequential_cnn_inference()
# flag_process_dataset()
aa=1