# scp -r ./logs/log1400/predictions_1400 /var/remote/lustre/project/k1404/pavel/DENISE-Black-Edition/predictions/predictions_1400
# scp -r ./logs/log1401/predictions_1401 /var/remote/lustre/project/k1404/pavel/DENISE-Black-Edition/predictions/predictions_1401
# scp -r ./logs/log1500/predictions_1500 /var/remote/lustre/project/k1404/pavel/DENISE-Black-Edition/predictions/predictions_1500
# scp -r ./logs/log548/predictions_548 /var/remote/lustre/project/k1404/pavel/DENISE-Black-Edition/predictions/predictions_548
# scp -r ./logs/log574/predictions_574 /var/remote/lustre/project/k1404/pavel/DENISE-Black-Edition/predictions/predictions_574
# scp -r ./logs/log75/predictions_75 /var/remote/lustre/project/k1404/pavel/DENISE-Black-Edition/predictions/predictions_75
# scp -r ./logs/log141/predictions_141 /var/remote/lustre/project/k1404/pavel/DENISE-Black-Edition/predictions/predictions_141
# scp -r ./logs/log147/predictions_147 /var/remote/lustre/project/k1404/pavel/DENISE-Black-Edition/predictions/predictions_147
# scp -r ./logs/log150/predictions_150/generator_130.pth /var/remote/lustre/project/k1404/pavel/DENISE-Black-Edition/predictions/predictions_147/generator_130.pth
# scp -r ./logs/log253/predictions_253 /var/remote/lustre/project/k1404/pavel/DENISE-Black-Edition/predictions/predictions_253
# scp -r ./logs/log269/predictions_269 /var/remote/lustre/project/k1404/pavel/DENISE-Black-Edition/predictions/predictions_269
# scp -r ./logs/log234/predictions_234 /var/remote/lustre/project/k1404/pavel/DENISE-Black-Edition/predictions/predictions_234
# scp -r ./logs/log235/predictions_235 /var/remote/lustre/project/k1404/pavel/DENISE-Black-Edition/predictions/predictions_235
# scp -r ./logs/log237/predictions_237 /var/remote/lustre/project/k1404/pavel/DENISE-Black-Edition/predictions/predictions_237
# scp -r ./logs/log236/predictions_236 /var/remote/lustre/project/k1404/pavel/DENISE-Black-Edition/predictions/predictions_236
# scp -r ./logs/log244/predictions_244 /var/remote/lustre/project/k1404/pavel/DENISE-Black-Edition/predictions/predictions_244
# scp -r ./logs/log242/predictions_242 /var/remote/lustre/project/k1404/pavel/DENISE-Black-Edition/predictions/predictions_242
# scp -r ./logs/log243/predictions_243 /var/remote/lustre/project/k1404/pavel/DENISE-Black-Edition/predictions/predictions_243
# scp -r ./logs/log245/predictions_245 /var/remote/lustre/project/k1404/pavel/DENISE-Black-Edition/predictions/predictions_245
# scp -r ./logs/log355/predictions_355 /var/remote/lustre/project/k1404/pavel/DENISE-Black-Edition/predictions/predictions_355
# /ibex/ai/home/plotnips/pix2pix
# cd /ibex/scratch/plotnips/intel/pytorch-gpu-data-science-project
# sbatch ./bin/launch-code-server.sbatch
# cd /ibex/scratch/plotnips/intel/MWE/Keras-GAN/pix2pix/bin
# sbatch launch-code-server.sbatch
# /ibex/scratch/plotnips/intel/MWE/Keras-GAN/pix2pix/code_server
# ipython nbconvert notebook.ipynb --to script
from imports_torch import *
from utils_low_wavenumbers_torch import *
from pytictoc import TicToc
##############################################
# sys.path.append('/var/remote/lustre/project/k1404/pavel/DENISE-Black-Edition')
# sys.path.append('/home/plotnips/Dropbox/Log_extrapolation/scripts/DENISE-Black-Edition-master/codes_server')
# import pyapi_denise_pavel as api
##############################################
print('torch.cuda.is_available()=',torch.cuda.is_available())
os.system('nvidia-smi')
# sys.path.append('/home/plotnips/Dropbox/Log_extrapolation/scripts/MWE/F_utils.py')
# sys.path.append('/home/plotnips/Dropbox/Log_extrapolation/scripts/MWE')
plt.rcParams.update({'font.size': 14})
logs_path='/ibex/scratch/plotnips/intel/MWE/Keras-GAN/pix2pix/logs'
logs_path='./logs'
os.makedirs(logs_path,exist_ok=True)
log_save_const = F_calculate_log_number(logs_path,'log','')
Save_pictures_path=logs_path+'/log'+str(log_save_const)
os.makedirs(Save_pictures_path,exist_ok=True)
logname = '/log' + str(log_save_const)+'.txt'
f = open(Save_pictures_path+logname,'w')
sys.stdout = Tee(sys.stdout,f)
flag_show_scaled_data=1
print('logname= ',logname)
print('datetime.now()=',datetime.datetime.now())
#####   dataset paths
dataset_path=['/ibex/scratch/plotnips/intel/MWE/datasets/dataset_oleg_individual_init_better_models_17730_300',
              '/ibex/scratch/plotnips/intel/MWE/datasets/dataset_vl_gen_individual_init_better_models_3000_300']
#########   multi-channel
dataset_path=['/ibex/scratch/plotnips/intel/MWE/datasets/dataset_not_preconditioned_same_new_params_multichannel_input_5000_from_dataset_training_data_10_it_oleg',
        '/ibex/scratch/plotnips/intel/MWE/datasets/dataset_not_preconditioned_same_new_params_multichannel_input_5000_from_dataset_training_data_10_it_vl_gen']
#########   50 iterations
dataset_path=['/ibex/scratch/plotnips/intel/MWE/datasets/dataset_not_preconditioned_same_new_params_50_iter_dv_input_3356_from_dataset_training_data_10_it_oleg',
              '/ibex/scratch/plotnips/intel/MWE/datasets/dataset_not_preconditioned_same_new_params_50_iter_dv_input_2197_from_dataset_training_data_10_it_vl_gen']
dataset_path=[
                '/ibex/scratch/plotnips/intel/MWE/datasets/dataset_not_preconditioned_same_new_params_50_iter_multichannel_input_4814_from_dataset_training_data_10_it_vl_gen',
                '/ibex/scratch/plotnips/intel/MWE/datasets/dataset_not_preconditioned_same_new_params_50_iter_multichannel_input_6318_from_dataset_training_data_10_it_oleg']
dataset_path=[
                '/home/plotnips/Dropbox/Log_extrapolation/scripts/MWE/datasets/dataset_not_preconditioned_same_new_params_50_iter_multichannel_input_4814_from_dataset_training_data_10_it_vl_gen',
                '/home/plotnips/Dropbox/Log_extrapolation/scripts/MWE/datasets/dataset_not_preconditioned_same_new_params_50_iter_multichannel_input_6318_from_dataset_training_data_10_it_oleg']
dataset_path=['/home/plotnips/Dropbox/Log_extrapolation/scripts/MWE/datasets/dataset_not_preconditioned_same_obc_50_iter_dv_input_147_from_dataset_training_data_10_it_vl_gen']
dataset_path=['/ibex/scratch/plotnips/intel/MWE/datasets/dataset_not_preconditioned_same_obc_50_iter_dv_input_147_from_dataset_training_data_10_it_vl_gen',
                '/ibex/scratch/plotnips/intel/MWE/datasets/dataset_not_preconditioned_same_obc_50_iter_dv_input_1513_from_dataset_training_data_10_it_oleg']
dataset_path=['/ibex/scratch/plotnips/intel/MWE/datasets/dataset_not_preconditioned_same_obc_10_iter_dv_input_4000_from_dataset_training_data_10_it_vl_gen']
dataset_path=['/ibex/scratch/plotnips/intel/MWE/datasets/generator1']
dataset_path=['/var/remote/lustre/project/k1404/pavel/DENISE-Black-Edition/datasets/generator1_2']
dataset_path=['/var/remote/lustre/project/k1404/pavel/DENISE-Black-Edition/datasets/generator1_3_strategy_cnn3']
dataset_path=['/var/remote/lustre/project/k1404/pavel/DENISE-Black-Edition/datasets/gen3_7_30_hz_data']
# dataset_path=['/var/remote/lustre/project/k1404/pavel/DENISE-Black-Edition/datasets/gen3_7_30_hz_data_sm_300']
dataset_path=['/var/remote/lustre/project/k1404/pavel/DENISE-Black-Edition/datasets/gen3_7_30_hz_data_cnn18_sm100']
dataset_path=['/var/remote/lustre/project/k1404/pavel/DENISE-Black-Edition/datasets/gen3_marine_data_cnn_fwi_strategy_13_nx_672']
# dataset_path=['/home/plotnips/Dropbox/Log_extrapolation/scripts/MWE/datasets/gen3_7_30_hz_data_cnn18_sm300']
# dataset_path=['/var/remote/lustre/project/k1404/pavel/DENISE-Black-Edition/datasets/gen3_7_30_hz_data_cnn18_sm300']
# dataset_path=['/var/remote/lustre/project/k1404/pavel/DENISE-Black-Edition/datasets/generator1_correct_save']
# dataset_path=['/var/remote/lustre/project/k1404/pavel/DENISE-Black-Edition/datasets/generator1_3_sm_100']
# dataset_path=['/var/remote/lustre/project/k1404/pavel/DENISE-Black-Edition/datasets/generator1_3_sm_300']
# dataset_path=['/home/plotnips/Dropbox/Log_extrapolation/scripts/MWE/datasets/generator1']     #   ws_cnn_5,generator1
#########   26.05.21
# dataset_path=['/ibex/scratch/plotnips/intel/MWE/datasets/dataset_not_preconditioned_same_obc_50_iter_dv_input_36_from_dataset_training_data_10_it_oleg',
#               '/ibex/scratch/plotnips/intel/MWE/datasets/dataset_not_preconditioned_same_obc_50_iter_dv_input_147_from_dataset_training_data_10_it_vl_gen']
# dataset_path=['/home/plotnips/Dropbox/Log_extrapolation/scripts/MWE/datasets/dataset_vl_gen_individual_init_models_36394']
###########################
dataset_path=['/var/remote/lustre/project/k1404/pavel/DENISE-Black-Edition/datasets/gen3_marine_data_cnn_fwi_strategy_13_nx_1200']
####### dataset_path=['/var/remote/lustre/project/k1404/pavel/DENISE-Black-Edition/datasets/gen3_marine_pseudofield_data_cnn_fwi_strategy_13_pseudo_field_nx_500']
####### dataset_path=['/var/remote/lustre/project/k1404/pavel/DENISE-Black-Edition/datasets/gen3_marine_pseudofield_data_cnn_fwi_strategy_13_dsrc_325_pseudo_field_nx_500']
dataset_path=['/var/remote/lustre/project/k1404/pavel/DENISE-Black-Edition/datasets/gen3_marine_pseudofield_data_cnn_fwi_strategy_13_dsrc_200_pseudo_field_nx_496']
dataset_path=['/var/remote/lustre/project/k1404/pavel/DENISE-Black-Edition/datasets/gen3_marine_pseudofield_data_cnn_fwi_strategy_13_dsrc_200_pseudo_field_nx_496_test2']
dataset_path=['/var/remote/lustre/project/k1404/pavel/DENISE-Black-Edition/datasets/gen3_marine_pseudofield_data_cnn_fwi_strategy_13_dsrc_200_pseudo_field_nx_496_test3']
dataset_path=['/var/remote/lustre/project/k1404/pavel/DENISE-Black-Edition/datasets/gen3_marine_pseudofield_data_cnn_fwi_strategy_13_dsrc_200_pseudo_field_nx_496_test4']
dataset_path=[
    '/var/remote/lustre/project/k1404/pavel/DENISE-Black-Edition/datasets/gen3_marine_pseudofield_data_cnn_fwi_strategy_13_dsrc_200_pseudo_field_nx_496',
    '/var/remote/lustre/project/k1404/pavel/DENISE-Black-Edition/datasets/gen3_marine_pseudofield_data_cnn_fwi_strategy_13_dsrc_200_pseudo_field_nx_496_random_trends_1st_attempt'
    ]
dataset_path=[
    # '/var/remote/lustre/project/k1404/pavel/DENISE-Black-Edition/datasets/gen3_marine_pseudofield_data_cnn_13_test_gradhor_start_above_water',
    '/var/remote/lustre/project/k1404/pavel/DENISE-Black-Edition/datasets/gen3_marine_pseudofield_data_cnn_13_test_gradhor_start_above_water_better_scaling']
dataset_path=['/var/remote/lustre/project/k1404/pavel/DENISE-Black-Edition/datasets/gen3_marine_pseudofield_data_cnn_13_test_gradhor_start_above_water_2_']
dataset_path=['/var/remote/lustre/project/k1404/pavel/DENISE-Black-Edition/datasets/gen3_marine_pseudofield_data_cnn_13_test_gradhor_start_above_water_3_21_08']
# dataset_path=['/home/plotnips/Dropbox/Log_extrapolation/scripts/DENISE-Black-Edition-master/datasets/gen3_marine_pseudofield_data_cnn_13_test_gradhor_start_above_water_3_21_08']
dataset_path=['/home/plotnips/Dropbox/Log_extrapolation/scripts/DENISE-Black-Edition-master/datasets/gen3_marine_pseudofield_data_cnn_13_test_gradhor_start_above_water_09_09_-11']
dataset_path=['/data/pavel/pix2pix/datasets/gen3_marine_pseudofield_data_cnn_13_test_gradhor_start_above_water_09_09_-11']
# dataset_path=['/home/plotnips/Dropbox/Log_extrapolation/scripts/DENISE-Black-Edition-master/datasets/gen3_marine_pseudofield_data_cnn_13_test_gradhor_start_above_water_09_09_standardization']
dataset_path=['/var/remote/lustre/project/k1404/pavel/DENISE-Black-Edition/datasets/gen3_marine_pseudofield_data_cnn_13_test_gradhor_start_above_water_09_09_-11']
dataset_path=['/var/remote/lustre/project/k1404/pavel/DENISE-Black-Edition/datasets/gen3_marine_pseudofield_data_cnn_13_test_gradhor_start_above_water_09_09_standardization']
dataset_path=['/var/remote/lustre/project/k1404/pavel/DENISE-Black-Edition/datasets/gen3_marine_pseudofield_data_cnn_13_test_gradhor_start_above_water_09_09_standardization_300']
dataset_path=['./datasets/gen3_marine_pseudofield_data_cnn_13_test_gradhor_start_above_water_09_09_standardization']
# dataset_path=['./datasets/gen3_marine_pseudofield_data_cnn_13_test_gradhor_start_above_water_09_09_standardization_300']
###########################
parser = argparse.ArgumentParser()
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--start_from_pretrained_model", type=int,  default=0,  help="start_from_pretrained_model")
parser.add_argument("--start_with_certain_dataset",  type=int,  default=0,  help="start_with_certain_dataset")
parser.add_argument("--img_height", type=int, default=4*128, help="size of image width, network nx")
parser.add_argument("--img_width", type=int, default=128, help="size of image height, network nz")
parser.add_argument("--architecture", type=int, default=0,help="gan architecture=1,cnn architecture=0")
parser.add_argument("--dataloading_pipeline", type=str, default='matrix', help="dataloading_pipeline=['dataloader','matrix']")
# parser.add_argument("--dataloading_pipeline", type=str, default='dataloader', help="dataloading_pipeline=['dataloader','matrix']")
parser.add_argument("--augment_training_dataset", type=bool, default=True,help="gan architecture=1,cnn architecture=0")
###########################     '1m_1taper','1dv_1taper','1grad_1dv','1dv',1dv_1init,1_fwi_res,dm_i,m_i
inp_channels_type='dm_i'    #   G
# inp_channels_type='m_i'
# inp_channels_type='1dv_1init'
# inp_channels_type='1dv_1init_1sign'
# inp_channels_type='dm_i_1init_1sign'    #   GI
# inp_channels_type='dm_i_1init_1sign_1taper'       #   GIW
# inp_channels_type='1dv'
##########################
inp_channels_type='dm_i'    #   dm_i  ok
inp_channels_type='dm_i_1init'    #   dm_i_1init  GI 
inp_channels_type='dm_i_1taper'    #   dm_i_1taper GW     ok
inp_channels_type='dm_i_1init_1taper'       #   dm_i_1init_1taper   GIW  ok
# ##########################
# inp_channels_type='dm_i_1init_1sign_1taper'       #   GISW
##########################
parser.add_argument("--inp_channels_type", type=str, default=inp_channels_type,help="type of input channels in generator")
if inp_channels_type=='dm_i' or inp_channels_type=='m_i':
    nch=10
elif inp_channels_type=='1m_1taper' or inp_channels_type=='1dv_1taper' or inp_channels_type=='1grad_1dv' or inp_channels_type=='1dv_1init':
    nch=2
elif inp_channels_type=='1dv_1init_1sign':  nch=3
elif inp_channels_type=='dm_i_1init_1sign_1taper':  nch=13
elif inp_channels_type=='dm_i_1init':  nch=11
elif inp_channels_type=='dm_i_1taper':  nch=11
elif inp_channels_type=='dm_i_1init_1taper':  nch=12
else:   nch=1
parser.add_argument("--channels", type=int, default=nch, help="number of input channels in generator")
parser.add_argument("--channels_discriminator", type=int, default=1, help="number of input channels in discriminator")
parser.add_argument("--shuffle_files",type=int,default=1,help="shuffle files")
####    short training
# parser.add_argument("--channels_discriminator", type=int, default=1, help="number of input channels in discriminator")
parser.add_argument("--dataset_path",type=int,default=dataset_path,help="path to the dataset")
# parser.add_argument("--n_epochs",type=int,default=300,help="number of epochs of training")
parser.add_argument("--n_epochs",type=int,default=2,help="number of epochs of training")  #   240
parser.add_argument("--batch_size", type=int,default=4,help="size of the batches") #16
parser.add_argument("--n_cpu",type=int,default=4,help="number of cpu threads to use during batch generation")   #8
# parser.add_argument("--batch_size", type=int,default=4,help="size of the batches") #16
# parser.add_argument("--n_cpu",type=int,default=2,help="number of cpu threads to use during batch generation")   #8
parser.add_argument("--checkpoint_interval", type=int,  default=30, help="interval between model checkpoints")
parser.add_argument("--first_checkpoint_interval", type=int,  default=10, help="interval between model checkpoints before epoch 10")
parser.add_argument("--plotting_interval", type=int,    default=30, help="plot results at each epoch")
# parser.add_argument("--checkpoint_interval", type=int,  default=100, help="interval between model checkpoints")
# parser.add_argument("--plotting_interval", type=int,    default=100, help="plot results at each epoch")
parser.add_argument("--N",type=int,default=-1,help="total number of samples in training and validation losses")
parser.add_argument("--noise_level",type=int,default=0.0,help="level of noise added to x data")
parser.add_argument("--dropout_value",type=int,default=0.25,help="dropout_value in conv_block3")
# parser.add_argument("--dropout_value",type=int,default=0.25,help="dropout_value in conv_block3")
parser.add_argument("--number_of_filters",type=int,default=6,help="number of filters in convolution blocks")      #110,120,160
parser.add_argument("--gpu_number",type=int,default=0,help="which gpu to use, provide number from 0 to 3")
parser.add_argument("--plot_samples",type=int,default=1,help="plot samples or not")
parser.add_argument("--save_weights_to_folder",type=int,default=1,help="save_early_stopping_weights_to_folder")
# "noise_level": 0.1, "dropout_value": 0.3,"number_of_filters": 110,
######################################
opt = parser.parse_args()
opt.save_path=Save_pictures_path;   opt.log_save_const=log_save_const;  start_from_pretrained_model=opt.start_from_pretrained_model
######################################
path=[];    counter=0
for dataset_path in opt.dataset_path:
    print('dataset_path=',dataset_path)
    files_path=fnmatch.filter(glob(dataset_path+'/*'),'*.npz'); 
    path_=sorted(files_path)
    # print(path_[-12:])
    if len(opt.dataset_path)>1:
        if counter==0:      
            val=len(path_)
            print('first dataset size=',val)
        elif counter==1:    
            val=len(path_)
            print('second dataset size=',val)
        ################
        # # val1=11208;     val2=15000
        # val1=2000;     val2=3000
        # # val1=11208;     val2=2000
        # val1=50;     val2=50
        # # val1=20;     val2=20
        # # val1=200;     val2=200
        # val1=6000;     val2=6000
        # val1=15000;     val2=3000
        # val1=3000;     val2=1000    #75%,25%, small dataset
        # sz=10000;   val1=int(sz*0.75);     val2=int(sz*0.25)    #75%,25%, small dataset
        path_=path_[-val:]
    path=path+path_
    counter=counter+1
print('len(path)=',len(path))
######################################  denise data
# path_test=path
# path=list(set(path)-set(path_test));
######################################  sfgpu data
path_test=fnmatch.filter(path,'*__*')
# path_test=fnmatch.filter(path_test,'*1d_lin_100*')
# path_test=fnmatch.filter(path_test,'*1d_lin_300*')
path=list(set(path)-set(path_test));    
path=sorted(path)
# path=path[0:30]
# path_test=fnmatch.filter(path_test,'*spacing800.npz')
######################################   define size of dataset  #   split dataset
if opt.N==-1:
    opt.N=len(path)
if len(opt.dataset_path)==2:
    if opt.shuffle_files==1:
        random.shuffle(path)
######################################  split by percent
# path=path[0:opt.N]
# opt.train_frac= 0.8
# path_train=path[0:int(len(path)*opt.train_frac)]
# path_valid=list(set(path)-set(path_train))
######################################   custom split 1. 'super tiny dataset'
path_valid=path[-20:]      #   -200
path_train= list(set(path)-set(path_valid))
path_train=path_train[0:50]
######################################   custom split 1. 'tiny dataset'
# path_valid=path[-50:]      #   -200
# path_train=list(set(path)-set(path_valid))
# path_train=path_train[0:200]
######################################   custom split 2. 'big dataset. all files'
# path_valid=path[-400:]      #   -200
# # path_valid=path[-100:]      #   -200
# path_train=list(set(path)-set(path_valid))
# # # path_train=path_train[0:1000] 
######################################
# path_valid=path[-500:]
# path_train=list(set(path)-set(path_valid))
# path_train=path_train[0:2000]
######################################   load model network weights
# with open('/var/remote/lustre/project/k1404/pavel/DENISE-Black-Edition/datasets/gen3_marine_pseudofield_data_cnn_13_test_gradhor_start_above_water_09_09_standardization/scaling_constants_dict.pkl','rb') as input:
#     scaling_constants_dict=pickle.load(input)
# with open('/var/remote/lustre/project/k1404/pavel/DENISE-Black-Edition/datasets/gen3_marine_pseudofield_data_cnn_13_test_gradhor_start_above_water_09_09_standardization/scaling_constants_dict.pkl','rb') as input:
#     scaling_constants_dict2=pickle.load(input)
if opt.n_epochs==-1 or start_from_pretrained_model==1 or opt.n_epochs==-11:
    log_name=175;   epoch_name=9
    log_name=176;   epoch_name=9
    log_name=174;   epoch_name=3
    log_name=200;   epoch_name=4499
    log_name=222;   epoch_name=99
    log_name=234;   epoch_name=300
    # log_name=235;   epoch_name=300
    log_name=284;   epoch_name=0
    log_name=256;   epoch_name=0
    log_name=277;   epoch_name=0
    log_name=254;   epoch_name=2000
    log_name=254;   epoch_name=4000
    # log_name=321;   epoch_name=570
    log_name=334;   epoch_name=999
    log_name=338;   epoch_name=450
    log_name=367;   epoch_name=299
    log_name=377;   epoch_name=60
    log_name=387;   epoch_name=20
    log_name=388;   epoch_name=96
    log_name=408;   epoch_name=1
    log_name=413;   epoch_name=3
    log_name=395;   epoch_name=900
    log_name=390;   epoch_name=330
    # log_name=403;   epoch_name=536
    # log_name=400;   epoch_name=1160
    # log_name=419;   epoch_name=840
    # log_name=419;   epoch_name=390
    # log_name=453;   epoch_name=444
    # log_name=453;   epoch_name=468  #403
    log_name=453;   epoch_name=892  #403
    #  log_name=452;   epoch_name=480  #390
    log_name=452;   epoch_name=915  #390
    # log_name=454;   epoch_name=420
    # log_name=465;   epoch_name=2370
    # log_name=467;   epoch_name=2280
    # log_name=550;   epoch_name=180
    # log_name=569;   epoch_name=340
    # log_name=570;   epoch_name=240
    log_name=656;   epoch_name=99
    log_name=663;   epoch_name=99
    log_name=703;   epoch_name=159
    log_name=717;   epoch_name=179
    log_name=718;   epoch_name=179
    log_name=749;   epoch_name=20
    log_name=751;   epoch_name=10
    log_name=755;   epoch_name=159
    # log_name=756;   epoch_name=159
    log_name=816;   epoch_name=3
    # log_name=831;   epoch_name=0
    log_name=845;   epoch_name=14
    # log_name=844;   epoch_name=14
    # log_name=842;   epoch_name=14
    # log_name=843;   epoch_name=14
    log_name=888;   epoch_name=199
    log_name=899;   epoch_name=399
    log_name=931;   epoch_name=199
    log_name=932;   epoch_name=199
    log_name=946;   epoch_name=60
    log_name=953;   epoch_name=1470
    log_name=982;   epoch_name=1099
    log_name=1064;   epoch_name=8
    log_name=1011;   epoch_name=390
    # log_name=1032;   epoch_name=570
    # log_name=1019;   epoch_name=510
    # log_name=1009;   epoch_name=420
    ##########  make statistics
    log_name=1084;   epoch_name=960
    log_name=1089;   epoch_name=120
    log_name=1098;   epoch_name=480 #   fusion-net started from 0.  the most used example
    # log_name=1099;   epoch_name=330
    # log_name=1100;   epoch_name=480
    log_name=1189;   epoch_name=14
    # log_name=1167;   epoch_name=2
    log_name=1202;   epoch_name=320
    log_name=1269;   epoch_name=160
    log_name=1290;   epoch_name=99
    log_name=1357;   epoch_name=19
    log_name=1208;   epoch_name=120
    log_name=1207;   epoch_name=899
    log_name=1227;   epoch_name=840
    log_name=1226;   epoch_name=999
    log_name=519;   epoch_name=99
    log_name=1263;   epoch_name=60
    log_name=1260;   epoch_name=120
    log_name=1275;   epoch_name=32
    log_name=1273;   epoch_name=360
    # log_name=1219;   epoch_name=999
    log_name=705;   epoch_name=1
    log_name=27;    epoch_name=60
    log_name=57;    epoch_name=100
    log_name=1350;   epoch_name=90
    log_name=1353;   epoch_name=90
    log_name=1390;   epoch_name=90
    log_name=1391;   epoch_name=40
    log_name=1497;   
    # log_name=1514;   
    ### smoothing 300
    log_name=79;    epoch_name=60
    #   raven.kaust.edu.sa
    log_name=67 #or 17,67, G
    # log_name=64 #GI
    log_name=65 #GW
    # log_name=66 #GIW
    ### smoothing 100
    # log_name=102 #or 17,67, G
    # log_name=103 #GI
    # log_name=104 #GW
    # log_name=105 #GIW
    ####    tests
    log_name=218
    log_name=219
    log_name=220
    # log_name=221
    # log_name=222
    # log_name=223
    # log_name=224
    # log_name=225
    # log_name=226
    # log_name=227
    # log_name=228
    # log_name=229
    # log_name=230
    # log_name=231
    # log_name=232
    # log_name=233
    ####
    tmp=logs_path+'/log'+str(log_name)+'/opt.txt'
    # if os.path.exists(tmp):
    #     parser = ArgumentParser();  opt = parser.parse_args()
    print('loading opt parameters from ',tmp)
    parser = ArgumentParser();  loaded_opt = parser.parse_args()
    with open(tmp,'r') as f:
        print('loading json.load')
        loaded_opt.__dict__=json.load(f)
    opt.save_path=Save_pictures_path
    opt.log_save_const=log_save_const
    opt.inp_channels_type=loaded_opt.inp_channels_type
    opt.channels=loaded_opt.channels
    opt.history=loaded_opt.history
    opt.noise_level=loaded_opt.noise_level
    opt.dropout_value=loaded_opt.dropout_value
    opt.number_of_filters=loaded_opt.number_of_filters
    ########################################################################################################
    tmp=fnmatch.filter(os.listdir(logs_path+'/log'+str(log_name)),'*.pth')
    epoch_list=[]
    for ii in tmp:
        epoch_list.append( int( ii.split('.pth')[0].split('generator_')[-1] ) )
    epoch_list=np.array(epoch_list)
    ########################################    R2 score plots  ################################################################
    ########    R2!!!!!!.find early stopping epoch name and then choose closest recorded epoch name
    r2_val=np.array(opt.history['loss_target_pred_valid'])      #a=np.amax(r2_val)
    early_stopping_epoch=np.where(r2_val==np.max(r2_val))[0][0]
    diff=np.abs(epoch_list-early_stopping_epoch).squeeze()
    c=np.where( diff==np.min(diff) )
    available_early_stopping_epoch= epoch_list [c[0][0]]
    early_stop_score=[opt.history['loss_target_pred_train'][early_stopping_epoch],opt.history['loss_target_pred_valid'][early_stopping_epoch]]
    avail_early_stop_score=[opt.history['loss_target_pred_train'][available_early_stopping_epoch],opt.history['loss_target_pred_valid'][available_early_stopping_epoch]]
    prediction_score1='early stop epoch=' +str(early_stopping_epoch)+',R2 train/val='+numstr(early_stop_score[0])+'/'+numstr(early_stop_score[1])
    prediction_score2='available early stop epoch=' +str(available_early_stopping_epoch)+',R2 train/val='+numstr(avail_early_stop_score[0])+'/'+numstr(avail_early_stop_score[1])
    Plot_r2_losses_for_geophysics(opt.history,available_early_stopping_epoch,save_name='log'+str(log_save_const)+'losses_r2_for_geophysics_early_stopping1',Title=prediction_score1,Save_pictures_path=Save_pictures_path,Save_flag=1)
    Plot_r2_losses_for_geophysics(opt.history,available_early_stopping_epoch,save_name='log'+str(log_save_const)+'losses_r2_for_geophysics_early_stopping2',Title=prediction_score2,Save_pictures_path=Save_pictures_path,Save_flag=1)
    ########################################    L1 score plots  ################################################################
    ########    ML loss plots. In this case L1
    L1_val=np.array(opt.history['loss_pixel_val'])      #a=np.amax(r2_val)
    early_stopping_epoch2=np.where(L1_val==np.min(L1_val))[0][0]
    diff2=np.abs(epoch_list-early_stopping_epoch2).squeeze()
    c2=np.where( diff2==np.min(diff2) )
    available_early_stopping_epoch2= epoch_list [c2[0][0]]
    early_stop_score=[opt.history['loss_pixel_train'][early_stopping_epoch2],opt.history['loss_pixel_val'][early_stopping_epoch2]]
    avail_early_stop_score=[opt.history['loss_pixel_train'][available_early_stopping_epoch2],opt.history['loss_pixel_val'][available_early_stopping_epoch2]]

    prediction_score1_='early stop epoch=' +str(early_stopping_epoch2)+',L1 train/val='+numstr4(early_stop_score[0])+'/'+numstr4(early_stop_score[1])
    prediction_score2_='available early stop epoch=' +str(available_early_stopping_epoch2)+',L1 train/val='+numstr4(avail_early_stop_score[0])+'/'+numstr4(avail_early_stop_score[1])
    Plot_L1_losses_for_geophysics(opt.history,available_early_stopping_epoch2,save_name='log'+str(log_save_const)+'losses_l1_for_geophysics_early_stopping1',Title=prediction_score1_,Save_pictures_path=Save_pictures_path,Save_flag=1)
    Plot_L1_losses_for_geophysics(opt.history,available_early_stopping_epoch2,save_name='log'+str(log_save_const)+'losses_l1_for_geophysics_early_stopping2',Title=prediction_score2_,Save_pictures_path=Save_pictures_path,Save_flag=1)
    ########
    epoch_name=available_early_stopping_epoch   #   pick up from r2 loss
    epoch_name=available_early_stopping_epoch2   #   pick up from l1 loss
    opt.generator_model_name=       './logs/log'+str(log_name)+'/generator_'+str(epoch_name)+'.pth'
    opt.discriminator_model_name=   './logs/log'+str(log_name)+'/discriminator_'+str(epoch_name)+'.pth'
    Plot_r2_losses_2(opt.history,Title='log'+str(log_name)+'_'+str(epoch_name) + 'losses_r2', Save_pictures_path=Save_pictures_path,Save_flag=1)
    Plot_unet_losses(opt.history,Title='log'+str(log_name)+'_'+str(epoch_name) + 'losses_unet', Save_pictures_path=Save_pictures_path,Save_flag=1)
    Plot_unet_losses_2(opt.history,Title='log'+str(log_name)+'_'+str(epoch_name) + 'losses_pixel', Save_pictures_path=Save_pictures_path,Save_flag=1)
    ss=1
######################################   load dataset paths
if opt.start_with_certain_dataset==1:
    if start_from_pretrained_model==1:
        do_nothing=1
    else:
        log_name=390    #combined dataset
        # log_name=403    #vladimir dataset
        # log_name=650    #testing dataset small
        # log_name=662    #testing dataset big
        log_name=698    #testing dataset 2, big
        log_name=717    #testing dataset 2, big
        log_name=749    #whole dataset, no preconditioning
        log_name=751    #whole dataset, with preconditioning
        log_name=753    #small dataset, with preconditioning
        log_name=791
        log_name=807
        log_name=816    #dv only
        log_name=899
        log_name=961
        log_name=953
        log_name=982
        log_name=1064
        log_name=1011
        # log_name=1032
        # log_name=1019
        # log_name=1047
        # log_name=1048
        # log_name=1049
        log_name=1089
        log_name=1098
        # log_name=1099
        # log_name=1100
        # log_name=1084
        log_name=1189
        log_name=1202
        log_name=1269
        log_name=1226
        log_name=1273
        log_name=1391
        log_name=79     # dataset standardization 300 smoothing
    tmp=logs_path+'/log'+str(log_name)+'/opt.txt'
    parser = ArgumentParser();  opt_old = parser.parse_args()
    print('loading same dataset files from ',tmp)
    with open(tmp,'r') as f:
        opt_old.__dict__=json.load(f)
    ##########################################  exact copy
    opt.path_test=path_test
    # opt.path_test=return_existing_file_list(opt_old.path_test,opt.dataset_path)
    opt.path_valid=return_existing_file_list(opt_old.path_valid,opt.dataset_path)
    opt.path_train=return_existing_file_list(opt_old.path_train,opt.dataset_path)
else:
    opt.path_test=path_test;    opt.path_valid=path_valid;  opt.path_train=path_train;
######################################
# opt.path_test=[opt.path_test[2]]
# opt.path_valid=opt.path_valid[-20:]
# opt.path_train=opt.path_train[0:50]
print('len(opt.path_train)=',len(opt.path_train))
print('len(opt.path_valid)=',len(opt.path_valid))
print('len(opt.path_test)=', len(opt.path_test))
print('(opt)=', (opt))
######################################
torch.backends.cudnn.benchmark = True
cuda = True if torch.cuda.is_available() else False
print('torch.cuda.is_available()=',torch.cuda.is_available())
# Loss functions
criterion_GAN = torch.nn.MSELoss()
criterion_pixelwise = torch.nn.L1Loss()
criterion_R2Loss=R2Loss()
criterion_R2Loss_custom=R2Loss_custom()
criterion_R2Loss_custom_average=R2Loss_custom_average()
criterion_NRMS=NRMS_loss_custom()
# Loss weight of L1 pixel-wise loss between translated image and real image
opt.lambda_pixel = 100
opt.lambda_pixel = 0.01
# Calculate output of image discriminator (PatchGAN)
patch=(1,opt.img_height // 2 ** 4, opt.img_width // 2 ** 4); opt.patch=patch
# Initialize generator and discriminator
torch.manual_seed(123)
torch.cuda.manual_seed(123)
np.random.seed(123)
random.seed(123)
###################
# gpu_number=opt.gpu_number        # choose number from 0 to 3
if cuda:
    gpu_ids = []
    gpu_ids += [gpu_id for gpu_id in range(torch.cuda.device_count())]
    opt.gpu_number=gpu_ids
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cuda:2")
    # opt.gpu_number=[2]
    # for gpu_number in opt.gpu_number:
    #     device = torch.device("cuda:"+str(gpu_number) if torch.cuda.is_available() else "cpu")
    #     print('Device',device)
###################    unet Generator
# generator = GeneratorUNet(in_channels=opt.channels,DROPOUT=opt.dropout_value);    gen_name='unet'
# # generator = GeneratorUNet_old_configuration(in_channels=opt.channels,DROPOUT=opt.dropout_value)
# # generator = GeneratorUNet_big(in_channels=opt.channels,DROPOUT=opt.dropout_value)
###################    fusion_net Generator
fusion_net_parameters=(opt.channels,1,opt.number_of_filters)
print('fusion net parameters=(input_nc,output_nc,ngf)',fusion_net_parameters)
# generator=nn.DataParallel(FusionGenerator(fusion_net_parameters[0],fusion_net_parameters[1],fusion_net_parameters[2]),device_ids=device )
# generator=nn.DataParallel(FusionGenerator(fusion_net_parameters[0],fusion_net_parameters[1],fusion_net_parameters[2]), output_device=device )
# generator=nn.DataParallel(FusionGenerator(fusion_net_parameters[0],fusion_net_parameters[1],fusion_net_parameters[2],opt),device_ids=[gpu_number])
generator=nn.DataParallel(FusionGenerator(fusion_net_parameters[0],fusion_net_parameters[1],fusion_net_parameters[2],opt),device_ids=opt.gpu_number)
discriminator = Discriminator(in_channels=opt.channels_discriminator)
gen_name='fusion'
summary(generator.to(device),(opt.channels,opt.img_height,opt.img_width))
# exit()
################## 
print('DL network name=',gen_name)
if cuda:
    # generator=generator.cuda(device=device)
    generator=generator.to(device)
    discriminator = discriminator.cuda(device=device)
    criterion_GAN.cuda(device=device)
    criterion_pixelwise.cuda(device=device)
    criterion_R2Loss.cuda(device=device)
    criterion_R2Loss_custom.cuda(device=device)
    criterion_NRMS.cuda(device=device)
if opt.n_epochs==-1 or start_from_pretrained_model==1:
    # opt.generator_model_name='./logs/log1351/predictions_1351/generator_weights.pth'
    print('Load pretrained models',opt.generator_model_name)
    generator.load_state_dict(torch.load(opt.generator_model_name))
    if opt.architecture==1:
        discriminator.load_state_dict(torch.load(opt.discriminator_model_name))
    history=opt.history
else:       # Initialize weights
    if gen_name=='unet':
        generator.apply(weights_init_normal)       
    discriminator.apply(weights_init_normal)
# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
# Configure dataloaders
transforms_=[]
print('opt.batch_size=',opt.batch_size)
with open(opt.save_path+'/'+'opt.txt', 'w') as f:
    json.dump(opt.__dict__, f, indent=2)
tracking_train_dataloader=DataLoader(ImageDataset(opt,
    mode=None,file_list=opt.path_train[0:4]),     #+path_test[-2:-1]+path_test[-4:-3]
    batch_size=1,shuffle=False,num_workers=1)
tracking_val_dataloader=DataLoader(ImageDataset(opt,
    mode=None,file_list=opt.path_valid[0:2]),
    batch_size=1,shuffle=False,num_workers=1)
tracking_train_dataloader2=DataLoader(ImageDataset(opt,
    mode=None,file_list=opt.path_train[0:9]),     #+path_test[-2:-1]+path_test[-4:-3]
    batch_size=1,shuffle=False,num_workers=1)
tracking_val_dataloader2=DataLoader(ImageDataset(opt,
    mode=None,file_list=opt.path_valid[0:1]),
    batch_size=1,shuffle=False,num_workers=1)
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# opt.path_test=opt.path_test[0:4]
# opt.path_test=[opt.path_test[-2]]
# opt.path_test=opt.path_test[-2:]
# opt.path_test=fnmatch.filter(opt.path_test,'*cgg*')
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
test_dataset = ImageDataset(opt, mode="test")
test_dataloader=DataLoader(test_dataset,batch_size=1,shuffle=False,num_workers=1)
# sample_models2(tracking_train_dataloader2,generator,opt,opt.n_epochs,flag_show_scaled_data=0)
# sample_models3(tracking_train_dataloader2,generator,opt,opt.n_epochs,flag_show_scaled_data=0)
# exit()
sample_dataset(tracking_train_dataloader,generator,opt,opt.n_epochs,flag_show_scaled_data=1,data_mode='test',record_weights=0)
# sample_dataset(tracking_val_dataloader,generator,opt,opt.n_epochs,flag_show_scaled_data=1,data_mode='test',record_weights=0)

if opt.n_epochs==-11:
    os.path.join('./logs','log1273')
    os.listdir(os.path.join('./logs','log1273'))
    weights=fnmatch.filter(glob(os.path.join('./logs','log1273')+'/*'),'*.pth')
    weights=sorted(weights)
    for i in range(len(weights)):
        epoch=int(weights[i].split('/generator_')[-1].split('.pth')[0] )
        generator.load_state_dict(torch.load(weights[i]))
        sample_dataset(test_dataloader,generator,opt,epoch,flag_show_scaled_data=1,data_mode='test',record_weights=0)    
    # exit()
if opt.n_epochs==-1:
    generator.eval()
    prediction_score='chosen epoch '+str(epoch_name+1)+', train/val score='+numstr3(history['loss_target_pred_train'][epoch_name])+'/'+numstr3(history['loss_target_pred_valid'][epoch_name])
    print(prediction_score)
    Plot_r2_losses_3(history,Title='log'+str(log_save_const) + 'losses_r2', Save_pictures_path=Save_pictures_path,Save_flag=1)
    Plot_r2_losses_for_geophysics(history,epoch_name,save_name='log'+str(log_save_const)+'losses_r2_for_geophysics_early_stopping',Title=prediction_score, Save_pictures_path=Save_pictures_path,Save_flag=1)
    MAX=np.max(history['loss_target_pred_valid'])
    ARGMAX=np.argmax(history['loss_target_pred_valid'])
    print('best accuracy=',MAX,'best epoch=',ARGMAX+1 )
    print(len(history['loss_target_pred_valid']))
    sample_dataset_geophysics(test_dataloader,generator,opt,opt.n_epochs,flag_show_scaled_data=1,record_weights=1,data_mode='test')
    # sample_dataset(test_dataloader,generator,opt,opt.n_epochs,flag_show_scaled_data=1,data_mode='test_8_pics_',record_weights=0)
    exit()
    # sample_dataset_geophysics(tracking_train_dataloader,generator,opt,opt.n_epochs,flag_show_scaled_data=1,data_mode='train')
    # sample_dataset(test_dataloader,generator,opt,opt.n_epochs,flag_show_scaled_data=1,data_mode='test',record_weights=1)
    # sample_dataset(tracking_train_dataloader,generator,opt,opt.n_epochs,flag_show_scaled_data=1,data_mode='train')
train_dataset = ImageDataset(opt, transforms_=transforms_, mode="train")
val_dataset = ImageDataset  (opt, transforms_=transforms_, mode="val")
if opt.dataloading_pipeline=='matrix' and opt.n_epochs!=-1:
    single_thread_loading=1
    ########################### extracting data from files
    output_train=[];    output_valid=[]
    if single_thread_loading==0:
        n_cpus=5
        n_cpus=multiprocessing.cpu_count()-5
        pool = multiprocessing.Pool(n_cpus)
        temp = partial(extract_ml_data_to_list2,opt,train_dataset)
        output_train = pool.map(func=temp, iterable=opt.path_train)
        pool.close();   pool.join()
        pool = multiprocessing.Pool(n_cpus)
        temp = partial(extract_ml_data_to_list2,opt,val_dataset)
        output_valid = pool.map(func=temp, iterable=opt.path_valid)
        pool.close();   pool.join()
    else:
        for path_ in opt.path_train:
            # a1=extract_ml_data_to_list(opt, path_)
            # a2=extract_ml_data_to_list2(opt,train_dataset, path_)
            output_train.append(extract_ml_data_to_list2(opt,train_dataset, path_))
        for path_ in opt.path_valid:
            output_valid.append(extract_ml_data_to_list2(opt,val_dataset,   path_))
    ########################### loading data into matrices
    x_train=[];t_train=[];  taper_train=[];
    for out in output_train:
        x_train.append(out[0])
        t_train.append(out[1])
        taper_train.append(out[2])
    x_valid=[];t_valid=[];  taper_valid=[];
    for out in output_valid:
        x_valid.append(out[0])
        if out[0].shape[0]<10 and opt.channels==10:
            print(' problem, out[0].shape[0]<10')
        t_valid.append(out[1])
        taper_valid.append(out[2])
    ###########################
    # if opt.augment_training_dataset==True:
    #     for out in output:
    #         ##  flip data from left to right along X axis
    #         x_train.append(out[0])
    #         x_train.append( np.flip(out[0],axis=1)  )
    #         t_train.append(out[1])
    #         t_train.append( np.flip(out[1],axis=1)  )
    #         # Plot_image(out[0].squeeze().T,Show_flag=0,Save_flag=1,Title='x_batch',Aspect='equal',Save_pictures_path=opt.save_path)
    #         # Plot_image(np.flip(out[0],axis=1).squeeze().T,Show_flag=0,Save_flag=1,Title='x_batch_flipped',Aspect='equal',Save_pictures_path=opt.save_path)
    #         # Plot_image(out[1].squeeze().T,Show_flag=0,Save_flag=1,Title='t_batch',Aspect='equal',Save_pictures_path=opt.save_path)
    #         # Plot_image(np.flip(out[1],axis=1).squeeze().T,Show_flag=0,Save_flag=1,Title='t_batch_flipped',Aspect='equal',Save_pictures_path=opt.save_path)
    # else:
    #     for out in output:
    #         x_train.append(out[0])
    #         t_train.append(out[1])
    ###########################
    train_dataset_from_array_original=torch.utils.data.TensorDataset(torch.FloatTensor(x_train),torch.FloatTensor(t_train),torch.FloatTensor(taper_train))
    valid_dataset_from_array_original=torch.utils.data.TensorDataset(torch.FloatTensor(x_valid),torch.FloatTensor(t_valid),torch.FloatTensor(taper_valid))
#######################################
####################################### Apply augmentation
if opt.augment_training_dataset==True and opt.n_epochs!=-1:
    ################################### dataloading from files
    # train_dataset = FlipLoader(train_dataset, 0.5)
    # train_dataset = NoiseLoader(train_dataset, 0.5)
    ################################### dataloading from array. 
    Noise=opt.noise_level
    train_dataset_from_array=FlipLoader_on_tensordataset(train_dataset_from_array_original,opt,p=0.0)
    train_dataset_from_array=NoiseLoader_on_tensordataset(train_dataset_from_array,opt,p=1.0,  c=Noise)
    ####################    Test
    # train_dataset_from_array=FlipLoader_on_tensordataset(train_dataset_from_array_original,opt,p=1.0)
    # train_dataset_from_array=NoiseLoader_on_tensordataset(train_dataset_from_array,opt,p=1.0,  c=Noise)
    ###     Plotting
    # a=train_dataset_from_array_original.__getitem__(0)[0]
    # b=train_dataset_from_array.__getitem__(0)[0]
    # a_=b[0,::].min()
    # b_=b[0,::].max()
    # fig_size = plt.rcParams["figure.figsize"]
    # fig_size[0] = 12.4
    # fig_size[1] = 5.0   #height
    # plt.rcParams["figure.figsize"] = fig_size
    # n_row=2;    n_col=1
    # gs = gridspec.GridSpec(n_row,n_col)
    # gs.update(left=0.0, right=0.94, wspace=0.0, hspace=0.03)
    # ax=[None]*6
    # labels=['a','b','c','d']
    # Fontsize=15
    # text_Fontsize=30
    # labelsize=14
    # fig=plt.figure()
    # ax[0]=fig.add_subplot(gs[0,0]);
    # ax[0].imshow(a[9,::].T,vmin=-b_,vmax=b_)
    # ax[0].text(30,30, labels[0], fontsize=text_Fontsize,color = "black",weight="bold")
    # ax[0].tick_params(axis='y',labelsize=17)
    # ax[0].axes.xaxis.set_visible(False)
    # ax[1]=fig.add_subplot(gs[1,0]);
    # last_image=ax[1].imshow(b[9,::].T,vmin=-b_,vmax=b_)
    # ax[1].text(30,30, labels[1], fontsize=text_Fontsize,color = "black",weight="bold")
    # ax[1].tick_params(axis='x',labelsize=17)
    # ax[1].tick_params(axis='y',labelsize=17)
    # cbar_ax = fig.add_axes([0.80, 0.12, 0.02, 0.75])
    # cbar=fig.colorbar(last_image,cax=cbar_ax)
    # for t in cbar.ax.get_yticklabels():
    #     t.set_fontsize(16)
    # print('saving to '+os.path.join(opt.save_path,'noise_augmentation2.png'))
    # plt.savefig(os.path.join(opt.save_path,'noise_augmentation2.png'),dpi=300,bbox_inches='tight')
    ###########################
    train_dataset_from_array2=FlipLoader_on_tensordataset(train_dataset_from_array_original,opt,p=1.0)
    train_dataset_from_array2=NoiseLoader_on_tensordataset(train_dataset_from_array2,opt,p=1.0,c=Noise)
    train_dataset_from_array3=torch.utils.data.ConcatDataset([train_dataset_from_array2,train_dataset_from_array])
    valid_dataset_from_array=FlipLoader_on_tensordataset(valid_dataset_from_array_original,opt,p=0.0)
    # valid_dataset_from_array=NoiseLoader_on_tensordataset(valid_dataset_from_array,opt,p=0.1,  c=Noise)
    valid_dataset_from_array2=FlipLoader_on_tensordataset(valid_dataset_from_array_original,opt,p=1.0)
    # valid_dataset_from_array2=NoiseLoader_on_tensordataset(valid_dataset_from_array2,opt,p=0.1,c=Noise)
    valid_dataset_from_array3=torch.utils.data.ConcatDataset([valid_dataset_from_array2,valid_dataset_from_array])
    ########################### check tensor dataset, plot it
    # ITER1=iter(train_dataset_from_array)
    # ITER2=iter(train_dataset_from_array2)
    # ITER3=iter(train_dataset_from_array3)
    # # for a1,a2 in enumerate():
    # #     Plot_image(np.concatenate((a1[-1,::],a2[-1,::],x_train[i][-1,::]),axis=1).T,Show_flag=0,Save_flag=1,Title='comparison_'+str(i),Aspect='equal',Save_pictures_path=opt.save_path)
    # len(train_dataset_from_array2)
    # for i in range(30):
    #     a1=next(ITER1)[0].cpu().detach().numpy()
    #     a2=next(ITER2)[0].cpu().detach().numpy()
    #     a3=next(ITER3)[0].cpu().detach().numpy()
    #     Plot_image(np.concatenate((a1[-1,::],a2[-1,::],a3[-1,::],x_train[i][-1,::]),axis=1).T,Show_flag=0,Save_flag=1,Title='comparison_'+str(i),Aspect='equal',Save_pictures_path=opt.save_path)
    #     sss=1
    ###########################
#######################################     Apply dataloader to tensor datasets
train_dataloader=DataLoader(train_dataset,batch_size=opt.batch_size,shuffle=True,num_workers=opt.n_cpu,pin_memory=True)
validation_dataloader=DataLoader(val_dataset,batch_size=opt.batch_size,shuffle=False,num_workers=opt.n_cpu,pin_memory=True)
if opt.n_epochs!=-1:
    train_dataloader_from_array=DataLoader(train_dataset_from_array3,batch_size=opt.batch_size,shuffle=True,num_workers=opt.n_cpu,pin_memory=True)
    valid_dataloader_from_array=DataLoader(valid_dataset_from_array3,batch_size=opt.batch_size,shuffle=True,num_workers=opt.n_cpu,pin_memory=True)
#######################################
opt.logging_loss_batch_size=len(train_dataloader)

# sample_models(tracking_train_dataloader,generator,opt,opt.n_epochs,flag_show_scaled_data=0)
# exit()
# sample_models(test_dataloader,generator,opt,opt.n_epochs,flag_show_scaled_data=0)
# sample_dataset(test_dataloader,generator,opt,opt.n_epochs,flag_show_scaled_data=1,record_weights=1,data_mode='test');   exit()
# Tensor type
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
#########  Training
g_loss_train=[];    g_loss_val=[];  
loss_GAN_train=[];  loss_GAN_val=[]
loss_pixel_train=[];    loss_pixel_val=[];  
d_loss_train=[];    d_loss_val=[];  
loss_real_train=[];    loss_real_val=[];
loss_fake_train=[];    loss_fake_val=[];
loss_target_pred_train=[];          loss_target_pred_valid=[]
loss_init_model_pred_train=[];      loss_init_model_pred_valid=[]
loss_true_model_pred_train=[];      loss_true_model_pred_valid=[]
loss_true_model_inversion_train=[]; loss_true_model_inversion_valid=[]

prev_time = time.time()
T1 = datetime.datetime.now()
print('datetime.now=',T1)
phases=['training','validation']
# phases=['training']
accumulated_target_pred_acc=0
if opt.n_epochs!=-1:
    for epoch in range(0,opt.n_epochs):
        for phase in phases:
            t = TicToc() #create instance of class
            t.tic() #Start timer
            if phase=='training':   
                if opt.dataloading_pipeline=='matrix':  dataloader=train_dataloader_from_array;
                else:   dataloader=train_dataloader
                generator.train();discriminator.train()
            elif phase=='validation':
                if opt.dataloading_pipeline=='matrix':  dataloader=valid_dataloader_from_array
                else:   dataloader=validation_dataloader;
                generator.eval();discriminator.eval()
            if opt.dataloading_pipeline=='matrix':
                i=0
                for x_batch,y_batch,taper_batch in dataloader:
                    len(dataloader)
                    real_A = x_batch.to(device)
                    real_B = y_batch.to(device)
                    real_taper = taper_batch.to(device)
                    # Plot_image(np.concatenate((real_A[-1,0,::].cpu().detach().numpy(),real_B[-1,0,::].cpu().detach().numpy(),
                    #     real_A[-1,1,::].cpu().detach().numpy() ),axis=1).T,
                    #     Show_flag=0,Save_flag=1,Title='batch_1_sample',Aspect='equal',Save_pictures_path=opt.save_path)
                    ss=1
                    # Plot_image(np.concatenate((real_A[-1,0,::].cpu().detach().numpy(),real_B[-1,0,::].cpu().detach().numpy() ),axis=1).T,
                    #     Show_flag=0,Save_flag=1,Title='batch_1_sample',Aspect='equal',Save_pictures_path=opt.save_path)
                    # print(real_A.shape)    
                    # Plot_image((real_A[-1,0,::].cpu().detach().numpy()).T,Show_flag=0,Save_flag=1,Title='batch_1_sampleA0',Aspect='equal',Save_pictures_path=opt.save_path)
                    # Plot_image((real_A[-1,1,::].cpu().detach().numpy()).T,Show_flag=0,Save_flag=1,Title='batch_1_sampleA1',Aspect='equal',Save_pictures_path=opt.save_path)
                    # Plot_image((real_B[-1,0,::].cpu().detach().numpy()).T,Show_flag=0,Save_flag=1,Title='batch_1_sampleB',Aspect='equal',Save_pictures_path=opt.save_path)
                    # Plot_image((real_taper[-1,0,::].cpu().detach().numpy()).T,Show_flag=0,Save_flag=1,Title='batch_1_sample_taper',Aspect='equal',Save_pictures_path=opt.save_path)
                # for i, batch in enumerate(dataloader):
                    # real_A = Variable(batch["A"].type(Tensor))
                    # real_B = Variable(batch["B"].type(Tensor))
                    #############  Train Generators
                    if phase=='training':   optimizer_G.zero_grad(set_to_none=True)
                    # GAN loss
                    fake_B = generator(real_A)
                    # Pixel-wise loss
                    loss_pixel = criterion_pixelwise(fake_B,real_B)
                    loss_G = 1*loss_pixel
                    ##############      calculate tracking losses
                    target_pred_acc=criterion_R2Loss(fake_B,real_B);
                    ##############  accumulate tracking losses
                    accumulated_target_pred_acc+=target_pred_acc.item()
                    ############### Total loss
                    if phase=='training':   loss_G.backward();  optimizer_G.step()
                    ##############  Log Progress
                    # Determine approximate time left
                    batches_done = epoch * len(dataloader) + i
                    batches_left = opt.n_epochs * len(dataloader) - batches_done
                    time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
                    prev_time = time.time()
                    # Print log
                    # If at sample interval save image
                    logging_loss_batch_size=len(dataloader)
                    i=i+1
            elif opt.dataloading_pipeline=='dataloader':
                for i, batch in enumerate(dataloader):
                    # Model inputs
                    real_A = Variable(batch["A"].type(Tensor))
                    real_B = Variable(batch["B"].type(Tensor))
                    #############  Train Generators
                    if phase=='training':   optimizer_G.zero_grad(set_to_none=True)
                    # GAN loss
                    fake_B = generator(real_A)
                    # Pixel-wise loss
                    loss_pixel = criterion_pixelwise(fake_B,real_B)
                    loss_G = 1*loss_pixel
                    ##############      calculate tracking losses
                    target_pred_acc=criterion_R2Loss(fake_B,real_B);
                    ##############  accumulate tracking losses
                    accumulated_target_pred_acc+=target_pred_acc.item()
                    ############### Total loss
                    if phase=='training':   loss_G.backward();  optimizer_G.step()
                    ##############  Log Progress
                    # Determine approximate time left
                    batches_done = epoch * len(dataloader) + i
                    batches_left = opt.n_epochs * len(dataloader) - batches_done
                    time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
                    prev_time = time.time()
                    # Print log
                    # If at sample interval save image
                    logging_loss_batch_size=len(dataloader)
            ###################################################
            sys.stdout.write("\rLog:%s PHASE: %s [Epoch %d/%d] [Batch %d/%d] [G loss: %f] ETA: %s"
                % ( str(log_save_const),phase,epoch,opt.n_epochs,i,len(dataloader),loss_G.item(),time_left,))          
            ################################################### record losses
            if phase=='training':
                loss_target_pred_train.append(accumulated_target_pred_acc/logging_loss_batch_size)
                loss_pixel_train.append(opt.lambda_pixel*loss_pixel.item())
                g_loss_train.append(loss_G.item())
                if opt.architecture==1:
                    loss_GAN_train.append(loss_GAN.item())
                    d_loss_train.append(loss_D.item())
                    loss_real_train.append(loss_real.item())
                    loss_fake_train.append(loss_fake.item())
            elif phase=='validation':
                loss_target_pred_valid.append(accumulated_target_pred_acc/logging_loss_batch_size)
                loss_pixel_val.append(opt.lambda_pixel*loss_pixel.item())
                g_loss_val.append(loss_G.item())     
                ################################################### plot some samples and losses
                # if epoch<15:    current_checkpoint_interval=2;current_plotting_interval=current_checkpoint_interval
                if epoch<=600:      current_checkpoint_interval=opt.first_checkpoint_interval;  current_plotting_interval=current_checkpoint_interval
                else:               current_checkpoint_interval=opt.checkpoint_interval;        current_plotting_interval=opt.plotting_interval
                if epoch%current_plotting_interval==0 and epoch!=0:     #   opt.plotting_interval
                # if epoch%15==0 and epoch!=0:     #   opt.plotting_interval
                    # sample_dataset(tracking_train_dataloader,generator,opt,epoch,flag_show_scaled_data=1) #tracking_dataloader.dataset.files
                    history={'g_loss_train':g_loss_train,'g_loss_val':g_loss_val,
                            'loss_pixel_train':loss_pixel_train,'loss_pixel_val':loss_pixel_val,
                            'loss_target_pred_train':loss_target_pred_train,'loss_target_pred_valid':loss_target_pred_valid}
                    print('hello')
                    Plot_r2_losses_2(history,Title='log'+str(log_save_const) + 'losses_r2', Save_pictures_path=Save_pictures_path,Save_flag=1)   
                    Plot_r2_losses_3(history,Title='log'+str(log_save_const) + 'losses3_r2', Save_pictures_path=Save_pictures_path,Save_flag=1)
                    Plot_r2_losses_for_geophysics(history,epoch,save_name='log'+str(log_save_const)+'losses_r2_for_geophysics',Title='epoch '+str(epoch+1),Save_pictures_path=Save_pictures_path,Save_flag=1)
                    Plot_unet_losses(history,Title='log'+str(log_save_const) + 'losses_unet', Save_pictures_path=Save_pictures_path,Save_flag=1)
                    Plot_unet_losses_2(history,Title='log'+str(log_save_const) + 'losses_pixel', Save_pictures_path=Save_pictures_path,Save_flag=1)
                    if opt.plot_samples==1:
                        ss=1
                        # sample_dataset_geophysics(test_dataloader,generator,opt,epoch,flag_show_scaled_data=1,record_weights=0)
                        # sample_dataset_geophysics(tracking_train_dataloader2,generator,opt,epoch,flag_show_scaled_data=1,data_mode='tracking_train',record_weights=0)
                        # sample_dataset_geophysics(tracking_val_dataloader2,generator,opt,epoch,flag_show_scaled_data=1,data_mode='tracking_val',record_weights=0)
                ################################################### save model weights and history
                if opt.checkpoint_interval != -1 and epoch%current_checkpoint_interval==0:
                    torch.save(generator.state_dict(), "%s/generator_%d.pth" % (Save_pictures_path, epoch))
                    ###########
                    history={'g_loss_train':g_loss_train,'g_loss_val':g_loss_val,
                            'loss_pixel_train':loss_pixel_train,'loss_pixel_val':loss_pixel_val,
                            'loss_target_pred_train':loss_target_pred_train,'loss_target_pred_valid':loss_target_pred_valid
                            }
                    opt.history=history
                    with open(opt.save_path+'/'+'opt.txt','w') as f:
                        json.dump(opt.__dict__, f, indent=2)
            accumulated_target_pred_acc=0
            ss=1
            # print('One epoch calculation time=',str())
            print('/n')
            t.toc() #Time elapsed since t.tic()
            # exit()
    torch.save(generator.state_dict(), "%s/generator_%d.pth" % (Save_pictures_path,epoch))
    print('time(now)=',datetime.datetime.now())
    print('Training time(sec)=',datetime.datetime.now()-T1)
generator.eval();discriminator.eval()
#########  plot losses
Plot_r2_losses_2(history,Title='log'+str(log_save_const) + 'losses_r2', Save_pictures_path=Save_pictures_path,Save_flag=1)   
Plot_unet_losses(history,Title='log'+str(log_save_const) + 'losses_unet', Save_pictures_path=Save_pictures_path,Save_flag=1)
Plot_unet_losses_2(history,Title='log'+str(log_save_const) + 'losses_pixel', Save_pictures_path=Save_pictures_path,Save_flag=1)
#########  Plotting testing results
if opt.plot_samples==1 and opt.n_epochs!=-1:
    sample_dataset_geophysics(test_dataloader,generator,opt,opt.n_epochs,flag_show_scaled_data=1,record_weights=0,data_mode='test')
    sample_dataset_geophysics(tracking_train_dataloader,generator,opt,opt.n_epochs,flag_show_scaled_data=1,data_mode='tracking_train',record_weights=0)
    # sample_dataset_geophysics(tracking_val_dataloader,generator,opt,opt.n_epochs,flag_show_scaled_data=1,data_mode='tracking_val',record_weights=0)
#########  save training session parameters to the file
with open(opt.save_path+'/'+'opt.txt', 'w') as f:
    json.dump(opt.__dict__, f, indent=2)
#########
calculate_prediction_histograms=1
if calculate_prediction_histograms==1:
    phases=['test','validation','training']
    phases=['test']
    phases=['training','validation','test']
    calculate_predictions=1;    save_predictions=1
    plot_samples=1;     
    val=multiprocessing.cpu_count()-3;
    # val=6
    batch_size_val=2  
    num_workers_val=2
    print('calculate misfits')
    ch1=[];ch2=[];ch3=[];ch4=[]
    with open(os.path.join(opt.dataset_path[0],'scaling_constants_dict.pkl'),'rb') as input:
        scaling_constants_dict=pickle.load(input)
    for phase in phases:
        if phase=='training':       
            dataloader=DataLoader(ImageDataset(opt,transforms_=transforms_,mode="train"),
                batch_size=batch_size_val,shuffle=False,num_workers=num_workers_val)
        elif phase=='validation':   
            dataloader=DataLoader(ImageDataset(opt,transforms_=transforms_,mode="val"),
                batch_size=batch_size_val,shuffle=False,num_workers=num_workers_val)
        elif phase=='test':         
            dataloader=DataLoader(ImageDataset(opt,transforms_=transforms_,mode="test"),
                batch_size=1,shuffle=False,num_workers=1)
        predictions=np.zeros((len(dataloader.dataset),10))
        batch_size=dataloader.batch_size
        for i,batch in enumerate(dataloader):
            """ A-input data,B-target data,C-initial model,D-true model,E-input data in real amplitudes
                ,F-output data in real amplitudes,sc_t-scaler_t"""
            print(phase+' '+str(i))
            real_A = Variable(batch["A"].type(Tensor))
            real_B = Variable(batch["B"].type(Tensor))
            init_model=Variable(batch["C"].type(Tensor))
            true_model=Variable(batch["D"].type(Tensor))
            real_A_ra = Variable(batch["E"].type(Tensor))
            real_B_ra = Variable(batch["F"].type(Tensor))
            # scaler_t=Variable(batch["sc_t"].type(Tensor)).cpu().detach().numpy()
            fake_B = generator(real_A)
            target_pred_acc=criterion_R2Loss_custom(fake_B,real_B);
            fake_B_numpy=fake_B.cpu().detach().numpy()
            real_B_numpy=real_B.cpu().detach().numpy()
            target_pred_1NRMS=calculate_metric_on_batch_(fake_B_numpy,real_B_numpy,metric_type='rms similarity');
            target_pred_SSIM=calculate_metric_on_batch_(fake_B_numpy,real_B_numpy,metric_type='ssim')
            ##############  make some preprocessing for extra losses, append water and scale predicted data back
            tmp=fake_B.cpu().detach().numpy()
            t_pred=np.empty((real_B_ra.shape))
            Nx=real_B_ra.shape[2]; Nz=real_B_ra.shape[3]
            for i_f in range(tmp.shape[0]):
                t_pred[i_f,0,:,:]=imresize(tmp[i_f,0,:, :],(Nx, Nz))
            fake_B_ra=np.ones_like(t_pred)
            for i_f in range(t_pred.shape[0]):
                if opt.inp_channels_type=='1_fwi_res':
                    tmp=scaling_data_back(t_pred[i_f,::],scaling_constants_dict,'fwi_res')
                else:
                    tmp=scaling_data_back(t_pred[i_f,::],scaling_constants_dict,'t')
                fake_B_ra[i_f,0,:,:]=np.squeeze(tmp)
            fake_B_ra=torch.from_numpy(fake_B_ra).to(device)
            ##############      calculate extra tracking losses
            # if opt.inp_channels_type=='1_fwi_res':
            #     init_model_pred_acc=criterion_R2Loss_custom(fake_B_ra,real_B_ra)
            #     init_model_inversion_acc=criterion_R2Loss_custom(real_A_ra,real_B_ra)
            #     true_model_pred_acc=criterion_R2Loss_custom(fake_B_ra,true_model)       
            #     true_model_inversion_acc=criterion_R2Loss_custom(real_A_ra,true_model)
            # else:
            #     init_model_pred_acc=criterion_R2Loss_custom(init_model+fake_B_ra,init_model+real_B_ra)
            #     init_model_inversion_acc=criterion_R2Loss_custom(init_model+real_A_ra,init_model+real_B_ra)
            #     true_model_pred_acc=criterion_R2Loss_custom(init_model+fake_B_ra,true_model)       
            #     true_model_inversion_acc=criterion_R2Loss_custom(init_model+real_A_ra,true_model)
            ##############      fill result matrix
            predictions[ ((i)*batch_size):((i)*batch_size)+len(target_pred_acc),0]=target_pred_acc  #R2
            predictions[ ((i)*batch_size):((i)*batch_size)+len(target_pred_acc),1]=target_pred_1NRMS
            predictions[ ((i)*batch_size):((i)*batch_size)+len(target_pred_acc),2]=target_pred_SSIM
            predictions[ ((i)*batch_size):((i)*batch_size)+len(target_pred_acc),3]=np.nan
            predictions[ ((i)*batch_size):((i)*batch_size)+len(target_pred_acc),4]=np.nan
            predictions[ ((i)*batch_size):((i)*batch_size)+len(target_pred_acc),5]=np.nan
            predictions[ ((i)*batch_size):((i)*batch_size)+len(target_pred_acc),6]=np.nan
            predictions[ ((i)*batch_size):((i)*batch_size)+len(target_pred_acc),7]=np.nan
            predictions[ ((i)*batch_size):((i)*batch_size)+len(target_pred_acc),8]=np.nan
            predictions[ ((i)*batch_size):((i)*batch_size)+len(target_pred_acc),9]=np.nan
        df = pd.DataFrame(predictions,columns=
            ['target_pred_acc' ,'target_pred_1NRMS','target_pred_SSIM','true_model_pred_acc','true_model_inversion_acc',
             'target_pred_nrms','init_model_pred_nrms','init_model_inversion_nrms','true_model_pred_nrms','true_model_inversion_nrms'])
        df.insert(0,'filename',dataloader.dataset.files)
        if phase=='training':       predictions_train=df
        elif phase=='validation':   predictions_valid=df
        elif phase=='test':         predictions_test=df
    # %%  save data to npz file
    if save_predictions==1:
        if 'predictions_train' not in globals():      
            predictions_train=np.array([])
            predictions_train=pd.DataFrame()
        if 'predictions_valid' not in globals():    
            predictions_valid=np.array([])
            predictions_valid=pd.DataFrame()
        if 'predictions_test' not in globals():     
            predictions_test=np.array([])
            predictions_test=pd.DataFrame()
        # np.savez(opt.save_path+'/r2_scores_inference.npz',predictions_train=predictions_train,
        #     predictions_valid=predictions_valid,predictions_test=predictions_test)
        compression_opts = dict(method='zip',archive_name='out.csv')
        predictions_train.to_csv(os.path.join(opt.save_path,'inference_scores_train.zip'),index=False,compression=compression_opts)
        predictions_valid.to_csv(os.path.join(opt.save_path,'inference_scores_valid.zip'), index=False,compression=compression_opts)
        predictions_test.to_csv(os.path.join(opt.save_path,'inference_scores_test.zip'), index=False,compression=compression_opts)
    print('calculating misfits finished')
#########
plot_prediction_histograms=1
if plot_prediction_histograms==1:
    # # name='./logs/log719/r2_scores_inference.npz'
    # name=opt.save_path+'/r2_scores_inference.npz'
    # # name='./logs/log1096/r2_scores_inference.npz'
    # # name='./logs/log'+str(log_name)+'/r2_scores_inference.npz'
    # with open(name,'rb') as f:
    #     data=np.load(f,allow_pickle=True)
    #     predictions_train=data['predictions_train']
    #     predictions_valid=data['predictions_valid']
    #     predictions_test=data['predictions_test']
    #     data.close()
    #################
    scores_path=opt.save_path
    # scores_path='./logs/log1402'
    # scores_path='./logs/log235'
    # scores_path='./logs/log236'
    predictions_train=  pd.read_csv(os.path.join(scores_path,'inference_scores_train.zip'))
    predictions_valid=  pd.read_csv(os.path.join(scores_path,'inference_scores_valid.zip'))
    predictions_test=   pd.read_csv(os.path.join(scores_path,'inference_scores_test.zip'))
    ##############
    Fontsize=17
    plt.rcParams.update({'font.size': Fontsize})
    FONTSIZE=18
    #################   check worst predicted files 
    predictions_train=predictions_train.sort_values(by=['target_pred_acc'])
    predictions_valid=predictions_valid.sort_values(by=['target_pred_acc'])
    predictions_test=predictions_test.sort_values(by=['target_pred_acc'])
    val_worst_values=predictions_valid[0:10]
    val_best_values=predictions_valid[-3:]
    train_worst_values=predictions_train[0:10]
    train_best_values=predictions_train[-10:]
    dl_val_best=DataLoader(ImageDataset(opt,transforms_=transforms_,
        mode=None,file_list=list(val_best_values.filename)),
        batch_size=1,shuffle=False,num_workers=1)
    dl_val_worst=DataLoader(ImageDataset(opt,transforms_=transforms_,
        mode=None,file_list=list(val_worst_values.filename)),
        batch_size=1,shuffle=False,num_workers=1)
    dl_train_worst=DataLoader(ImageDataset(opt,transforms_=transforms_,
        mode=None,file_list=list(train_worst_values.filename)),
        batch_size=1,shuffle=False,num_workers=1)
    ############################################################################################################################################################################################################
    ############## choose metric from target_pred_acc,target_pred_1NRMS,target_pred_SSIM
    # metric_name='r2'
    # metric_name='SSIM'
    metric_name='rms_similarity'  # RMS similarity = 1 - rms(x - y) / rms(x)
    if metric_name=='r2':
        pred_train= predictions_train.target_pred_acc.to_numpy().T
        pred_valid= predictions_valid.target_pred_acc.to_numpy().T
        pred_test=  predictions_test.target_pred_acc.to_numpy().T
    elif metric_name=='SSIM':
        pred_train= predictions_train.target_pred_SSIM.to_numpy().T
        pred_valid= predictions_valid.target_pred_SSIM.to_numpy().T
        pred_test=  predictions_test.target_pred_SSIM.to_numpy().T
    elif metric_name=='rms_similarity':
        pred_train= predictions_train.target_pred_1NRMS.to_numpy().T
        pred_valid= predictions_valid.target_pred_1NRMS.to_numpy().T
        pred_test=  predictions_test.target_pred_1NRMS.to_numpy().T
    #################   scores
    mean_score={ 'train':np.mean(pred_train),
            'valid':np.mean(pred_valid),
            'test':np.mean(pred_test)}
    median_score={ 'train':np.median(pred_train),
            'valid':np.median(pred_valid),
            'test':np.median(pred_test)}
    std_score={'train': np.std(pred_train),
            'valid':    np.std(pred_valid),
            'test':     np.std(pred_test)}
    #################
    plt.figure(figsize=(10,4),dpi=300)
    # plt.hist(np.vstack((predictions_train.target_pred_acc.to_numpy(),predictions_train.init_model_pred_acc.to_numpy())).T,
    plt.hist(pred_train,range=[np.min(pred_train), 1], bins=20, cumulative=False)    # range=[-2,1], bins=20, cumulative=False)
    plt.ylabel('Number of samples',fontsize=FONTSIZE)
    plt.xlabel('R2(predicted,true)',fontsize=FONTSIZE)
    plt.title('Training,total#='+str(predictions_train.shape[0])+',mean/std='+numstr3(mean_score['train'])+'/'+numstr3(std_score['train']),fontsize=FONTSIZE)
    # plt.legend(['target_pred_acc','init_model_pred_acc','init_model_pred_acc-init_model_inversion_acc'],fontsize=FONTSIZE)
    plt.legend([metric_name,'target_pred_1NRMS','target_pred_SSIM'],fontsize=FONTSIZE)
    plt.xlim([np.min(pred_train),1])
    tmp=os.path.join(Save_pictures_path,'hist_training_misfits_mean'+metric_name+'.png')
    print('Saving to '+tmp)
    plt.tight_layout()
    plt.savefig(tmp,dpi=300,bbox_inches='tight')
    plt.show(block=False)
    plt.close()

    plt.figure(figsize=(10,4),dpi=300)
    plt.hist(pred_valid,range=[np.min(pred_valid), 1], bins=20, cumulative=False)
    plt.ylabel('Number of samples',fontsize=FONTSIZE)
    plt.xlabel('R2(predicted,true)',fontsize=FONTSIZE)
    plt.title('Validation,total#='+str(predictions_valid.shape[0])+',mean/std='+numstr3(mean_score['valid'])+'/'+numstr3(std_score['valid']),fontsize=FONTSIZE)
    plt.legend([metric_name,'target_pred_1NRMS','target_pred_SSIM'],fontsize=FONTSIZE)
    plt.xlim([np.min(pred_valid),1])
    tmp=os.path.join(Save_pictures_path,'hist_valid_misfits_mean'+metric_name+'.png')
    print('Saving to '+tmp)
    plt.tight_layout()
    plt.savefig(tmp,dpi=300,bbox_inches='tight')
    plt.show(block=False)
    plt.close()

    plt.figure(figsize=(10,4),dpi=300)
    # plt.hist(np.vstack((predictions_train.target_pred_acc.to_numpy(),predictions_train.init_model_pred_acc.to_numpy())).T,
    plt.hist(pred_train,range=[np.min(pred_train), 1], bins=20, cumulative=False)    # range=[-2,1], bins=20, cumulative=False)
    plt.ylabel('Number of samples',fontsize=FONTSIZE)
    plt.xlabel('R2(predicted,true)',fontsize=FONTSIZE)
    plt.title('Training,total#='+str(predictions_train.shape[0])+',median/std='+numstr3(median_score['train'])+'/'+numstr3(std_score['train']),fontsize=FONTSIZE)
    # plt.legend(['target_pred_acc','init_model_pred_acc','init_model_pred_acc-init_model_inversion_acc'],fontsize=FONTSIZE)
    plt.legend([metric_name,'target_pred_1NRMS','target_pred_SSIM'],fontsize=FONTSIZE)
    plt.xlim([np.min(pred_train),1])
    tmp=os.path.join(Save_pictures_path,'hist_training_misfits_median'+metric_name+'.png')
    print('Saving to '+tmp)
    plt.tight_layout()
    plt.savefig(tmp,dpi=300,bbox_inches='tight')
    plt.show(block=False)
    plt.close()

    plt.figure(figsize=(10,4),dpi=300)
    plt.hist(pred_valid,range=[np.min(pred_valid), 1], bins=20, cumulative=False)
    plt.ylabel('Number of samples',fontsize=FONTSIZE)
    plt.xlabel('R2(predicted,true)',fontsize=FONTSIZE)
    plt.title('Validation,total#='+str(predictions_valid.shape[0])+',median/std='+numstr3(median_score['valid'])+'/'+numstr3(std_score['valid']),fontsize=FONTSIZE)
    plt.legend([metric_name,'target_pred_1NRMS','target_pred_SSIM'],fontsize=FONTSIZE)
    plt.xlim([np.min(pred_valid),1])
    tmp=os.path.join(Save_pictures_path,'hist_valid_misfits_median'+metric_name+'.png')
    print('Saving to '+tmp)
    plt.tight_layout()
    plt.savefig(tmp,dpi=300,bbox_inches='tight')
    plt.show(block=False)
    plt.close()

    plt.figure(figsize=(10,8),dpi=300)
    plt.hist(pred_test,range=[np.min(pred_test), 1], bins=20, cumulative=False)
    plt.ylabel('Number of samples')
    plt.xlabel('R2(predicted,true)')
    plt.title('Testing,total= '+str(predictions_test.shape[0])+' samples',fontsize=16)
    plt.legend([metric_name,'target_pred_1NRMS','target_pred_SSIM'],fontsize=FONTSIZE)
    plt.xlim([np.min(pred_test),1])
    tmp=os.path.join(Save_pictures_path,'hist_testing_misfits_r2.png')
    print('Saving to '+tmp)
    plt.tight_layout()
    plt.savefig(tmp,dpi=300)
    plt.show(block=False)
    plt.close()
    ############################################################################################################################################################################################################
    metric_name='r2'
    # metric_name='SSIM'
    # metric_name='rms_similarity'  # RMS similarity = 1 - rms(x - y) / rms(x)
    if metric_name=='r2':
        pred_train= predictions_train.target_pred_acc.to_numpy().T
        pred_valid= predictions_valid.target_pred_acc.to_numpy().T
        pred_test=  predictions_test.target_pred_acc.to_numpy().T
    elif metric_name=='SSIM':
        pred_train= predictions_train.target_pred_SSIM.to_numpy().T
        pred_valid= predictions_valid.target_pred_SSIM.to_numpy().T
        pred_test=  predictions_test.target_pred_SSIM.to_numpy().T
    elif metric_name=='rms_similarity':
        pred_train= predictions_train.target_pred_1NRMS.to_numpy().T
        pred_valid= predictions_valid.target_pred_1NRMS.to_numpy().T
        pred_test=  predictions_test.target_pred_1NRMS.to_numpy().T
    #################   scores
    mean_score={ 'train':np.mean(pred_train),
            'valid':np.mean(pred_valid),
            'test':np.mean(pred_test)}
    median_score={ 'train':np.median(pred_train),
            'valid':np.median(pred_valid),
            'test':np.median(pred_test)}
    std_score={'train': np.std(pred_train),
            'valid':    np.std(pred_valid),
            'test':     np.std(pred_test)}
    #################
    plt.figure(figsize=(10,4),dpi=300)
    # plt.hist(np.vstack((predictions_train.target_pred_acc.to_numpy(),predictions_train.init_model_pred_acc.to_numpy())).T,
    plt.hist(pred_train,range=[np.min(pred_train), 1], bins=20, cumulative=False)    # range=[-2,1], bins=20, cumulative=False)
    plt.ylabel('Number of samples',fontsize=FONTSIZE)
    plt.xlabel('R2(predicted,true)',fontsize=FONTSIZE)
    plt.title('Training,total#='+str(predictions_train.shape[0])+',mean/std='+numstr3(mean_score['train'])+'/'+numstr3(std_score['train']),fontsize=FONTSIZE)
    # plt.legend(['target_pred_acc','init_model_pred_acc','init_model_pred_acc-init_model_inversion_acc'],fontsize=FONTSIZE)
    plt.legend([metric_name,'target_pred_1NRMS','target_pred_SSIM'],fontsize=FONTSIZE)
    plt.xlim([np.min(pred_train),1])
    tmp=os.path.join(Save_pictures_path,'hist_training_misfits_mean'+metric_name+'.png')
    print('Saving to '+tmp)
    plt.tight_layout()
    plt.savefig(tmp,dpi=300,bbox_inches='tight')
    plt.show(block=False)
    plt.close()

    plt.figure(figsize=(10,4),dpi=300)
    plt.hist(pred_valid,range=[np.min(pred_valid), 1], bins=20, cumulative=False)
    plt.ylabel('Number of samples',fontsize=FONTSIZE)
    plt.xlabel('R2(predicted,true)',fontsize=FONTSIZE)
    plt.title('Validation,total#='+str(predictions_valid.shape[0])+',mean/std='+numstr3(mean_score['valid'])+'/'+numstr3(std_score['valid']),fontsize=FONTSIZE)
    plt.legend([metric_name,'target_pred_1NRMS','target_pred_SSIM'],fontsize=FONTSIZE)
    plt.xlim([np.min(pred_valid),1])
    tmp=os.path.join(Save_pictures_path,'hist_valid_misfits_mean'+metric_name+'.png')
    print('Saving to '+tmp)
    plt.tight_layout()
    plt.savefig(tmp,dpi=300,bbox_inches='tight')
    plt.show(block=False)
    plt.close()

    plt.figure(figsize=(10,4),dpi=300)
    # plt.hist(np.vstack((predictions_train.target_pred_acc.to_numpy(),predictions_train.init_model_pred_acc.to_numpy())).T,
    plt.hist(pred_train,range=[np.min(pred_train), 1], bins=20, cumulative=False)    # range=[-2,1], bins=20, cumulative=False)
    plt.ylabel('Number of samples',fontsize=FONTSIZE)
    plt.xlabel('R2(predicted,true)',fontsize=FONTSIZE)
    plt.title('Training,total#='+str(predictions_train.shape[0])+',median/std='+numstr3(median_score['train'])+'/'+numstr3(std_score['train']),fontsize=FONTSIZE)
    # plt.legend(['target_pred_acc','init_model_pred_acc','init_model_pred_acc-init_model_inversion_acc'],fontsize=FONTSIZE)
    plt.legend([metric_name,'target_pred_1NRMS','target_pred_SSIM'],fontsize=FONTSIZE)
    plt.xlim([np.min(pred_train),1])
    tmp=os.path.join(Save_pictures_path,'hist_training_misfits_median'+metric_name+'.png')
    print('Saving to '+tmp)
    plt.tight_layout()
    plt.savefig(tmp,dpi=300,bbox_inches='tight')
    plt.show(block=False)
    plt.close()

    plt.figure(figsize=(10,4),dpi=300)
    plt.hist(pred_valid,range=[np.min(pred_valid), 1], bins=20, cumulative=False)
    plt.ylabel('Number of samples',fontsize=FONTSIZE)
    plt.xlabel('R2(predicted,true)',fontsize=FONTSIZE)
    plt.title('Validation,total#='+str(predictions_valid.shape[0])+',median/std='+numstr3(median_score['valid'])+'/'+numstr3(std_score['valid']),fontsize=FONTSIZE)
    plt.legend([metric_name,'target_pred_1NRMS','target_pred_SSIM'],fontsize=FONTSIZE)
    plt.xlim([np.min(pred_valid),1])
    tmp=os.path.join(Save_pictures_path,'hist_valid_misfits_median'+metric_name+'.png')
    print('Saving to '+tmp)
    plt.tight_layout()
    plt.savefig(tmp,dpi=300,bbox_inches='tight')
    plt.show(block=False)
    plt.close()

    plt.figure(figsize=(10,8),dpi=300)
    plt.hist(pred_test,range=[np.min(pred_test), 1], bins=20, cumulative=False)
    plt.ylabel('Number of samples')
    plt.xlabel('R2(predicted,true)')
    plt.title('Testing,total= '+str(predictions_test.shape[0])+' samples',fontsize=16)
    plt.legend([metric_name,'target_pred_1NRMS','target_pred_SSIM'],fontsize=FONTSIZE)
    plt.xlim([np.min(pred_test),1])
    tmp=os.path.join(Save_pictures_path,'hist_testing_misfits_r2.png')
    print('Saving to '+tmp)
    plt.tight_layout()
    plt.savefig(tmp,dpi=300)
    plt.show(block=False)
    plt.close()
    ############################################################################################################################################################################################################
    #################   sample worst and best predictions
    data=val_worst_values
    for i in range(data.shape[0]):
        print(data[i:i+1].filename.values[0],data[i:i+1].target_pred_acc)
    data=val_best_values
    for i in range(data.shape[0]):
        print(data[i:i+1].filename.values[0],data[i:i+1].target_pred_acc)   
    data=train_worst_values
    for i in range(data.shape[0]):
        print(data[i:i+1].filename.values[0],data[i:i+1].target_pred_acc)
    # sample_dataset_geophysics(dl_val_best,generator,opt,opt.n_epochs,flag_show_scaled_data=1,record_weights=0,data_mode='val_best')
    # sample_dataset_geophysics(dl_val_worst,generator,opt,opt.n_epochs,flag_show_scaled_data=1,record_weights=0,data_mode='val_worst')
    # sample_dataset_geophysics(dl_train_worst,generator,opt,opt.n_epochs,flag_show_scaled_data=1,record_weights=0,data_mode='train_worst')
    #################   NRMS scores
    # plt.figure(figsize=(10,8),dpi=300)
    # plt.hist(np.vstack((predictions_train.target_pred_nrms.to_numpy(),predictions_train.true_model_pred_nrms.to_numpy())).T,
    #     range=[np.min(predictions_train.target_pred_nrms), 1], bins=20, cumulative=False)
    # plt.ylabel('Number of samples')
    # plt.xlabel('R2(predicted,true)')
    # plt.title('Training samples, total= '+str(predictions_train.shape[0])+' samples')
    # # plt.legend(['target_pred_nrms','true_model_pred_nrms'])
    # plt.legend(['target_pred_nrms','true_model_pred_nrms','true_model_pred_nrms-true_model_inversiom_nrms'])
    # tmp=os.path.join(Save_pictures_path,'hist_train_misfits_nrms.png')
    # print('Saving to '+tmp)
    # plt.tight_layout()
    # plt.savefig(tmp,dpi=300)
    # plt.show()
    # plt.close()

    # plt.figure(figsize=(10,8),dpi=300)
    # plt.hist(np.vstack((predictions_valid.target_pred_nrms.to_numpy(),predictions_valid.true_model_pred_nrms.to_numpy())).T,
    #     range=[np.min(predictions_valid.target_pred_nrms), 1], bins=20, cumulative=False)
    # plt.ylabel('Number of samples')
    # plt.xlabel('R2(predicted,true)')
    # plt.title('Validation samples, total= '+str(predictions_valid.shape[0])+' samples')
    # plt.legend(['target_pred_nrms','true_model_pred_nrms','true_model_pred_nrms-true_model_inversiom_nrms'])
    # tmp=Save_pictures_path + '/' +'hist_'+'valid_misfits'+ '.png'
    # tmp=os.path.join(Save_pictures_path,'hist_valid_misfits_nrms.png')
    # print('Saving to '+tmp)
    # plt.tight_layout()
    # plt.savefig(tmp,dpi=300)
    # plt.show()
    # plt.close()

    # plt.figure(figsize=(10,8),dpi=300)
    # plt.hist(np.vstack((predictions_test.target_pred_nrms.to_numpy(),predictions_test.true_model_pred_nrms.to_numpy())).T,
    #     range=[np.min(predictions_test.target_pred_nrms), 1], bins=20, cumulative=False)
    # plt.ylabel('Number of samples')
    # plt.xlabel('R2(predicted,true)')
    # plt.title('Testing samples, total= '+str(predictions_test.shape[0])+' samples')
    # plt.legend(['target_pred_nrms','true_model_pred_nrms'])
    # tmp=os.path.join(Save_pictures_path,'hist_testing_misfits_nrms.png')
    # print('Saving to '+tmp)
    # plt.tight_layout()
    # plt.savefig(tmp,dpi=300)
    # plt.show()
    # plt.close()

# summary(discriminator.to(0),[(1,opt.img_height,opt.img_width),(opt.channels_discriminator,opt.img_height,opt.img_width)] )