# scp -r ./logs/log1189/predictions_1189 /var/remote/lustre/project/k1404/pavel/DENISE-Black-Edition/predictions/predictions_1189
# scp -r ./logs/log1192/predictions_1192 /var/remote/lustre/project/k1404/pavel/DENISE-Black-Edition/predictions/predictions_1192
# scp -r ./logs/log1202/predictions_1202 /var/remote/lustre/project/k1404/pavel/DENISE-Black-Edition/predictions/predictions_1202
# scp -r ./logs/log1204/predictions_1204 /var/remote/lustre/project/k1404/pavel/DENISE-Black-Edition/predictions/predictions_1204
# scp -r ./logs/log1206/predictions_1206 /var/remote/lustre/project/k1404/pavel/DENISE-Black-Edition/predictions/predictions_1206

# ipython nbconvert notebook.ipynb --to script
#   1092,1093 histogram results (1094,1095)
from imports_torch import *
from utils_low_wavenumbers_torch import *
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
#####   dataset paths
dataset_path='/ibex/scratch/plotnips/intel/MWE/datasets/dataset_vl_gen_scaled_60000_smooth_450m_scaler_picture_01'
dataset_path='/home/plotnips/Dropbox/Log_extrapolation/scripts/MWE/datasets/dataset_vl_gen_scaled_3005_smooth_450m_scaler_picture_individual_scaling_false'
dataset_path='/ibex/scratch/plotnips/intel/MWE/datasets/dataset_vl_gen_scaled_287145_smooth_450m_6860_scaler_picture_individual_scaling'
dataset_path=['/ibex/scratch/plotnips/intel/MWE/datasets/dataset_dense_source_scaled_2454_smooth_450m_scaler_picture_individual_scaling_true']
# dataset_path='/ibex/scratch/plotnips/intel/MWE/datasets/dataset_vl_gen_scaled_287145_smooth_450m_scaler_picture_individual_scaling_true_x_precon_t_no_precond'
dataset_path=['/ibex/scratch/plotnips/intel/MWE/datasets/dataset_oleg_scaled_44562_smooth_450m_scaler_picture_individual_scaling_true_x_precon_t_no_precond']
# dataset_path=[
#     '/ibex/scratch/plotnips/intel/MWE/datasets/dataset_oleg_scaled_44562_smooth_450m_scaler_picture_individual_scaling_true_x_precon_t_no_precond',
#     '/ibex/scratch/plotnips/intel/MWE/datasets/dataset_vl_gen_scaled_287145_smooth_450m_scaler_picture_individual_scaling_true_x_precon_t_no_precond']
# dataset_path=['/ibex/scratch/plotnips/intel/MWE/datasets/dataset_vl_gen_scaled_287145_smooth_450m_scaler_picture_individual_scaling_true_x_precon_t_no_precond']
# dataset_path=['/ibex/scratch/plotnips/intel/MWE/datasets/dataset_vl_gen_scaled_287145_smooth_450m_scaler_picture_individual_scaling_false']
# dataset_path=['/ibex/scratch/plotnips/intel/MWE/datasets/dataset_vl_gen_individual_init_models_36394']
# dataset_path=['/home/plotnips/Dropbox/Log_extrapolation/scripts/MWE/datasets/dataset_oleg_individual_init_models_1000']
# dataset_path=['/home/plotnips/Dropbox/Log_extrapolation/scripts/MWE/datasets/dataset_oleg_individual_init_models_1000']
# dataset_path=['/ibex/scratch/plotnips/intel/MWE/datasets/dataset_vl_gen']
# dataset_path=['/ibex/scratch/plotnips/intel/MWE/datasets/dataset_oleg_scaled_44562_smooth_450m_scaler_picture_individual_scaling_false',
#               '/ibex/scratch/plotnips/intel/MWE/datasets/dataset_vl_gen_scaled_287145_smooth_450m_scaler_picture_individual_scaling_false']
dataset_path=['/ibex/scratch/plotnips/intel/MWE/datasets/dataset_oleg_individual_init_models_1000',
              '/ibex/scratch/plotnips/intel/MWE/datasets/dataset_vl_gen_individual_init_models_36394']
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
    '/var/remote/lustre/project/k1404/pavel/DENISE-Black-Edition/datasets/gen3_marine_pseudofield_data_cnn_fwi_strategy_13_dsrc_200_pseudo_field_nx_496_test4'
    ]
dataset_path=[
    # 'home/plotnips/Dropbox/Log_extrapolation/scripts/DENISE-Black-Edition-master/datasets/gen3_marine_pseudofield_data_cnn_fwi_strategy_13_dsrc_200_pseudo_field_nx_496_random_trends_1st_attempt',
    '/home/plotnips/Dropbox/Log_extrapolation/scripts/DENISE-Black-Edition-master/datasets/gen3_marine_pseudofield_data_cnn_fwi_strategy_13_dsrc_200_pseudo_field_nx_496_random_trends_1st_attempt'
    ]
# dataset_path=['/home/plotnips/Dropbox/Log_extrapolation/scripts/MWE/datasets/gen3_marine_pseudofield_data_cnn_fwi_strategy_13_dsrc_200_pseudo_field_nx_496']
###########################
####    torch code  ,multiprocessing.cpu_count()-3
parser = argparse.ArgumentParser()
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--start_from_pretrained_model", type=int,  default=0,  help="start_from_pretrained_model")
parser.add_argument("--start_with_certain_dataset",  type=int,  default=0,  help="start_with_certain_dataset")
parser.add_argument("--img_height", type=int, default=4*128, help="size of image width, network nx")
parser.add_argument("--img_width", type=int, default=128, help="size of image height, network nz")
parser.add_argument("--architecture", type=int, default=0,help="gan architecture=1,cnn architecture=0")
# parser.add_argument("--channels", type=int, default=2, help="number of input channels in generator")
parser.add_argument("--channels", type=int, default=2, help="number of input channels in generator")
###########################     '1m_1taper','1dv_1taper','1grad_1dv','1dv',1dv_1init
parser.add_argument("--inp_channels_type", type=str, default='1dv_1init', help="type of input channels in generator")
parser.add_argument("--channels_discriminator", type=int, default=1, help="number of input channels in discriminator")
# parser.add_argument("--cnn_dropout",type=int,default=0.1,help="dropout in cnn")
parser.add_argument("--cnn_dropout",type=int,default=0.2,help="dropout in cnn") #   0.05
# parser.add_argument("--cnn_dropout",type=int,default=0.5,help="dropout in cnn")
parser.add_argument("--shuffle_files",type=int,default=1,help="shuffle files")
####    short training
# parser.add_argument("--channels_discriminator", type=int, default=1, help="number of input channels in discriminator")
parser.add_argument("--dataset_path",type=int,default=dataset_path,help="path to the dataset")
parser.add_argument("--n_epochs",type=int,default=5,help="number of epochs of training")
parser.add_argument("--batch_size", type=int,default=4,help="size of the batches") #16
parser.add_argument("--n_cpu",type=int,default=4,help="number of cpu threads to use during batch generation")   #8
# parser.add_argument("--batch_size", type=int,default=4,help="size of the batches") #16
# parser.add_argument("--n_cpu",type=int,default=2,help="number of cpu threads to use during batch generation")   #8
parser.add_argument("--checkpoint_interval", type=int,  default=40, help="interval between model checkpoints")
parser.add_argument("--plotting_interval", type=int,    default=30, help="plot results at each epoch")
# parser.add_argument("--checkpoint_interval", type=int,  default=100, help="interval between model checkpoints")
# parser.add_argument("--plotting_interval", type=int,    default=100, help="plot results at each epoch")
parser.add_argument("--N",type=int,default=-1,help="total number of samples in training and validation losses")
####    long training
# parser.add_argument("--dataset_path",type=int,default=dataset_path,help="path to the dataset")
# parser.add_argument("--n_epochs",type=int,default=2800,help="number of epochs of training")
# parser.add_argument("--batch_size", type=int, default=32,help="size of the batches")
# parser.add_argument("--n_cpu",type=int,default=8,help="number of cpu threads to use during batch generation")
# parser.add_argument("--checkpoint_interval", type=int, default=30, help="interval between model checkpoints")
# parser.add_argument("--plotting_interval", type=int, default=30, help="plot results at each epoch")
# parser.add_argument("--N", type=int,default=-1, help="total number of samples in training and validation losses")
####    super long training
# # dataset_path=['/ibex/scratch/plotnips/intel/MWE/datasets/dataset_vl_gen_scaled_287145_smooth_450m_scaler_picture_individual_scaling_false']
# parser.add_argument("--dataset_path",type=int,default=dataset_path,help="path to the dataset")
# parser.add_argument("--n_epochs",type=int,default=10,help="number of epochs of training")
# parser.add_argument("--batch_size", type=int, default=2,help="size of the batches")
# parser.add_argument("--n_cpu",type=int,default=1,help="number of cpu threads to use during batch generation")
# parser.add_argument("--checkpoint_interval", type=int, default=50, help="interval between model checkpoints")
# parser.add_argument("--plotting_interval", type=int, default=50, help="plot results at each epoch")
# parser.add_argument("--N", type=int, default=30, help="total number of samples in training and validation losses")
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
        path_=path_[-val:];
    path=path+path_
    counter=counter+1
print('len(path)=',len(path))
######################################  denise data
# path_test=path
# path=list(set(path)-set(path_test));
######################################  sfgpu data
path_test=fnmatch.filter(path,'*__*')
path=list(set(path)-set(path_test));    
path=sorted(path)
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
######################################  big dataset
# path_train=path[0:10000];   path_valid=path[10000:10500]
######################################  small dataset
# path_train=path[0:3000];   path_valid=path[-500:]
######################################   custom split 1
# path_valid=path[-200:]      #   -200
path_valid=path[-10:]      #   -200
path_train=list(set(path)-set(path_valid))
# path_train=path_train[0:30]
######################################   custom split 2
# path_valid=path[-200:]      #   -200
# # path_valid=path[-100:]      #   -200
# path_train=list(set(path)-set(path_valid))
# # path_train=path_train[0:30]
######################################
# path_valid=path[-500:]
# path_train=list(set(path)-set(path_valid))
# path_train=path_train[0:2000]
######################################
if opt.start_with_certain_dataset==1:
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
    tmp=logs_path+'/log'+str(log_name)+'/opt.txt'
    parser = ArgumentParser();  opt_old = parser.parse_args()
    print('loading opt parameters from ',tmp)
    with open(tmp,'r') as f:
        opt_old.__dict__=json.load(f)
    ##########################################  exact copy
    opt.path_test=opt_old.path_test;
    opt.path_valid=return_existing_file_list(opt_old.path_valid,opt.dataset_path)
    opt.path_train=return_existing_file_list(opt_old.path_train,opt.dataset_path)
    # opt.path_valid=opt_old.path_valid;
    # opt.path_train=opt_old.path_train;
else:
    opt.path_test=path_test;    opt.path_valid=path_valid;  opt.path_train=path_train;
if opt.n_epochs==-1 or start_from_pretrained_model==1:
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
    tmp=logs_path+'/log'+str(log_name)+'/opt.txt'
    n_epochs=opt.n_epochs;
    # if os.path.exists(tmp):
    #     parser = ArgumentParser();  opt = parser.parse_args()
    print('loading opt parameters from ',tmp)
    parser = ArgumentParser();  loaded_opt = parser.parse_args()
    with open(tmp,'r') as f:
        print('loading json.load')
        loaded_opt.__dict__=json.load(f)
    print(opt.path_test)
    opt.channels=loaded_opt.channels
    opt.history=loaded_opt.history
    opt.save_path=Save_pictures_path
    opt.log_save_const=log_save_const
    opt.n_epochs=n_epochs
    opt.generator_model_name=       './logs/log'+str(log_name)+'/generator_'+str(epoch_name)+'.pth'
    opt.discriminator_model_name=   './logs/log'+str(log_name)+'/discriminator_'+str(epoch_name)+'.pth'
    Plot_r2_losses_2(opt.history,Title='log'+str(log_name) + 'losses_r2', Save_pictures_path=Save_pictures_path,Save_flag=1)
    Plot_unet_losses(opt.history,Title='log'+str(log_name) + 'losses_unet', Save_pictures_path=Save_pictures_path,Save_flag=1)
    Plot_unet_losses_2(opt.history,Title='log'+str(log_name) + 'losses_pixel', Save_pictures_path=Save_pictures_path,Save_flag=1)
print('len(opt.path_train)=',len(opt.path_train))
print('len(opt.path_valid)=',len(opt.path_valid))
print('len(opt.path_test)=', len(opt.path_test))
# print(opt.path_train);    print(opt.path_valid);  print(opt.path_test)
# aa=fnmatch.filter(opt.path_valid,'*oleg*')
# aa2=fnmatch.filter(opt.path_train,'*oleg*');    len(aa2)
# path_all=opt.path_train+opt.path_valid# oleg=fnmatch.filter(path_all,'*oleg*')# vl_gen=fnmatch.filter(path_all,'*vl_gen*')
#####   !!!!!!!!!!!!!!!!!!!!!!!!!!!11DELETE!!!!!!!!!!!!!!!!!!!!!!!!!!!11
# opt.path_train=opt.path_train[0:100]
# opt.path_valid=opt.path_valid[0:100]
#####
torch.backends.cudnn.benchmark = True
cuda = True if torch.cuda.is_available() else False
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
###################    unet Generator
generator = GeneratorUNet(in_channels=opt.channels,DROPOUT=opt.cnn_dropout);    gen_name='unet'
# generator = GeneratorUNet_old_configuration(in_channels=opt.channels,DROPOUT=opt.cnn_dropout)
# generator = GeneratorUNet_big(in_channels=opt.channels,DROPOUT=opt.cnn_dropout)
###################    fusion_net Generator
fusion_net_parameters=  (opt.channels,1,2)
# fusion_net_parameters=(opt.channels,1,8)
# fusion_net_parameters=(opt.channels,1,12)

# fusion_net_parameters=(opt.channels,1,24)

# fusion_net_parameters=(opt.channels,1,48)
print('fusion net parameters=(input_nc,output_nc,ngf)',fusion_net_parameters)
generator=nn.DataParallel(FusionGenerator(fusion_net_parameters[0],fusion_net_parameters[1],fusion_net_parameters[2]));   gen_name='fusion'
# summary(fusion.to(0),(1,256,256))
# summary(fusion.to(0),(1,128,128))
###################
summary(generator.to(0),(opt.channels,opt.img_height,opt.img_width))
# summary(generator.to(0), (opt.channels,256,128))
# exit()
###################     Discriminator
discriminator = Discriminator(in_channels=opt.channels_discriminator)
# summary(discriminator.to(0),[(1,opt.img_height,opt.img_width),(opt.channels_discriminator,opt.img_height,opt.img_width)] )
################## 
print('DL network name=',gen_name)
if cuda:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    generator = generator.cuda()
    discriminator = discriminator.cuda()
    criterion_GAN.cuda()
    criterion_pixelwise.cuda()
    criterion_R2Loss.cuda()
    criterion_R2Loss_custom.cuda()
    criterion_NRMS.cuda()
if opt.n_epochs==-1 or start_from_pretrained_model==1:
    print('Load pretrained models',opt.generator_model_name,opt.discriminator_model_name)
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
# print(opt)
# print(opt.path_test+opt.path_train+opt.path_valid)
max_for_initial_models=ImageDataset(opt,transforms_=transforms_,mode="all").calculate_max_model_init()
train_dataset = ImageDataset(opt, transforms_=transforms_, mode="train", max_for_initial_models=max_for_initial_models)
val_dataset = ImageDataset(opt, transforms_=transforms_, mode="val", max_for_initial_models=max_for_initial_models)
test_dataset = ImageDataset(opt, transforms_=transforms_, mode="test", max_for_initial_models=max_for_initial_models)
# next(iter(train_dataset))
# max_for_initial_models=6600
train_dataloader=DataLoader(train_dataset,batch_size=opt.batch_size,shuffle=False,num_workers=opt.n_cpu,pin_memory=True)
validation_dataloader=DataLoader(val_dataset,batch_size=opt.batch_size,shuffle=False,num_workers=opt.n_cpu,pin_memory=True)
test_dataloader=DataLoader(test_dataset,batch_size=1,shuffle=False,num_workers=1)
tracking_train_dataloader=DataLoader(ImageDataset(opt,transforms_=transforms_,
    mode=None,file_list=opt.path_train[0:2],max_for_initial_models=max_for_initial_models),     #+path_test[-2:-1]+path_test[-4:-3]
    batch_size=1,shuffle=False,num_workers=1)
tracking_val_dataloader=DataLoader(ImageDataset(opt,transforms_=transforms_,
    mode=None,file_list=opt.path_valid[0:5],max_for_initial_models=max_for_initial_models),
    batch_size=1,shuffle=False,num_workers=1)
opt.logging_loss_batch_size=len(train_dataloader)
# sample_models(tracking_dataloader2,generator,opt,opt.n_epochs,flag_show_scaled_data=0)
# sample_models(test_dataloader,generator,opt,opt.n_epochs,flag_show_scaled_data=0)
# sample_dataset(test_dataloader,generator,opt,opt.n_epochs,flag_show_scaled_data=1,record_weights=1,data_mode='test')
# sample_dataset(test_dataloader,generator,opt,opt.n_epochs,flag_show_scaled_data=1,data_mode='test')
with open(opt.save_path+'/'+'opt.txt', 'w') as f:
    json.dump(opt.__dict__, f, indent=2)
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
accumulated_target_pred_nrms=0
accumulated_init_model_pred_acc=0
accumulated_init_model_pred_nrms=0
accumulated_true_model_pred_acc=0
accumulated_true_model_inversion_acc=0
if opt.n_epochs!=-1:
    for epoch in range(0,opt.n_epochs):
        for phase in phases:
            if phase=='training':   
                dataloader=train_dataloader;
                generator.train();discriminator.train()
            elif phase=='validation':   
                dataloader=validation_dataloader;
                generator.eval();discriminator.eval()
            for i, batch in enumerate(dataloader):
                # Model inputs
                real_A = Variable(batch["A"].type(Tensor))
                real_B = Variable(batch["B"].type(Tensor))
                init_model=Variable(batch["C"].type(Tensor))
                true_model=Variable(batch["D"].type(Tensor))
                real_A_ra = Variable(batch["E"].type(Tensor))   # real amplitudes (m/sec)
                real_B_ra = Variable(batch["F"].type(Tensor))   # real amplitudes (m/sec)
                # scaler_t=Variable(batch["sc_t"].type(Tensor)).cpu().detach().numpy()   # real amplitudes (m/sec)
                # Adversarial ground truths
                valid = Variable(Tensor(np.ones((real_A.size(0), *patch))), requires_grad=False)
                fake = Variable(Tensor(np.zeros((real_A.size(0), *patch))), requires_grad=False)
                #############  Train Generators
                if phase=='training':   optimizer_G.zero_grad(set_to_none=True)
                # GAN loss
                fake_B = generator(real_A)
                # Pixel-wise loss
                loss_pixel = criterion_pixelwise(fake_B,real_B)
                if opt.architecture==1:
                    pred_fake = discriminator(fake_B,real_A)
                    loss_GAN = criterion_GAN(pred_fake,valid)
                    loss_G = loss_GAN + opt.lambda_pixel*loss_pixel
                else:
                    loss_G = 1*loss_pixel
                ##############  make some preprocessing for extra losses, append water and scale predicted data back
                tmp=fake_B.cpu().detach().numpy()
                t_pred=np.empty((real_B_ra.shape))
                Nx=real_B_ra.shape[2]; Nz=real_B_ra.shape[3]
                for i_f in range(tmp.shape[0]):
                    t_pred[i_f,0,:,:]=imresize(tmp[i_f,0,:, :],(Nx, Nz))
                fake_B_ra=np.ones_like(t_pred)
                for i_f in range(t_pred.shape[0]):
                    tmp=transforming_data_inverse(t_pred[i_f,::],scaler_t[i_f])
                    fake_B_ra[i_f,0,:,:]=np.squeeze(tmp)
                water_thickness=true_model.shape[3]-Nz
                water=np.zeros((fake_B_ra.shape[0],fake_B_ra.shape[1],fake_B_ra.shape[2],water_thickness))
                water=torch.from_numpy(water).to(device)
                fake_B_ra=torch.from_numpy(fake_B_ra).to(device)
                fake_B_ra=torch.cat((water,fake_B_ra),axis=3)
                real_A_ra=torch.cat((water,real_A_ra),axis=3)
                real_B_ra=torch.cat((water,real_B_ra),axis=3)
                ##############      calculate tracking losses
                target_pred_acc=criterion_R2Loss(fake_B,real_B);
                # target_pred_nrms=criterion_NRMS(fake_B,real_B)
                init_model_pred_acc=criterion_R2Loss(init_model+fake_B_ra,init_model+real_B_ra)
                # init_model_pred_nrms=criterion_NRMS(init_model+fake_B_ra,init_model+real_B_ra)
                true_model_pred_acc=criterion_R2Loss(init_model+fake_B_ra,true_model)       #true_model_prediction_acc
                true_model_inversion_acc=criterion_R2Loss(init_model+real_A_ra,true_model)  #true_model_inversion_acc 
                ##############  accumulate tracking losses
                accumulated_target_pred_acc+=target_pred_acc.item()
                # accumulated_target_pred_nrms+=target_pred_nrms.item()
                accumulated_init_model_pred_acc+=init_model_pred_acc.item()
                # accumulated_init_model_pred_nrms+=init_model_pred_nrms.item()
                accumulated_true_model_pred_acc+=true_model_pred_acc.item()
                accumulated_true_model_inversion_acc+=true_model_inversion_acc.item()
                ############### Total loss
                if phase=='training':   loss_G.backward();  optimizer_G.step()
                ##############  Train Discriminator
                if opt.architecture==1:
                    if phase=='training':   optimizer_D.zero_grad(set_to_none=True)
                    # Real loss
                    pred_real = discriminator(real_B, real_A)
                    loss_real = criterion_GAN(pred_real, valid)
                    # Fake loss
                    pred_fake = discriminator(fake_B.detach(), real_A)
                    loss_fake = criterion_GAN(pred_fake, fake)
                    # Total loss
                    loss_D = 0.5 * (loss_real+loss_fake)
                    if phase=='training':   loss_D.backward();  optimizer_D.step()
                ##############  Log Progress
                # Determine approximate time left
                batches_done = epoch * len(dataloader) + i
                batches_left = opt.n_epochs * len(dataloader) - batches_done
                time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
                prev_time = time.time()
                # Print log
                # If at sample interval save image
                logging_loss_batch_size=len(dataloader)
            # if i % logging_loss_batch_size == logging_loss_batch_size-1:
            #     print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, accumulated_target_pred_acc / logging_loss_batch_size))
            #     # R2_score=accumulated_target_pred_acc/logging_loss_batch_size
            #     # R2_score_init_model=accumulated_init_model_pred_acc/logging_loss_batch_size
            if opt.architecture==1:
                sys.stdout.write("\rLog:%s PHASE: %s [Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, pixel: %f, adv: %f] ETA: %s"
                        % ( str(log_save_const),phase,epoch,opt.n_epochs,i,len(dataloader),
                        loss_D.item(),loss_G.item(),opt.lambda_pixel*loss_pixel.item(),loss_GAN.item(),time_left,))
            else:
                sys.stdout.write("\rLog:%s PHASE: %s [Epoch %d/%d] [Batch %d/%d] [G loss: %f] ETA: %s"
                    % ( str(log_save_const),phase,epoch,opt.n_epochs,i,len(dataloader),loss_G.item(),time_left,))          
            if phase=='training':
                loss_target_pred_train.append(accumulated_target_pred_acc/logging_loss_batch_size)
                loss_init_model_pred_train.append(accumulated_init_model_pred_acc/logging_loss_batch_size)
                loss_true_model_pred_train.append(accumulated_true_model_pred_acc/logging_loss_batch_size)
                loss_true_model_inversion_train.append(accumulated_true_model_inversion_acc/logging_loss_batch_size)
                ###                
                loss_pixel_train.append(opt.lambda_pixel*loss_pixel.item())
                g_loss_train.append(loss_G.item())
                if opt.architecture==1:
                    loss_GAN_train.append(loss_GAN.item())
                    d_loss_train.append(loss_D.item())
                    loss_real_train.append(loss_real.item())
                    loss_fake_train.append(loss_fake.item())
            elif phase=='validation':
                loss_target_pred_valid.append(accumulated_target_pred_acc/logging_loss_batch_size)
                loss_init_model_pred_valid.append(accumulated_init_model_pred_acc/logging_loss_batch_size)
                loss_true_model_pred_valid.append(accumulated_true_model_pred_acc/logging_loss_batch_size)
                loss_true_model_inversion_valid.append(accumulated_true_model_inversion_acc/logging_loss_batch_size)
                ###
                loss_pixel_val.append(opt.lambda_pixel*loss_pixel.item())
                g_loss_val.append(loss_G.item())
                if opt.architecture==1:
                    loss_GAN_val.append(loss_GAN.item())
                    d_loss_val.append(loss_D.item())
                    loss_real_val.append(loss_real.item())
                    loss_fake_val.append(loss_fake.item())
                if opt.architecture==1 and epoch%opt.plotting_interval==0:
                    history={'g_loss_train':g_loss_train,'g_loss_val':g_loss_val,
                            'loss_GAN_train':loss_GAN_train,'loss_GAN_val':loss_GAN_val,
                            'loss_pixel_train':loss_pixel_train,'loss_pixel_val':loss_pixel_val,
                            'd_loss_train':d_loss_train,'d_loss_val':d_loss_val,
                            'loss_real_train':loss_real_train,    'loss_real_val':loss_real_val,
                            'loss_fake_train':loss_fake_train,    'loss_fake_val':loss_fake_val,
                            'loss_R2_score_train':loss_R2_score_train,'loss_R2_score_val':loss_R2_score_val};
                    Plot_g_losses(history,Title='log'+str(log_save_const) + 'losses_g', Save_pictures_path=Save_pictures_path,Save_flag=1)
                    Plot_r2_losses(history,Title='log'+str(log_save_const) + 'losses_r2', Save_pictures_path=Save_pictures_path,Save_flag=1)
                    Plot_d_losses(history,Title='log'+str(log_save_const) + 'losses_d', Save_pictures_path=Save_pictures_path,Save_flag=1)
                elif opt.architecture==0 and epoch%opt.plotting_interval==0:
                    history={'g_loss_train':g_loss_train,'g_loss_val':g_loss_val,
                            'loss_pixel_train':loss_pixel_train,'loss_pixel_val':loss_pixel_val,
                            'loss_target_pred_train':loss_target_pred_train,'loss_target_pred_valid':loss_target_pred_valid,
                            'loss_init_model_pred_train':loss_init_model_pred_train,'loss_init_model_pred_valid':loss_init_model_pred_valid,
                            'loss_true_model_pred_train':loss_true_model_pred_train,'loss_true_model_pred_valid':loss_true_model_pred_valid,
                            'loss_true_model_inversion_train':loss_true_model_inversion_train,'loss_true_model_inversion_valid':loss_true_model_inversion_valid}
                elif epoch%5==0:
                    Plot_r2_losses_2(history,Title='log'+str(log_save_const) + 'losses_r2', Save_pictures_path=Save_pictures_path,Save_flag=1)   
                    Plot_unet_losses(history,Title='log'+str(log_save_const) + 'losses_unet', Save_pictures_path=Save_pictures_path,Save_flag=1)
                    Plot_unet_losses_2(history,Title='log'+str(log_save_const) + 'losses_pixel', Save_pictures_path=Save_pictures_path,Save_flag=1)
                opt.history=history
                if epoch<15:    current_checkpoint_interval=2;current_plotting_interval=current_checkpoint_interval
                else:           current_checkpoint_interval=opt.checkpoint_interval;current_plotting_interval=opt.plotting_interval
                if opt.checkpoint_interval != -1 and epoch%current_checkpoint_interval == 0:
                    torch.save(generator.state_dict(), "%s/generator_%d.pth" % (Save_pictures_path, epoch))
                    if opt.architecture==1:
                        torch.save(discriminator.state_dict(), "%s/discriminator_%d.pth" % (Save_pictures_path,epoch))
                    with open(opt.save_path+'/'+'opt.txt', 'w') as f:
                        json.dump(opt.__dict__, f, indent=2)
                ###################################################
                if epoch==opt.n_epochs-1:
                    torch.save(generator.state_dict(), "%s/generator_%d.pth" % (Save_pictures_path,epoch))
                    torch.save(discriminator.state_dict(), "%s/discriminator_%d.pth" % (Save_pictures_path,epoch))
                    print('time(now)=',datetime.datetime.now())
                if epoch % current_plotting_interval==0 and epoch!=0:
                    # sample_dataset(tracking_dataloader,generator,opt,epoch,flag_show_scaled_data=1) #tracking_dataloader.dataset.files
                    sample_dataset(test_dataloader,generator,opt,epoch,flag_show_scaled_data=1,record_weights=1)
            accumulated_target_pred_acc=0
            accumulated_init_model_pred_acc=0
            accumulated_true_model_pred_acc=0
            accumulated_true_model_inversion_acc=0
            ss=1
    print('Training time(sec)=',datetime.datetime.now()-T1)
generator.eval();discriminator.eval()
#########  Plotting testing results
sample_dataset(test_dataloader,generator,opt,opt.n_epochs,flag_show_scaled_data=1,record_weights=1,data_mode='test')
#########  save training session parameters to the file
with open(opt.save_path+'/'+'opt.txt', 'w') as f:
    json.dump(opt.__dict__, f, indent=2)
#########  plot losses
if opt.architecture==1:
    Plot_r2_losses(history,Title='log'+str(log_save_const) + 'losses_r2', Save_pictures_path=Save_pictures_path,Save_flag=1)
    Plot_g_losses(history,Title='log'+str(log_save_const) + 'losses_g', Save_pictures_path=Save_pictures_path,Save_flag=1)
    Plot_d_losses(history,Title='log'+str(log_save_const) + 'losses_d', Save_pictures_path=Save_pictures_path,Save_flag=1)
else:
    Plot_r2_losses_2(history,Title='log'+str(log_save_const) + 'losses_r2', Save_pictures_path=Save_pictures_path,Save_flag=1)
    Plot_unet_losses(history,Title='log'+str(log_save_const) + 'losses_unet', Save_pictures_path=Save_pictures_path,Save_flag=1)
    Plot_unet_losses_2(history,Title='log'+str(log_save_const) + 'losses_pixel', Save_pictures_path=Save_pictures_path,Save_flag=1)
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
    batch_size_val=40  
    num_workers_val=10
    print('calculate misfits')
    ch1=[];ch2=[];ch3=[];ch4=[]
    for phase in phases:
        if phase=='training':       
            dataloader=DataLoader(ImageDataset(opt,transforms_=transforms_,mode="train",max_for_initial_models=max_for_initial_models),
                batch_size=batch_size_val,shuffle=False,num_workers=num_workers_val)
        elif phase=='validation':   
            dataloader=DataLoader(ImageDataset(opt,transforms_=transforms_,mode="val",max_for_initial_models=max_for_initial_models),
                batch_size=batch_size_val,shuffle=False,num_workers=num_workers_val)
        elif phase=='test':         
            dataloader=DataLoader(ImageDataset(opt,transforms_=transforms_,mode="test",max_for_initial_models=max_for_initial_models),
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
            scaler_t=Variable(batch["sc_t"].type(Tensor)).cpu().detach().numpy()
            fake_B = generator(real_A)
            target_pred_acc=criterion_R2Loss_custom(fake_B,real_B);
            ##############  make some preprocessing for extra losses, append water and scale predicted data back
            tmp=fake_B.cpu().detach().numpy()
            t_pred=np.empty((real_B_ra.shape))
            Nx=real_B_ra.shape[2]; Nz=real_B_ra.shape[3]
            for i_f in range(tmp.shape[0]):
                t_pred[i_f,0,:,:]=imresize(tmp[i_f,0,:, :],(Nx, Nz))
            fake_B_ra=np.ones_like(t_pred)
            for i_f in range(t_pred.shape[0]):
                tmp=transforming_data_inverse(t_pred[i_f,::],scaler_t[i_f])
                fake_B_ra[i_f,0,:,:]=np.squeeze(tmp)
            water_thickness=true_model.shape[3]-Nz
            water=np.zeros((fake_B_ra.shape[0],fake_B_ra.shape[1],fake_B_ra.shape[2],water_thickness))
            water=torch.from_numpy(water).to(device)
            fake_B_ra=torch.from_numpy(fake_B_ra).to(device)
            fake_B_ra=torch.cat((water,fake_B_ra),axis=3)
            real_A_ra=torch.cat((water,real_A_ra),axis=3)
            real_B_ra=torch.cat((water,real_B_ra),axis=3)
            ##############      calculate extra tracking losses
            init_model_pred_acc=criterion_R2Loss_custom(init_model+fake_B_ra,init_model+real_B_ra)
            init_model_inversion_acc=criterion_R2Loss_custom(init_model+real_A_ra,init_model+real_B_ra)
            true_model_pred_acc=criterion_R2Loss_custom(init_model+fake_B_ra,true_model)       #true_model_prediction_acc
            true_model_inversion_acc=criterion_R2Loss_custom(init_model+real_A_ra,true_model)  #true_model_inversion_acc    
            ##############      fill result matrix
            predictions[ ((i)*batch_size):((i)*batch_size)+len(target_pred_acc),0]=target_pred_acc
            predictions[ ((i)*batch_size):((i)*batch_size)+len(target_pred_acc),1]=init_model_pred_acc
            predictions[ ((i)*batch_size):((i)*batch_size)+len(target_pred_acc),2]=init_model_inversion_acc
            predictions[ ((i)*batch_size):((i)*batch_size)+len(target_pred_acc),3]=true_model_pred_acc
            predictions[ ((i)*batch_size):((i)*batch_size)+len(target_pred_acc),4]=true_model_inversion_acc
            predictions[ ((i)*batch_size):((i)*batch_size)+len(target_pred_acc),5]=criterion_NRMS(fake_B,real_B)
            predictions[ ((i)*batch_size):((i)*batch_size)+len(target_pred_acc),6]=criterion_NRMS(init_model+fake_B_ra,init_model+real_B_ra)
            predictions[ ((i)*batch_size):((i)*batch_size)+len(target_pred_acc),7]=criterion_NRMS(init_model+real_A_ra,init_model+real_B_ra)
            predictions[ ((i)*batch_size):((i)*batch_size)+len(target_pred_acc),8]=criterion_NRMS(init_model+fake_B_ra,true_model)
            predictions[ ((i)*batch_size):((i)*batch_size)+len(target_pred_acc),9]=criterion_NRMS(init_model+real_A_ra,true_model)
            ss=1
        df = pd.DataFrame(predictions,columns=
            ['target_pred_acc' ,'init_model_pred_acc','init_model_inversion_acc','true_model_pred_acc','true_model_inversion_acc',
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
    # scores_path='./logs/log1048'
    # scores_path='./logs/log1047'
    # scores_path='./logs/log1049'
    # scores_path='./logs/log1092'
    # scores_path='./logs/log1134'
    predictions_train=  pd.read_csv(os.path.join(scores_path,'inference_scores_train.zip'))
    predictions_valid=  pd.read_csv(os.path.join(scores_path,'inference_scores_valid.zip'))
    predictions_test=   pd.read_csv(os.path.join(scores_path,'inference_scores_test.zip'))
    predictions_train=predictions_train.sort_values(by=['true_model_pred_acc'])
    predictions_valid=predictions_valid.sort_values(by=['true_model_pred_acc'])
    predictions_test=predictions_test.sort_values(by=['true_model_pred_acc'])
    #################   check worst predicted files 
    val_worst_values=predictions_valid[0:10]
    val_best_values=predictions_valid[-3:]
    train_worst_values=predictions_train[0:10]
    dl_val_best=DataLoader(ImageDataset(opt,transforms_=transforms_,
        mode=None,file_list=list(val_best_values.filename),max_for_initial_models=max_for_initial_models),
        batch_size=1,shuffle=False,num_workers=1)
    dl_val_worst=DataLoader(ImageDataset(opt,transforms_=transforms_,
        mode=None,file_list=list(val_worst_values.filename),max_for_initial_models=max_for_initial_models),
        batch_size=1,shuffle=False,num_workers=1)
    dl_train_worst=DataLoader(ImageDataset(opt,transforms_=transforms_,
        mode=None,file_list=list(train_worst_values.filename),max_for_initial_models=max_for_initial_models),
        batch_size=1,shuffle=False,num_workers=1)
    #################
    data=val_worst_values
    for i in range(data.shape[0]):
        print(data[i:i+1].filename.values[0],data[i:i+1].target_pred_acc)
    data=val_best_values
    for i in range(data.shape[0]):
        print(data[i:i+1].filename.values[0],data[i:i+1].target_pred_acc)   
    data=train_worst_values
    for i in range(data.shape[0]):
        print(data[i:i+1].filename.values[0],data[i:i+1].target_pred_acc)
    sample_dataset(dl_val_worst,generator,opt,opt.n_epochs,flag_show_scaled_data=1,record_weights=1,data_mode='val_worst')
    sample_dataset(dl_val_best,generator,opt,opt.n_epochs,flag_show_scaled_data=1,record_weights=0,data_mode='val_best')
    sample_dataset(dl_train_worst,generator,opt,opt.n_epochs,flag_show_scaled_data=1,record_weights=0,data_mode='train_worst')
    #################   R2 scores
    Fontsize=24
    plt.rcParams.update({'font.size': Fontsize})
    plt.figure(figsize=(10,8),dpi=300)
    plt.hist(np.vstack((predictions_train.target_pred_acc.to_numpy(),predictions_train.true_model_pred_acc.to_numpy())).T,
        # range=[np.min(predictions_train.target_pred_acc), 1], bins=20, cumulative=False)
        range=[-2,1], bins=20, cumulative=False)
    plt.ylabel('Number of samples')
    plt.xlabel('R2(predicted,true)')
    plt.title('Training samples, total= '+str(predictions_train.shape[0])+' samples')
    plt.legend(['target_pred_acc','true_model_pred_acc','init_model_pred_acc-init_model_inversion_acc'])   
    tmp=os.path.join(Save_pictures_path,'hist_training_misfits_r2.png')
    print('Saving to '+tmp)
    plt.tight_layout()
    plt.savefig(tmp,dpi=300)
    plt.show()
    plt.close()

    plt.figure(figsize=(10,8),dpi=300)
    plt.hist(np.vstack((predictions_valid.target_pred_acc.to_numpy(),predictions_valid.true_model_pred_acc.to_numpy())).T,
        range=[np.min(predictions_valid.target_pred_acc), 1], bins=20, cumulative=False)
    plt.ylabel('Number of samples')
    plt.xlabel('R2(predicted,true)')
    plt.title('Validation samples, total= '+str(predictions_valid.shape[0])+' samples')
    plt.legend(['target_pred_acc','init_model_pred_acc','init_model_pred_acc-init_model_inversion_acc'])
    tmp=os.path.join(Save_pictures_path,'hist_valid_misfits_r2.png')
    print('Saving to '+tmp)
    plt.tight_layout()
    plt.savefig(tmp,dpi=300)
    plt.show()
    plt.close()

    plt.figure(figsize=(10,8),dpi=300)
    plt.hist(np.vstack((predictions_test.target_pred_acc.to_numpy(),predictions_test.true_model_pred_acc.to_numpy())).T,
        range=[np.min(predictions_test.target_pred_acc), 1], bins=20, cumulative=False)
    plt.ylabel('Number of samples')
    plt.xlabel('R2(predicted,true)')
    plt.title('Testing samples, total= '+str(predictions_test.shape[0])+' samples')
    plt.legend(['target_pred_acc','init_model_pred_acc'])
    tmp=os.path.join(Save_pictures_path,'hist_testing_misfits_r2.png')
    print('Saving to '+tmp)
    plt.tight_layout()
    plt.savefig(tmp,dpi=300)
    plt.show()
    plt.close()
    #################   NRMS scores
    plt.figure(figsize=(10,8),dpi=300)
    plt.hist(np.vstack((predictions_train.target_pred_nrms.to_numpy(),predictions_train.true_model_pred_nrms.to_numpy())).T,
        range=[np.min(predictions_train.target_pred_nrms), 1], bins=20, cumulative=False)
    plt.ylabel('Number of samples')
    plt.xlabel('R2(predicted,true)')
    plt.title('Training samples, total= '+str(predictions_train.shape[0])+' samples')
    # plt.legend(['target_pred_nrms','true_model_pred_nrms'])
    plt.legend(['target_pred_nrms','true_model_pred_nrms','true_model_pred_nrms-true_model_inversiom_nrms'])
    tmp=os.path.join(Save_pictures_path,'hist_train_misfits_nrms.png')
    print('Saving to '+tmp)
    plt.tight_layout()
    plt.savefig(tmp,dpi=300)
    plt.show()
    plt.close()

    plt.figure(figsize=(10,8),dpi=300)
    plt.hist(np.vstack((predictions_valid.target_pred_nrms.to_numpy(),predictions_valid.true_model_pred_nrms.to_numpy())).T,
        range=[np.min(predictions_valid.target_pred_nrms), 1], bins=20, cumulative=False)
    plt.ylabel('Number of samples')
    plt.xlabel('R2(predicted,true)')
    plt.title('Validation samples, total= '+str(predictions_valid.shape[0])+' samples')
    plt.legend(['target_pred_nrms','true_model_pred_nrms','true_model_pred_nrms-true_model_inversiom_nrms'])
    tmp=Save_pictures_path + '/' +'hist_'+'valid_misfits'+ '.png'
    tmp=os.path.join(Save_pictures_path,'hist_valid_misfits_nrms.png')
    print('Saving to '+tmp)
    plt.tight_layout()
    plt.savefig(tmp,dpi=300)
    plt.show()
    plt.close()

    plt.figure(figsize=(10,8),dpi=300)
    plt.hist(np.vstack((predictions_test.target_pred_nrms.to_numpy(),predictions_test.true_model_pred_nrms.to_numpy())).T,
        range=[np.min(predictions_test.target_pred_nrms), 1], bins=20, cumulative=False)
    plt.ylabel('Number of samples')
    plt.xlabel('R2(predicted,true)')
    plt.title('Testing samples, total= '+str(predictions_test.shape[0])+' samples')
    plt.legend(['target_pred_nrms','true_model_pred_nrms'])
    tmp=os.path.join(Save_pictures_path,'hist_testing_misfits_nrms.png')
    print('Saving to '+tmp)
    plt.tight_layout()
    plt.savefig(tmp,dpi=300)
    plt.show()
    plt.close()
#########  Plotting testing results
# sample_dataset(dl_train_worst,generator,opt,opt.n_epochs,flag_show_scaled_data=1,record_weights=0,data_mode='train_worst')
# sample_dataset(tracking_val_dataloader,generator,opt,opt.n_epochs,flag_show_scaled_data=1,record_weights=0,data_mode='val')
# sample_dataset(tracking_train_dataloader,generator,opt,opt.n_epochs,flag_show_scaled_data=1,record_weights=0,data_mode='train')

# name_to_plot=opt.path_train[np.argmin(predictions_train)]
    # bad_files_dataloader=DataLoader(ImageDataset(opt,transforms_=transforms_,
    #     mode=None,file_list=name_to_plot,max_for_initial_models=max_for_initial_models),
    #     batch_size=1,shuffle=False,num_workers=1)
    # sample_dataset(bad_files_dataloader,generator,opt,opt.n_epochs,flag_show_scaled_data=0)
######################################
# tmp='/ibex/scratch/plotnips/intel/MWE/Keras-GAN/pix2pix/logs/log403/opt.txt'
# tmp='/ibex/scratch/plotnips/intel/MWE/Keras-GAN/pix2pix/logs/log390/opt.txt'
# parser=ArgumentParser();  opt = parser.parse_args()
# with open(tmp,'r') as f:
#     opt.__dict__=json.load(f)
# history1=opt.history
# tmp='/ibex/scratch/plotnips/intel/MWE/Keras-GAN/pix2pix/logs/log453/opt.txt'
# tmp='/ibex/scratch/plotnips/intel/MWE/Keras-GAN/pix2pix/logs/log452/opt.txt'
# parser=ArgumentParser();  opt = parser.parse_args()
# with open(tmp,'r') as f:
#     opt.__dict__=json.load(f)
# history2=opt.history
# history_=history1
# history_['loss_R2_score_train']=np.append(history1['loss_R2_score_train'],history2['loss_R2_score_train'])
# history_['loss_R2_score_val']=np.append(history1['loss_R2_score_val'],history2['loss_R2_score_val'])
# history_['g_loss_train']=np.append(history1['g_loss_train'],history2['g_loss_train'])
# history_['g_loss_val']=np.append(history1['g_loss_val'],history2['g_loss_val'])
# history_['d_loss_train']=np.append(history1['d_loss_train'],history2['d_loss_train'])
# history_['d_loss_val']=np.append(history1['d_loss_val'],history2['d_loss_val'])
# history_['loss_real_train']=np.append(history1['loss_real_train'],history2['loss_real_train'])
# history_['loss_fake_train']=np.append(history1['loss_fake_train'],history2['loss_fake_train'])
# history_['loss_GAN_train']=np.append(history1['loss_GAN_train'],history2['loss_GAN_train'])
# history_['loss_pixel_train']=np.append(history1['loss_pixel_train'],history2['loss_pixel_train'])
# Plot_r2_losses(history_,Title='log'+str(log_save_const) + 'losses_r2',Save_pictures_path=Save_pictures_path,Save_flag=1)
# Plot_g_losses( history_,Title='log'+str(log_save_const) + 'losses_g', Save_pictures_path=Save_pictures_path,Save_flag=1)
# Plot_d_losses( history_,Title='log'+str(log_save_const) + 'losses_d', Save_pictures_path=Save_pictures_path,Save_flag=1)
# exit()

# names,scores=calculate_accuracy_on_dataset(tracking_dataloader,generator,opt,opt.n_epochs);  print(names);print(scores)
# Plot_loss_torch(history,Title='log'+str(log_save_const) + 'losses', Save_pictures_path=Save_pictures_path,Save_flag=1)
# generator.load_state_dict(torch.load("saved_models/%s/generator_%d.pth" % (opt.dataset_name, opt.epoch)))
# discriminator.load_state_dict(torch.load("saved_models/%s/discriminator_%d.pth" % (opt.dataset_name, opt.epoch)))
# If at sample interval save image
# if batches_done % opt.sample_interval == 0:
#     sample_images(batches_done)
#     if os.path.exists(tmp):
    #         parser = ArgumentParser();  opt_old=parser.parse_args()
    #         with open(tmp,'r') as f:
    #             opt_old.__dict__=json.load(f)
############################
# opt.path_train[1000]
# NAME=opt.path_test[0]
# with open(NAME,'rb') as f:
#     data=np.load(f,allow_pickle=True)
#     M0=data['models'][0,:,:,0]; dz=data['dz'];  dx=data['dx']
#     M1=data['input_data'];  M2=data['output_data']
#     Nx=M1.shape[1];     Nz=M1.shape[2];
#     Minit1=data['models_init'][0,:,:,0]
#     if 'scaler_x' in data.keys():
#         scaler_type='_individual_scaling'
#         scaler_x=data['scaler_x']
#         scaler_t=data['scaler_t']
#     data.close()
# Plot_image(Minit1.T,Show_flag=1,Save_flag=1,Title='Minit1',Aspect='equal',
#     c_lim=[1500,4000],Save_pictures_path=Save_pictures_path)
# ############################
# NAME=opt.path_test[3]
# with open(NAME,'rb') as f:
#     data=np.load(f,allow_pickle=True)
#     M0=data['models'][0,:,:,0]; dz=data['dz'];  dx=data['dx']
#     M1=data['input_data'];  M2=data['output_data']
#     Nx=M1.shape[1];     Nz=M1.shape[2];
#     Minit2=data['models_init'][0,:,:,0]
#     if 'scaler_x' in data.keys():
#         scaler_type='_individual_scaling'
#         scaler_x=data['scaler_x']
#         scaler_t=data['scaler_t']
#     data.close()
# Plot_image(Minit2.T,Show_flag=1,Save_flag=1,Title='Minit2',Aspect='equal',
#     c_lim=[1500,4000],Save_pictures_path=Save_pictures_path)
# Plot_image((Minit1-Minit2).T,Show_flag=1,Save_flag=1,Title='diff',Aspect='equal',
#     Save_pictures_path=Save_pictures_path)
# ############################
# NAME=opt.path_train[-2]
# with open(NAME,'rb') as f:
#     data=np.load(f,allow_pickle=True)
#     M0=data['models'][0,:,:,0]; dz=data['dz'];  dx=data['dx']
#     M1=data['input_data'];  M2=data['output_data']
#     Nx=M1.shape[1];     Nz=M1.shape[2];
#     Minit3=data['models_init'][0,:,:,0]
#     if 'scaler_x' in data.keys():
#         scaler_type='_individual_scaling'
#         scaler_x=data['scaler_x']
#         scaler_t=data['scaler_t']
#     data.close()
# Plot_image(Minit3.T,Show_flag=1,Save_flag=1,Title='Minit3',Aspect='equal',
#     c_lim=[1500,4000],Save_pictures_path=Save_pictures_path)
# ############################
# NAME=opt.path_test[-5]
# with open(NAME,'rb') as f:
#     data=np.load(f,allow_pickle=True)
#     M0=data['models'][0,:,:,0]; dz=data['dz'];  dx=data['dx']
#     M1=data['input_data'];  M2=data['output_data']
#     Nx=M1.shape[1];     Nz=M1.shape[2];
#     Minit4=data['models_init'][0,:,:,0]
#     if 'scaler_x' in data.keys():
#         scaler_type='_individual_scaling'
#         scaler_x=data['scaler_x']
#         scaler_t=data['scaler_t']
#     data.close()
# Plot_image(Minit4.T,Show_flag=1,Save_flag=1,Title='Minit4',Aspect='equal',
#     c_lim=[1500,4000],Save_pictures_path=Save_pictures_path)


######################################
# # tmp='/ibex/scratch/plotnips/intel/MWE/Keras-GAN/pix2pix/logs/log403/opt.txt'
# tmp='/ibex/scratch/plotnips/intel/MWE/Keras-GAN/pix2pix/logs/log390/opt.txt'
# parser=ArgumentParser();  opt = parser.parse_args()
# with open(tmp,'r') as f:
#     opt.__dict__=json.load(f)
# history1=opt.history
# tmp='/ibex/scratch/plotnips/intel/MWE/Keras-GAN/pix2pix/logs/log453/opt.txt'
# tmp='/ibex/scratch/plotnips/intel/MWE/Keras-GAN/pix2pix/logs/log452/opt.txt'
# tmp='/ibex/scratch/plotnips/intel/MWE/Keras-GAN/pix2pix/logs/log461/opt.txt'
# parser=ArgumentParser();  opt = parser.parse_args()
# with open(tmp,'r') as f:
#     opt.__dict__=json.load(f)
# history2=opt.history
# history_=history1
# history_['loss_R2_score_train']=np.append(history1['loss_R2_score_train'],history2['loss_R2_score_train'])
# history_['loss_R2_score_val']=np.append(history1['loss_R2_score_val'],history2['loss_R2_score_val'])
# # history_['g_loss_train']=np.append(history1['g_loss_train'],history2['g_loss_train'])
# # history_['g_loss_val']=np.append(history1['g_loss_val'],history2['g_loss_val'])
# # history_['d_loss_train']=np.append(history1['d_loss_train'],history2['d_loss_train'])
# # history_['d_loss_val']=np.append(history1['d_loss_val'],history2['d_loss_val'])
# # history_['loss_real_train']=np.append(history1['loss_real_train'],history2['loss_real_train'])
# # history_['loss_fake_train']=np.append(history1['loss_fake_train'],history2['loss_fake_train'])
# # history_['loss_GAN_train']=np.append(history1['loss_GAN_train'],history2['loss_GAN_train'])
# # history_['loss_pixel_train']=np.append(history1['loss_pixel_train'],history2['loss_pixel_train'])
# Plot_r2_losses(history_,Title='log'+str(log_save_const) + 'losses_r2',Save_pictures_path=Save_pictures_path,Save_flag=1)
# # Plot_g_losses( history_,Title='log'+str(log_save_const) + 'losses_g', Save_pictures_path=Save_pictures_path,Save_flag=1)
# # Plot_d_losses( history_,Title='log'+str(log_save_const) + 'losses_d', Save_pictures_path=Save_pictures_path,Save_flag=1)
# exit()

# for i, batch in enumerate(dataloader):
#     print('phase=',phase,' i=',i)
#     NAME=dataloader.dataset.files[i]
#     tmp= dataloader.dataset.files[i].split('/')[-1];    Name=tmp.split('.npz')[0]
#     real_A = Variable(batch["A"].type(Tensor))
#     real_B = Variable(batch["B"].type(Tensor))
#     fake_B = generator(real_A)
#     with open(NAME,'rb') as f:
#         data=np.load(f,allow_pickle=True)
#         M0=data['models'][0,:,:,0]; dz=data['dz'];  dx=data['dx']
#         M1=data['input_data'];  M2=data['output_data']
#         Nx=M1.shape[1];     Nz=M1.shape[2];
#         Minit=data['models_init'][0,:,:,0]
#         if 'scaler_x' in data.keys():
#             scaler_type='_individual_scaling'
#             scaler_x=data['scaler_x']
#             scaler_t=data['scaler_t']
#         else:
#             from joblib import dump,load
#             scaler_type='_1_scaler'
#             scaler_x=load(data_path+'/scaler_x.bin')
#             scaler_t=load(data_path+'/scaler_t.bin')
#         data.close()
#     M1=imresize(real_A[0,0,:,:].cpu().data.numpy().squeeze(),(Nx,Nz))
#     M1=np.expand_dims(M1,axis=[0,-1])
#     M2=imresize(real_B.cpu().data.numpy().squeeze(),(Nx,Nz))
#     M2=np.expand_dims(M2,axis=[0,-1])
#     M3=imresize(fake_B.cpu().data.numpy().squeeze(),(Nx,Nz))
#     M3=np.expand_dims(M3,axis=0);M3=np.expand_dims(M3,axis=-1);
#     if flag_show_scaled_data==0:
#         M1=transforming_data_inverse(M1,scaler_x)
#         M2=transforming_data_inverse(M2,scaler_t)
#         M3=transforming_data_inverse(M3,scaler_t)
#         Predicted_update=imresize(M3[0,:,:,0],[M1.shape[1],M1.shape[2]])
#         True_update=M2
#     else:
#         tmp=transforming_data_inverse(M3,scaler_t)
#         Predicted_update=tmp[0,:,:,0]
#         True_update=transforming_data_inverse(M2,scaler_t)
#     True_update=True_update[0,:,:,0]
#     M1=M1[0,:,:,0];M2=M2[0,:,:,0];M3=M3[0,:,:,0]

# pred_acc1=F_r2(tpred_,t_)
# print('pred_acc1',pred_acc1,'pred_acc2',pred_acc2,'pred_acc1-pred_acc2',pred_acc1-pred_acc2)
# if abs(pred_acc1-pred_acc2)>0.01:
#     print('Alarm!!!!!!!!!!!Alarm!!!!!!!!!!!',pred_acc1-pred_acc2)
# fig = plt.figure()
# gs = fig.add_gridspec(nrows=1,ncols=3,hspace=0,wspace=0)
# axs = gs.subplots(sharex=True, sharey=True)
# # fig.suptitle('Sharing both axes')
# axs[0].imshow(tpred_.T)
# axs[1].imshow(t_.T)
# axs[2].imshow(t_-tpred_.T)
# name=Save_pictures_path+'/test1.png'
# plt.savefig(name,bbox_inches='tight')
# plt.show(block=False)
# plt.close()
# fig = plt.figure()
# gs = fig.add_gridspec(nrows=3,ncols=1,hspace=0,wspace=0)
# axs = gs.subplots(sharex=True, sharey=True)
# # fig.suptitle('Sharing both axes')
# axs[0].imshow(tpred.T)
# axs[1].imshow(t.T)              
# axs[2].imshow((t-tpred).T)
# name=Save_pictures_path+'/test2.png'
# plt.savefig(name,bbox_inches='tight')
# plt.show(block=False)
# plt.close()

# R2val=F_r2(testing_model,M0_show)
####################
# Prediction_accuracy=F_r2(M3,M2)
# a=1
# with open(NAME,'rb') as f:
#     data=np.load(f,allow_pickle=True)
#     M0=data['models'][0,:,:,0]; dz=data['dz'];  dx=data['dx']
#     M1=data['input_data'];  M2=data['output_data']
#     Nx=M1.shape[1];     Nz=M1.shape[2];
#     Minit=data['models_init'][0,:,:,0]
#     if 'scaler_x' in data.keys():
#         scaler_type='_individual_scaling'
#         scaler_x=data['scaler_x']
#         scaler_t=data['scaler_t']
#     else:
#         from joblib import dump,load
#         scaler_type='_1_scaler'
#         scaler_x=load(data_path+'/scaler_x.bin')
#         scaler_t=load(data_path+'/scaler_t.bin')
#     data.close()
# M1=imresize(real_A[0,0,:,:].cpu().data.numpy().squeeze(),(Nx,Nz))
# M1=np.expand_dims(M1,axis=[0,-1])
# M2=imresize(real_B.cpu().data.numpy().squeeze(),(Nx,Nz))
# M2=np.expand_dims(M2,axis=[0,-1])
# M3=imresize(fake_B.cpu().data.numpy().squeeze(),(Nx,Nz))
# M3=np.expand_dims(M3,axis=0);M3=np.expand_dims(M3,axis=-1);
# if flag_show_scaled_data==0:
#     M1=transforming_data_inverse(M1,scaler_x)
#     M2=transforming_data_inverse(M2,scaler_t)
#     M3=transforming_data_inverse(M3,scaler_t)
#     Predicted_update=imresize(M3[0,:,:,0],[M1.shape[1],M1.shape[2]])
#     True_update=M2
# else:
#     tmp=transforming_data_inverse(M3,scaler_t)
#     Predicted_update=tmp[0,:,:,0]
#     True_update=transforming_data_inverse(M2,scaler_t)
# True_update=True_update[0,:,:,0]
# M1=M1[0,:,:,0];M2=M2[0,:,:,0];M3=M3[0,:,:,0]
# ####################
# # Models_init=Minit
# # if Models_init.shape!=Predicted_update.shape:
# #     Predicted_update=imresize(Predicted_update,Models_init.shape)
# # if Models_init.shape!=True_update.shape:
# #     True_update=imresize(True_update,Models_init.shape)
# # testing_model=Models_init+Predicted_update
# # ideal_init_model=Models_init+True_update
# # M0_show=M0
# # ################### Crop testing models for better visualization
# # water=np.ones((M0_show.shape[0],18))*1500
# # M0_show=np.concatenate([water,M0_show],axis=1)
# # testing_model=np.concatenate([water,testing_model],axis=1)    
# # ideal_init_model=np.concatenate([water,ideal_init_model],axis=1)
# # init_model=np.concatenate([water,Models_init],axis=1)
# # pics_6=[M1,M2,M3,init_model,testing_model,ideal_init_model]
# # R2val=F_r2(testing_model,M0_show)
# ####################
# Prediction_accuracy=F_r2(M3,M2)
# prediction_string='Prediction, R2(prediction, target) = ' + numstr(Prediction_accuracy)
# # picture_name=opt.save_path+'/'+'log'+str(opt.log_save_const)+Name+'_epoch_'+str(epoch)+'_'+numstr(Prediction_accuracy)+'.png'
# ####################

######################################
# tmp='/ibex/scratch/plotnips/intel/MWE/Keras-GAN/pix2pix/logs/log403/opt.txt'
# tmp='/ibex/scratch/plotnips/intel/MWE/Keras-GAN/pix2pix/logs/log390/opt.txt'
# parser=ArgumentParser();  opt = parser.parse_args()
# with open(tmp,'r') as f:
#     opt.__dict__=json.load(f)
# history1=opt.history
# tmp='/ibex/scratch/plotnips/intel/MWE/Keras-GAN/pix2pix/logs/log453/opt.txt'
# tmp='/ibex/scratch/plotnips/intel/MWE/Keras-GAN/pix2pix/logs/log452/opt.txt'
# parser=ArgumentParser();  opt = parser.parse_args()
# with open(tmp,'r') as f:
#     opt.__dict__=json.load(f)
# history2=opt.history
# history_=history1
# history_['loss_R2_score_train']=np.append(history1['loss_R2_score_train'],history2['loss_R2_score_train'])
# history_['loss_R2_score_val']=np.append(history1['loss_R2_score_val'],history2['loss_R2_score_val'])
# history_['g_loss_train']=np.append(history1['g_loss_train'],history2['g_loss_train'])
# history_['g_loss_val']=np.append(history1['g_loss_val'],history2['g_loss_val'])
# history_['d_loss_train']=np.append(history1['d_loss_train'],history2['d_loss_train'])
# history_['d_loss_val']=np.append(history1['d_loss_val'],history2['d_loss_val'])
# history_['loss_real_train']=np.append(history1['loss_real_train'],history2['loss_real_train'])
# history_['loss_fake_train']=np.append(history1['loss_fake_train'],history2['loss_fake_train'])
# history_['loss_GAN_train']=np.append(history1['loss_GAN_train'],history2['loss_GAN_train'])
# history_['loss_pixel_train']=np.append(history1['loss_pixel_train'],history2['loss_pixel_train'])
# Plot_r2_losses(history_,Title='log'+str(log_save_const) + 'losses_r2',Save_pictures_path=Save_pictures_path,Save_flag=1)
# Plot_g_losses( history_,Title='log'+str(log_save_const) + 'losses_g', Save_pictures_path=Save_pictures_path,Save_flag=1)
# Plot_d_losses( history_,Title='log'+str(log_save_const) + 'losses_d', Save_pictures_path=Save_pictures_path,Save_flag=1)
# exit()

# zz = np.arange(60) * 50
# models_init_=1500+2.3*zz
# models_init_=models_init_-1500
# models_init_2=1500+2.1*zz
# models_init_2=models_init_2-1500

# name='./logs/log599/r2_scores_inference.npz'
# with open(name,'rb') as f:
#     data=np.load(f,allow_pickle=True)
#     predictions_train=data['predictions_train']
#     predictions_valid=data['predictions_valid']
#     predictions_test=data['predictions_test']
#     data.close()
# name_to_plot=opt.path_train[np.argmin(predictions_train)]
# path=fnmatch.filter(sorted(glob(opt.dataset_path+'/*')),'*.npz')        #old variant

# train_dataloader_for_histograms=DataLoader(ImageDataset(opt,transforms_=transforms_,
#     mode=None,file_list=opt.path_train[0:12],max_for_initial_models=max_for_initial_models,
#     load_precondition_data=1),batch_size=1,shuffle=False,num_workers=1)

# debug_dataloader=DataLoader(train_dataset,batch_size=8,shuffle=False,num_workers=opt.n_cpu,pin_memory=True)
# print(debug_dataloader.dataset.files)
# for i, batch in enumerate(debug_dataloader):
#     NAME = debug_dataloader.dataset.files[i];print(NAME)
#     # Model inputs
#     real_A = Variable(batch["A"].type(Tensor))
#     real_B = Variable(batch["B"].type(Tensor))
#     init_model=Variable(batch["C"].type(Tensor))
#     true_model=Variable(batch["D"].type(Tensor))
#     real_A_ra = Variable(batch["E"].type(Tensor))   
#     real_B_ra = Variable(batch["F"].type(Tensor))   
#     print('real_A=',real_A.shape)
#     print('real_B=',real_B.shape)
#     print('init_model=',init_model.shape)
#     print('true_model=',true_model.shape)
#     print('real_A_ra=',real_A_ra.shape)
#     print('real_B_ra=',real_B.shape)
#     a=1
# exit()

###########################     calculate histograms
# calculate_prediction_histograms=1
# if calculate_prediction_histograms==1:
#     phases=['test','validation','training']
#     phases=['test']
#     phases=['training','validation','test'];
#     calculate_predictions=1;    save_predictions=1
#     plot_samples=1;     
#     val=multiprocessing.cpu_count()-3;      
#     val=1
#     batch_size_val=val  
#     num_workers_val=val
#     train_dataloader_for_histograms=DataLoader(ImageDataset(opt,transforms_=transforms_,mode="train",max_for_initial_models=max_for_initial_models),
#         batch_size=batch_size_val,shuffle=False,num_workers=num_workers_val)
#     val_dataloader_for_histograms=DataLoader(ImageDataset(opt,transforms_=transforms_,mode="val",max_for_initial_models=max_for_initial_models),
#         batch_size=batch_size_val,shuffle=False,num_workers=num_workers_val)
#     test_dataloader_for_histograms=DataLoader(ImageDataset(opt,transforms_=transforms_,mode="test",max_for_initial_models=max_for_initial_models),
#         batch_size=batch_size_val,shuffle=False,num_workers=num_workers_val)
#     print('calculate misfits')
#     ch1=[];ch2=[];ch3=[];ch4=[];
#     for phase in phases:
#         if phase=='training':   
#             dataloader=train_dataloader_for_histograms
#             predictions_train=np.empty((len(dataloader.dataset),4))
#         elif phase=='validation':   
#             dataloader=val_dataloader_for_histograms
#             predictions_valid=np.empty((len(dataloader.dataset),4))
#         elif phase=='test':   
#             dataloader=test_dataloader_for_histograms
#             predictions_test=np.empty(( len(dataloader.dataset),4))
#         cuda = True if torch.cuda.is_available() else False
#         Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
#         for i,batch in enumerate(dataloader):
#             real_A = Variable(batch["A"].type(Tensor))
#             fake_B = generator(real_A)
#             real_B = Variable(batch["B"].type(Tensor))
#             real_C=Variable(batch["C"].type(Tensor))
#             real_D=Variable(batch["D"].type(Tensor))
#             batch_size=real_A.shape[0]
#             for k in range(batch_size):
#                 print('phase=',phase,'batch number i=',i,'/',batch_size,',file_number=',i*batch_size+k)
#                 NAME=dataloader.dataset.files[i*batch_size+k];  tmp=NAME.split('/')[-1];    Name=tmp.split('.npz')[0]
#                 if calculate_predictions==1:
#                     x=real_A[k,0,:,:].cpu().data.numpy()
#                     tpred=fake_B[k,0,:,:].cpu().data.numpy()
#                     t=real_B[k,0,:,:].cpu().data.numpy()
#                     init_model=real_C[k,0,:,:].cpu().data.numpy()
#                     true_model=real_D[k,0,:,:].cpu().data.numpy()
#                     target_pred_acc=F_r2(tpred,t)   #target_prediction_acc
#                     init_model_pred_acc=F_r2(init_model+tpred,init_model+t)     #init_model_prediction_acc
#                     true_model_pred_acc=F_r2(init_model+tpred,true_model)       #true_model_prediction_acc
#                     true_model_inversion_acc=F_r2(init_model+x,true_model)  #true_model_inversion_acc
#                     if phase=='training':   
#                         predictions_train[(i-1)*batch_size+k]=[target_pred_acc,init_model_pred_acc,true_model_pred_acc,true_model_inversion_acc]
#                     elif phase=='validation':   
#                         predictions_valid[(i-1)*batch_size+k]=[target_pred_acc,init_model_pred_acc,true_model_pred_acc,true_model_inversion_acc]
#                     elif phase=='test':   
#                         predictions_test[(i-1)*batch_size+k]=[target_pred_acc,init_model_pred_acc,true_model_pred_acc,true_model_inversion_acc]
#                 #####################################
#                 ch1.append(real_A[k,0,:,:].cpu().data.numpy())
#                 ch4.append(real_B[k,0,:,:].cpu().data.numpy())
#                 if opt.channels==3:
#                     ch2.append(real_A[k,1,:,:].cpu().data.numpy())
#                     ch3.append(real_A[k,2,:,:].cpu().data.numpy())
#                 #####################################
#                 if plot_samples==1:
#                     subtitles=['x','initial model','depth matrix','target']
#                     if opt.channels==3:
#                         fig = plt.figure(); fig.suptitle(Name)
#                         gs = fig.add_gridspec(nrows=2,ncols=2,hspace=0.2,wspace=0)
#                         axs = gs.subplots(sharex=True, sharey=True)
#                         ax=axs[0,0];    pt=ax.imshow(real_A[k,0,:,:].cpu().data.numpy().T);     
#                         fig.colorbar(pt,ax=ax);     ax.title.set_text(subtitles[0])
#                         ax=axs[0,1];    pt=ax.imshow(real_A[k,1,:,:].cpu().data.numpy().T);  
#                         fig.colorbar(pt,ax=ax);     ax.title.set_text(subtitles[1])
#                         ax=axs[1,0];    pt=ax.imshow(real_A[k,2,:,:].cpu().data.numpy().T);  
#                         fig.colorbar(pt,ax=ax);     ax.title.set_text(subtitles[2])
#                         ax=axs[1,1];    pt=ax.imshow(real_B[k,0,:,:].cpu().data.numpy().T);
#                         fig.colorbar(pt,ax=ax);     ax.title.set_text(subtitles[3])
#                         name=Save_pictures_path+'/sample__'+Name+'.png'
#                         plt.savefig(name,bbox_inches='tight')
#                         plt.show()  # plt.show(block=False)
#                         plt.close()
#                     elif opt.channels==1:
#                         fig = plt.figure(); fig.set_size_inches(10, 4); fig.suptitle(Name)
#                         gs = fig.add_gridspec(nrows=1,ncols=2,wspace=0.2,hspace=0)
#                         axs = gs.subplots(sharex=True, sharey=True)
#                         ax=axs[0];    pt=ax.imshow(real_A[k,0,:,:].cpu().data.numpy().T);     
#                         divider = make_axes_locatable(ax);  cax = divider.append_axes("right", size="5%", pad=0.05)
#                         fig.colorbar(pt,cax=cax);     ax.title.set_text(subtitles[0])
#                         ax=axs[1];    pt=ax.imshow(real_B[k,0,:,:].cpu().data.numpy().T);
#                         divider = make_axes_locatable(ax);  cax = divider.append_axes("right", size="5%", pad=0.05)
#                         fig.colorbar(pt,cax=cax);     ax.title.set_text(subtitles[3])
#                         name=Save_pictures_path+'/sample__'+Name+'.png'
#                         plt.savefig(name,bbox_inches='tight');  # plt.savefig(name);
#                         plt.show()  # plt.show(block=False)
#                         plt.close()
#                     a=1
# ch1=np.asarray(ch1);    print('ch1,input_data,min=',np.min(ch1),',max=',np.max(ch1))
# ch4=np.asarray(ch4);    print('ch4,target,min=',np.min(ch4),',max=',np.max(ch4))
# if opt.channels==3:
#     ch2=np.asarray(ch2);    print('ch2,initial_model=',np.min(ch2),',max=',np.max(ch2))
#     ch3=np.asarray(ch3);    print('ch3,depth_matrix,min=',np.min(ch3),',max=',np.max(ch3))