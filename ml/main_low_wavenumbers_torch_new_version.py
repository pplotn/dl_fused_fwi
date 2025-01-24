print('Torch CODE Torch CODE Torch CODE Torch CODE!!!!!')
from imports_torch import *
from utils_low_wavenumbers_torch import *
####    create folder and start logging to txt file 
logs_path='/ibex/scratch/plotnips/intel/MWE/Keras-GAN/pix2pix/logs'
os.makedirs(logs_path,exist_ok=True)
log_save_const = F_calculate_log_number(logs_path,'log','')
Save_pictures_path=logs_path+'/log'+str(log_save_const)
os.makedirs(Save_pictures_path,exist_ok=True)
logname = '/log' + str(log_save_const)+'.txt'
f = open(Save_pictures_path+logname,'w')
sys.stdout = Tee(sys.stdout,f)
if not os.path.exists(logs_path+'/log'+str(log_save_const)+'/opt.txt'):
    parser = ArgumentParser();  opt = parser.parse_args()
######################################
# if start_from_pretrained_model==0:
#     opt.path_train=path_train;opt.path_valid=path_valid;opt.path_test=path_test
# opt.channels_discriminator=1
# opt.architecture=1
n_epochs=10
start_from_pretrained_model=0
if n_epochs==-1 or start_from_pretrained_model==1:
    log_to_load_name=175;   epoch_name=9
    log_to_load_name=176;   epoch_name=9
    log_to_load_name=174;   epoch_name=3
    log_to_load_name=200;   epoch_name=4499
    log_to_load_name=222;   epoch_name=99
    log_to_load_name=234;   epoch_name=300
    # log_to_load_name=235;   epoch_name=300
    log_to_load_name=284;   epoch_name=0
    log_to_load_name=256;   epoch_name=0
    log_to_load_name=277;   epoch_name=0
    log_to_load_name=254;   epoch_name=2000
    log_to_load_name=254;   epoch_name=4000
    # log_to_load_name=321;   epoch_name=570
    log_to_load_name=334;   epoch_name=999
    log_to_load_name=338;   epoch_name=450
    log_to_load_name=367;   epoch_name=299
    log_to_load_name=377;   epoch_name=60
    log_to_load_name=387;   epoch_name=20
    log_to_load_name=388;   epoch_name=96
    log_to_load_name=408;   epoch_name=1
    log_to_load_name=413;   epoch_name=3
    log_to_load_name=395;   epoch_name=900
    log_to_load_name=390;   epoch_name=330
    # log_to_load_name=403;   epoch_name=536
    # log_to_load_name=400;   epoch_name=1160
    # log_to_load_name=419;   epoch_name=840
    # log_to_load_name=419;   epoch_name=390
    # log_to_load_name=453;   epoch_name=444
    # log_to_load_name=453;   epoch_name=468
    # log_to_load_name=452;   epoch_name=480
    log_to_load_name=454;   epoch_name=420
    log_to_load_name=465;   epoch_name=2370
    log_to_load_name=467;   epoch_name=2280
    tmp=logs_path+'/log'+str(log_to_load_name)+'/opt.txt'
    if not os.path.exists(tmp):
        parser = ArgumentParser();  opt = parser.parse_args()
    with open(tmp,'r') as f:
        opt.__dict__=json.load(f)
    opt.n_epochs=n_epochs
    opt.generator_model_name=       './logs/log'+str(log_to_load_name)+'/generator_'+str(epoch_name)+'.pth'
    opt.discriminator_model_name=   './logs/log'+str(log_to_load_name)+'/discriminator_'+str(epoch_name)+'.pth'
if start_with_dataset_from_opt_file==1:
    log_to_load_name=527
    tmp=logs_path+'/log'+str(log_to_load_name)+'/opt.txt'
    if os.path.exists(tmp):
        parser = ArgumentParser();  opt_old = parser.parse_args()
    with open(tmp,'r') as f:
        opt_old.__dict__=json.load(f)
    len(opt.path_train)
    len(opt_old.path_train)

opt.save_path=Save_pictures_path
opt.log_save_const=log_save_const
######################################   dataset paths
dataset_path='/ibex/scratch/plotnips/intel/MWE/datasets/dataset_vl_gen_scaled_60000_smooth_450m_scaler_picture_01'
dataset_path='/home/plotnips/Dropbox/Log_extrapolation/scripts/MWE/datasets/dataset_vl_gen_scaled_3005_smooth_450m_scaler_picture_individual_scaling_false'
dataset_path='/ibex/scratch/plotnips/intel/MWE/datasets/dataset_vl_gen_scaled_287145_smooth_450m_6860_scaler_picture_individual_scaling'
dataset_path=['/ibex/scratch/plotnips/intel/MWE/datasets/dataset_dense_source_scaled_2454_smooth_450m_scaler_picture_individual_scaling_true']
# dataset_path='/ibex/scratch/plotnips/intel/MWE/datasets/dataset_vl_gen_scaled_287145_smooth_450m_scaler_picture_individual_scaling_true_x_precon_t_no_precond'
dataset_path=['/ibex/scratch/plotnips/intel/MWE/datasets/dataset_oleg_scaled_44562_smooth_450m_scaler_picture_individual_scaling_true_x_precon_t_no_precond']
# dataset_path=[
#     '/ibex/scratch/plotnips/intel/MWE/datasets/dataset_oleg_scaled_44562_smooth_450m_scaler_picture_individual_scaling_true_x_precon_t_no_precond',
#     '/ibex/scratch/plotnips/intel/MWE/datasets/dataset_vl_gen_scaled_287145_smooth_450m_scaler_picture_individual_scaling_true_x_precon_t_no_precond']
dataset_path=['/ibex/scratch/plotnips/intel/MWE/datasets/dataset_vl_gen_scaled_287145_smooth_450m_scaler_picture_individual_scaling_true_x_precon_t_no_precond']
dataset_path=['/ibex/scratch/plotnips/intel/MWE/datasets/dataset_vl_gen_scaled_287145_smooth_450m_scaler_picture_individual_scaling_false']
dataset_path=['/ibex/scratch/plotnips/intel/MWE/datasets/dataset_vl_gen_individual_init_models_36394']
dataset_path=['/home/plotnips/Dropbox/Log_extrapolation/scripts/MWE/datasets/dataset_oleg_individual_init_models_1000']
dataset_path=['/home/plotnips/Dropbox/Log_extrapolation/scripts/MWE/datasets/dataset_oleg_individual_init_models_1000']
# dataset_path=['/ibex/scratch/plotnips/intel/MWE/datasets/dataset_vl_gen']
# dataset_path=['/ibex/scratch/plotnips/intel/MWE/datasets/dataset_oleg_scaled_44562_smooth_450m_scaler_picture_individual_scaling_false',
#               '/ibex/scratch/plotnips/intel/MWE/datasets/dataset_vl_gen_scaled_287145_smooth_450m_scaler_picture_individual_scaling_false']
dataset_path=['/ibex/scratch/plotnips/intel/MWE/datasets/dataset_oleg_individual_init_models_1000',
              '/ibex/scratch/plotnips/intel/MWE/datasets/dataset_vl_gen_individual_init_models_36394']
# dataset_path=['/ibex/scratch/plotnips/intel/MWE/datasets/dataset_oleg_individual_init_models_1000']
dataset_path=['/ibex/scratch/plotnips/intel/MWE/datasets/dataset_vl_gen_individual_init_models_36394']
# dataset_path=['/home/plotnips/Dropbox/Log_extrapolation/scripts/MWE/datasets/dataset_vl_gen_individual_init_models_36394']
####    torch code  ,multiprocessing.cpu_count()-3
parser = argparse.ArgumentParser()
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--start_from_pretrained_model", type=int, default=0, help="start_from_pretrained_model")
parser.add_argument("--start_with_certain_dataset", type=int, default=1, help="start_with_certain_dataset")
parser.add_argument("--img_height", type=int, default=256, help="size of image height")
parser.add_argument("--img_width", type=int, default=256, help="size of image width")
parser.add_argument("--architecture", type=int, default=1,help="gan architecture=1,cnn architecture=0")
parser.add_argument("--channels", type=int, default=3, help="number of input channels in generator")
parser.add_argument("--channels_discriminator", type=int, default=3, help="number of input channels in discriminator")
####    long training
# parser.add_argument("--dataset_path",type=int,default=dataset_path,help="path to the dataset")
# parser.add_argument("--n_epochs",type=int,default=2800,help="number of epochs of training")
# parser.add_argument("--batch_size", type=int, default=32,help="size of the batches")
# parser.add_argument("--n_cpu",type=int,default=8,help="number of cpu threads to use during batch generation")
# parser.add_argument("--checkpoint_interval", type=int, default=30, help="interval between model checkpoints")
# parser.add_argument("--plotting_interval", type=int, default=30, help="plot results at each epoch")
# parser.add_argument("--N", type=int,default=-1, help="total number of samples in training and validation losses")
####    short training
# # parser.add_argument("--channels_discriminator", type=int, default=1, help="number of input channels in discriminator")
# parser.add_argument("--dataset_path",type=int,default=dataset_path,help="path to the dataset")
# parser.add_argument("--n_epochs",type=int,default=2,help="number of epochs of training")
# parser.add_argument("--batch_size", type=int, default=16,help="size of the batches") #16
# parser.add_argument("--n_cpu",type=int,default=2,help="number of cpu threads to use during batch generation")
# parser.add_argument("--checkpoint_interval", type=int, default=20, help="interval between model checkpoints")
# parser.add_argument("--plotting_interval", type=int, default=20, help="plot results at each epoch")
# parser.add_argument("--N",type=int,default=30,help="total number of samples in training and validation losses")
####    super long training
# dataset_path=['/ibex/scratch/plotnips/intel/MWE/datasets/dataset_vl_gen_scaled_287145_smooth_450m_scaler_picture_individual_scaling_false']
parser.add_argument("--dataset_path",type=int,default=dataset_path,help="path to the dataset")
parser.add_argument("--n_epochs",type=int,default=100,help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=2,help="size of the batches")
parser.add_argument("--n_cpu",type=int,default=1,help="number of cpu threads to use during batch generation")
parser.add_argument("--checkpoint_interval", type=int, default=50, help="interval between model checkpoints")
parser.add_argument("--plotting_interval", type=int, default=50, help="plot results at each epoch")
parser.add_argument("--N", type=int, default=30, help="total number of samples in training and validation losses")
######################################
opt = parser.parse_args()
opt.save_path=Save_pictures_path;   opt.log_save_const=log_save_const;  start_from_pretrained_model=opt.start_from_pretrained_model
######################################
path=[];    counter=0
for dataset_path in opt.dataset_path:
    files_path=fnmatch.filter(glob(dataset_path+'/*'),'*.npz'); 
    path_=sorted(files_path)
    # print(path_[-12:])
    if len(opt.dataset_path)>1:
        # val1=11208;     val2=15000
        val1=2000;     val2=3000
        # val1=11208;     val2=2000
        val1=300;     val2=300
        if counter==0:  #   remove test files out of the first dataset
            path_=path_[-val1:]
            files_to_delete=[]
            for ii in path_:
                if '__' in ii:
                    files_to_delete.append(ii)
            for ii in files_to_delete:
                print('removing ',ii)
                path_.remove(ii)
        elif counter==1:
            path_=path_[-val2:]
    path=path+path_
    counter=counter+1
path=sorted(path)
path_test=path[-10:]
path=list(set(path)-set(path_test));    path=sorted(path)
#   define size of dataset  #   split dataset
if opt.N==-1:
    opt.N=len(path)
if len(opt.dataset_path)==2:
    opt.shuffle_files=0
    # opt.shuffle_files=1
    # random.shuffle(path)
else:
    opt.shuffle_files=0
path=path[0:opt.N]
opt.train_frac= 0.8
path_train=path[0:int(len(path)*opt.train_frac)]
path_valid=list(set(path)-set(path_train))
# path_train=path[0:3];  path_valid=path[5:6]
opt.path_test=path_test
print(opt)
opt.path_valid=path_valid;  opt.path_train=path_train;
print('len(opt.path_train)=',len(opt.path_train))
print('len(opt.path_test)=',len(opt.path_test))
print('len(opt.path_valid)=',len(opt.path_valid))
with open(opt.save_path+'/'+'opt.txt', 'w') as f:
    json.dump(opt.__dict__, f, indent=2)
############################
cuda = True if torch.cuda.is_available() else False
# Loss functions
criterion_GAN = torch.nn.MSELoss()
criterion_pixelwise = torch.nn.L1Loss()
criterion_R2Loss=R2Loss()
# Loss weight of L1 pixel-wise loss between translated image and real image
opt.lambda_pixel = 100
# Calculate output of image discriminator (PatchGAN)
patch = (1, opt.img_height // 2 ** 4, opt.img_width // 2 ** 4); opt.patch=patch
############################# Initialize generator and discriminator
torch.manual_seed(123)
np.random.seed(123)
random.seed(123)
generator = GeneratorUNet(in_channels=opt.channels)
discriminator = Discriminator(in_channels=opt.channels_discriminator)
summary(generator.to(0), (opt.channels,256,256))
summary(discriminator.to(0),[(1,256,256),(opt.channels_discriminator,256,256)] )
if cuda:
    generator = generator.cuda()
    discriminator = discriminator.cuda()
    criterion_GAN.cuda()
    criterion_pixelwise.cuda()
    criterion_R2Loss.cuda()
if opt.n_epochs==-1 or start_from_pretrained_model==1:
    print('Load pretrained models',opt.generator_model_name,opt.discriminator_model_name)
    generator.load_state_dict(torch.load(opt.generator_model_name))
    discriminator.load_state_dict(torch.load(opt.discriminator_model_name))
    history=opt.history
else:       # Initialize weights
    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)
# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
# Configure dataloaders
transforms_=[]
print('opt.batch_size=',opt.batch_size)
# exit()
max_for_initial_models=ImageDataset(opt,transforms_=transforms_,mode="all").calculate_max_model_init()
train_dataloader = DataLoader(ImageDataset(opt,transforms_=transforms_,mode="train",max_for_initial_models=max_for_initial_models),
    batch_size=opt.batch_size,shuffle=False,num_workers=opt.n_cpu)
val_dataloader=DataLoader(ImageDataset(opt,transforms_=transforms_,mode="val",max_for_initial_models=max_for_initial_models),
    batch_size=opt.batch_size,shuffle=False,num_workers=opt.n_cpu)
test_dataloader=DataLoader( ImageDataset(opt,transforms_=transforms_,mode="test",max_for_initial_models=max_for_initial_models),
    batch_size=1,shuffle=False,num_workers=1)
tracking_dataloader=DataLoader(ImageDataset(opt,transforms_=transforms_,
    mode=None,file_list=path_train[0:1]+path_test[-2:-1]+path_test[-4:-3],max_for_initial_models=max_for_initial_models),
    batch_size=1,shuffle=False,num_workers=1)
tracking_dataloader2=DataLoader(ImageDataset(opt,transforms_=transforms_,
    mode=None,file_list=path_train[0:2],max_for_initial_models=max_for_initial_models),
    batch_size=1,shuffle=False,num_workers=1)
# Tensor type
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
#########  Training
g_loss_train=[];    g_loss_val=[];  
loss_GAN_train=[];  loss_GAN_val=[]
loss_pixel_train=[];    loss_pixel_val=[];  
d_loss_train=[];    d_loss_val=[];  
loss_real_train=[];    loss_real_val=[];
loss_fake_train=[];    loss_fake_val=[];
loss_R2_score_train=[];loss_R2_score_val=[]

prev_time = time.time()
T1 = datetime.datetime.now()
phases=['training','validation']
# phases=['training']
accumulated_loss_r2=0
if opt.n_epochs!=-1:
    for epoch in range(0,opt.n_epochs):
        for phase in phases:
            if phase=='training':   
                dataloader=train_dataloader;
                generator.train();discriminator.train()
            elif phase=='validation':   
                dataloader=val_dataloader;
                generator.eval();discriminator.eval()
            for i, batch in enumerate(dataloader):
                # Model inputs
                real_A = Variable(batch["A"].type(Tensor))
                real_B = Variable(batch["B"].type(Tensor))
                # Adversarial ground truths
                valid = Variable(Tensor(np.ones((real_A.size(0), *patch))), requires_grad=False)
                fake = Variable(Tensor(np.zeros((real_A.size(0), *patch))), requires_grad=False)
                #############  Train Generators
                if phase=='training':   optimizer_G.zero_grad()
                # GAN loss
                fake_B = generator(real_A)
                loss_r2=criterion_R2Loss(fake_B,real_B);
                pred_fake = discriminator(fake_B,real_A)
                loss_GAN = criterion_GAN(pred_fake,valid)
                # Pixel-wise loss
                loss_pixel = criterion_pixelwise(fake_B,real_B)
                # Total loss
                loss_G = loss_GAN + opt.lambda_pixel*loss_pixel
                if phase=='training':   loss_G.backward();  optimizer_G.step()
                ##############  Train Discriminator
                if opt.architecture==1:
                    if phase=='training':   optimizer_D.zero_grad()
                    # Real loss
                    pred_real = discriminator(real_B, real_A)
                    loss_real = criterion_GAN(pred_real, valid)
                    # Fake loss
                    pred_fake = discriminator(fake_B.detach(), real_A)
                    loss_fake = criterion_GAN(pred_fake, fake)
                    # Total loss
                    loss_D = 0.5 * (loss_real+loss_fake)
                    if phase=='training':   loss_D.backward();  optimizer_D.step()
                ##############  calculate tracking losses
                accumulated_loss_r2+=loss_r2.item()
                ##############  Log Progress
                # Determine approximate time left
                batches_done = epoch * len(dataloader) + i
                batches_left = opt.n_epochs * len(dataloader) - batches_done
                time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
                prev_time = time.time()
                # Print log
                if opt.architecture==1:
                    sys.stdout.write("\rLog:%s PHASE: %s [Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, pixel: %f, adv: %f] ETA: %s"
                            % ( str(log_save_const),phase,epoch,opt.n_epochs,i,len(dataloader),
                            loss_D.item(),loss_G.item(),opt.lambda_pixel*loss_pixel.item(),loss_GAN.item(),time_left,))
                else:
                    sys.stdout.write("\rLog:%s PHASE: %s [Epoch %d/%d] [Batch %d/%d] [G loss: %f] ETA: %s"
                        % ( str(log_save_const),phase,epoch,opt.n_epochs,i,len(dataloader),loss_G.item(),time_left,))
                # If at sample interval save image
                logging_loss_batch_size=len(dataloader)
                if i % logging_loss_batch_size == logging_loss_batch_size-1:
                    print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, accumulated_loss_r2 / logging_loss_batch_size))
                    R2_score=accumulated_loss_r2/logging_loss_batch_size
                    accumulated_loss_r2 = 0.0
            # Save model checkpoints
            if phase=='training':
                g_loss_train.append(loss_G.item())
                loss_pixel_train.append(opt.lambda_pixel*loss_pixel.item())
                loss_R2_score_train.append(R2_score)
                if opt.architecture==1:
                    loss_GAN_train.append(loss_GAN.item())
                    d_loss_train.append(loss_D.item())
                    loss_real_train.append(loss_real.item())
                    loss_fake_train.append(loss_fake.item())
            elif phase=='validation':
                if opt.architecture==1:
                    g_loss_val.append(loss_G.item())
                    loss_GAN_val.append(loss_GAN.item())
                    loss_pixel_val.append(opt.lambda_pixel*loss_pixel.item())
                    d_loss_val.append(loss_D.item())
                    loss_real_val.append(loss_real.item())
                    loss_fake_val.append(loss_fake.item())
                    loss_R2_score_val.append(R2_score)
                else:
                    g_loss_val.append(loss_G.item())
                    loss_pixel_val.append(opt.lambda_pixel*loss_pixel.item())
                    loss_R2_score_val.append(R2_score)
                
                if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
                    torch.save(generator.state_dict(), "%s/generator_%d.pth" % (Save_pictures_path, epoch))
                    torch.save(discriminator.state_dict(), "%s/discriminator_%d.pth" % (Save_pictures_path,epoch))
                    if opt.architecture==1:
                        history={
                                'g_loss_train':g_loss_train,'g_loss_val':g_loss_val,
                                'loss_GAN_train':loss_GAN_train,'loss_GAN_val':loss_GAN_val,
                                'loss_pixel_train':loss_pixel_train,'loss_pixel_val':loss_pixel_val,
                                'd_loss_train':d_loss_train,'d_loss_val':d_loss_val,
                                'loss_real_train':loss_real_train,    'loss_real_val':loss_real_val,
                                'loss_fake_train':loss_fake_train,    'loss_fake_val':loss_fake_val,
                                'loss_R2_score_train':loss_R2_score_train,'loss_R2_score_val':loss_R2_score_val};
                    else:
                        history={
                                'g_loss_train':g_loss_train,'g_loss_val':g_loss_val,
                                'loss_pixel_train':loss_pixel_train,'loss_pixel_val':loss_pixel_val,
                                'loss_R2_score_train':loss_R2_score_train,'loss_R2_score_val':loss_R2_score_val};
                    opt.history=history
                    with open(opt.save_path+'/'+'opt.txt', 'w') as f:
                        json.dump(opt.__dict__, f, indent=2)
                ###################################################
                if epoch==opt.n_epochs-1:
                    torch.save(generator.state_dict(), "%s/generator_%d.pth" % (Save_pictures_path,epoch))
                    torch.save(discriminator.state_dict(), "%s/discriminator_%d.pth" % (Save_pictures_path,epoch))
                if epoch % opt.plotting_interval==0:
                    # sample_dataset(tracking_dataloader,generator,opt,epoch) #tracking_dataloader.dataset.files
                    # sample_dataset(test_dataloader,generator,opt,epoch,flag_show_scaled_data=0)
                    if opt.architecture==1:
                        Plot_g_losses(history,Title='log'+str(log_save_const) + 'losses_g', Save_pictures_path=Save_pictures_path,Save_flag=1)
                        Plot_r2_losses(history,Title='log'+str(log_save_const) + 'losses_r2', Save_pictures_path=Save_pictures_path,Save_flag=1)
                        Plot_d_losses(history,Title='log'+str(log_save_const) + 'losses_d', Save_pictures_path=Save_pictures_path,Save_flag=1)
                    else:
                        Plot_r2_losses_2(history,Title='log'+str(log_save_const) + 'losses_r2', Save_pictures_path=Save_pictures_path,Save_flag=1)   
                        Plot_unet_losses(history,Title='log'+str(log_save_const) + 'losses_unet', Save_pictures_path=Save_pictures_path,Save_flag=1)
                        Plot_unet_losses_2(history,Title='log'+str(log_save_const) + 'losses_pixel', Save_pictures_path=Save_pictures_path,Save_flag=1)
    ###############            
    print('Training time(sec)=',datetime.datetime.now()-T1)
    if opt.architecture==1:
        history={
                'g_loss_train':g_loss_train,'g_loss_val':g_loss_val,
                'loss_GAN_train':loss_GAN_train,'loss_GAN_val':loss_GAN_val,
                'loss_pixel_train':loss_pixel_train,'loss_pixel_val':loss_pixel_val,
                'd_loss_train':d_loss_train,'d_loss_val':d_loss_val,
                'loss_real_train':loss_real_train,    'loss_real_val':loss_real_val,
                'loss_fake_train':loss_fake_train,    'loss_fake_val':loss_fake_val,
                'loss_R2_score_train':loss_R2_score_train,'loss_R2_score_val':loss_R2_score_val};
    else:
        history={'g_loss_train':g_loss_train,'g_loss_val':g_loss_val,
                'loss_pixel_train':loss_pixel_train,'loss_pixel_val':loss_pixel_val,
                'loss_R2_score_train':loss_R2_score_train,'loss_R2_score_val':loss_R2_score_val}
    opt.history=history
else:
    generator.eval();discriminator.eval()
#########  Plotting testing results
sample_dataset(test_dataloader,generator,opt,opt.n_epochs,flag_show_scaled_data=0)
sample_dataset(tracking_dataloader2,generator,opt,opt.n_epochs,flag_show_scaled_data=0)
if opt.n_epochs==-1:
    sample_dataset(test_dataloader,generator,opt,opt.n_epochs,flag_show_scaled_data=0,record_weights=1)
# names,scores=calculate_accuracy_on_dataset(tracking_dataloader2,generator,opt,opt.n_epochs);  print(names);print(scores)
with open(opt.save_path+'/'+'opt.txt', 'w') as f:
    json.dump(opt.__dict__, f, indent=2)
if opt.architecture==1:
    Plot_r2_losses(history,Title='log'+str(log_save_const) + 'losses_r2', Save_pictures_path=Save_pictures_path,Save_flag=1)
    Plot_g_losses(history,Title='log'+str(log_save_const) + 'losses_g', Save_pictures_path=Save_pictures_path,Save_flag=1)
    Plot_d_losses(history,Title='log'+str(log_save_const) + 'losses_d', Save_pictures_path=Save_pictures_path,Save_flag=1)
else:
    Plot_r2_losses_2(history,Title='log'+str(log_save_const) + 'losses_r2', Save_pictures_path=Save_pictures_path,Save_flag=1)
    Plot_unet_losses(history,Title='log'+str(log_save_const) + 'losses_unet', Save_pictures_path=Save_pictures_path,Save_flag=1)
    Plot_unet_losses_2(history,Title='log'+str(log_save_const) + 'losses_pixel', Save_pictures_path=Save_pictures_path,Save_flag=1)

