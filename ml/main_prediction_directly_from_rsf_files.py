print('Torch CODE Torch CODE Torch CODE Torch CODE!!!!!')
from imports_torch import *
from utils_low_wavenumbers_torch import *
s=1
####    create logging
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
dataset_path=['/home/plotnips/Dropbox/Log_extrapolation/scripts/MWE/datasets/data_from_server/training_data_10_it_oleg',
              '/home/plotnips/Dropbox/Log_extrapolation/scripts/MWE/datasets/data_from_server/training_data_10_it_vl_gen']
dataset_path=['/home/plotnips/Dropbox/Log_extrapolation/scripts/MWE/datasets/data_from_server/training_data_10_it_vl_gen']
####    torch code  ,multiprocessing.cpu_count()-3
parser = argparse.ArgumentParser()
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--start_from_pretrained_model", type=int, default=1, help="start_from_pretrained_model")
parser.add_argument("--start_with_certain_dataset",type=int,default=0,help="start_with_certain_dataset")
parser.add_argument("--img_height", type=int, default=256, help="size of image height")
parser.add_argument("--img_width", type=int, default=256, help="size of image width")
parser.add_argument("--architecture", type=int, default=0,help="gan architecture=1,cnn architecture=0")
# parser.add_argument("--channels", type=int, default=2, help="number of input channels in generator")
parser.add_argument("--channels", type=int, default=11, help="number of input channels in generator")
# parser.add_argument("--inp_channels_type", type=str, default='1dv', help="type of input channels in generator")
parser.add_argument("--inp_channels_type", type=str, default='only_grads', help="type of input channels in generator")
parser.add_argument("--channels_discriminator", type=int, default=1, help="number of input channels in discriminator")
# parser.add_argument("--cnn_dropout",type=int,default=0.1,help="dropout in cnn")
parser.add_argument("--cnn_dropout",type=int,default=0.2,help="dropout in cnn")
# parser.add_argument("--cnn_dropout",type=int,default=0.5,help="dropout in cnn")
parser.add_argument("--shuffle_files",type=int,default=1,help="shuffle files")
####    short training
# parser.add_argument("--channels_discriminator", type=int, default=1, help="number of input channels in discriminator")
parser.add_argument("--dataset_path",type=int,default=dataset_path,help="path to the dataset")
parser.add_argument("--n_epochs",type=int,default=160,help="number of epochs of training")
parser.add_argument("--batch_size", type=int,default=8,help="size of the batches") #16
parser.add_argument("--n_cpu",type=int,default=8,help="number of cpu threads to use during batch generation")   #8
# parser.add_argument("--batch_size", type=int,default=4,help="size of the batches") #16
# parser.add_argument("--n_cpu",type=int,default=2,help="number of cpu threads to use during batch generation")   #8
parser.add_argument("--checkpoint_interval", type=int,  default=10, help="interval between model checkpoints")
parser.add_argument("--plotting_interval", type=int,    default=10, help="plot results at each epoch")
# parser.add_argument("--checkpoint_interval", type=int,  default=100, help="interval between model checkpoints")
# parser.add_argument("--plotting_interval", type=int,    default=100, help="plot results at each epoch")
parser.add_argument("--N",type=int,default=-1,help="total number of samples in training and validation losses")
######################################
opt = parser.parse_args()
opt.save_path=Save_pictures_path;   opt.log_save_const=log_save_const;  start_from_pretrained_model=opt.start_from_pretrained_model
######################################
path=[];    counter=0
for dataset_path in opt.dataset_path:
    print('dataset_path=',dataset_path)
    files_path=glob(dataset_path+'/*')
    files_path=sorted(files_path);  #files_path.reverse()
    files_path=fnmatch.filter(files_path,'*_cropped*'); path_=files_path;    # print(path_[-12:])
    if len(opt.dataset_path)>1:
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
        ################
        if counter==0:      val1=len(path_);    print('val1=',val1)
        elif counter==1:    val2=len(path_);    print('val2=',val2)
        if counter==0:  #   remove test files out of the first dataset
            path_=path_[-val1:];    print('first dataset size=',len(path_))
            files_to_delete=[]
            for ii in path_:
                if '__' in ii:
                    files_to_delete.append(ii)
            for ii in files_to_delete:
                print('removing ',ii)
                path_.remove(ii)
        elif counter==1:
            path_=path_[-val2:];    print('second dataset size=',len(path_))
    path=path+path_
    counter=counter+1
print('len(path)=',len(path))
path_test=sorted(path)
######################################  custom folders
path_test=[]
path_test.append('/home/plotnips/Dropbox/Log_extrapolation/scripts/MWE/datasets/data_from_server/training_data_10_it_vl_gen/model_Marmousi_cropped_model/Pictures7')
path_test.append('/home/plotnips/Dropbox/Log_extrapolation/scripts/MWE/datasets/data_from_server/training_data_10_it_vl_gen/model_Overthrust_cropped_model/Pictures5')
path_test.append('/home/plotnips/Dropbox/Log_extrapolation/scripts/MWE/datasets/data_from_server/training_data_10_it_vl_gen/model_Seam2_cropped_model/Pictures0')
######################################
path_valid=path[0:2]
path_train=path[3:4]
######################################
if opt.n_epochs==-1 or start_from_pretrained_model==1:
    log_name=842;   epoch_name=14
    log_name=843;   epoch_name=14
    log_name=845;   epoch_name=14
    tmp=logs_path+'/log'+str(log_name)+'/opt.txt'
    n_epochs=opt.n_epochs;
    if os.path.exists(tmp):
        parser = ArgumentParser();  opt = parser.parse_args()
    print('loading opt parameters from ',tmp)
    # exit()
    with open(tmp,'r') as f:
        print('loading json.load')
        opt.__dict__=json.load(f)
    print(opt.path_test)
    opt.save_path=Save_pictures_path
    opt.log_save_const=log_save_const
    opt.n_epochs=n_epochs
    opt.generator_model_name=       './logs/log'+str(log_name)+'/generator_'+str(epoch_name)+'.pth'
opt.path_test=path_test;    opt.path_valid=path_valid;  opt.path_train=path_train;# print(opt)
print('len(opt.path_train)=',len(opt.path_train))
print('len(opt.path_valid)=',len(opt.path_valid))
print('len(opt.path_test)=', len(opt.path_test))
#####
torch.backends.cudnn.benchmark = True
cuda = True if torch.cuda.is_available() else False
# Loss functions
criterion_GAN = torch.nn.MSELoss()
criterion_pixelwise = torch.nn.L1Loss()
criterion_R2Loss=R2Loss()
# Loss weight of L1 pixel-wise loss between translated image and real image
opt.lambda_pixel = 100
opt.lambda_pixel = 0.01
# Calculate output of image discriminator (PatchGAN)
patch = (1, opt.img_height // 2 ** 4, opt.img_width // 2 ** 4); opt.patch=patch
# Initialize generator and discriminator
torch.manual_seed(123)
np.random.seed(123)
random.seed(123)
############################
# generator = GeneratorUNet(in_channels=opt.channels,DROPOUT=opt.cnn_dropout)
generator = GeneratorUNet_old_configuration(in_channels=opt.channels,DROPOUT=opt.cnn_dropout)
discriminator = Discriminator(in_channels=opt.channels_discriminator)
summary(generator.to(0), (opt.channels,256,256))
summary(discriminator.to(0),[(1,256,256),(opt.channels_discriminator,256,256)] )
################## generator = GeneratorUNet_big(in_channels=opt.channels,DROPOUT=opt.cnn_dropout)
if cuda:
    generator = generator.cuda()
    discriminator = discriminator.cuda()
    criterion_GAN.cuda()
    criterion_pixelwise.cuda()
    criterion_R2Loss.cuda()
if opt.n_epochs==-1 or start_from_pretrained_model==1:
    print('Load pretrained models',opt.generator_model_name)
    generator.load_state_dict(torch.load(opt.generator_model_name))
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
# train_dataloader = DataLoader(ImageDataset(opt,transforms_=transforms_,mode="train",max_for_initial_models=max_for_initial_models),
#     batch_size=opt.batch_size,shuffle=False,num_workers=opt.n_cpu,pin_memory=True)
# val_dataloader=DataLoader(ImageDataset(opt,transforms_=transforms_,mode="val",max_for_initial_models=max_for_initial_models),
#     batch_size=opt.batch_size,shuffle=False,num_workers=opt.n_cpu,pin_memory=True)
# test_dataloader=DataLoader( ImageDataset(opt,transforms_=transforms_,mode="test",max_for_initial_models=max_for_initial_models),
#     batch_size=1,shuffle=False,num_workers=1)
# tracking_dataloader=DataLoader(ImageDataset(opt,transforms_=transforms_,
#     mode=None,file_list=opt.path_train[0:2],max_for_initial_models=max_for_initial_models),     #+path_test[-2:-1]+path_test[-4:-3]
#     batch_size=1,shuffle=False,num_workers=1)
# tracking_dataloader2=DataLoader(ImageDataset(opt,transforms_=transforms_,
#     mode=None,file_list=opt.path_train[0:60],max_for_initial_models=max_for_initial_models),
#     batch_size=1,shuffle=False,num_workers=1)
# opt.logging_loss_batch_size=len(train_dataloader)
# sample_models(tracking_dataloader2,generator,opt,opt.n_epochs,flag_show_scaled_data=0)
# sample_models(test_dataloader,generator,opt,opt.n_epochs,flag_show_scaled_data=0)
with open(opt.save_path+'/'+'opt.txt', 'w') as f:
    json.dump(opt.__dict__, f, indent=2)
# exit()
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
generator.eval();discriminator.eval()
sample_dataset_specially(generator,opt,opt.n_epochs,flag_show_scaled_data=1,record_weights=1)
with open(opt.save_path+'/'+'opt.txt', 'w') as f:
    json.dump(opt.__dict__, f, indent=2)

#########  Plotting testing results
# test_dataloader=DataLoader( Dataset_from_rsf_files(opt,transforms_=transforms_,mode="test"),
#     batch_size=1,shuffle=False,num_workers=1)
# sample_dataset(tracking_dataloader,generator,opt,opt.n_epochs,flag_show_scaled_data=1)
# sample_dataset(tracking_dataloader2,generator,opt,opt.n_epochs,flag_show_scaled_data=1)
# if opt.n_epochs==-1:
#     sample_dataset(test_dataloader,generator,opt,opt.n_epochs,flag_show_scaled_data=1,record_weights=1)
# names,scores=calculate_accuracy_on_dataset(tracking_dataloader2,generator,opt,opt.n_epochs);  print(names);print(scores)