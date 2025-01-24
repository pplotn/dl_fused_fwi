from imports_torch import *
Fontsize_for_losses = 14
from sklearn.metrics import mean_squared_error
from skimage.metrics import structural_similarity as ssim
plt.rcParams.update({'font.size': 14})
# sys.path.append('/var/remote/lustre/project/k1404/pavel/DENISE-Black-Edition')
# sys.path.append('/home/plotnips/Dropbox/Log_extrapolation/scripts/DENISE-Black-Edition-master/codes_server')
# import pyapi_denise_pavel as api
# scaling_constants_dict={'x':[2010],'t':[2700],'init_model':[5500]}
def return_existing_file_list(list,path):
    existing_list=[]
    print(os.getcwd())
    existing_files=os.listdir(path[0])
    for file in list:
        if file.split('/')[-1] in existing_files:
            existing_list.append(os.path.join(path[0],file.split('/')[-1]))
        else:
            a=1
    return existing_list

def RepresentsInt(s):
    try:
        int(s)
        return True
    except ValueError:
        return False

def is_defined(v):
    return False if v is None else True

def sample_images(batches_done):
    """Saves a generated sample from the validation set"""
    imgs = next(iter(val_dataloader))
    real_A = Variable(imgs["B"].type(Tensor))
    real_B = Variable(imgs["A"].type(Tensor))
    fake_B = generator(real_A)
    m1 = torch.transpose(real_A.data, 2, 3)
    m2 = torch.transpose(fake_B.data, 2, 3)
    m3 = torch.transpose(real_B.data, 2, 3)
    img_sample = torch.cat((m1, m2, m3), -2)
    save_image(img_sample, Save_pictures_path+"/"+'Batch_'+str(batches_done)+".png", nrow=5, normalize=True)

##################################  my metrics
def nrms(T_pred,T_true):
    return np.linalg.norm(T_pred-T_true)/np.linalg.norm(T_true)

class R2Loss(nn.Module):
    # loss_r2_2=calculate_metric_on_batch(fake_B.cpu().data.numpy(),real_B.cpu().data.numpy());
    def forward(self, y_pred, y):
        var_y = torch.var(y, unbiased=False)
        return 1.0 - F.mse_loss(y_pred, y, reduction="mean") / var_y

class R2Loss_custom(nn.Module):
    # https://github.com/pytorch/ignite/blob/master/tests/ignite/contrib/metrics/regression/test_r2_score.py
    def forward(self, y_pred, y):
        y_pred = y_pred.cpu().detach().numpy()
        y = y.cpu().detach().numpy()
        batch_size=y.shape[0]
        scores=np.zeros((batch_size))
        for i in range(y.shape[0]):
            v1 = y_pred[i,::].squeeze().flatten()
            v2 = y[i,::].squeeze().flatten()
            r2_2 = r2_score(v1, v2)
            scores[i]=r2_2
        return scores

class R2Loss_custom_average(nn.Module):
    # https://github.com/pytorch/ignite/blob/master/tests/ignite/contrib/metrics/regression/test_r2_score.py
    def forward(self, y_pred, y):
        y_pred = y_pred.cpu().detach().numpy()
        y = y.cpu().detach().numpy()
        batch_size=y.shape[0]
        scores=np.zeros((batch_size))
        for i in range(y.shape[0]):
            v1 = y_pred[i,::].squeeze().flatten()
            v2 = y[i,::].squeeze().flatten()
            r2_2 = r2_score(v1, v2)
            scores[i]=r2_2
        averaged_score=scores.sum()/y.shape[0]
        return averaged_score

class NRMS_loss_custom(nn.Module):
    # https://github.com/pytorch/ignite/blob/master/tests/ignite/contrib/metrics/regression/test_r2_score.py
    def forward(self, y_pred, y):
        loss = nn.MSELoss()
        scores=np.zeros((y.shape[0]))
        for i in range(y.shape[0]):
            v1 = y_pred[i,::].squeeze()
            v2 = y[i,::].squeeze()
            NRMS=torch.pow( torch.sum(torch.pow((v1-v2),2))/v2.numel() ,0.5)
            ### normalization
            NRMS=NRMS/torch.mean(v2)
            scores[i]=NRMS
        return scores

class R2Loss2(nn.Module):
    def forward(self, y_pred, y):
        y_pred = y_pred.cpu().data.numpy()
        y = y.cpu().data.numpy()
        v1 = y_pred.flatten()
        v2 = y.flatten()
        r2_2 = r2_score(v1, v2)
        return r2_2
        # var_y = torch.var(y, unbiased=False)
        # return 1.0 - F.mse_loss(y_pred,y,reduction="mean") / var_y

def calculate_metric_on_batch(mat, mat_true):
    mat = mat.squeeze()
    mat_true = mat_true.squeeze()
    v1 = mat.flatten()
    v2 = mat_true.flatten()
    r2_2 = r2_score(v1, v2)
    return r2_2

def tracewise_rms(x):
    return np.sqrt(np.sum(np.square(x), axis=-1) / x.shape[-1])
def rms(x):
    return np.sqrt(np.sum(np.square(x)) / np.prod(x.shape))
def relative_rms(ref, x, digits=2):
    return np.round(rms(x) / rms(ref), 2)
def handle_zeros(t, eps):
    """ Avoid division by zero """
    # t[t < eps] = 1.
    t2=np.copy(t)
    t2[t2 < eps] = 1.
    return t2
def calculate_metric_on_batch_(predictions,labels,eps=1e-6,metric_type='ssim'):
    # RMS similarity = 1 - rms(x - y) / rms(x). Best when 1, worst 0
    """https://stats.stackexchange.com/questions/255276/normalized-root-mean-square-error-nrmse-with-zero-mean-of-observed-value"""
    bl=predictions.shape[0]
    scores=np.zeros((bl))
    for i in range(bl):
        pred=predictions[i,::].squeeze()
        lbl=labels[i,::].squeeze()
        diff=pred-lbl
        if metric_type=='rms similarity':
            diff_rms = np.sqrt( np.sum( (diff**2).flatten() ) ).item()
            lbl_rms = np.sqrt(np.sum(lbl ** 2)).item()
            coeff_rms1 = 1 - diff_rms/handle_zeros(lbl_rms, eps)
            coeff_rms2 = 1 - np.mean(tracewise_rms(lbl - pred) / _handle_np_zeros(tracewise_rms(lbl)))
            scores[i]=coeff_rms1
        elif metric_type=='ssim':
            scores[i]=ssim(pred,lbl)
    return scores
def F_r2(mat, mat_true):
    # r2 = 1 - (np.std(mat_true.flatten() - mat.flatten()) / np.std(mat_true.flatten())) ** 2
    v1 = mat.flatten()
    v2 = mat_true.flatten()
    r2_2 = r2_score(v1, v2)
    return r2_2

def F_r2_tracewise(mat,mat_true,calculate_nz_loss=0,output_size=1):
    ########### non-order 
    v1 = mat.flatten()
    v2 = mat_true.flatten()
    r2_2 = r2_score(v1, v2)
    ###########
    n1,n2 = mat.shape
    if n1<n2:   nx=n2;  nz=n1;  
    else:       nx=n1;  nz=n2;  mat=mat.T;  mat_true=mat_true.T
    x_tracewise_r2_score=np.empty((nx))
    for j in range(nx):
        x_tracewise_r2_score[j]=sklearn.metrics.r2_score(mat[:,j],mat_true[:,j])
    average_x_tracewise_r2_score=np.mean(x_tracewise_r2_score)
    ###########
    # calculate_nz_loss=1
    if calculate_nz_loss==1:
        z_tracewise_r2_score=np.empty((nz))
        for i in range(nz):
            z_tracewise_r2_score[i]=sklearn.metrics.r2_score(mat[i,:],mat_true[i,:])
        average_z_tracewise_r2_score=np.mean(z_tracewise_r2_score)
    if output_size==1:
        return average_x_tracewise_r2_score
    else:
        if calculate_nz_loss==1:
            return average_x_tracewise_r2_score,x_tracewise_r2_score,average_z_tracewise_r2_score,z_tracewise_r2_score
        else:
            return average_x_tracewise_r2_score,x_tracewise_r2_score

def Plot_accuracy2(history, Title='Title', Save_pictures_path='./Pictures', Save_flag=0):
    plt.figure()
    plt.plot(history['g_accuracy_history'])
    plt.plot(history['d_accuracy_history'])
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.axis('tight')
    plt.ylim(0, 100)
    string = ', R2 accuracy'
    plt.title(Title+string)
    plt.legend(['generator accuracy', 'discriminator accuracy'],
               loc='lower right', fontsize=Fontsize)
    if Save_flag == 1:
        name = Save_pictures_path + '/' + Title + '.png'
        print(name)
        plt.savefig(name)
    plt.show(block=False)
    plt.close()
    return None

def Plot_loss_torch(history, Title='Title', Save_pictures_path='./Pictures', Save_flag=0):
    Fontsize = Fontsize_for_losses
    plt.figure()
    plt.plot(history['g_loss_history'])
    plt.plot(history['d_loss_history'])
    plt.tick_params(labelsize=Fontsize)
    plt.yscale('log')
    plt.ylabel('Loss function', fontsize=Fontsize)
    plt.xlabel('Epochs', fontsize=Fontsize)
    plt.axis('tight')
    string = ', R2 loss'
    plt.title(Title, fontsize=Fontsize)
    plt.legend(['generator loss', 'discriminator loss'],
               loc='lower right', fontsize=Fontsize)
    if Save_flag == 1:
        name = Save_pictures_path + '/' + Title + '.png'
        print(name)
        plt.savefig(name)
    plt.show(block=False)
    plt.close()
    return None

def Plot_r2_losses(history, Title='Title', Save_pictures_path='./Pictures', Save_flag=0):
    Fontsize = Fontsize_for_losses
    plt.figure()
    plt.plot(history['loss_R2_score_train'])
    plt.plot(history['loss_R2_score_val'])
    plt.tick_params(labelsize=Fontsize)
    plt.ylabel('Loss function', fontsize=Fontsize)
    plt.xlabel('Epochs', fontsize=Fontsize)
    aa = np.append(history['loss_R2_score_train'],
                   history['loss_R2_score_val'])
    # bb=np.min(history['loss_R2_score_train']+history['loss_R2_score_val'])
    # plt.ylim([0.7,0.8])
    # plt.ylim([0.85,0.95])
    plt.ylim([0.0, 1])
    # plt.ylim([np.min(history['loss_R2_score_train']+history['loss_R2_score_val']),1])
    # plt.axis('tight')
    string = ', R2 loss'
    plt.title('Average R2 misfit(prediction, target), '+Title+', train/val='+numstr(
        history['loss_R2_score_train'][-1])+'/'+numstr(history['loss_R2_score_val'][-1]), fontsize=Fontsize)
    plt.legend(['training R2 loss', 'validation R2 loss'],
               loc='lower right', fontsize=Fontsize)
    if Save_flag == 1:
        name = Save_pictures_path + '/' + Title + '.png'
        print(name)
        plt.savefig(name)
    plt.show(block=False)
    plt.close()
    return None

def Plot_r2_losses_2(history, Title='Title', Save_pictures_path='./Pictures', Save_flag=0):
    Fontsize = Fontsize_for_losses
    plt.figure(figsize=(8, 6), dpi=80)
    loss_target_pred_train=np.asarray(history['loss_target_pred_train'])
    loss_target_pred_valid=np.asarray(history['loss_target_pred_valid'])
    plt.plot(loss_target_pred_train,'b-')
    plt.plot(loss_target_pred_valid,'b--')
    # plt.plot(history['loss_init_model_pred_train'], '-', color="orange")
    # plt.plot(history['loss_init_model_pred_valid'], '--', color="orange")

    # plt.plot(history['loss_true_model_pred_train'],'g-')
    # plt.plot(history['loss_true_model_pred_valid'],'g--')
    # plt.plot(history['loss_true_model_inversion_train'],'r-')
    # plt.plot(history['loss_true_model_inversion_valid'],'r--')
    plt.tick_params(labelsize=Fontsize)
    plt.ylabel('Loss function', fontsize=Fontsize)
    plt.xlabel('Epochs', fontsize=Fontsize)
    # plt.ylim([np.min(history['loss_R2_score_train']+history['loss_R2_score_val']),1])
    # plt.axis('tight')
    # plt.ylim(top=0.6)
    plt.ylim(top=1)
    string = ', R2 loss'
    # plt.title('Average R2 misfit(prediction, target), '+Title,fontsize=Fontsize)
    plt.title('Average R2 misfit(prediction, target), '+Title+', train/val=' +
            numstr(loss_target_pred_train[-1])+'/'+numstr(loss_target_pred_valid[-1] )+
            ', train/val='
            , fontsize=Fontsize)
    plt.legend([
                'Target prediction R2,train', 'Target prediction R2,validation',
                # 'Initial model prediction R2,train','Initial model prediction R2,validation',
                # 'True model prediction R2,train','True model prediction R2,validation',
                # 'True model inversion R2,train', 'True model inversion R2,validation'
                ],loc='lower right', fontsize=Fontsize)
    # plt.tight_layout()
    if Save_flag == 1:
        name = Save_pictures_path + '/' + Title + '.png'
        print(name)
        plt.savefig(name,dpi=300,bbox_inches='tight')
    plt.show(block=False)
    # plt.show()
    plt.close()
    return None

def Plot_r2_losses_3(history, Title='Title', Save_pictures_path='./Pictures', Save_flag=0):
    Fontsize = Fontsize_for_losses
    plt.figure(figsize=(8,3), dpi=300)
    loss_target_pred_train=np.asarray(history['loss_target_pred_train'])
    loss_target_pred_valid=np.asarray(history['loss_target_pred_valid'])
    plt.plot(loss_target_pred_train,'b-')
    plt.plot(loss_target_pred_valid,'b--')
    # plt.plot(history['loss_init_model_pred_train'], '-', color="orange")
    # plt.plot(history['loss_init_model_pred_valid'], '--', color="orange")

    # plt.plot(history['loss_true_model_pred_train'],'g-')
    # plt.plot(history['loss_true_model_pred_valid'],'g--')
    # plt.plot(history['loss_true_model_inversion_train'],'r-')
    # plt.plot(history['loss_true_model_inversion_valid'],'r--')
    plt.tick_params(labelsize=Fontsize)
    plt.ylabel('Loss function', fontsize=Fontsize)
    plt.xlabel('Epochs', fontsize=Fontsize)
    # plt.ylim([np.min(history['loss_R2_score_train']+history['loss_R2_score_val']),1])
    # plt.axis('tight')
    plt.ylim(top=0.6)
    # plt.ylim(top=1)
    string = ', R2 loss'
    plt.legend([
                'Target prediction R2,train', 'Target prediction R2,validation',
                # 'Initial model prediction R2,train','Initial model prediction R2,validation',
                # 'True model prediction R2,train','True model prediction R2,validation',
                # 'True model inversion R2,train', 'True model inversion R2,validation'
                ],loc='lower right', fontsize=Fontsize)
    # plt.tight_layout()
    if Save_flag == 1:
        name = Save_pictures_path + '/' + Title + '.png'
        print(name)
        plt.savefig(name,dpi=300,bbox_inches='tight')
    # plt.show(block=False)
    # plt.show()
    plt.close()
    return None

def Plot_r2_losses_for_geophysics(history,early_stop,save_name='save',Title='Title', Save_pictures_path='./Pictures', Save_flag=0):
    Fontsize = Fontsize_for_losses
    plt.figure(figsize=(8,3), dpi=300)
    loss_target_pred_train=np.asarray(history['loss_target_pred_train'])
    loss_target_pred_valid=np.asarray(history['loss_target_pred_valid'])
    plt.plot(loss_target_pred_train,'b-',label='Target prediction R2,train')
    plt.plot(loss_target_pred_valid,'b--',label='Target prediction R2,validation')
    plt.axvline(x=early_stop,color='k',lw=2,label='early stopping')
    plt.tick_params(labelsize=Fontsize)
    plt.ylabel('Loss function', fontsize=Fontsize)
    plt.xlabel('Epochs', fontsize=Fontsize)
    # plt.ylim(top=1.0)
    plt.ylim(top=np.max(history['loss_target_pred_train']),bottom=0)
    plt.title(Title)
    plt.legend(loc='lower right', fontsize=Fontsize)
    if Save_flag == 1:
        name = Save_pictures_path + '/' + save_name + '.png'
        print(name)
        plt.savefig(name,dpi=300,bbox_inches='tight')
    plt.close()
    return None

def Plot_L1_losses_for_geophysics(history,early_stop,save_name='save',Title='Title', Save_pictures_path='./Pictures', Save_flag=0):
    Fontsize = Fontsize_for_losses
    plt.figure(figsize=(8,3), dpi=300)
    loss_target_pred_train=np.asarray(history['loss_pixel_train'])
    loss_target_pred_valid=np.asarray(history['loss_pixel_val'])
    plt.plot(loss_target_pred_train,'b-',label='Target prediction R2,train')
    plt.plot(loss_target_pred_valid,'b--',label='Target prediction R2,validation')
    plt.axvline(x=early_stop,color='k',lw=2,label='early stopping')
    plt.tick_params(labelsize=Fontsize)
    plt.ylabel('Loss function', fontsize=Fontsize)
    plt.xlabel('Epochs', fontsize=Fontsize)
    # plt.ylim(top=1.0)
    plt.ylim(top=np.max(history['loss_pixel_train']),bottom=0)
    plt.title(Title)
    plt.legend(loc='lower right', fontsize=Fontsize)
    if Save_flag == 1:
        name = Save_pictures_path + '/' + save_name + '.png'
        print(name)
        plt.savefig(name,dpi=300,bbox_inches='tight')
    plt.close()
    return None

def Plot_unet_losses(history, Title='Title', Save_pictures_path='./Pictures', Save_flag=0):
    Fontsize = Fontsize_for_losses
    plt.figure()
    plt.plot(history['g_loss_train'])
    plt.plot(history['g_loss_val'])
    plt.tick_params(labelsize=Fontsize)
    # plt.yscale('log')
    plt.ylabel('Loss function', fontsize=Fontsize)
    plt.xlabel('Epochs', fontsize=Fontsize)
    # plt.axis('tight')
    string = ', R2 loss'
    plt.title('Generator losses, '+Title, fontsize=Fontsize)
    plt.legend(['total loss, train',
                'total loss, val',
                ], loc='lower right', fontsize=Fontsize)
    if Save_flag == 1:
        name = Save_pictures_path + '/' + Title + '.png'
        print(name)
        plt.savefig(name)
    plt.show(block=False)
    plt.close()
    return None

def Plot_unet_losses_2(history, Title='Title', Save_pictures_path='./Pictures', Save_flag=0):
    Fontsize = Fontsize_for_losses
    plt.figure()
    plt.plot(history['loss_pixel_train'])
    plt.plot(history['loss_pixel_val'])
    plt.tick_params(labelsize=Fontsize)
    plt.yscale('log')
    plt.ylabel('Loss function', fontsize=Fontsize)
    plt.xlabel('Epochs', fontsize=Fontsize)
    # plt.axis('tight')
    string = ', R2 loss'
    plt.title('Generator losses, '+Title, fontsize=Fontsize)
    plt.legend(['loss_pixel, train',
                'loss_pixel, val',
                ], loc='lower right', fontsize=Fontsize)
    if Save_flag == 1:
        name = Save_pictures_path + '/' + Title + '.png'
        print(name)
        plt.savefig(name)
    plt.show(block=False)
    plt.close()
    return None

def Plot_g_losses(history, Title='Title', Save_pictures_path='./Pictures', Save_flag=0):
    Fontsize = Fontsize_for_losses
    plt.figure()
    plt.plot(history['g_loss_train'])
    plt.plot(history['loss_GAN_train'])
    plt.plot(history['loss_pixel_train'])
    # plt.plot(history['g_loss_val'])
    # plt.plot(history['loss_GAN_val'])
    # plt.plot(history['loss_pixel_val'])
    plt.tick_params(labelsize=Fontsize)
    plt.yscale('log')
    plt.ylabel('Loss function', fontsize=Fontsize)
    plt.xlabel('Epochs', fontsize=Fontsize)
    # plt.axis('tight')
    string = ', R2 loss'
    plt.title('Generator losses, '+Title, fontsize=Fontsize)
    plt.legend(['total loss, train',
                'adversarial loss D(x,G(x)), train',
                'loss_pixel ||y-G(x)||, train',
                # 'total loss, val',
                # 'adversarial loss D(x,G(x)), val',
                # 'loss_pixel ||y-G(x)||, val',
                ], loc='lower right', fontsize=Fontsize)
    if Save_flag == 1:
        name = Save_pictures_path + '/' + Title + '.png'
        print(name)
        plt.savefig(name)
    plt.show(block=False)
    plt.close()
    return None

def Plot_d_losses(history, Title='Title', Save_pictures_path='./Pictures', Save_flag=0):
    Fontsize = Fontsize_for_losses
    plt.figure()
    plt.plot(history['d_loss_train'])
    plt.plot(history['loss_real_train'])
    plt.plot(history['loss_fake_train'])
    # plt.plot(history['d_loss_val'])
    # plt.plot(history['loss_real_val'])
    # plt.plot(history['loss_fake_val'])
    plt.tick_params(labelsize=Fontsize)
    plt.yscale('log')
    plt.ylabel('Loss function', fontsize=Fontsize)
    plt.xlabel('Epochs', fontsize=Fontsize)
    # plt.axis('tight')
    string = ', R2 loss'
    plt.title('Discriminator losses, '+Title, fontsize=Fontsize)
    plt.legend(['total loss, train',
                'loss_real, train',
                'loss_fake, train',
                # 'total loss, val',
                # 'loss_real, val',
                # 'loss_fake, val',
                ], loc='upper right', fontsize=Fontsize)
    if Save_flag == 1:
        name = Save_pictures_path + '/' + Title + '.png'
        print(name)
        plt.savefig(name)
    plt.show(block=False)
    plt.close()
    return None

class Tee(object):
    # Write terminal output to log
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()

def F_calculate_log_number(path, Word, type='.png'):
    Const = len(fnmatch.filter(os.listdir(path), Word + '*'))
    Name = path + '/' + Word + str(Const) + type
    while os.path.exists(Name):
        Const = Const + 1
        Name = path + '/' + Word + str(Const) + type
    return Const
def unpack_file(NAME):
    pts=NAME.split('/')
    f_name=pts[-1]
    f_name=f_name[:-4]; 
    # print('reading ',f_name)
    with open(NAME, 'rb') as f:
        data=np.load(f)
        model=data['models'].squeeze()
        initial_model=data['models_init'].squeeze()
        input_data=data['input_data'].squeeze()
        output_data=data['output_data'].squeeze()
        input_data_real_amplitudes=data['input_data_real_amplitudes'].squeeze()
        output_data_real_amplitudes=data['output_data_real_amplitudes'].squeeze()
        data.close()
    data_structure={'model':model.T,'initial_model':initial_model.T,'input_data':input_data.T,'input_data_real_amplitudes':input_data_real_amplitudes.T
        ,'output_data_real_amplitudes':output_data_real_amplitudes.T,'name':f_name,'output_data':output_data.T}
    # Plot_image(M0.T,Show_flag=0,Save_flag=1,Save_pictures_path=Save_pictures_path,Title=f_name+'_M0',Aspect='equal')
    # Plot_image(M2.T,Show_flag=0,Save_flag=1,Save_pictures_path=Save_pictures_path,Title=f_name+'_M2',Aspect='equal')
    return data_structure
def extract_structure_field(list_of_structures,field):
    list_of_arrays=[]
    for i in range(len(list_of_structures)):
        list_of_arrays.append( list_of_structures[i][field] )
    return list_of_arrays
def unpack_array_from_file(NAME):
    pts=NAME.split('/')
    f_name=pts[-1]
    f_name=f_name[:-4]; 
    # print('reading ',f_name)
    with open(NAME, 'rb') as f:
        data=np.load(f)
        model=data['models']
        initial_model=data['models_init']
        input_data=data['input_data']
        output_data=data['output_data']
        input_data_real_amplitudes=data['input_data_real_amplitudes']
        output_data_real_amplitudes=data['output_data_real_amplitudes']
        data.close()
    data_structure={'model':model,'initial_model':initial_model,'input_data':input_data,'input_data_real_amplitudes':input_data_real_amplitudes
        ,'output_data_real_amplitudes':output_data_real_amplitudes,'name':f_name,'output_data':output_data}
    # Plot_image(M0.T,Show_flag=0,Save_flag=1,Save_pictures_path=Save_pictures_path,Title=f_name+'_M0',Aspect='equal')
    # Plot_image(M2.T,Show_flag=0,Save_flag=1,Save_pictures_path=Save_pictures_path,Title=f_name+'_M2',Aspect='equal')
    return data_structure

def extract_ml_data_to_list(opt,full_name):
    type_of_input=opt.inp_channels_type
    #######################################################
    Name=full_name.split('/')[-1]
    Name=Name.split('.npz')[0]
    with open(full_name,'rb') as f:
        data = np.load(f,allow_pickle=True)
        # print('data[input_data]=',data['input_data'].shape)
        x = data['input_data']
        t = data['output_data']
        x_ra= data['input_data_real_amplitudes']
        t_ra= data['output_data_real_amplitudes']
        models_init =data['models_init']
        models_init_scaled=data['models_init_scaled']
        models = data['models']
        fwi_result_scaled=data['fwi_result_scaled']
        smoothed_true_model_scaled=data['smoothed_true_model_scaled']
        dm_i_ra=data['dm_i_ra']
        dm_i_ra_scaled=data['dm_i_ra_scaled']
        m_i_ra=data['m_i_ra']
        m_i_ra_scaled=data['m_i_ra_scaled']
        # scaler_t=data['scaler_t']
        taper=data['taper']
        dz = data['dz']
        dx = data['dx']
        data.close()
    # nch = x.shape[0]
    nch =opt.channels
    nx = models.shape[1]
    nz = models.shape[2]
    ################################    record x
    taper=imresize(taper.squeeze(),[opt.img_height,opt.img_width]); taper[taper<1]=0
    model_initial=np.squeeze(models_init_scaled)
    model_initial=imresize(model_initial,[opt.img_height,opt.img_width])
    x_ = np.empty((1,opt.img_height, opt.img_width))
    x_[0,::]=imresize(x.squeeze(),[opt.img_height,opt.img_width])
    img_A = np.zeros((opt.channels,opt.img_height,opt.img_width))
    if type_of_input == '1dv':
        img_A = np.zeros((1, opt.img_height, opt.img_width))
        img_A[0, :, :] = x_[-1, ::]
    if type_of_input == '1_fwi_res':
        img_A = np.zeros((1, opt.img_height, opt.img_width))
        img_A[0,:,:]=imresize(fwi_result_scaled.squeeze(),[opt.img_height,opt.img_width])
    elif type_of_input == '1m_1taper':
        img_A = np.zeros((2, opt.img_height, opt.img_width))
        model_initial=np.squeeze(models_init)
        model_initial=imresize(model_initial,[opt.img_height,opt.img_width])
        img_A[0, :, :] = x_[-1, ::]+model_initial
        img_A[1, :, :] = taper
    elif type_of_input == '1dv_1taper':
        img_A = np.zeros((2, opt.img_height, opt.img_width))
        img_A[0, :, :] = x_[-1, ::]
        img_A[1, :, :] = taper
    elif type_of_input == '1dv_1init':
        img_A = np.zeros((2, opt.img_height, opt.img_width))
        model_initial=np.squeeze(models_init_scaled)
        model_initial=imresize(model_initial,[opt.img_height,opt.img_width])
        img_A[0, :, :] = x_[-1, ::]
        img_A[1, :, :] = model_initial
    elif type_of_input == '1dv_1init_1sign':
        model_initial=np.squeeze(models_init_scaled)
        model_initial=imresize(model_initial,[opt.img_height,opt.img_width])
        img_A[0, :, :] = x_[-1, ::]
        img_A[1, :, :] =model_initial
        img_A[2, :, :]=np.sign(x_[-1, ::])
    elif type_of_input == '1grad_1dv':
        img_A = np.zeros((2, opt.img_height, opt.img_width))
        img_A[0, :, :] = x_[-2, ::]
        img_A[1, :, :] = x_[-1, ::]
    elif type_of_input == 'allgrad_1dv':
        img_A = x_[-opt.channels:, ::]
    elif type_of_input == 'only_grads':
        x_ = x_[0:-1, ::]  # select only gradients for input
        selected_channels = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 49]
        img_A = x_[selected_channels, ::]  # variant 1
    elif type_of_input == 'dm_i':
        img_A=np.empty((dm_i_ra_scaled.shape[0],opt.img_height, opt.img_width))
        for i in range(img_A.shape[0]):
            img_A[i,::]=imresize(dm_i_ra_scaled[i,::].squeeze(),[opt.img_height,opt.img_width])
    elif type_of_input == 'm_i':
        img_A=np.empty((m_i_ra_scaled.shape[0],opt.img_height, opt.img_width))
        for i in range(img_A.shape[0]):
            img_A[i,::]=imresize(m_i_ra_scaled[i,::].squeeze(),[opt.img_height,opt.img_width])
    elif type_of_input == 'dm_i_1init_1sign_1taper':
        for i in range(10):
            img_A[i,::]=imresize(dm_i_ra_scaled[i,::].squeeze(),[opt.img_height,opt.img_width])
        img_A[10, :, :]=model_initial
        img_A[11, :, :]=np.sign(x_[-1, ::])
        img_A[12, :, :]=taper
    ################################    record t
    img_B = np.zeros((1,opt.img_height,opt.img_width))
    if type_of_input == '1_fwi_res' or type_of_input == 'm_i':
        img_B[0,:,:]=imresize(smoothed_true_model_scaled.squeeze(),[opt.img_height,opt.img_width])
    else:
        t = np.squeeze(t)
        t = imresize(t, [opt.img_height,opt.img_width])
        img_B[0, :, :] = t
    taper=np.expand_dims(taper,axis=0)
    return [img_A,img_B,taper]

def extract_ml_data_to_list2(opt,dataset,full_name):
    batch=dataset._getitem_by_name(full_name)
    taper=np.expand_dims(batch['taper'],axis=0)
    return [batch['A'],batch['B'],taper]


def scaling_data(data,scaling_constants_dict,data_type,scaling_range='-11'):
    max_=scaling_constants_dict[data_type][0]
    min_=scaling_constants_dict[data_type][1]
    mean_=scaling_constants_dict[data_type][2]
    std_=scaling_constants_dict[data_type][3]
    data_=np.squeeze(data)
    if scaling_range=='-11':
        data_=(data_-min_)/(max_-min_)
        data_=data_*2-1
    elif scaling_range=='standardization':
        data_=(data_-mean_)/(std_)
    ####### plotting
    visualize_scaling_results=0
    if visualize_scaling_results==1:
        save_pictures_path='./pictures_for_check'
        subtitles=['x','initial model','depth matrix','target'];    Name=''
        fig = plt.figure()
        fig.suptitle(Name)
        gs = fig.add_gridspec(nrows=len(data_processing_history),ncols=1,hspace=0.2,wspace=0)
        axs = gs.subplots(sharex=True, sharey=True)
        for i in range(len(data_processing_history)):
            ax=axs[i];    pt=ax.imshow(data_processing_history[i].T);     
            fig.colorbar(pt,ax=ax);     ax.title.set_text('stage '+str(i))
        print('saving picture to '+save_pictures_path)
        plt.savefig(save_pictures_path,bbox_inches='tight')
        plt.show()  # plt.show(block=False)
        plt.close()
    ####### in my notation data should be 4-channel
    if data_.ndim==2:
        data_=np.expand_dims(data_,axis=0) 
        data_=np.expand_dims(data_,axis=-1)
    return data_
def scaling_data_back(data,scaling_constants_dict,data_type,scaling_range='-11'):
    data_=np.squeeze(data);  
    max_=scaling_constants_dict[data_type][0]
    min_=scaling_constants_dict[data_type][1]
    mean_=scaling_constants_dict[data_type][2]
    std_=scaling_constants_dict[data_type][3]
    if scaling_range=='-11':
        data_=(data_+1)/2
        data_=data_*(max_-min_)+min_
    elif scaling_range=='standardization':
        data_=data_*std_ +mean_
    if data_.ndim==2:
        data_=np.expand_dims(data_,axis=0) 
        data_=np.expand_dims(data_,axis=-1)
    return data_

def numstr(x):
    string = str('{0:.2f}'.format(x))
    return string

def numstr3(x):
    string = str('{0:.3f}'.format(x))
    return string
def numstr4(x):
    string = str('{0:.4f}'.format(x))
    return string

def load_numpy_file(name):
    with open(name, 'rb') as f:
        data = np.load(f, allow_pickle=True)
        input_data = data['input_data']
        output_data = data['output_data']
        dx = data['dx']
        data.close()
    return input_data, output_data, dx


class Pix2Pix():
    def __init__(self, Save_pictures_path, log_save_const):
        # Input shape
        self.img_rows = 256
        self.img_cols = 256
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.save_path = Save_pictures_path
        self.log_save_const = log_save_const
        self.keras_models_path = './keras_models'
        os.makedirs(self.keras_models_path, exist_ok=True)
        self.training_proceeded = 0
        # Configure data loader
        self.dataset_path = './datasets'
        self.dataset_name = 'facades'
        # self.dataset_path='/ibex/scratch/plotnips/intel/MWE/datasets/'
        self.dataset_path = '/ibex/scratch/projects/c2107/MWE/datasets'
        self.dataset_name = 'dataset_vl_gen_scaled_287145_smooth_450m_scaler_picture_max_abs'

        self.dataset_path = '/home/plotnips/Dropbox/Log_extrapolation/scripts/MWE/datasets'
        self.dataset_name = 'dataset_vl_gen_scaled_130_smooth_450m_scaler_picture_max_abs'
        self.dataset_name = 'dataset_vl_gen_scaled_3000_smooth_450m_scaler_picture_-11'

        self.dataset_path = '/ibex/scratch/plotnips/intel/MWE/datasets/'
        # self.dataset_name='dataset_vl_gen_scaled_60000_smooth_450m_scaler_picture_01'
        self.dataset_name = 'dataset_vl_gen_scaled_3000_smooth_450m_scaler_picture_individual_scaling'
        self.dataset_name = 'dataset_vl_gen_scaled_287145_smooth_450m_6860_scaler_picture_individual_scaling'
        # self.dataset_name='dataset_vl_gen_scaled_3005_smooth_450m_scaler_picture_individual_scaling_true'
        self.dataset_name = 'dataset_vl_gen_scaled_3005_smooth_450m_scaler_picture_individual_scaling_false'
        self.dataset_name = 'dataset_vl_gen_scaled_287145_smooth_450m_scaler_picture_individual_scaling_false'

        # self.dataset_name='dataset_vl_gen_scaled_3000_smooth_450m_scaler_picture_01'
        print('saving data to '+self.dataset_path+self.dataset_name)
        path = glob('%s%s/*' % (self.dataset_path, self.dataset_name))
        path = fnmatch.filter(sorted(path), '*.npz')
        path_test = path[-10:]
        path = list(set(path)-set(path_test))
        self.N = len(path)
        self.N = 50
        self.train_frac = 0.9
        path = path[0:self.N]
        # path=random.sample(path,len(path))    # randomize??
        path_train = path[0:int(len(path)*self.train_frac)]
        print(path_train[0:10])
        path_valid = list(set(path)-set(path_train))
        self.path_train = path_train
        self.path_test = path_test
        self.path_valid = path_valid
        print('Models for training:',  len(path_train))
        print('Models for validation:', len(path_valid))
        print('Models for testing:',   len(path_test))
        self.data_loader = DataLoader(dataset_name=self.dataset_name,
                                      dataset_path=self.dataset_path,
                                      path_train=self.path_train,
                                      path_valid=self.path_valid,
                                      path_test=self.path_test,
                                      img_res=(self.img_rows, self.img_cols),
                                      save_pictures_path=Save_pictures_path)
        input_data, output_data, dx = load_numpy_file(path[0])
        self.orig_input_shape = input_data.shape
        self.orig_output_shape = output_data.shape
        # Calculate output shape of D (PatchGAN)
        patch = int(self.img_rows / 2**4)
        self.disc_patch = (patch, patch, 1)
        # Number of filters in the first layer of G and D
        self.gf = 6
        self.df = 6
        # optimizer = Adam(0.0002, 0.5)
        optimizer = Adam(0.02, 0.5)
        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='mse',
                                   optimizer=optimizer,
                                   metrics=['accuracy'])
        # -------------------------
        # Construct Computational
        #   Graph of Generator
        # -------------------------
        # Build the generator
        self.generator = self.build_generator()
        # Input images and their conditioning images
        img_A = Input(shape=self.img_shape)
        img_B = Input(shape=self.img_shape)
        # By conditioning on B generate a fake version of A
        fake_A = self.generator(img_B)
        # For the combined model we will only train the generator
        self.discriminator.trainable = False
        # Discriminators determines validity of translated images / condition pairs
        valid = self.discriminator([fake_A, img_B])
        self.combined = Model(inputs=[img_A, img_B], outputs=[valid, fake_A])
        self.combined.compile(loss=['mse', 'mae'],  # mae in second position????
                              loss_weights=[1, 100],
                              optimizer=optimizer)
        print('generator.summary()')
        self.generator.summary()
        print('discriminator.summary()')
        self.discriminator.summary()
        # print('combined.summary()')
        # self.combined.summary()
        print(optimizer._hyper)
        print('gf=', self.gf)
        print('df=', self.df)
        aa = 1

    def build_generator(self):
        """U-Net Generator"""
        def conv2d(layer_input, filters, f_size=4, bn=True):
            """Layers used during downsampling"""
            d = Conv2D(filters, kernel_size=f_size,
                       strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            return d

        def deconv2d(layer_input, skip_input, filters, f_size=4, dropout_rate=0):
            """Layers used during upsampling"""
            u = UpSampling2D(size=2)(layer_input)
            u = Conv2D(filters, kernel_size=f_size, strides=1,
                       padding='same', activation='relu')(u)
            if dropout_rate:
                u = Dropout(dropout_rate)(u)
            u = BatchNormalization(momentum=0.8)(u)
            u = Concatenate()([u, skip_input])
            return u
        # Image input
        d0 = Input(shape=self.img_shape)
        # Downsampling
        d1 = conv2d(d0, self.gf, bn=False)
        d2 = conv2d(d1, self.gf*2)
        d3 = conv2d(d2, self.gf*4)
        d4 = conv2d(d3, self.gf*8)
        d5 = conv2d(d4, self.gf*8)
        d6 = conv2d(d5, self.gf*8)
        d7 = conv2d(d6, self.gf*8)
        # Upsampling
        u1 = deconv2d(d7, d6, self.gf*8)
        u2 = deconv2d(u1, d5, self.gf*8)
        u3 = deconv2d(u2, d4, self.gf*8)
        u4 = deconv2d(u3, d3, self.gf*4)
        u5 = deconv2d(u4, d2, self.gf*2)
        u6 = deconv2d(u5, d1, self.gf)
        u7 = UpSampling2D(size=2)(u6)
        # output_img = Conv2D(self.channels, kernel_size=4, strides=1, padding='same', activation='linear')(u7)
        output_img = Conv2D(self.channels, kernel_size=4,
                            strides=1, padding='same', activation='relu')(u7)
        return Model(d0, output_img)

    def build_discriminator(self):
        def d_layer(layer_input, filters, f_size=4, bn=True):
            """Discriminator layer"""
            d = Conv2D(filters, kernel_size=f_size,
                       strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            return d
        img_A = Input(shape=self.img_shape)
        img_B = Input(shape=self.img_shape)
        # Concatenate image and conditioning image by channels to produce input
        combined_imgs = Concatenate(axis=-1)([img_A, img_B])
        d1 = d_layer(combined_imgs, self.df, bn=False)
        d2 = d_layer(d1, self.df*2)
        d3 = d_layer(d2, self.df*4)
        d4 = d_layer(d3, self.df*8)
        final = d4
        validity = Conv2D(1, kernel_size=4, strides=1, padding='same')(final)
        return Model([img_A, img_B], validity)

    def train(self, epochs, batch_size=1, sample_interval=1, plotting_interval=1):
        print('epochs=',    epochs)
        print('batch_size=', batch_size)
        print("Training parameters, epochs:%d, batch_size:%d, sample_interval:%d" % (
            epochs, batch_size, sample_interval))
        start_time = datetime.datetime.now()
        # Adversarial loss ground truths
        valid = np.ones((batch_size,) + self.disc_patch)
        fake = np.zeros((batch_size,) + self.disc_patch)
        # aa=enumerate(self.data_loader.load_batch(batch_size))
        aa = self.data_loader.load_data(batch_size)
        g_loss_history = []
        d_loss_history = []
        g_accuracy_history = []
        d_accuracy_history = []
        for epoch in range(epochs):
            for batch_i, (imgs_A, imgs_B) in enumerate(self.data_loader.load_batch(batch_size)):
                # ---------------------
                #  Train Discriminator
                # ---------------------
                # Condition on B and generate a translated version
                fake_A = self.generator.predict(imgs_B)
                # Train the discriminators (original images = real / generated = Fake)
                d_loss_real = self.discriminator.train_on_batch(
                    [imgs_A, imgs_B], valid)
                d_loss_fake = self.discriminator.train_on_batch(
                    [fake_A, imgs_B], fake)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
                # -----------------
                #  Train Generator
                # -----------------
                # Train the generators
                g_loss = self.combined.train_on_batch(
                    [imgs_A, imgs_B], [valid, imgs_A])
                elapsed_time = datetime.datetime.now() - start_time
                # Plot the progress
                print("[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %3d%%] [G loss: %f] [G accuracy: %f] time: %s" % (epoch, epochs,
                                                                                                                       batch_i, self.data_loader.n_batches,
                                                                                                                       d_loss[0], 100 *
                                                                                                                       d_loss[1],
                                                                                                                       g_loss[0], 100 *
                                                                                                                       g_loss[1],
                                                                                                                       elapsed_time))
                # If at save interval => save generated model
            g_loss_history.append(g_loss[0])
            d_loss_history.append(d_loss[0])
            g_accuracy_history.append(100*g_loss[1])
            d_accuracy_history.append(100*d_loss[1])
            if epoch % sample_interval == 0:
                # self.combined.save_weights( self.keras_models_path+'/model_combined'+str(self.log_save_const)+'epoch_'+str(epoch)+'.hdf5')
                self.generator.save_weights(
                    self.keras_models_path+'/generator'+str(self.log_save_const)+'epoch_' + str(epoch)+'.hdf5')
            if epoch % plotting_interval == 0:
                # self.sample_images(0,1,original_plotting=1)
                self.sample_list(epoch, self.path_train[0:1])
        history = {'g_loss_history': g_loss_history, 'd_loss_history': d_loss_history,
                   'g_accuracy_history': g_accuracy_history, 'd_accuracy_history': d_accuracy_history}
        Plot_loss(history, Title='log' + str(self.log_save_const) +
                  'loss_mse', Save_pictures_path=self.save_path, Save_flag=1)
        Plot_accuracy2(history, Title='log' + str(self.log_save_const) +
                       'r2accuracy', Save_pictures_path=self.save_path, Save_flag=1)
        print('saving model')
        self.combined.save_weights(
            self.keras_models_path+'/model_combined'+str(self.log_save_const)+'.hdf5')  # +str()
        self.generator.save_weights(
            self.keras_models_path+'/generator'+str(self.log_save_const)+'.hdf5')  # +str()
        self.training_proceeded = 1

    def sample_images(self, epoch, batch_i, original_plotting=0):
        original_plotting = 0
        if original_plotting == 1:
            os.makedirs('images/%s' % self.dataset_name, exist_ok=True)
            r, c = 3, 3
            imgs_A, imgs_B = self.data_loader.load_data(
                batch_size=3, is_testing=True)
            fake_A = self.generator.predict(imgs_B)
            gen_imgs = np.concatenate([imgs_B, fake_A, imgs_A])
            titles = ['Condition', 'Generated', 'Original']
            fig, axs = plt.subplots(r, c)
            cnt = 0
            for i in range(r):
                for j in range(c):
                    mat = gen_imgs[cnt, :, :, 0].T
                    im = axs[i, j].imshow(mat)
                    axs[i, j].set_title(titles[i])
                    axs[i, j].axis('off')
                    cbar = fig.colorbar(im, extend='both',
                                        shrink=0.9, ax=axs[i, j])
                    cnt += 1
            fig.savefig(self.save_path+"/%d_%d.png" % (epoch, batch_i))
            plt.close()
        else:
            flag_show_scaled_data = 1
            # list_train=self.path_train[0::];list_valid=self.path_valid[0:1];list_test=self.path_test[0::]
            list_train = self.path_train[0:7]
            list_valid = self.path_valid[0:2]
            list_test = self.path_test[0::]
            list_all = list_test+list_train+list_valid
            [x_all_, t_all_] = self.data_loader.record_dataset_spec_ids(
                list_all)
            t_predicted_all_ = self.generator.predict(x_all_, verbose=1)
            t_predicted_all = np.zeros((len(list_all), self.orig_output_shape[1],
                                       self.orig_output_shape[2], 1))
            t_all = np.zeros((len(list_all), self.orig_output_shape[1],
                             self.orig_output_shape[2], 1))
            x_all = np.zeros((len(list_all), self.orig_input_shape[1],
                             self.orig_input_shape[2], 1))
            for i_x, NAME in enumerate(list_all):
                t_predicted_all[i_x, :, :, 0] = imresize(t_predicted_all_[i_x, :, :, 0],
                                                         (self.orig_output_shape[1], self.orig_output_shape[2]))
                t_all[i_x, :, :, 0] = imresize(t_all_[i_x, :, :, 0],
                                               (self.orig_output_shape[1], self.orig_output_shape[2]))
                x_all[i_x, :, :, 0] = imresize(x_all_[i_x, :, :, 0],
                                               (self.orig_input_shape[1], self.orig_input_shape[2]))
                with open(NAME, 'rb') as f:
                    data = np.load(f, allow_pickle=True)
                    M0 = data['models'][0, :, :, 0]
                    dz = data['dz']
                    dx = data['dx']
                    Minit = data['models_init'][0, :, :, 0]
                    if 'scaler_x' in data.keys():
                        scaler_type = '_individual_scaling'
                        scaler_x = data['scaler_x']
                        scaler_t = data['scaler_t']
                    else:
                        from joblib import dump, load
                        scaler_type = '_1_scaler'
                        scaler_x = load(data_path+'/scaler_x.bin')
                        scaler_t = load(data_path+'/scaler_t.bin')
                    data.close()
                M1 = x_all[i_x, :, :, :]
                M1 = np.expand_dims(M1, axis=0)
                M2 = t_all[i_x, :, :, :]
                M2 = np.expand_dims(M2, axis=0)
                M3 = t_predicted_all[i_x, :, :, :]
                M3 = np.expand_dims(M3, axis=0)
                if flag_show_scaled_data == 0:
                    M1 = transforming_data_inverse(M1, scaler_x)
                    M2 = transforming_data_inverse(M2, scaler_t)
                    M3 = transforming_data_inverse(M3, scaler_t)
                    Predicted_update = imresize(
                        M3[0, :, :, 0], [M1.shape[1], M1.shape[2]])
                    True_update = M2
                else:
                    tmp = transforming_data_inverse(M3, scaler_t)
                    Predicted_update = tmp[0, :, :, 0]
                    True_update = transforming_data_inverse(M2, scaler_t)
                True_update = True_update[0, :, :, 0]
                M1 = M1[0, :, :, 0]
                M2 = M2[0, :, :, 0]
                M3 = M3[0, :, :, 0]
                Models_init = Minit
                if Models_init.shape != Predicted_update.shape:
                    Predicted_update = imresize(
                        Predicted_update, Models_init.shape)
                if Models_init.shape != True_update.shape:
                    True_update = imresize(True_update, Models_init.shape)
                testing_model = Models_init+Predicted_update
                ideal_init_model = Models_init+True_update
                M0_show = M0
                # Crop testing models for better visualization
                water = np.ones((M0_show.shape[0], 18))*1500
                M0_show = np.concatenate([water, M0_show], axis=1)
                testing_model = np.concatenate([water, testing_model], axis=1)
                ideal_init_model = np.concatenate(
                    [water, ideal_init_model], axis=1)
                inp_orig_sizes = [M1, M2, M3, testing_model, M0_show]
                pics_6 = [M1, M2, M3, M3-M2, testing_model, ideal_init_model]
                # inp_orig_sizes=[M1,M2,M3,testing_model,ideal_init_model]
                saving_name = NAME.split('augmented_marmousi_10_it')[-1]
                saving_name = saving_name.split('.npy')[0]
                # Plot_image(testing_model.T,Show_flag=1,Save_flag=1,Title='testing_model1'+saving_name,Aspect='equal',Save_pictures_path=Save_pictures_path)
                ####
                Prediction_accuracy = F_r2(M3, M2)
                R2val = F_r2(testing_model, M0_show)
                # R2val=F_r2(pics_6[-2],pics_6[-1])
                # R2val2=F_r2(testing_model,ideal_init_model)
                if NAME in list_train:
                    data_type = 'Train'
                elif NAME in list_test:
                    data_type = 'Test'
                    tmp2 = NAME.split('augmented_marmousi')
                    path = self.save_path + '/' + \
                        tmp2[1][0:-4]+'_weights_'+str(self.log_save_const)
                    np.savez(path, input_data=M1, output_data=M2,
                             models_init=Models_init, models=M0, predicted_update=Predicted_update, dx=dx, dz=dz)
                elif NAME in list_valid:
                    data_type = 'Valid'
                tmp = NAME.split('augmented_marmousi_10_it')[-1]
                tmp = tmp.split('.npz')[0]
                if NAME in list_test:
                    data_type = '_' + data_type+tmp + \
                        '_'+numstr(Prediction_accuracy)
                    title = 'Prediction, R2(prediction, target) = ' + \
                        numstr(Prediction_accuracy)
                else:
                    data_type = '_' + data_type+tmp + \
                        '_'+numstr(Prediction_accuracy)
                    title = 'Prediction, R2(prediction, target) = ' + \
                        numstr(Prediction_accuracy)
                Name = self.save_path + '/' + 'log' + \
                    str(self.log_save_const) + data_type+'.png'
                Name = self.save_path + '/' + 'log' + \
                    str(self.log_save_const) + data_type+'_6pics'+'.png'
                #   PLOT_ML_Result_fixed_colorbar
                PLOT_ML_Result_adaptive_colorbar(pics_6, numstr(R2val), history_flag=0,
                    history=None, Boundaries=[], save_file_path=Name,
                    dx=dx, dy=dz, Title=title, Title2='', Save_flag=1, adaptive_colorbar=3)
                i_x = i_x+1
            #################################
            aa = 1
            # imgs_A,imgs_B=self.data_loader.load_data(batch_size=3,is_testing=True)
            # fake_A = self.generator.predict(imgs_B)

    def sample_list(self, epoch, sample_list):
        flag_show_scaled_data = 1
        list_all = sample_list
        [x_all_, t_all_] = self.data_loader.record_dataset_spec_ids(list_all)
        t_predicted_all_ = self.generator.predict(x_all_, verbose=1)
        t_predicted_all = np.zeros((len(list_all), self.orig_output_shape[1],
                                   self.orig_output_shape[2], 1))
        t_all = np.zeros((len(list_all), self.orig_output_shape[1],
                         self.orig_output_shape[2], 1))
        x_all = np.zeros((len(list_all), self.orig_input_shape[1],
                         self.orig_input_shape[2], 1))
        for i_x, NAME in enumerate(list_all):
            t_predicted_all[i_x, :, :, 0] = imresize(t_predicted_all_[i_x, :, :, 0],
                                                     (self.orig_output_shape[1], self.orig_output_shape[2]))
            t_all[i_x, :, :, 0] = imresize(t_all_[i_x, :, :, 0],
                                           (self.orig_output_shape[1], self.orig_output_shape[2]))
            x_all[i_x, :, :, 0] = imresize(x_all_[i_x, :, :, 0],
                                           (self.orig_input_shape[1], self.orig_input_shape[2]))
            with open(NAME, 'rb') as f:
                data = np.load(f, allow_pickle=True)
                M0 = data['models'][0, :, :, 0]
                dz = data['dz']
                dx = data['dx']
                Minit = data['models_init'][0, :, :, 0]
                if 'scaler_x' in data.keys():
                    scaler_type = '_individual_scaling'
                    scaler_x = data['scaler_x']
                    scaler_t = data['scaler_t']
                else:
                    from joblib import dump, load
                    scaler_type = '_1_scaler'
                    scaler_x = load(data_path+'/scaler_x.bin')
                    scaler_t = load(data_path+'/scaler_t.bin')
                data.close()
            M1 = x_all[i_x, :, :, :]
            M1 = np.expand_dims(M1, axis=0)
            M2 = t_all[i_x, :, :, :]
            M2 = np.expand_dims(M2, axis=0)
            M3 = t_predicted_all[i_x, :, :, :]
            M3 = np.expand_dims(M3, axis=0)
            if flag_show_scaled_data == 0:
                M1 = transforming_data_inverse(M1, scaler_x)
                M2 = transforming_data_inverse(M2, scaler_t)
                M3 = transforming_data_inverse(M3, scaler_t)
                Predicted_update = imresize(
                    M3[0, :, :, 0], [M1.shape[1], M1.shape[2]])
                True_update = M2
            else:
                tmp = transforming_data_inverse(M3, scaler_t)
                Predicted_update = tmp[0, :, :, 0]
                True_update = transforming_data_inverse(M2, scaler_t)
            True_update = True_update[0, :, :, 0]
            M1 = M1[0, :, :, 0]
            M2 = M2[0, :, :, 0]
            M3 = M3[0, :, :, 0]
            Models_init = Minit
            if Models_init.shape != Predicted_update.shape:
                Predicted_update = imresize(
                    Predicted_update, Models_init.shape)
            if Models_init.shape != True_update.shape:
                True_update = imresize(True_update, Models_init.shape)
            testing_model = Models_init+Predicted_update
            ideal_init_model = Models_init+True_update
            M0_show = M0
            # Crop testing models for better visualization
            water = np.ones((M0_show.shape[0], 18))*1500
            M0_show = np.concatenate([water, M0_show], axis=1)
            testing_model = np.concatenate([water, testing_model], axis=1)
            ideal_init_model = np.concatenate(
                [water, ideal_init_model], axis=1)
            inp_orig_sizes = [M1, M2, M3, testing_model, M0_show]
            pics_6 = [M1, M2, M3, M3-M2, testing_model, ideal_init_model]
            # inp_orig_sizes=[M1,M2,M3,testing_model,ideal_init_model]
            saving_name = NAME.split('augmented_marmousi_10_it')[-1]
            saving_name = saving_name.split('.npy')[0]
            # Plot_image(testing_model.T,Show_flag=1,Save_flag=1,Title='testing_model1'+saving_name,Aspect='equal',Save_pictures_path=Save_pictures_path)
            ####
            Prediction_accuracy = F_r2(M3, M2)
            R2val = F_r2(testing_model, M0_show)
            # R2val=F_r2(pics_6[-2],pics_6[-1])
            # R2val2=F_r2(testing_model,ideal_init_model)
            tmp = NAME.split('augmented_marmousi_10_it')[-1]
            tmp = tmp.split('.npz')[0]
            data_type = tmp
            title = 'Prediction, R2(prediction, target) = ' + \
                numstr(Prediction_accuracy)
            Name = self.save_path+'/'+'log' + \
                str(self.log_save_const) + data_type+'_epoch_' + \
                str(epoch)+'_'+numstr(Prediction_accuracy)+'.png'
            #   PLOT_ML_Result_fixed_colorbar
            PLOT_ML_Result_adaptive_colorbar(pics_6, numstr(R2val), history_flag=0,
                history=None, Boundaries=[], save_file_path=Name,
                dx=dx, dy=dz, Title=title, Title2='', Save_flag=1, adaptive_colorbar=3)
            i_x=i_x+1

    def inference(self, Model_to_load_const, model_name='', batch_size=1):
        if model_name == '':
            model_name = 'generator'+str(Model_to_load_const)+'.hdf5'
        if self.training_proceeded == 1:
            imgs_A, imgs_B = self.data_loader.load_data(
                batch_size=3, is_testing=True)
            fake_A = self.generator.predict(imgs_B)
            aa = 1
        else:
            print('model loaded=', model_name)
            self.generator.load_weights(
                self.keras_models_path+'/'+model_name, by_name=1)
            # self.combined.load_weights(self.keras_models_path+'/model_combined'+str(Model_to_load_const)+'.hdf5')
        # history = F_load_history_from_file(Keras_models_path,Model_to_load_const)


class DataLoader_tf():
    def __init__(self, dataset_name, path_train,
                 path_valid, path_test, dataset_path, img_res=(128, 128), save_pictures_path=''):
        self.dataset_name = dataset_name
        self.dataset_path = dataset_path
        self.path_train = path_train
        self.path_valid = path_valid
        self.path_test = path_test
        self.img_res = img_res
        self.save_pictures_path = save_pictures_path

    def unpack_npz_file(self, path):
        if os.stat(path).st_size == 0:
            print('File is empty')
            os.remove(path)
            return None
        with open(path, 'rb') as f:
            data = np.load(f, allow_pickle=True)
            x = data['input_data']
            t = data['output_data']
            data.close()
        x = np.squeeze(x)
        t = np.squeeze(t)
        # Plot_image(x.T,Show_flag=1,Save_flag=1,Title='x',
        #     Aspect='equal',Save_pictures_path=self.save_pictures_path)
        # Plot_image(t.T,Show_flag=1,Save_flag=1,Title='t',
        #     Aspect='equal',Save_pictures_path=self.save_pictures_path)
        x = imresize(x, [self.img_res[0], self.img_res[1]])
        t = imresize(t, [self.img_res[0], self.img_res[1]])
        # Plot_image(x.T,Show_flag=1,Save_flag=1,Title='x_imresized',
        #     Aspect='equal',Save_pictures_path=self.save_pictures_path)
        # Plot_image(t.T,Show_flag=1,Save_flag=1,Title='t_imresized',
        #     Aspect='equal',Save_pictures_path=self.save_pictures_path)
        x = np.expand_dims(x, axis=0)
        x = np.expand_dims(x, axis=-1)
        t = np.expand_dims(t, axis=0)
        t = np.expand_dims(t, axis=-1)
        return x, t

    def load_data(self, batch_size=1, is_testing=False):
        data_type = "train" if not is_testing else "test"
        # path = glob('%s/%s/%s/*' % (self.dataset_path,self.dataset_name, data_type))
        if data_type == 'test':
            path = self.path_test
        elif data_type == 'train':
            path = self.path_train
        # batch_images = np.random.choice(path, size=batch_size)
        # batch_images2=np.random.choice(np.arange(100), size=batch_size)
        batch_images = random.sample(path, batch_size)
        imgs_A = []
        imgs_B = []
        for img_path in batch_images:
            x, t = self.unpack_npz_file(img_path)
            img_A = t[0, ::]
            img_B = x[0, ::]
            imgs_A.append(img_A)
            imgs_B.append(img_B)
        imgs_A = np.array(imgs_A)
        imgs_B = np.array(imgs_B)
        return imgs_A, imgs_B

    def load_batch(self, batch_size=1, is_testing=False):
        #   same as load_data function
        data_type = "train" if not is_testing else "val"
        # path = glob('./datasets/%s/%s/*' % (self.dataset_name, data_type))
        if data_type == 'test':
            path = self.path_test
        elif data_type == 'train':
            path = self.path_train
        self.n_batches = int(len(path) / batch_size)
        for i in range(self.n_batches-1):
            batch = path[i*batch_size:(i+1)*batch_size]
            imgs_A, imgs_B = [], []
            for img in batch:
                x, t = self.unpack_npz_file(img)
                img_A = t[0, ::]
                img_B = x[0, ::]
                imgs_A.append(img_A)
                imgs_B.append(img_B)
            imgs_A = np.array(imgs_A)
            imgs_B = np.array(imgs_B)
            yield imgs_A, imgs_B

    def imread(self, path):
        return scipy.misc.imread(path, mode='RGB').astype(np.float)

    def record_dataset_spec_ids(self, file_names):
        out_x = np.empty(
            (len(file_names), self.img_res[0], self.img_res[1], 1))
        out_t = np.empty(
            (len(file_names), self.img_res[0], self.img_res[1], 1))
        for count, name in enumerate(file_names):
            x, t = self.unpack_npz_file(name)
            out_x[count, ::] = x[0, ::]
            out_t[count, ::] = t[0, ::]
        return out_x, out_t

# torch code

def calculate_file_max(file_):
    # print(file_)
    data = np.load(file_, allow_pickle=True)
    models_init = data['models_init']
    data.close()
    file_max = np.max(models_init)
    return file_max

class ImageDataset(Dataset):
    def __init__(self, opt, transforms_=None, mode="train", file_list=None,
                 load_precondition_data=1, max_for_initial_models=5000):
        self.transform = transforms.Compose(transforms_)
        # self.files = sorted(glob.glob(os.path.join(root, mode) + "/*.*"))
        self.opt = opt
        self.dataset_path = opt.dataset_path
        self.max_for_initial_models = max_for_initial_models
        self.load_precondition_data = load_precondition_data
        if mode == "train":
            self.files = opt.path_train
        elif mode == "val":
            self.files = opt.path_valid
        elif mode == "test":
            # self.files=opt.path_test+opt.path_train[0:3]+opt.path_valid[0:1]
            self.files = opt.path_test
        elif mode == "all":
            self.files = opt.path_test+opt.path_train+opt.path_valid
        elif mode == None:
            self.files = file_list
        else:
            self.files = file_list

    def calculate_max_model_init(self):
        const = 0
        flag_single_thread_processing = 0
        if flag_single_thread_processing == 1:
            list_of_max = []
            for file_ in self.files:
                data = np.load(file_, allow_pickle=True)
                models_init = data['models_init']
                data.close()
                file_max = np.max(models_init)
                list_of_max.append(file_max)
        else:
            from functools import partial
            pool = multiprocessing.Pool(multiprocessing.cpu_count()-3)
            temp = partial(calculate_file_max)
            list_of_max = pool.map(func=temp, iterable=self.files)
            pool.close()
            pool.join()
        const = np.array(list_of_max).max()
        print('Maximum of initial models', const)
        return const

    def __getitem__(self, index):
        """ A-input data,B-target data,C-initial model,D-true model,E-input data in real amplitudes
        ,F-output data in real amplitudes,sc_t-scaler_t"""
        # print('get item from ',self.files[index % len(self.files)])
        Name=self.files[index % len(self.files)].split('/')[-1]
        Name=Name.split('.npz')[0];     # print(Name)
        with open(self.files[index % len(self.files)], 'rb') as f:
            data = np.load(f, allow_pickle=True)
            x = data['input_data']
            t = data['output_data']
            x_ra= data['input_data_real_amplitudes']
            t_ra= data['output_data_real_amplitudes']
            models_init =data['models_init']
            models_init_scaled=data['models_init_scaled']
            models = data['models']
            fwi_result_scaled=data['fwi_result_scaled']
            smoothed_true_model_scaled=data['smoothed_true_model_scaled']
            smoothed_true_model=data['smoothed_true_model']
            taper=data['taper']
            dm_i_ra=data['dm_i_ra']
            dm_i_ra_scaled=data['dm_i_ra_scaled']
            m_i_ra=data['m_i_ra']
            m_i_ra_scaled=data['m_i_ra_scaled']
            dz = data['dz']
            dx = data['dx']
            data.close()
        nch = x.shape[0]
        nx = models.shape[1]
        nz = models.shape[2]
        #######################
        taper=taper.squeeze()
        taper=imresize(taper,[self.opt.img_height, self.opt.img_width]);    taper[taper<1]=0;   # imresize introduces real number
        model_initial=np.squeeze(models_init_scaled)
        model_initial=imresize(model_initial,[self.opt.img_height,self.opt.img_width])
        #######################    A,E
        if x.ndim == 4:
            x = np.squeeze(x, axis=3)
            x_ra=np.squeeze(x_ra,axis=3)
        x_ = np.empty((nch, self.opt.img_height, self.opt.img_width))
        for i in range(nch):
            x_[i, ::] = imresize(x[i, ::], [self.opt.img_height, self.opt.img_width])
        if is_defined(self.opt.inp_channels_type):
            type_of_input = self.opt.inp_channels_type
        #######################
        img_A = np.zeros((self.opt.channels,self.opt.img_height, self.opt.img_width))
        if type_of_input == '1dv':
            img_A[0, :, :] = x_[-1, ::]
        elif type_of_input == '1_fwi_res':
            img_A[0,:,:]=imresize(fwi_result_scaled.squeeze(),[self.opt.img_height,self.opt.img_width])
        elif type_of_input == '1m_1taper':
            img_A[0, :, :] = x_[-1, ::]+model_initial
            img_A[1, :, :] = taper
        elif type_of_input == '1dv_1taper':
            img_A[0, :, :] = x_[-1, ::]
            img_A[1, :, :] = taper
        elif type_of_input == '1dv_1init':
            img_A[0, :, :] = x_[-1, ::]
            img_A[1, :, :] = model_initial
        elif type_of_input == '1dv_1init_1sign':
            img_A[0, :, :] = x_[-1, ::]
            img_A[1, :, :] =model_initial
            img_A[2, :, :]=np.sign(x_[-1, ::])
        elif type_of_input == 'dm_i_1init':
            for i in range(10):
                img_A[i,::]=imresize(dm_i_ra_scaled[i,::].squeeze(),[self.opt.img_height,self.opt.img_width])
            img_A[10, :, :]=model_initial
        elif type_of_input == 'dm_i_1taper':
            for i in range(10):
                img_A[i,::]=imresize(dm_i_ra_scaled[i,::].squeeze(),[self.opt.img_height,self.opt.img_width])
            img_A[10, :, :]=taper
        elif type_of_input == 'dm_i_1init_1taper':
            for i in range(10):
                img_A[i,::]=imresize(dm_i_ra_scaled[i,::].squeeze(),[self.opt.img_height,self.opt.img_width])
            img_A[10, :, :]=model_initial
            img_A[11, :, :]=taper
        elif type_of_input == 'dm_i_1init_1sign_1taper':
            for i in range(10):
                img_A[i,::]=imresize(dm_i_ra_scaled[i,::].squeeze(),[self.opt.img_height,self.opt.img_width])
            img_A[10, :, :]=model_initial
            img_A[11, :, :]=np.sign(x_[-1, ::])
            img_A[12, :, :]=taper
        elif type_of_input == '1grad_1dv':
            img_A[0, :, :] = x_[-2, ::]
            img_A[1, :, :] = x_[-1, ::]
        elif type_of_input == 'allgrad_1dv':
            img_A = x_[-self.opt.channels:, ::]
        elif type_of_input == 'only_grads':
            x_ = x_[0:-1, ::]  # select only gradients for input
            selected_channels = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 49]
            img_A = x_[selected_channels, ::]  # variant 1
        elif type_of_input == 'dm_i':
            for i in range(img_A.shape[0]):
                img_A[i,::]=imresize(dm_i_ra_scaled[i,::].squeeze(),[self.opt.img_height,self.opt.img_width])
        elif type_of_input == 'm_i':
            for i in range(img_A.shape[0]):
                img_A[i,::]=imresize(m_i_ra_scaled[i,::].squeeze(),[self.opt.img_height,self.opt.img_width])
        #######################
        img_E = np.zeros((1,nx,x.shape[2]))
        img_E[0, :, :] = x_ra[-1, ::]
        ########    B
        img_B = np.zeros((1,self.opt.img_height,self.opt.img_width))
        if type_of_input == '1_fwi_res' or type_of_input == 'm_i':
            img_B[0,:,:]=imresize(smoothed_true_model_scaled.squeeze(),[self.opt.img_height,self.opt.img_width])
        else:
            t = np.squeeze(t)
            t = imresize(t, [self.opt.img_height,self.opt.img_width])
            img_B[0, :, :] = t
        ########    F
        img_F = np.zeros((1,nx,x.shape[2]))
        t_ra=np.squeeze(t_ra)
        img_F[0, :, :]=t_ra
        ########    C
        img_С = np.zeros((1,nx,nz))
        models_init = np.squeeze(models_init)
        img_С[0, :, :] = models_init
        ########    D
        img_D = np.zeros((1,nx,nz))
        models = np.squeeze(models)
        img_D[0,:,:]=models
        ########
        if np.isnan(img_A).any():   print('img_A_has_nan')
        if np.isnan(img_B).any():   print('img_B_has_nan')
        return {"A": img_A, "B": img_B, "C": img_С, "D": img_D,"E":img_E,"F":img_F,"taper":taper}
        # return {"A": img_A, "B": img_B, "C": img_С, "D": img_D,"E":img_E,"F":img_F,"taper":taper,"scaling_constants_dict":scaling_constants_dict}
        # return {"A": img_A, "B": img_B, "C": img_С, "D": img_D,"E":img_E,"F":img_F,"sc_t":scaler_t,"sc_t":scaler_t,"taper":taper}
        # return {"A": img_A, "B": img_B, "C": img_С, "D": img_D,"E":img_E,"F":img_F,"sc_t":scaler_t,"sc_t":scaler_t}
    
    def _getitem_by_name(self, name):
        index = self.files.index(name)
        batch=self.__getitem__(index)
        return batch

    def __len__(self):
        return len(self.files)

# self.loader.dataset.tensors[index]=augmented_tensor
# rewrite like NoiseLoader_on_tensordataset#

class FlipLoader_on_tensordataset(torch.utils.data.Dataset):
    def __init__(self, dataset,opt,p=0.5):
        """Args:p(float): probability of flip, 0 means no flip, 1 means always flip"""
        self.dataset = dataset   # tensor_dataset
        self.p = p
        self.opt=opt
    def __getitem__(self,index):
        self.dataset.tensors[0].shape
        original_sample=[];
        for ii in self.dataset.tensors:
            original_sample.append(ii[index])
        augmented_sample=original_sample
        if np.random.rand()<self.p:
            for count,ii in enumerate(original_sample):
                tensor=ii.clone()
                augmented_tensor=ii.clone()
                for i in range(augmented_tensor.shape[0]):
                    augmented_tensor[i,:,:]=torch.flip(tensor[i,:,:],[0])   # horizontal flip
                # Plot_image(np.concatenate((augmented_tensor[i,:,:],tensor[i,:,:]),axis=1).T,Show_flag=0,Save_flag=1,Title='comparison_'+str(index),Aspect='equal',Save_pictures_path=self.opt.save_path)
                augmented_sample[count]=augmented_tensor
        augmented_sample=tuple(augmented_sample)
        return augmented_sample
    def __len__(self):
        return len(self.dataset)
class NoiseLoader_on_tensordataset(torch.utils.data.Dataset):
    """   add noise to cnn input (x)  """
    def __init__(self, dataset,opt,p=0.5,c=0.01):
        """Args:p(float): probability of flip, 0 means no flip, 1 means always flip"""
        self.dataset = dataset
        self.p = p
        self.c = c
        self.opt=opt
    def __getitem__(self, index):
        inp_channels_type=self.opt.inp_channels_type
        if hasattr(self.dataset,'dataset'):
            original_sample=self.dataset.__getitem__(index)
            original_sample=list(original_sample)
            ss=1
            sample=[]
            for ii in self.dataset.dataset.tensors:
                sample.append(ii[index])
        else:
            original_sample=[]
            for ii in self.dataset.tensors:
                original_sample.append(ii[index])
        #################################
        # self.dataset.dataset.tensors[0][0,0,::]
        # Plot_image(self.dataset.dataset.tensors[0][0,0,::].cpu().detach().numpy().T,Show_flag=0,Save_flag=1,Title='data_'+str(index),Aspect='equal',Save_pictures_path=self.opt.save_path)
        # Plot_image(self.dataset.dataset.tensors[1][0,0,::].cpu().detach().numpy().T,Show_flag=0,Save_flag=1,Title='data_'+str(index),Aspect='equal',Save_pictures_path=self.opt.save_path)
        # Plot_image(self.dataset.dataset.tensors[2][0,0,::].cpu().detach().numpy().T,Show_flag=0,Save_flag=1,Title='data_'+str(index),Aspect='equal',Save_pictures_path=self.opt.save_path)
        taper=original_sample[2].squeeze()
        augmented_sample=original_sample
        if np.random.rand()<self.p:
            for count,ii in enumerate(original_sample):
                tensor=ii.clone()
                orig_tensor=sample[count]
                augmented_tensor=ii.clone()
                if count==0:    # choose x tensor
                    if inp_channels_type=='dm_i' or inp_channels_type=='m_i':
                        channels_to_process=range(10)
                    elif inp_channels_type=='1m_1taper' or inp_channels_type=='1dv_1taper' or inp_channels_type=='1grad_1dv' or inp_channels_type=='1dv_1init':
                        channels_to_process=[0]
                    elif inp_channels_type=='1dv_1init_1sign':
                        channels_to_process=[0]
                    elif inp_channels_type=='dm_i_1init_1sign_1taper':
                        channels_to_process=range(10)
                    elif inp_channels_type=='dm_i_1taper' or inp_channels_type=='dm_i_1init' or inp_channels_type=='dm_i_1init_1taper':
                        channels_to_process=range(10)
                    else:
                        channels_to_process=[0]
                    augmentation_percent=self.c*np.random.rand()
                    dmin,dmax=tensor[-1,:,:].min(),tensor[-1,:,:].max()
                    # print('augmentation percent=',augmentation_percent)
                    # print('augmentation amplitude=',augmentation_percent*(dmax-dmin) )
                    for i in channels_to_process:
                        data=orig_tensor[i,:,:]
                        # data=tensor[i,:,:]
                        dmin,dmax=data.min(),data.max()
                        # https://stackoverflow.com/questions/44328530/how-to-get-a-uniform-distribution-in-a-range-r1-r2-in-pytorch  How to get a uniform distribution in a range [r1,r2] in PyTorch?
                        noise=(torch.rand(size=data.shape)-0.5)*2*(dmax-dmin)*augmentation_percent
                        # Plot_image(np.concatenate((augmented_tensor[i,:,:],tensor[i,:,:]),axis=1).T,Show_flag=0,Save_flag=1,Title='noise_before_taper_'+str(index),Aspect='equal',Save_pictures_path=self.opt.save_path)
                        noise2=noise.clone()
                        noise2[taper==0]=0
                        # Plot_image(np.concatenate((taper,noise2,noise,noise-noise2),axis=1).T,Show_flag=0,Save_flag=1,Title='noise_after_taper_'+str(index),Aspect='equal',Save_pictures_path=self.opt.save_path)
                        augmented_tensor[i,:,:]+=noise2
                        ################################
                        plotting_flag=1
                        if plotting_flag==1:
                            if i==channels_to_process[-1]:
                                print('augmentation percent=',augmentation_percent)
                                print('augmentation amplitude=',augmentation_percent*(dmax-dmin) )
                                # Plot_image(np.concatenate((data,augmented_tensor[i,:,:]),axis=1).T,Show_flag=0,Save_flag=1,
                                #     Title='noise_after_taper_'+str(index),Aspect='equal',Save_pictures_path=self.opt.save_path)
                                fig_size = plt.rcParams["figure.figsize"]
                                fig_size[0] = 12.4
                                fig_size[1] = 5.0   #height
                                plt.rcParams["figure.figsize"] = fig_size
                                n_row=2;    n_col=1
                                gs = gridspec.GridSpec(n_row,n_col)
                                gs.update(left=0.0, right=0.94, wspace=0.0, hspace=0.03)
                                # gs=GridSpec(n_row,n_col+1, width_ratios=[1,1,0.1], height_ratios=[1,1,1])
                                # gs.update(wspace=0.0,hspace=0.0)
                                ax=[None]*6
                                labels=['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o']
                                Fontsize=15
                                text_Fontsize=30
                                labelsize=14
                                a=torch.min(augmented_tensor[i,:,:])
                                b=torch.max(augmented_tensor[i,:,:])
                                a=-dmax;     b=dmax
                                fig=plt.figure()
                                ax[0]=fig.add_subplot(gs[0,0]);
                                ax[0].imshow(data.T,vmin=a,vmax=b)
                                ax[0].text(30,30, labels[0], fontsize=text_Fontsize,color = "black",weight="bold")
                                ax[0].tick_params(axis='y',labelsize=17)
                                ax[0].axes.xaxis.set_visible(False)
                                ax[1]=fig.add_subplot(gs[1,0]);
                                last_image=ax[1].imshow(augmented_tensor[i,:,:].T,vmin=a,vmax=b)
                                ax[1].text(30,30, labels[1], fontsize=text_Fontsize,color = "black",weight="bold")
                                ax[1].tick_params(axis='x',labelsize=17)
                                ax[1].tick_params(axis='y',labelsize=17)
                                cbar_ax = fig.add_axes([0.80, 0.12, 0.02, 0.75])
                                cbar=fig.colorbar(last_image,cax=cbar_ax)
                                # cbar.set_label('V (km/sec)',labelpad=-40, y=1.05, rotation=0)   #horizontalalignment='center'
                                # cbar.ax.set_title('V (m/sec)',fontsize=18,pad=9.3)
                                for t in cbar.ax.get_yticklabels():
                                    t.set_fontsize(16)
                                print('saving to '+os.path.join(self.opt.save_path,'noise_augmentation'+str(count)+'_'+str(augmentation_percent)+'.png'))
                                plt.savefig(os.path.join(self.opt.save_path,'noise_augmentation'+str(count)+'_'+str(augmentation_percent)+'.png'),dpi=300,bbox_inches='tight')
                                print('mean(data)=',torch.mean(data))
                                print('mean(augmented)=',torch.mean(augmented_tensor[i,:,:]))
                                ss=1
                    # Plot_image(np.concatenate((augmented_tensor[i,:,:],tensor[i,:,:]),axis=1).T,Show_flag=0,Save_flag=1,Title='comparison_'+str(index),Aspect='equal',Save_pictures_path=self.opt.save_path)
                    augmented_sample[count]=augmented_tensor
        augmented_sample=tuple(augmented_sample)
        return augmented_sample
    def __len__(self):
        return len(self.dataset)

class FlipLoader(Dataset):
    def __init__(self, loader, p=0.5):
        """
        Args:
            p(float): probability of flip, 0 means no flip, 1 means always flip
        """
        self.loader = loader
        self.p = p
        self.files = loader.files
        self.opt = loader.opt

    def __getitem__(self, index):
        data_dict = self.loader.__getitem__(index)
        if np.random.rand() < self.p:
            new_dict = {}
            for key, data in data_dict.items():
                new_dict[key] = np.flip(data, -2).copy()
            return new_dict
        return data_dict
    
    def __len__(self):
        return len(self.loader)
class NoiseLoader(Dataset):
    def __init__(self, loader, p=0.5):
        """
        Args:
            p(float): probability of flip, 0 means no flip, 1 means always flip
        """
        self.loader = loader
        self.p = p
        self.files = loader.files
        self.opt = loader.opt

    def __getitem__(self, index):
        data_dict = self.loader.__getitem__(index)
        if np.random.rand() < self.p:
            dmin,dmax=data.min(),data.max()
            noise=dmin+(np.random.rand()*(dmax-dmin))*np.random.rand(shape=data.shape)
            data_dict["A"]+=noise
        return data_dict
    
    def __len__(self):
        return len(self.loader)

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        # torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)
        # torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
##############################
# https://github.com/eriklindernoren/PyTorch-GAN/tree/master/implementations/pix2pix
#          generator U-NET
##############################

class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super(UNetDown, self).__init__()
        layers = [nn.Conv2d(in_size, out_size, 4, 2, 1, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_size))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        # print('size before layer',x.size())
        x = self.model(x)
        # print('size after layer',x.size())
        return x


class UNetUp(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0):
        super(UNetUp, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_size, out_size, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(out_size),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        # print('size before layer',x.size())
        x = self.model(x)
        # print('size after layer',x.size())
        x = torch.cat((x, skip_input), 1)
        # print('size after concatenation',skip_input.size())
        return x


class GeneratorUNet(nn.Module):
    """Unet with cropped deep layers  """

    def __init__(self, in_channels=1, out_channels=1, DROPOUT=0.5):
        # k=4;    print('Divide number of filters by k,',k)
        # print('DROPOUT,',DROPOUT)
        k = 1
        print('Divide number of filters by k,', k)
        print('DROPOUT=', DROPOUT)
        super(GeneratorUNet, self).__init__()
        self.down1 = UNetDown(in_channels, int(64/k), normalize=False)
        self.down2 = UNetDown(int(64/k), int(128/k))
        self.down3 = UNetDown(int(128/k), int(256/k))
        self.down4 = UNetDown(int(256/k), int(512/k), dropout=DROPOUT)
        self.down8 = UNetDown(int(512/k), int(512/k),
                              normalize=False, dropout=DROPOUT)
        self.up1 = UNetUp(int(512/k), int(512/k),  dropout=DROPOUT)
        self.up5 = UNetUp(int(1024/k), int(256/k))
        self.up6 = UNetUp(int(512/k), int(128/k))
        self.up7 = UNetUp(int(256/k), int(64/k))
        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(int(128/k), out_channels, 4, padding=1),
            nn.Tanh(),)

    def forward(self, x):
        # U-Net generator with skip connections from encoder to decoder
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d8 = self.down8(d4)
        u1 = self.up1(d8, d4)
        u5 = self.up5(u1, d3)
        u6 = self.up6(u5, d2)
        u7 = self.up7(u6, d1)
        return self.final(u7)


class GeneratorUNet_big(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, DROPOUT=0.5):
        # k=4;    print('Divide number of filters by k,',k)
        # print('DROPOUT,',DROPOUT)
        k = 4
        print('Divide number of filters by k,', k)
        print('DROPOUT=', DROPOUT)
        super(GeneratorUNet_big, self).__init__()
        self.down1 = UNetDown(in_channels, int(64/k), normalize=False)
        self.down2 = UNetDown(int(64/k), int(128/k))
        self.down3 = UNetDown(int(128/k), int(256/k))
        self.down4 = UNetDown(int(256/k), int(512/k), dropout=DROPOUT)
        self.down5 = UNetDown(int(512/k), int(512/k), dropout=DROPOUT)
        self.down6 = UNetDown(int(512/k), int(512/k), dropout=DROPOUT)
        self.down7 = UNetDown(int(512/k), int(512/k), dropout=DROPOUT)
        self.down8 = UNetDown(int(512/k), int(512/k),
                              normalize=False, dropout=DROPOUT)
        self.up1 = UNetUp(int(512/k),  int(512/k),  dropout=DROPOUT)
        self.up2 = UNetUp(int(1024/k), int(512/k),  dropout=DROPOUT)
        self.up3 = UNetUp(int(1024/k), int(512/k),  dropout=DROPOUT)
        self.up4 = UNetUp(int(1024/k), int(512/k),  dropout=DROPOUT)
        self.up5 = UNetUp(int(1024/k), int(256/k))
        self.up6 = UNetUp(int(512/k), int(128/k))
        self.up7 = UNetUp(int(256/k), int(64/k))
        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(int(128/k), out_channels, 4, padding=1),
            nn.Tanh(),)

    def forward(self, x):
        # U-Net generator with skip connections from encoder to decoder
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)
        u1 = self.up1(d8, d7)
        u2 = self.up2(u1, d6)
        u3 = self.up3(u2, d5)
        u4 = self.up4(u3, d4)
        u5 = self.up5(u4, d3)
        u6 = self.up6(u5, d2)
        u7 = self.up7(u6, d1)
        return self.final(u7)


class GeneratorUNet_old_configuration(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, DROPOUT=0.5):
        k = 1
        print('Divide number of filters by k,', k)
        print('DROPOUT,', DROPOUT)
        # k=2;    print('Divide number of filters by k,',k)
        # DROPOUT=0.5
        super(GeneratorUNet_old_configuration, self).__init__()
        self.down1 = UNetDown(in_channels, int(64/k), normalize=False)
        self.down2 = UNetDown(int(64/k), int(128/k))
        self.down3 = UNetDown(int(128/k), int(256/k))
        self.down4 = UNetDown(int(256/k), int(512/k), dropout=DROPOUT)
        self.down5 = UNetDown(int(512/k), int(512/k), dropout=DROPOUT)
        self.down6 = UNetDown(int(512/k), int(512/k), dropout=DROPOUT)
        self.down7 = UNetDown(int(512/k), int(512/k), dropout=DROPOUT)
        self.down8 = UNetDown(int(512/k), int(512/k),
                              normalize=False, dropout=DROPOUT)
        self.up1 = UNetUp(int(512/k), int(512/k),  dropout=DROPOUT)
        self.up2 = UNetUp(int(1024/k), int(512/k), dropout=DROPOUT)
        self.up3 = UNetUp(int(1024/k), int(512/k), dropout=DROPOUT)
        self.up4 = UNetUp(int(1024/k), int(512/k), dropout=DROPOUT)
        self.up5 = UNetUp(int(1024/k), int(256/k))
        self.up6 = UNetUp(int(512/k), int(128/k))
        self.up7 = UNetUp(int(256/k), int(64/k))
        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(int(128/k), out_channels, 4, padding=1),
            nn.Tanh(),)

    def forward(self, x):
        # U-Net generator with skip connections from encoder to decoder
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)
        u1 = self.up1(d8, d7)
        u2 = self.up2(u1, d6)
        # u3 = self.up3(d6, d5)
        u3 = self.up3(u2, d5)
        u4 = self.up4(u3, d4)
        u5 = self.up5(u4, d3)
        u6 = self.up6(u5, d2)
        u7 = self.up7(u6, d1)
        return self.final(u7)
##############################
# https://github.com/eriklindernoren/PyTorch-GAN/tree/master/implementations/pix2pix
#         Discriminator

class Discriminator(nn.Module):
    def __init__(self, in_channels=1):
        super(Discriminator, self).__init__()
        self.in_channels = in_channels

        def discriminator_block(in_filters, out_filters, normalization=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters,
                                4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        self.model = nn.Sequential(
            # *discriminator_block(in_channels * 2, 64, normalization=False),
            *discriminator_block(in_channels+1, 64, normalization=False),
            # *discriminator_block(4,64, normalization=False),
            # *discriminator_block(in_channels * 2, 64, normalization=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1, bias=False))

    def forward(self, img_A, img_B):
        # Concatenate image and condition image by channels to produce input
        if self.in_channels == 1:
            tmp = torch.unsqueeze(img_B[:, 0, :, :], dim=1)
        else:
            tmp = img_B
        img_input = torch.cat((img_A, tmp), 1)
        return self.model(img_input)
# https://github.com/GunhoChoi/FusionNet-Pytorch

#   Fusion net
##########################################################################################

class Conv_residual_conv(nn.Module):
    def __init__(self, in_dim, out_dim, act_fn,opt):
        super(Conv_residual_conv, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.opt = opt
        act_fn = act_fn
        self.conv_1 = conv_block(self.in_dim, self.out_dim, act_fn,self.opt)
        self.conv_2 = conv_block_3(self.out_dim, self.out_dim, act_fn,self.opt)
        self.conv_3 = conv_block(self.out_dim, self.out_dim, act_fn,self.opt)
    def forward(self, input):
        conv_1 = self.conv_1(input)
        conv_2 = self.conv_2(conv_1)
        res = conv_1 + conv_2
        conv_3 = self.conv_3(res)
        return conv_3

class FusionGenerator(nn.Module):
    def __init__(self,input_nc,output_nc,ngf,opt):
        super(FusionGenerator, self).__init__()
        self.in_dim = input_nc
        self.out_dim = ngf
        self.final_out_dim = output_nc
        self.opt=opt
        # act_fn = nn.LeakyReLU(0.2, inplace=True)
        # act_fn_2 = nn.ReLU()
        act_fn = nn.PReLU()
        act_fn_2 = nn.PReLU()
        print("\n------Initiating FusionNet------\n")
        # encoder
        self.down_1 = Conv_residual_conv(self.in_dim, self.out_dim, act_fn,self.opt)
        self.pool_1 = maxpool()
        self.down_2 = Conv_residual_conv(self.out_dim, self.out_dim * 2, act_fn,self.opt)
        self.pool_2 = maxpool()
        self.down_3 = Conv_residual_conv(self.out_dim * 2, self.out_dim * 4, act_fn,self.opt)
        self.pool_3 = maxpool()
        self.down_4 = Conv_residual_conv(self.out_dim * 4, self.out_dim * 8, act_fn,self.opt)
        self.pool_4 = maxpool()

        # bridge

        self.bridge = Conv_residual_conv(self.out_dim * 8, self.out_dim * 16, act_fn,self.opt)

        # decoder

        self.deconv_1 = conv_trans_block(self.out_dim * 16, self.out_dim * 8, act_fn_2)
        self.up_1 = Conv_residual_conv(self.out_dim * 8, self.out_dim * 8, act_fn_2,self.opt)
        self.deconv_2 = conv_trans_block(self.out_dim * 8, self.out_dim * 4, act_fn_2)
        self.up_2 = Conv_residual_conv(self.out_dim * 4, self.out_dim * 4, act_fn_2,self.opt)
        self.deconv_3 = conv_trans_block(self.out_dim * 4, self.out_dim * 2, act_fn_2)
        self.up_3 = Conv_residual_conv(self.out_dim * 2, self.out_dim * 2, act_fn_2,self.opt)
        self.deconv_4 = conv_trans_block(self.out_dim * 2, self.out_dim, act_fn_2)
        self.up_4 = Conv_residual_conv(self.out_dim, self.out_dim, act_fn_2,self.opt)

        # original first output layer
        self.out = nn.Conv2d(self.out_dim, self.final_out_dim,
                             kernel_size=3, stride=1, padding=1)
        # new final output layer, before it was nn.tanh()
        self.out_2 = nn.Conv2d(self.final_out_dim, self.final_out_dim,
            kernel_size=1, stride=1, padding=0)
        # self.out.in_channels
        # self.out.out_channels
        # self.out_2 = nn.Tanh()
        # self.out_2 = nn.Linear(1024,128)
        # initialization

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
                m.bias.data.fill_(0)

            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)

    def forward(self, input):
        down_1 = self.down_1(input)
        pool_1 = self.pool_1(down_1)
        down_2 = self.down_2(pool_1)
        pool_2 = self.pool_2(down_2)
        down_3 = self.down_3(pool_2)
        pool_3 = self.pool_3(down_3)
        down_4 = self.down_4(pool_3)
        pool_4 = self.pool_4(down_4)

        bridge = self.bridge(pool_4)

        deconv_1 = self.deconv_1(bridge)
        skip_1 = (deconv_1 + down_4)/2
        up_1 = self.up_1(skip_1)
        deconv_2 = self.deconv_2(up_1)
        skip_2 = (deconv_2 + down_3)/2
        up_2 = self.up_2(skip_2)
        deconv_3 = self.deconv_3(up_2)
        skip_3 = (deconv_3 + down_2)/2
        up_3 = self.up_3(skip_3)
        deconv_4 = self.deconv_4(up_3)
        skip_4 = (deconv_4 + down_1)/2
        up_4 = self.up_4(skip_4)

        out = self.out(up_4)
        out2= self.out_2(out)

        # m = nn.Linear(20, 30)
        # input = torch.randn(128, 20)
        # output = m(input)
        # print(output.size())
        #out = torch.clamp(out, min=-1, max=1)

        return out2


def conv_block(in_dim,out_dim,act_fn,opt):
    dropout_value=opt.dropout_value
    model = nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_dim),
        nn.Dropout2d(p=dropout_value),
        act_fn,
    )
    return model


def conv_trans_block(in_dim, out_dim, act_fn):
    model = nn.Sequential(
        nn.ConvTranspose2d(in_dim, out_dim, kernel_size=3,
                           stride=2, padding=1, output_padding=1),
        nn.BatchNorm2d(out_dim),
        act_fn,)
    return model


def maxpool():
    pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
    return pool


def conv_block_3(in_dim, out_dim, act_fn,opt):
    dropout_value=opt.dropout_value
    # dropout_value=0.8
    print('nn.Dropout2d(p=dropout_value), in conv_block_3=',dropout_value)
    model = nn.Sequential(
        conv_block(in_dim, out_dim, act_fn,opt),
        conv_block(out_dim, out_dim, act_fn,opt),
        nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_dim),
        nn.Dropout2d(p=dropout_value),
        )
    return model

def _handle_np_zeros(t, eps=1e-6):
    t[np.abs(t) < eps] = 1
    return t
def metric_pearson2(x, y):
    """ Pearson's coefficient, rho = \frac{cov(x, y)}{std_x * std_y}
    https://en.wikipedia.org/wiki/Pearson_correlation_coefficient
    Args:
        x: [b, c, h, w] np.array
        y: [b, c, h, w] np.array
    Returns:
        [b, c, h] np.array
    """
    ex = x - np.expand_dims(np.mean(x, axis=-1), -1)
    ey = y - np.expand_dims(np.mean(y, axis=-1), -1)
    m = np.mean(ex * ey, axis=-1) \
        / _handle_np_zeros(np.std(x, axis=-1)) \
        / _handle_np_zeros(np.std(y, axis=-1))
    return m
def PCC(x, y):
    """ Pearson's coefficient, rho = \frac{cov(x, y)}{std_x * std_y}
    https://en.wikipedia.org/wiki/Pearson_correlation_coefficient
    Args:
        x: [h, w] np.array
        y: [h, w] np.array
    Returns:
        [1] np.array
    """
    x=x.flatten()
    y=y.flatten()
    y2=-y
    ex = x - np.mean(x)
    ey = y - np.mean(y)
    m = np.mean(ex * ey) \
        / (np.std(x)) \
        / (np.std(y))

    ex2 = x - np.mean(x)
    ey2 = y2 - np.mean(y2)
    m2 = np.mean(ex * ey2) \
        / (np.std(x)) \
        / (np.std(y2))
    return m
def SSIM(x, y):
    """ SSIM:
        x: [h, w] np.array
        y: [h, w] np.array
    Returns:
        [1] np.array
    """
    return None
##########################################################################################

def sample_dataset_original(dataloader, generator, opt, epoch, flag_show_scaled_data=1, record_weights=0, data_mode='test'):
    """ A-input data,B-target data,C-initial model,D-true model,E-input data in real amplitudes
            ,F-output data in real amplitudes,sc_t-scaler_t"""
    cuda = True if torch.cuda.is_available() else False
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for i, batch in enumerate(dataloader):
        NAME=dataloader.dataset.files[i]
        tmp=dataloader.dataset.files[i].split('/')[-1]
        Name=tmp.split('.npz')[0]
        real_A = Variable(batch["A"].type(Tensor))
        real_B = Variable(batch["B"].type(Tensor))
        init_model=Variable(batch["C"].type(Tensor))
        true_model=Variable(batch["D"].type(Tensor))
        real_A_ra = Variable(batch["E"].type(Tensor))
        real_B_ra = Variable(batch["F"].type(Tensor))
        scaler_t=Variable(batch["sc_t"].type(Tensor)).cpu().detach().numpy()
        dx=20;dz=20
        ##############  predict from x
        fake_B=generator(real_A)
        ##############  imresize predicted data
        t_pred=np.empty((real_B_ra.shape))
        Nx=real_B_ra.shape[2]; Nz=real_B_ra.shape[3]
        tmp=fake_B.cpu().detach().numpy()
        for i_f in range(tmp.shape[0]):
            t_pred[i_f,0,:,:]=imresize(tmp[i_f,0,:, :],(Nx, Nz))
        ##############  scale predicted data back to original amplitudes
        fake_B_ra=np.ones_like(t_pred)
        for i_f in range(t_pred.shape[0]):
            tmp=transforming_data_inverse(t_pred[i_f,::],scaler_t[i_f])
            fake_B_ra[i_f,0,:,:]=np.squeeze(tmp)
        ##############  append tapered part of initial model to predicted data to compare it with true model
        upper_taper_thickness=true_model.shape[3]-Nz
        zero_tapered_area=np.zeros((fake_B_ra.shape[0],fake_B_ra.shape[1],fake_B_ra.shape[2],upper_taper_thickness))
        zero_tapered_area=torch.from_numpy(zero_tapered_area).to(device)
        fake_B_ra=torch.from_numpy(fake_B_ra).to(device)
        fake_B_ra=torch.cat((zero_tapered_area,fake_B_ra),axis=3)
        real_A_ra=torch.cat((zero_tapered_area,real_A_ra),axis=3)
        real_B_ra=torch.cat((zero_tapered_area,real_B_ra),axis=3)
        ##############
        M1=real_A.cpu().detach().numpy().squeeze()
        M2=real_B.cpu().detach().numpy().squeeze()
        M3=fake_B.cpu().detach().numpy().squeeze()
        fake_B_ra=fake_B_ra.cpu().detach().numpy().squeeze()
        real_A_ra=real_A_ra.cpu().detach().numpy().squeeze()
        real_B_ra=real_B_ra.cpu().detach().numpy().squeeze()
        init_model=init_model.cpu().detach().numpy().squeeze()
        true_model=true_model.cpu().detach().numpy().squeeze()
        ##############
        fwi_result=init_model+real_A_ra
        ideal_init_model=init_model+real_B_ra
        predicted_init_model=init_model+fake_B_ra
        pics=[M1,M2,M3,init_model,predicted_init_model,ideal_init_model,fwi_result,true_model]
        # pics_6=[M1,M2,M3,init_model,testing_model,ideal_init_model,M0_show]
        picture_name = opt.save_path+'/'+'log_'+data_mode+str(opt.log_save_const)+Name+'_epoch_'+str(epoch)+'_'+numstr(F_r2(M3,M2))+'.png'
        titles = ['Input','Target',
            'Prediction, R2(prediction,target) = '+numstr(F_r2(M3,M2))+',NRMS='+numstr(nrms(M3,M2)),
            'initial model, R2(initial,true)='+numstr(F_r2(init_model,true_model))+',NRMS='+numstr(nrms(init_model,true_model)),
            'Predicted initial model for fwi:R2(predicted initial,true)='+numstr(F_r2(predicted_init_model,true_model))+',NRMS='+numstr(nrms(predicted_init_model,true_model)),
            'Ideal initial model, R2(initial model,true)='+numstr(F_r2(ideal_init_model,true_model))+',NRMS='+numstr(nrms(ideal_init_model,true_model)),
            'FWI result,R2(FWI result,true)='+numstr(F_r2(fwi_result,true_model))+',NRMS='+numstr(nrms(fwi_result,true_model)),
            'True model']
        # PLOT_ML_Result_7_pics(pics, numstr(F_r2(testing_model,M0_show)), history_flag=0,
        #     history=None, Boundaries=[], save_file_path=picture_name,
        #     dx=dx, dy=dz, Title=titles, Title2='', Save_flag=1, adaptive_colorbar=3)
        PLOT_ML_Result_8_pics(pics, numstr(F_r2(predicted_init_model,true_model)),history_flag=0,
            history=None, Boundaries=[], save_file_path=picture_name,
            dx=dx,dy=dz,Title=titles, Title2='', Save_flag=1, adaptive_colorbar=3)
        ###################
        if record_weights==1:
            tmp2 = NAME.split('/')[-1]
            path = opt.save_path+'/'+tmp2[0:-4]+'_weights_'+str(opt.log_save_const)
            os.makedirs(os.path.join(opt.save_path,'predictions_'+str(opt.log_save_const)),exist_ok=True)
            path=os.path.join(opt.save_path,'predictions_'+str(opt.log_save_const),tmp2[0:-4]+'_weights_'+str(opt.log_save_const)+'.npz')
            np.savez(path,input_data=M1,output_data=M2,predicted_update=M3,models_init=init_model,
                models=true_model,predicted_initial_model=predicted_init_model,ideal_init_model=ideal_init_model,
                fwi_result=fwi_result,dx=dx, dz=dz)
    return None

def sample_dataset(dataloader, generator, opt, epoch, flag_show_scaled_data=1, record_weights=0, data_mode='test'):
    """ A-input data,B-target data,C-initial model,D-true model,E-input data in real amplitudes
            ,F-output data in real amplitudes,sc_t-scaler_t"""
    cuda=True if torch.cuda.is_available() else False
    Tensor=torch.cuda.FloatTensor if cuda else torch.FloatTensor
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    chosen_metric=F_r2;     metric_name=',R2='
    chosen_metric=F_r2_tracewise;     metric_name=',R2_tw='
    chosen_metric=PCC;     metric_name=',PCC='
    for i, batch in enumerate(dataloader):
        NAME=dataloader.dataset.files[i]
        p = pathlib.Path(NAME)
        file_dataset_path=p.parents[0]
        with open(os.path.join(file_dataset_path,'scaling_constants_dict.pkl'),'rb') as input:
            scaling_constants_dict=pickle.load(input)
        with open(NAME,'rb') as f:
            data = np.load(f, allow_pickle=True)
            x = data['input_data']
            t = data['output_data']
            x_ra= data['input_data_real_amplitudes']
            t_ra= data['output_data_real_amplitudes']
            models_init =data['models_init']
            models_init_scaled=data['models_init_scaled']
            models = data['models']
            fwi_result_scaled=data['fwi_result_scaled']
            smoothed_true_model_scaled=data['smoothed_true_model_scaled']
            smoothed_true_model=data['smoothed_true_model']
            dm_i_ra=data['dm_i_ra']
            dm_i_ra_scaled=data['dm_i_ra_scaled']
            m_i_ra=data['m_i_ra']
            m_i_ra_scaled=data['m_i_ra_scaled']
            taper=data['taper']
            dz = data['dz']
            dx = data['dx']
            data.close()
        tmp=dataloader.dataset.files[i].split('/')[-1]
        Name=tmp.split('.npz')[0]
        real_A = Variable(batch["A"].type(Tensor))
        real_B = Variable(batch["B"].type(Tensor))
        init_model=Variable(batch["C"].type(Tensor))
        true_model=Variable(batch["D"].type(Tensor))
        real_A_ra = Variable(batch["E"].type(Tensor))
        real_B_ra = Variable(batch["F"].type(Tensor))
        dx=25;dz=25
        ##############  predict from x
        fake_B=generator(real_A)
        ##############  imresize predicted data
        t_pred=np.empty((real_B_ra.shape))
        Nx=real_B_ra.shape[2]; Nz=real_B_ra.shape[3]
        tmp=fake_B.cpu().detach().numpy()
        for i_f in range(tmp.shape[0]):
            t_pred[i_f,0,:,:]=imresize(tmp[i_f,0,:, :],(Nx,Nz))
        ##############  scale predicted data back to original amplitudes
        fake_B_ra=np.ones_like(t_pred)
        if np.max(np.abs(x))<=1:    scaling_range='-11'
        else:   scaling_range='standardization'
        print('Data analyze shows dataset scaling_range=',scaling_range)
        for i_f in range(t_pred.shape[0]):
            # tmp=transforming_data_inverse(t_pred[i_f,::],scaler_t[i_f])
            if opt.inp_channels_type=='1_fwi_res' or opt.inp_channels_type=='m_i':
                tmp=scaling_data_back(t_pred[i_f,::],scaling_constants_dict,'fwi_res',  scaling_range=scaling_range)
            else:
                tmp=scaling_data_back(t_pred[i_f,::],scaling_constants_dict,'t',        scaling_range=scaling_range)
            fake_B_ra[i_f,0,:,:]=np.squeeze(tmp)
        ##############  append tapered part of initial model to predicted data to compare it with true model
        # upper_taper_thickness=true_model.shape[3]-Nz
        # zero_tapered_area=np.zeros((fake_B_ra.shape[0],fake_B_ra.shape[1],fake_B_ra.shape[2],upper_taper_thickness))
        # zero_tapered_area=torch.from_numpy(zero_tapered_area).to(device)
        # fake_B_ra=torch.from_numpy(fake_B_ra).to(device)
        # fake_B_ra=torch.cat((zero_tapered_area,fake_B_ra),axis=3)
        # real_A_ra=torch.cat((zero_tapered_area,real_A_ra),axis=3)
        # real_B_ra=torch.cat((zero_tapered_area,real_B_ra),axis=3)
        ##############
        M1=real_A.cpu().detach().numpy().squeeze()
        if M1.ndim==3:
            M1=M1[9,::]    #   9,-1
        elif opt.inp_channels_type=='1dv_1init':
            M1=M1[0,::]
        M2=real_B.cpu().detach().numpy().squeeze()
        M3=fake_B.cpu().detach().numpy().squeeze()
        if torch.is_tensor(fake_B_ra)==True:
            fake_B_ra=fake_B_ra.cpu().detach().numpy().squeeze()
        else:
            fake_B_ra=fake_B_ra.squeeze()
        real_A_ra=real_A_ra.cpu().detach().numpy().squeeze()
        real_B_ra=real_B_ra.cpu().detach().numpy().squeeze()
        init_model=init_model.cpu().detach().numpy().squeeze()
        true_model=true_model.cpu().detach().numpy().squeeze()
        ##############
        # fake_B_ra[taper==0]=0
        ##############
        fwi_result=init_model+real_A_ra
        ideal_init_model=smoothed_true_model
        if opt.inp_channels_type=='1_fwi_res' or opt.inp_channels_type=='m_i':
            predicted_init_model=fake_B_ra
        else:
            predicted_init_model=init_model+fake_B_ra
        ##############
        # fake_B_ra2=scaling_data_back(real_B.cpu().detach().numpy().squeeze(),scaling_constants_dict,'fwi_res')
        # Plot_image(fake_B_ra2.squeeze().T,Show_flag=0,Save_flag=1,Title='fake_B_ra2',Aspect='equal',Save_pictures_path=opt.save_path)
        # Plot_image(real_B_ra.cpu().detach().numpy().squeeze().T,Show_flag=0,Save_flag=1,Title='real_B_ra',Aspect='equal',Save_pictures_path=opt.save_path)
        # Plot_image(fake_B.cpu().detach().numpy().squeeze().T,Show_flag=0,Save_flag=1,Title='fake_B',Aspect='equal',Save_pictures_path=opt.save_path)
        # Plot_image(real_B.cpu().detach().numpy().squeeze().T,Show_flag=0,Save_flag=1,Title='real_B',Aspect='equal',Save_pictures_path=opt.save_path)
        # Plot_image(predicted_init_model.squeeze().T,Show_flag=0,Save_flag=1,Title='predicted_init_model',Aspect='equal',Save_pictures_path=opt.save_path)
        ##############  comparison between predicted initial and true model
        # pics=[M1,M2,M3,init_model,predicted_init_model,ideal_init_model,fwi_result,true_model]
        # picture_name = opt.save_path+'/'+'log_'+data_mode+str(opt.log_save_const)+Name+'_epoch_'+str(epoch)+'_'+numstr(chosen_metric(M3,M2))+'.png'
        # # titles = ['Input','Target',
        # #     'Prediction, R2(prediction,target) = '+numstr(chosen_metric(M3,M2))+',R2='+numstr(F_r2(M3,M2)),
        # #     'initial model, R2(initial,true)='+numstr(chosen_metric(init_model,true_model))+',R2='+numstr(F_r2(init_model,true_model)),
        # #     'Predicted initial model for fwi:R2(predicted initial,true)='+numstr(chosen_metric(predicted_init_model,true_model))+',R2='+numstr(F_r2(predicted_init_model,true_model)),
        # #     'Ideal initial model, R2(initial model,true)='+numstr(chosen_metric(ideal_init_model,true_model))+',R2='+numstr(F_r2(ideal_init_model,true_model)),
        # #     'FWI result,R2(FWI result,true)='+numstr(chosen_metric(fwi_result,true_model))+',R2='+numstr(F_r2(fwi_result,true_model)),
        # #     'True model']
        # titles=['Input','Target',
        #     'Prediction, R2_tw(prediction,target) = '+numstr(chosen_metric(M3,M2))+',R2='+numstr(F_r2(M3,M2)),
        #     'initial model, R2_tw(initial,true)='+numstr(chosen_metric(init_model,true_model))+',R2='+numstr(F_r2(init_model,true_model)),
        #     'Predicted initial model for fwi:R2_tw(predicted initial,true)='+numstr(chosen_metric(predicted_init_model,true_model))+',R2='+numstr(F_r2(predicted_init_model,true_model)),
        #     'Ideal initial model, R2_tw(initial model,true)='+numstr(chosen_metric(ideal_init_model,true_model))+',R2='+numstr(F_r2(ideal_init_model,true_model)),
        #     'FWI result,R2_tw(FWI result,true)='+numstr(chosen_metric(fwi_result,true_model))+',R2='+numstr(F_r2(fwi_result,true_model)),
        #     'True model']
        # PLOT_ML_Result_8_pics(pics, numstr(chosen_metric(predicted_init_model,true_model)),history_flag=0,
        #     history=None, Boundaries=[], save_file_path=picture_name,
        #     dx=dx,dy=dz,Title=titles, Title2='',Show_flag=0,Save_flag=1,adaptive_colorbar=3)
        ##############  comparison between predicted initial and ideal initial model
        # pics=[M1,M2,M3,init_model,predicted_init_model,ideal_init_model,fwi_result,true_model]
        # F_r2(M3,M2)
        # PCC(M3,M2)
        # picture_name = opt.save_path+'/'+'log_'+data_mode+str(opt.log_save_const)+Name+'_epoch_'+str(epoch)+'_'+numstr(F_r2(M3,M2))+'.png'
        # titles=[
        #     'Input',
        #     'Target',
        #     'Prediction, '+'R2(prediction,target)='+numstr(F_r2(M3,M2))+metric_name+numstr(chosen_metric(M3,M2)),
        #     'initial model, '+'R2(initial,ideal_init_model)='+numstr(F_r2(init_model,ideal_init_model))+metric_name+numstr(chosen_metric(init_model,ideal_init_model)),
        #     'Predicted initial model,'+'R2(_-||-,ideal_init_model)='+numstr(F_r2(predicted_init_model,ideal_init_model))+metric_name+numstr(chosen_metric(predicted_init_model,ideal_init_model)),
        #     'Ideal initial model, '+'R2(initial model,true)='+numstr(F_r2(ideal_init_model,true_model))+metric_name+numstr(chosen_metric(ideal_init_model,true_model)),
        #     'FWI result,'+'R2(FWI result,ideal_init_model)='+numstr(F_r2(fwi_result,ideal_init_model))+metric_name+numstr(chosen_metric(fwi_result,ideal_init_model)),
        #     'True model']
        # PLOT_ML_Result_8_pics(pics, numstr(chosen_metric(predicted_init_model,true_model)),history_flag=0,
        #     history=None, Boundaries=[], save_file_path=picture_name,
        #     dx=dx,dy=dz,Title=titles, Title2='',Show_flag=0,Save_flag=1,adaptive_colorbar=3)
        ###############
        # features=real_A.cpu().detach().numpy().squeeze()
        # pics=[M1,M2,M3,init_model,predicted_init_model,ideal_init_model,fwi_result,true_model,features]
        # picture_name = opt.save_path+'/'+'log_'+data_mode+str(opt.log_save_const)+Name+'_epoch_'+str(epoch)+'_'+numstr(F_r2(M3,M2))+'.png'
        # titles=[
        #     'Input',
        #     'Target',
        #     'Prediction, '+'R2(prediction,target)='+numstr(F_r2(M3,M2))+metric_name+numstr(chosen_metric(M3,M2)),
        #     'initial model, '+'R2(initial,ideal_init_model)='+numstr(F_r2(init_model,ideal_init_model))+metric_name+numstr(chosen_metric(init_model,ideal_init_model)),
        #     'Predicted initial model,'+'R2(_-||-,ideal_init_model)='+numstr(F_r2(predicted_init_model,ideal_init_model))+metric_name+numstr(chosen_metric(predicted_init_model,ideal_init_model)),
        #     'Ideal initial model, '+'R2(initial model,true)='+numstr(F_r2(ideal_init_model,true_model))+metric_name+numstr(chosen_metric(ideal_init_model,true_model)),
        #     'FWI result,'+'R2(FWI result,ideal_init_model)='+numstr(F_r2(fwi_result,ideal_init_model))+metric_name+numstr(chosen_metric(fwi_result,ideal_init_model)),
        #     'True model']
        # PLOT_ML_Result_all_inputs2(pics, numstr(chosen_metric(predicted_init_model,true_model)),history_flag=0,
        #     history=None, Boundaries=[], save_file_path=picture_name,
        #     dx=dx,dy=dz,Title=titles, Title2='',Show_flag=0,Save_flag=1,adaptive_colorbar=3)
        ###############
        features=real_A.cpu().detach().numpy().squeeze()
        pics=[M1,M2,M3,init_model,predicted_init_model,ideal_init_model,fwi_result,true_model,features]
        picture_name = opt.save_path+'/'+'log_'+data_mode+str(opt.log_save_const)+Name+'_epoch_'+str(epoch)+'_'+numstr(F_r2(M3,M2))+'.png'
        titles=[
            'Input',
            'Target',
            'Prediction, '+'R2(prediction,target)='+numstr(F_r2(M3,M2))+metric_name+numstr(chosen_metric(M3,M2)),
            'initial model, '+'R2(initial,ideal_init_model)='+numstr(F_r2(init_model,ideal_init_model))+metric_name+numstr(chosen_metric(init_model,ideal_init_model)),
            'Predicted initial model,'+'R2(_-||-,ideal_init_model)='+numstr(F_r2(predicted_init_model,ideal_init_model))+metric_name+numstr(chosen_metric(predicted_init_model,ideal_init_model)),
            'Ideal initial model, '+'R2(initial model,true)='+numstr(F_r2(ideal_init_model,true_model))+metric_name+numstr(chosen_metric(ideal_init_model,true_model)),
            'FWI result,'+'R2(FWI result,ideal_init_model)='+numstr(F_r2(fwi_result,ideal_init_model))+metric_name+numstr(chosen_metric(fwi_result,ideal_init_model)),
            'True model']
        PLOT_channels_separately(pics, numstr(chosen_metric(predicted_init_model,true_model)),opt,history_flag=0,
            history=None, Boundaries=[], save_file_path=picture_name,model_name=Name,
            dx=dx,dy=dz,Title=titles, Title2='',Show_flag=0,Save_flag=1,adaptive_colorbar=3)
        PLOT_ML_Result_8_pics(pics, numstr(chosen_metric(predicted_init_model,true_model)),history_flag=0,
            history=None, Boundaries=[], save_file_path=picture_name,
            dx=dx,dy=dz,Title=titles, Title2='',Show_flag=0,Save_flag=1,adaptive_colorbar=3)
        ###############     test water velocity
        # features.shape
        # update_m=M3*(-features[-1,::]+1)
        # print(M3)
        # print(update_m)
        # Plot_image(M3.T,Show_flag=0,Save_flag=1,Title='M3'+Name,Aspect='equal',Save_pictures_path=opt.save_path)
        # Plot_image(M1.T,Show_flag=0,Save_flag=1,Title='M1'+Name,Aspect='equal',Save_pictures_path=opt.save_path)
        # Plot_image(features[-1,::].T,Show_flag=0,Save_flag=1,Title='water_taper'+Name,Aspect='equal',Save_pictures_path=opt.save_path)
        # Plot_image(update_m.T,Show_flag=0,Save_flag=1,Title='update_m'+Name,Aspect='equal',Save_pictures_path=opt.save_path)
        # Plot_image(real_A[0,11,::].cpu().detach().numpy().T,Show_flag=0,Save_flag=1,Title='11 th channel'+Name,Aspect='equal',Save_pictures_path=opt.save_path)
        # Plot_image(real_A[0,1,::].cpu().detach().numpy().T,Show_flag=0,Save_flag=1,Title='1 th channel'+Name,Aspect='equal',Save_pictures_path=opt.save_path)
        # print(real_A[0,5,20,0:30].cpu().detach().numpy())
        # ss=1
        ###############
        if record_weights==1:
            tmp2 = NAME.split('/')[-1]
            path = opt.save_path+'/'+tmp2[0:-4]+'_weights_'+str(opt.log_save_const)
            os.makedirs(os.path.join(opt.save_path,'predictions_'+str(opt.log_save_const) ),exist_ok=True)
            path=os.path.join(opt.save_path,'predictions_'+str(opt.log_save_const),tmp2[0:-4]+'_weights_'+str(opt.log_save_const)+'.npz')
            np.savez(path,input_data=M1,output_data=M2,predicted_update=M3,
                models_init=init_model,
                models=true_model,
                predicted_initial_model=predicted_init_model,
                ideal_init_model=ideal_init_model,
                water_taper=taper,
                fwi_result=fwi_result,dx=dx, dz=dz)
            with open(os.path.join(opt.save_path,'predictions_'+str(opt.log_save_const),'opt.txt' ),'w') as f:
                json.dump(opt.__dict__, f, indent=2)
            with open(os.path.join(opt.save_path,'scaling_constants_dict.pkl'),'wb') as output:
                pickle.dump(scaling_constants_dict,output,protocol=4)
            torch.save(generator.state_dict(),os.path.join(opt.save_path,'predictions_'+str(opt.log_save_const),'generator_weights.pth') )
    return None

def sample_dataset_geophysics(dataloader, generator, opt, epoch, flag_show_scaled_data=1,
     record_weights=0,record_predictions=1,data_mode='test'):
    """ A-input data,B-target data,C-initial model,D-true model,E-input data in real amplitudes
            ,F-output data in real amplitudes,sc_t-scaler_t"""
    cuda=True if torch.cuda.is_available() else False
    Tensor=torch.cuda.FloatTensor if cuda else torch.FloatTensor
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    chosen_metric=F_r2;     metric_name=',R2='
    # chosen_metric=F_r2_tracewise;     metric_name=',R2_tw='
    chosen_metric=PCC;     metric_name=',PCC='
    chosen_metric=ssim;     metric_name=',SSIM='
    for i, batch in enumerate(dataloader):
        NAME=dataloader.dataset.files[i]
        p = pathlib.Path(NAME)
        file_dataset_path=p.parents[0]
        with open(os.path.join(file_dataset_path,'scaling_constants_dict.pkl'),'rb') as input:
            scaling_constants_dict=pickle.load(input)
        with open(NAME,'rb') as f:
            data = np.load(f, allow_pickle=True)
            x = data['input_data']
            t = data['output_data']
            x_ra= data['input_data_real_amplitudes']
            t_ra= data['output_data_real_amplitudes']
            models_init =data['models_init']
            models_init_scaled=data['models_init_scaled']
            models = data['models']
            fwi_result_scaled=data['fwi_result_scaled']
            smoothed_true_model_scaled=data['smoothed_true_model_scaled']
            smoothed_true_model=data['smoothed_true_model']
            dm_i_ra=data['dm_i_ra']
            dm_i_ra_scaled=data['dm_i_ra_scaled']
            m_i_ra=data['m_i_ra']
            m_i_ra_scaled=data['m_i_ra_scaled']
            taper=data['taper']
            dz = data['dz']
            dx = data['dx']
            data.close()
        tmp=dataloader.dataset.files[i].split('/')[-1]
        Name=tmp.split('.npz')[0]
        real_A = Variable(batch["A"].type(Tensor))
        real_B = Variable(batch["B"].type(Tensor))
        init_model=Variable(batch["C"].type(Tensor))
        true_model=Variable(batch["D"].type(Tensor))
        real_A_ra = Variable(batch["E"].type(Tensor))
        real_B_ra = Variable(batch["F"].type(Tensor))
        dx=25;dz=25
        # Plot_image(true_model.cpu().detach().numpy().squeeze().T,Show_flag=0,Save_flag=1,Title='true_model_batch',Aspect='equal',Save_pictures_path=opt.save_path)
        ##############  predict from x
        fake_B=generator(real_A)
        ##############  imresize predicted data
        t_pred=np.empty((real_B_ra.shape))
        Nx=real_B_ra.shape[2]; Nz=real_B_ra.shape[3]
        tmp=fake_B.cpu().detach().numpy()
        for i_f in range(tmp.shape[0]):
            t_pred[i_f,0,:,:]=imresize(tmp[i_f,0,:, :],(Nx,Nz))
        ##############  scale predicted data back to original amplitudes
        fake_B_ra=np.ones_like(t_pred)
        if np.max(np.abs(x))<=1:    scaling_range='-11'
        else:   scaling_range='standardization'
        print('Data analyze shows dataset scaling_range=',scaling_range)
        for i_f in range(t_pred.shape[0]):
            # tmp=transforming_data_inverse(t_pred[i_f,::],scaler_t[i_f])
            if opt.inp_channels_type=='1_fwi_res' or opt.inp_channels_type=='m_i':
                tmp=scaling_data_back(t_pred[i_f,::],scaling_constants_dict,'fwi_res',  scaling_range=scaling_range)
            else:
                tmp=scaling_data_back(t_pred[i_f,::],scaling_constants_dict,'t',        scaling_range=scaling_range)
            fake_B_ra[i_f,0,:,:]=np.squeeze(tmp)
        ##############
        M1=real_A.cpu().detach().numpy().squeeze()
        if M1.ndim==3:
            M1=M1[-1,::]
        elif opt.inp_channels_type=='1dv_1init':
            M1=M1[0,::]
        M2=real_B.cpu().detach().numpy().squeeze()
        M3=fake_B.cpu().detach().numpy().squeeze()
        if torch.is_tensor(fake_B_ra)==True:
            fake_B_ra=fake_B_ra.cpu().detach().numpy().squeeze()
        else:
            fake_B_ra=fake_B_ra.squeeze()
        real_A_ra=real_A_ra.cpu().detach().numpy().squeeze()
        real_B_ra=real_B_ra.cpu().detach().numpy().squeeze()
        init_model=init_model.cpu().detach().numpy().squeeze()
        true_model=true_model.cpu().detach().numpy().squeeze()
        ##############
        # fake_B_ra[taper==0]=0
        ##############
        fwi_result=init_model+real_A_ra
        ideal_init_model=smoothed_true_model
        if opt.inp_channels_type=='1_fwi_res' or opt.inp_channels_type=='m_i':
            predicted_init_model=fake_B_ra
        else:
            predicted_init_model=init_model+fake_B_ra
        features_orig=real_A.cpu().detach().numpy().squeeze()
        ##############
        # fake_B_ra2=scaling_data_back(real_B.cpu().detach().numpy().squeeze(),scaling_constants_dict,'fwi_res')
        # Plot_image(fake_B_ra2.squeeze().T,Show_flag=0,Save_flag=1,Title='fake_B_ra2',Aspect='equal',Save_pictures_path=opt.save_path)
        # Plot_image(real_B_ra.cpu().detach().numpy().squeeze().T,Show_flag=0,Save_flag=1,Title='real_B_ra',Aspect='equal',Save_pictures_path=opt.save_path)
        # Plot_image(fake_B.cpu().detach().numpy().squeeze().T,Show_flag=0,Save_flag=1,Title='fake_B',Aspect='equal',Save_pictures_path=opt.save_path)
        # Plot_image(real_B.cpu().detach().numpy().squeeze().T,Show_flag=0,Save_flag=1,Title='real_B',Aspect='equal',Save_pictures_path=opt.save_path)
        # Plot_image(predicted_init_model.squeeze().T,Show_flag=0,Save_flag=1,Title='predicted_init_model',Aspect='equal',Save_pictures_path=opt.save_path)
        ##############  comparison between predicted initial and ideal initial model
        features=np.zeros((features_orig.shape[0],Nx,Nz))
        for i_f in range(features_orig.shape[0]):
            features[i_f,:,:]=imresize(features_orig[i_f,:,:],(Nx,Nz))
        M1=imresize(M1,(Nx,Nz))
        M2=imresize(M2,(Nx,Nz))
        M3=imresize(M3,(Nx,Nz))
        pics=[M1,M2,M3,init_model,predicted_init_model,ideal_init_model,fwi_result,true_model,features]
        picture_name = opt.save_path+'/'+'log_'+data_mode+str(opt.log_save_const)+Name+'_epoch_'+str(epoch)+'_'+numstr(F_r2(M3,M2))
        if '_cgg_' in Name:
            picture_name2 = opt.save_path+'/'+'log_'+data_mode+str(opt.log_save_const)+Name+'_epoch_'+str(epoch)+'_'+numstr(F_r2(M3,M2))+'_2nd_plot.png'
            nx_cropped=Nx-320
            # nx_cropped=600
            nx_cropped=Nx
            pics=[M1[0:nx_cropped,:],M2[0:nx_cropped,:],M3[0:nx_cropped,:],init_model[0:nx_cropped,:],predicted_init_model[0:nx_cropped,:],ideal_init_model[0:nx_cropped,:],fwi_result[0:nx_cropped,:],true_model[0:nx_cropped,:],features[:,0:nx_cropped,:]]
            titles=[
                'Input, 10th iteration high-wavenumber model update',
                'Target, low-wavenumber model update',
                'Prediction',
                'Apriori initial model, 11th CNN input channel',
                'Predicted initial model',
                'Target initial model, '+'R2(-||-,true model)='+numstr(F_r2(ideal_init_model,true_model))+metric_name+numstr(chosen_metric(ideal_init_model,true_model)),
                # 'FWI result,'+'R2(FWI result,ideal init. model)='+numstr(F_r2(fwi_result,ideal_init_model))+metric_name+numstr(chosen_metric(fwi_result,ideal_init_model)),
                'FWI result,'+'R2(FWI result,ideal init. model)='+numstr(F_r2(fwi_result,true_model))+metric_name+numstr(chosen_metric(fwi_result,true_model)),
                'True model']
            PLOT_ML_Result_geophysics_field_data2(pics, numstr(chosen_metric(predicted_init_model,true_model)),history_flag=0,
                history=None, Boundaries=[], save_file_path=picture_name+'.png',
                dx=dx,dy=dz,Title=titles, Title2='',Show_flag=0,Save_flag=1,adaptive_colorbar=3)
            PLOT_cgg_field_data(pics, numstr(chosen_metric(predicted_init_model,true_model)),history_flag=0,
                history=None, Boundaries=[], save_file_path=picture_name2,
                dx=dx,dy=dz,Title=titles, Title2='',Show_flag=0,Save_flag=1,adaptive_colorbar=3)
        else:
            pics=[M1,M2,M3,init_model,predicted_init_model,ideal_init_model,fwi_result,true_model,features]
            titles=[
                'Input, 10th iteration high-wavenumber model update',
                'Target, low-wavenumber model update',
                'Prediction, '+'R2(prediction,target)='+numstr(F_r2(M3,M2))+metric_name+numstr(chosen_metric(M3,M2)),
                ### comparison with cnn target initial model
                'Apriori initial model, '+'R2(-||-,target init. model)='+numstr(F_r2(init_model,ideal_init_model))+metric_name+numstr(chosen_metric(init_model,ideal_init_model)),
                'Predicted init. model, '+'R2(-||-,target init. model)='+numstr(F_r2(predicted_init_model,ideal_init_model))+metric_name+numstr(chosen_metric(predicted_init_model,ideal_init_model)),
                'Target initial model', # +'R2(-||-,true model)='+numstr(F_r2(ideal_init_model,true_model))+metric_name+numstr(chosen_metric(ideal_init_model,ideal_init_model)),
                'FWI result,'+'R2(FWI result,target init. model)='+numstr(F_r2(fwi_result,ideal_init_model))+metric_name+numstr(chosen_metric(fwi_result,ideal_init_model)),
                'True model']
            # PLOT_ML_Result_geophysics3(pics, numstr(chosen_metric(predicted_init_model,true_model)),history_flag=0,
            #     history=None, Boundaries=[], save_file_path=picture_name+'.png',
            #     dx=dx,dy=dz,Title=titles, Title2='',Show_flag=0,Save_flag=1,adaptive_colorbar=3)
            PLOT_ML_Result_geophysics4(pics, numstr(chosen_metric(predicted_init_model,true_model)),history_flag=0,
                history=None, Boundaries=[], save_file_path=picture_name+'_1.png',
                dx=dx,dy=dz,Title=titles, Title2='',Show_flag=0,Save_flag=1,adaptive_colorbar=3)
            ####
            pics=[M1,M2,M3,init_model,predicted_init_model,ideal_init_model,fwi_result,true_model,features]
            titles=[
                'Input, 10th iteration high-wavenumber model update',
                'Target, low-wavenumber model update',
                'Prediction, '+'R2(prediction,target)='+numstr(F_r2(M3,M2))+metric_name+numstr(chosen_metric(M3,M2)),
                ### comparison with true model
                'Apriori initial model, '+'R2(-||-,true model)='+numstr(F_r2(init_model,true_model))+metric_name+numstr(chosen_metric(init_model,true_model)),
                'Predicted init. model, '+'R2(-||-,true model)='+numstr(F_r2(predicted_init_model,true_model))+metric_name+numstr(chosen_metric(predicted_init_model,true_model)),
                'Target initial model, ' +'R2(-||-,true model)='+numstr(F_r2(ideal_init_model,true_model))+metric_name+numstr(chosen_metric(ideal_init_model,true_model)),
                'FWI result,'+'R2(FWI result,true model)='+numstr(F_r2(fwi_result,true_model))+metric_name+numstr(chosen_metric(fwi_result,true_model)),
                'True model']
            # PLOT_ML_Result_geophysics3(pics, numstr(chosen_metric(predicted_init_model,true_model)),history_flag=0,
            #     history=None, Boundaries=[], save_file_path=picture_name+'_comp_with_true.png',
            #     dx=dx,dy=dz,Title=titles, Title2='',Show_flag=0,Save_flag=1,adaptive_colorbar=3)
            PLOT_ML_Result_geophysics4(pics, numstr(chosen_metric(predicted_init_model,true_model)),history_flag=0,
                history=None, Boundaries=[], save_file_path=picture_name+'_2.png',
                dx=dx,dy=dz,Title=titles, Title2='',Show_flag=0,Save_flag=1,adaptive_colorbar=3)
            ss=1
            ####
            # PLOT_channels_separately(pics, numstr(chosen_metric(predicted_init_model,true_model)),opt,history_flag=0,
            #     history=None, Boundaries=[], save_file_path=picture_name,model_name=Name,
            #     dx=dx,dy=dz,Title=titles, Title2='',Show_flag=0,Save_flag=1,adaptive_colorbar=3)
        ##############  comparison between predicted initial and ideal initial model
        # features=np.zeros((features_orig.shape[0],Nx,Nz))
        # for i_f in range(features_orig.shape[0]):
        #     features[i_f,:,:]=imresize(features_orig[i_f,:,:],(Nx,Nz))
        # M1=imresize(M1,(Nx,Nz))
        # M2=imresize(M2,(Nx,Nz))
        # M3=imresize(M3,(Nx,Nz))
        # pics=[M1,M2,M3,init_model,predicted_init_model,ideal_init_model,fwi_result,true_model,features]
        # F_r2(M3,M2)
        # PCC(M3,M2)
        # picture_name = opt.save_path+'/'+'log_'+data_mode+str(opt.log_save_const)+Name+'_epoch_'+str(epoch)+'_'+numstr(F_r2(M3,M2))+'.png'
        # titles=[
        #     'Input, 10th iteration high-wavenumber model update',
        #     'Target, low-wavenumber model update',
        #     'Prediction, '+'R2(prediction,target)='+numstr(F_r2(M3,M2))+metric_name+numstr(chosen_metric(M3,M2)),
        #     'Apriori initial model, '+'R2(-||-,target init. model)='+numstr(F_r2(init_model,ideal_init_model))+metric_name+numstr(chosen_metric(init_model,ideal_init_model)),
        #     'Predicted init. model, '+'R2(-||-,target init. model)='+numstr(F_r2(predicted_init_model,ideal_init_model))+metric_name+numstr(chosen_metric(predicted_init_model,ideal_init_model)),
        #     'Target initial model, '+'R2(-||-,true model)='+numstr(F_r2(ideal_init_model,true_model))+metric_name+numstr(chosen_metric(ideal_init_model,true_model)),
        #     'FWI result,'+'R2(FWI result,ideal init. model)='+numstr(F_r2(fwi_result,ideal_init_model))+metric_name+numstr(chosen_metric(fwi_result,ideal_init_model)),
        #     'True model']
        # PLOT_ML_Result_wavenumbers(pics, numstr(chosen_metric(predicted_init_model,true_model)),history_flag=0,
        #     history=None, Boundaries=[], save_file_path=picture_name,
        #     dx=dx,dy=dz,Title=titles, Title2='',Show_flag=0,Save_flag=1,adaptive_colorbar=3)
        os.makedirs(os.path.join(opt.save_path,'predictions_'+str(opt.log_save_const) ),exist_ok=True)        
        if record_weights==1:
            torch.save(generator.state_dict(),os.path.join(opt.save_path,'predictions_'+str(opt.log_save_const),opt.generator_model_name.split('/')[-1]) )
        if record_predictions==1:
            tmp2 = NAME.split('/')[-1]
            path=os.path.join(opt.save_path,'predictions_'+str(opt.log_save_const),tmp2[0:-4]+'_weights_'+str(opt.log_save_const)+'.npz')
            np.savez(path,input_data=M1,output_data=M2,predicted_update=M3,
                models_init=init_model,
                models=true_model,
                predicted_initial_model=predicted_init_model,
                ideal_init_model=ideal_init_model,
                water_taper=taper,
                fwi_result=fwi_result,dx=dx, dz=dz)
            with open(os.path.join(opt.save_path,'predictions_'+str(opt.log_save_const),'opt.txt' ),'w') as f:
                json.dump(opt.__dict__, f, indent=2)
            with open(os.path.join(opt.save_path,'scaling_constants_dict.pkl'),'wb') as output:
                pickle.dump(scaling_constants_dict,output,protocol=4)
    return None

def sample_dataset_backup(dataloader, generator, opt, epoch, flag_show_scaled_data=1, record_weights=0, data_mode='test'):
    """ A-input data,B-target data,C-initial model,D-true model,E-input data in real amplitudes
            ,F-output data in real amplitudes,sc_t-scaler_t
            04.08.21"""
    cuda=True if torch.cuda.is_available() else False
    Tensor=torch.cuda.FloatTensor if cuda else torch.FloatTensor
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for i, batch in enumerate(dataloader):
        NAME=dataloader.dataset.files[i]
        with open(NAME,'rb') as f:
            data = np.load(f, allow_pickle=True)
            taper=data['taper']
            data.close()
        tmp=dataloader.dataset.files[i].split('/')[-1]
        Name=tmp.split('.npz')[0]
        real_A = Variable(batch["A"].type(Tensor))
        real_B = Variable(batch["B"].type(Tensor))
        init_model=Variable(batch["C"].type(Tensor))
        true_model=Variable(batch["D"].type(Tensor))
        real_A_ra = Variable(batch["E"].type(Tensor))
        real_B_ra = Variable(batch["F"].type(Tensor))
        scaler_t=Variable(batch["sc_t"].type(Tensor)).cpu().detach().numpy()
        dx=25;dz=25
        ##############  predict from x
        fake_B=generator(real_A)
        ##############  imresize predicted data
        t_pred=np.empty((real_B_ra.shape))
        Nx=real_B_ra.shape[2]; Nz=real_B_ra.shape[3]
        tmp=fake_B.cpu().detach().numpy()
        for i_f in range(tmp.shape[0]):
            t_pred[i_f,0,:,:]=imresize(tmp[i_f,0,:, :],(Nx,Nz))
        ##############  scale predicted data back to original amplitudes
        fake_B_ra=np.ones_like(t_pred)
        for i_f in range(t_pred.shape[0]):
            tmp=transforming_data_inverse(t_pred[i_f,::],scaler_t[i_f])
            fake_B_ra[i_f,0,:,:]=np.squeeze(tmp)
        ##############  append tapered part of initial model to predicted data to compare it with true model
        # upper_taper_thickness=true_model.shape[3]-Nz
        # zero_tapered_area=np.zeros((fake_B_ra.shape[0],fake_B_ra.shape[1],fake_B_ra.shape[2],upper_taper_thickness))
        # zero_tapered_area=torch.from_numpy(zero_tapered_area).to(device)
        # fake_B_ra=torch.from_numpy(fake_B_ra).to(device)
        # fake_B_ra=torch.cat((zero_tapered_area,fake_B_ra),axis=3)
        # real_A_ra=torch.cat((zero_tapered_area,real_A_ra),axis=3)
        # real_B_ra=torch.cat((zero_tapered_area,real_B_ra),axis=3)
        ##############
        M1=real_A.cpu().detach().numpy().squeeze()
        if M1.ndim==3:
            M1=M1[0,::]
        M2=real_B.cpu().detach().numpy().squeeze()
        M3=fake_B.cpu().detach().numpy().squeeze()
        if torch.is_tensor(fake_B_ra)==True:
            fake_B_ra=fake_B_ra.cpu().detach().numpy().squeeze()
        else:
            fake_B_ra=fake_B_ra.squeeze()
        real_A_ra=real_A_ra.cpu().detach().numpy().squeeze()
        real_B_ra=real_B_ra.cpu().detach().numpy().squeeze()
        init_model=init_model.cpu().detach().numpy().squeeze()
        true_model=true_model.cpu().detach().numpy().squeeze()
        ##############
        fake_B_ra[taper==0]=0
        ##############
        fwi_result=init_model+real_A_ra
        ideal_init_model=init_model+real_B_ra
        predicted_init_model=init_model+fake_B_ra
        pics=[M1,M2,M3,init_model,predicted_init_model,ideal_init_model,fwi_result,true_model]
        picture_name = opt.save_path+'/'+'log_'+data_mode+str(opt.log_save_const)+Name+'_epoch_'+str(epoch)+'_'+numstr(F_r2(M3,M2))+'.png'
        titles = ['Input','Target',
            'Prediction, R2(prediction,target) = '+numstr(F_r2(M3,M2))+',NRMS='+numstr(nrms(M3,M2)),
            'initial model, R2(initial,true)='+numstr(F_r2(init_model,true_model))+',NRMS='+numstr(nrms(init_model,true_model)),
            'Predicted initial model for fwi:R2(predicted initial,true)='+numstr(F_r2(predicted_init_model,true_model))+',NRMS='+numstr(nrms(predicted_init_model,true_model)),
            'Ideal initial model, R2(initial model,true)='+numstr(F_r2(ideal_init_model,true_model))+',NRMS='+numstr(nrms(ideal_init_model,true_model)),
            'FWI result,R2(FWI result,true)='+numstr(F_r2(fwi_result,true_model))+',NRMS='+numstr(nrms(fwi_result,true_model)),
            'True model']
        # PLOT_ML_Result_7_pics(pics, numstr(F_r2(testing_model,M0_show)), history_flag=0,
        #     history=None, Boundaries=[], save_file_path=picture_name,
        #     dx=dx, dy=dz, Title=titles, Title2='', Save_flag=1, adaptive_colorbar=3)
        PLOT_ML_Result_8_pics(pics, numstr(F_r2(predicted_init_model,true_model)),history_flag=0,
            history=None, Boundaries=[], save_file_path=picture_name,
            dx=dx,dy=dz,Title=titles, Title2='', Save_flag=1, adaptive_colorbar=3)
        ###################
        if record_weights==1:
            tmp2 = NAME.split('/')[-1]
            path = opt.save_path+'/'+tmp2[0:-4]+'_weights_'+str(opt.log_save_const)
            os.makedirs(os.path.join(opt.save_path,'predictions_'+str(opt.log_save_const)),exist_ok=True)
            path=os.path.join(opt.save_path,'predictions_'+str(opt.log_save_const),tmp2[0:-4]+'_weights_'+str(opt.log_save_const)+'.npz')
            np.savez(path,input_data=M1,output_data=M2,predicted_update=M3,models_init=init_model,
                models=true_model,predicted_initial_model=predicted_init_model,ideal_init_model=ideal_init_model,
                fwi_result=fwi_result,dx=dx, dz=dz)
    return None

def sample_dataset_old(dataloader, generator, opt, epoch, flag_show_scaled_data=1, record_weights=0, data_mode='test'):
    """DEPRECATED in 07.07.21"""
    cuda = True if torch.cuda.is_available() else False
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    for i, batch in enumerate(dataloader):
        NAME = dataloader.dataset.files[i]
        tmp = dataloader.dataset.files[i].split('/')[-1]
        Name = tmp.split('.npz')[0]
        real_A = Variable(batch["A"].type(Tensor))
        real_B = Variable(batch["B"].type(Tensor))
        # print('sample file ',Name)
        # print('i in the loop ',i)
        # print('batch["A"]=',batch["A"].shape)
        # print('batch["B"]=',batch["B"].shape)
        # print('len(batch)',len(batch))
        # Adversarial ground truths
        valid = Variable(Tensor(np.ones((real_A.size(0), *opt.patch))), requires_grad=False)
        fake = Variable(Tensor(np.zeros((real_A.size(0), *opt.patch))), requires_grad=False)
        fake_B = generator(real_A)
        with open(NAME, 'rb') as f:
            data = np.load(f, allow_pickle=True)
            M0 = data['models'][0, :, :, 0]
            dz = data['dz']
            dx = data['dx']
            M1 = data['input_data']
            M2 = data['output_data']
            # M1_orig_amplitudes=data['input_data_real_amplitudes'];
            # M2_orig_amplitudes=data['output_data_real_amplitudes'];
            Nx = M1.shape[1]
            Nz = M1.shape[2]
            Minit = data['models_init'][0, :, :, 0]
            if 'scaler_x' in data.keys():
                scaler_type = '_individual_scaling'
                scaler_x = data['scaler_x']
                scaler_t = data['scaler_t']
            else:
                from joblib import dump, load
                scaler_type = '_1_scaler'
                scaler_x = load(data_path+'/scaler_x.bin')
                scaler_t = load(data_path+'/scaler_t.bin')
            data.close()
        # imresize square pictures to original size
        M1 = imresize(real_A[0, 0, :, :].cpu().data.numpy().squeeze(), (Nx, Nz))
        M1 = np.expand_dims(M1, axis=[0, -1])
        M2 = imresize(real_B.cpu().data.numpy().squeeze(), (Nx, Nz))
        M2 = np.expand_dims(M2, axis=[0, -1])
        M3 = imresize(fake_B.cpu().data.numpy().squeeze(), (Nx, Nz))
        M3 = np.expand_dims(M3, axis=0)
        M3 = np.expand_dims(M3, axis=-1)
        if flag_show_scaled_data == 0:
            M1 = transforming_data_inverse(M1, scaler_x)
            M2 = transforming_data_inverse(M2, scaler_t)
            M3 = transforming_data_inverse(M3, scaler_t)
            Predicted_update = imresize(
                M3[0, :, :, 0], [M1.shape[1], M1.shape[2]])
            True_update = M2
        else:
            M1_orig_amplitudes = transforming_data_inverse(M1, scaler_x)
            M1_orig_amplitudes = np.squeeze(M1_orig_amplitudes)
            M2_orig_amplitudes = transforming_data_inverse(M2, scaler_t)
            M2_orig_amplitudes = np.squeeze(M2_orig_amplitudes)
            tmp = transforming_data_inverse(M3, scaler_t)
            Predicted_update = tmp[0, :, :, 0]
            True_update = transforming_data_inverse(M2, scaler_t)
        True_update = True_update.squeeze()
        M1 = M1[0, :, :, 0]
        M2 = M2[0, :, :, 0]
        M3 = M3[0, :, :, 0]
        Models_init = Minit
        # append water part to target and prediction
        # if Models_init.shape!=Predicted_update.shape:
        #     Predicted_update=imresize(Predicted_update,Models_init.shape)
        # if Models_init.shape!=True_update.shape:
        #     True_update=imresize(True_update,Models_init.shape)
        water_sz = Models_init.shape[1]-Predicted_update.shape[1]
        water = np.zeros((Nx, water_sz))
        testing_model = Models_init + \
            np.concatenate([water, Predicted_update], axis=1)
        ideal_init_model = Models_init + \
            np.concatenate([water, True_update], axis=1)
        M1_orig_amplitudes = np.concatenate(
            [water, M1_orig_amplitudes], axis=1)
        init_model = Models_init
        M0_show = M0
        # Crop testing models for better visualization
        # water=np.ones((M0_show.shape[0],18))*1500
        # M0_show=np.concatenate([water,M0_show],axis=1)
        # testing_model=np.concatenate([water,testing_model],axis=1)
        # ideal_init_model=np.concatenate([water,ideal_init_model],axis=1)
        # init_model=np.concatenate([water,Models_init],axis=1)
        pics_6=[M1,M2,M3,init_model,testing_model,ideal_init_model,M0_show]
        picture_name = opt.save_path+'/'+'log_'+data_mode+str(opt.log_save_const)+Name+'_epoch_'+str(epoch)+'_'+numstr(F_r2(M3,M2))+'.png'
        titles = ['Input','Target',
            'Prediction, R2(prediction, target) = '+numstr(F_r2(M3,M2))+',NRMS='+numstr(nrms(M3,M2)),
            'initial model, R2(initial,true)='+numstr(F_r2(init_model,M0_show))+',NRMS='+numstr(nrms(init_model,M0_show)),
            'Predicted initial model for fwi:R2(predicted initial,true)='+numstr(F_r2(testing_model,M0_show))+',NRMS='+numstr(nrms(testing_model,M0_show)),
            'Ideal initial model, R2(initial model,true)='+numstr(F_r2(ideal_init_model,M0_show))+',NRMS='+numstr(nrms(ideal_init_model,M0_show)),
            'True model']
        PLOT_ML_Result_7_pics(pics_6, numstr(F_r2(testing_model,M0_show)), history_flag=0,
                                         history=None, Boundaries=[], save_file_path=picture_name,
                                         dx=dx, dy=dz, Title=titles, Title2='', Save_flag=1, adaptive_colorbar=3)
        ###################
        # pics_6 = [M1_orig_amplitudes, M2_orig_amplitudes, M1_orig_amplitudes+init_model, init_model, testing_model, ideal_init_model]
        # # M1_orig_amplitudes M2_orig_amplitudes
        # Prediction_accuracy = F_r2(M1,M2)
        # # prediction_string='input_data+initial model, R2(input_data,target) = ' + numstr(Prediction_accuracy)
        # prediction_string = 'input_data+initial model, R2(-||-,true) = '+numstr(F_r2(pics_6[2], M0_show))
        # picture_name = opt.save_path+'/'+'log_'+data_mode + \
        #     str(opt.log_save_const)+Name+'_epoch_'+str(epoch) + \
        #     '_'+numstr(Prediction_accuracy)+'_analog_pic.png'
        # titles = ['Input', 'Target', prediction_string, 'initial model, R2(initial,true)='+numstr(F_r2(init_model, M0_show)),
        #           'Predicted initial model for fwi, R2(predicted initial,true)='+numstr(F_r2(testing_model,M0_show)), 
        #           'Ideal initial model, R2(initial model,true)='+numstr(F_r2(ideal_init_model, M0_show))]
        # PLOT_ML_Result_adaptive_colorbar(pics_6, numstr(F_r2(testing_model, M0_show)), history_flag=0,
        #                                  history=None, Boundaries=[], save_file_path=picture_name,
        #                                  dx=dx, dy=dz, Title=titles, Title2='', Save_flag=1, adaptive_colorbar=4)
        ###################
        if record_weights == 1:
            # tmp2=NAME.split('augmented_marmousi')
            # path=opt.save_path+'/'+tmp2[1][0:-4]+'_weights_'+str(opt.log_save_const)
            tmp2 = NAME.split('/')[-1]
            path = opt.save_path+'/'+tmp2[0:-4]+'_weights_'+str(opt.log_save_const)
            os.makedirs(os.path.join(opt.save_path,'predictions_'+str(opt.log_save_const)),exist_ok=True)
            path=os.path.join(opt.save_path,'predictions',tmp2[0:-4]+'_weights_'+str(opt.log_save_const))
            np.savez(path, input_data=M1, output_data=M2_orig_amplitudes, models_init=Models_init,
                     models=M0, predicted_update=Predicted_update, predicted_initial_model=testing_model, dx=dx, dz=dz)
        ###################
    return None

def sample_models(dataloader, generator, opt, epoch, flag_show_scaled_data=1, record_weights=0):
    cuda = True if torch.cuda.is_available() else False
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    for i, batch in enumerate(dataloader):
        NAME = dataloader.dataset.files[i]
        tmp = dataloader.dataset.files[i].split('/')[-1]
        Name = tmp.split('.npz')[0]
        real_A = Variable(batch["A"].type(Tensor))
        real_B = Variable(batch["B"].type(Tensor))
        # Adversarial ground truths
        valid = Variable(
            Tensor(np.ones((real_A.size(0), *opt.patch))), requires_grad=False)
        fake = Variable(
            Tensor(np.zeros((real_A.size(0), *opt.patch))), requires_grad=False)
        fake_B = generator(real_A)
        with open(NAME, 'rb') as f:
            data = np.load(f, allow_pickle=True)
            M0_show = data['models'][0, :, :, 0]
            dz = data['dz']
            dx = data['dx']
            data.close()
        # Crop testing models for better visualization
        water = np.ones((M0_show.shape[0], 18))*1500
        M0_show = np.concatenate([water, M0_show], axis=1)
        Plot_image(M0_show.T, Show_flag=1, Save_flag=1, Title=Name, Aspect='equal',
                   c_lim=[1500, 4200], Save_pictures_path=opt.save_path)
    return None

def sample_models2(dataloader, generator, opt, epoch, flag_show_scaled_data=1, record_weights=0):
    cuda = True if torch.cuda.is_available() else False
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    for i, batch in enumerate(dataloader):
        NAME = dataloader.dataset.files[i]
        tmp = dataloader.dataset.files[i].split('/')[-1]
        Name = tmp.split('.npz')[0]
        real_A = Variable(batch["A"].type(Tensor))
        real_B = Variable(batch["B"].type(Tensor))
        # Adversarial ground truths
        valid = Variable(
            Tensor(np.ones((real_A.size(0), *opt.patch))), requires_grad=False)
        fake = Variable(
            Tensor(np.zeros((real_A.size(0), *opt.patch))), requires_grad=False)
        fake_B = generator(real_A)
        with open(NAME, 'rb') as f:
            data = np.load(f, allow_pickle=True)
            M0_show = data['models'][0, :, :, 0]
            dz = data['dz']
            dx = data['dx']
            data.close()
    Data=M0_show

    fig = plt.figure()
    fig.dpi = 330
    plt.set_cmap('RdBu_r')
    # plt.imshow(np.flipud(Data),interpolation='nearest')  #,extent=extent
    # plt.gca().invert_yaxis()
    plt.imshow(Data.T/1000)
    # plt.title(Title)
    ax = plt.gca()
    divider1 = make_axes_locatable((ax))
    cax1 = divider1.append_axes("right", size="2%", pad=0.05)
    cbar = plt.colorbar(cax=cax1)
    cbar.ax.tick_params(labelsize=8)
    # cbar.set_label("(km/s)",fontsize=8)
    cbar.ax.set_title('(km/s)',fontsize=8,pad=5.3,loc='left')
    plt.clim([1.5,4.5])
    ax.axes.xaxis.set_visible(False);  
    ax.axes.yaxis.set_visible(False)
    # plt.ylabel(y_label)
    # plt.xlabel(x_label)
    name = opt.save_path + '/' + Name + '.png'
    plt.savefig(name, bbox_inches='tight')
    print('to '+name)
    # plt.show()
    plt.close()
    return None

def sample_models3(dataloader, generator, opt, epoch, flag_show_scaled_data=1, record_weights=0):
    cuda = True if torch.cuda.is_available() else False
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    for i, batch in enumerate(dataloader):
        NAME = dataloader.dataset.files[i]
        tmp = dataloader.dataset.files[i].split('/')[-1]
        Name = tmp.split('.npz')[0]
        real_A = Variable(batch["A"].type(Tensor))
        real_B = Variable(batch["B"].type(Tensor))
        # Adversarial ground truths
        valid = Variable(
            Tensor(np.ones((real_A.size(0), *opt.patch))), requires_grad=False)
        fake = Variable(
            Tensor(np.zeros((real_A.size(0), *opt.patch))), requires_grad=False)
        fake_B = generator(real_A)
        with open(NAME, 'rb') as f:
            data = np.load(f, allow_pickle=True)
            M0_show = data['models'][0, :, :, 0]
            dz = data['dz']
            dx = data['dx']
            data.close()
        Data=M0_show
        fig = plt.figure()
        fig.dpi = 330
        plt.set_cmap('RdBu_r')
        # plt.imshow(np.flipud(Data),interpolation='nearest')  #,extent=extent
        # plt.gca().invert_yaxis()
        plt.imshow(Data.T/1000)
        ax = plt.gca()
        # divider1 = make_axes_locatable((ax))
        # cax1 = divider1.append_axes("right", size="2%", pad=0.05)
        # cbar = plt.colorbar(cax=cax1)
        # cbar.ax.tick_params(labelsize=8)
        # cbar.set_label("(km/s)",fontsize=8)
        # plt.clim([1.5,4.5])
        ax.axes.xaxis.set_visible(False);  
        ax.axes.yaxis.set_visible(False)
        # plt.ylabel(y_label)
        # plt.xlabel(x_label)
        name = opt.save_path + '/' + Name + '_3.png'
        plt.savefig(name, bbox_inches='tight')
        print('to '+name)
        # plt.show()
        plt.close()
    return None

def sample_dataset_specially(generator, opt, epoch, flag_show_scaled_data=1, record_weights=0):
    """ deprecated """
    import m8r as sf
    cuda = True if torch.cuda.is_available() else False
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    for data_directory in opt.path_test:
        os.listdir(data_directory)
        Name = data_directory.split('/')[-2]
        files_list = glob(data_directory+'/*')
        vsnaps_file = fnmatch.filter(files_list, '*vsnaps*')
        grads_file = fnmatch.filter(files_list, '*grads*')
        f = sf.Input(vsnaps_file[0])
        Nx = f.int("n2")
        Nz = f.int("n1")
        dx = f.int("d2")
        dz = f.int("d1")
        vsnaps = rsf_to_np(vsnaps_file[0])
        grads = rsf_to_np(vsnaps_file[0])
        base_directory = os.path.split(data_directory)[0]
        M0_show = rsf_to_np(os.path.join(base_directory, 'vel.rsf'))
        Minit = rsf_to_np(os.path.join(base_directory, 'init_model.rsf'))
        Models_init = rsf_to_np(os.path.join(
            base_directory, 'init_model_temporary.rsf'))
        Models_init_orig = Models_init
        Models_upsampled = rsf_to_np(os.path.join(base_directory, 'model.rsf'))
        Models_upsampled_ = Models_upsampled
        input_data_real_amplitudes = vsnaps[-1]-Models_init
        output_data_real_amplitudes = np.zeros_like(Models_upsampled)
        output_data_smooth_100 = np.zeros_like(Models_upsampled)
        v_log = Models_upsampled[0, :]
        ind = np.where(v_log == 1500)
        water_bottom = (ind[0][-1]+1)
        # plot data
        Plot_image(M0_show.T,                Show_flag=0, Save_flag=1, Title='M0_show',
                   Aspect='equal', c_lim=[1500, 4200], Save_pictures_path=opt.save_path)
        Plot_image(Models_upsampled.T,  Show_flag=0, Save_flag=1, Title='Models_upsampled',
                   Aspect='equal', c_lim=[1500, 4200], Save_pictures_path=opt.save_path)
        Plot_image(Models_upsampled[:, water_bottom:].T, Show_flag=0, Save_flag=1,
                   Title='Models_upsampled2', Aspect='equal', c_lim=[1500, 4200], Save_pictures_path=opt.save_path)
        Plot_image(Models_init.T,   Show_flag=0, Save_flag=1, Title='Minit_upsampled',
                   Aspect='equal', c_lim=[1500, 4200], Save_pictures_path=opt.save_path)
        Plot_image(Models_init[:, water_bottom:].T,  Show_flag=0, Save_flag=1, Title='Minit_upsampled2',
                   Aspect='equal', c_lim=[1500, 4200], Save_pictures_path=opt.save_path)
        Plot_image(input_data_real_amplitudes.T,        Show_flag=0, Save_flag=1,
                   Title='input_data', Aspect='equal', Save_pictures_path=opt.save_path)
        Plot_image(input_data_real_amplitudes[:, water_bottom:].T,       Show_flag=0,
                   Save_flag=1, Title='input_data2', Aspect='equal', Save_pictures_path=opt.save_path)
        # crop water
        input_data = input_data_real_amplitudes[:, water_bottom:]
        output_data = output_data_real_amplitudes[:, water_bottom:]
        Models_init = Models_init[:, water_bottom:]
        Models_upsampled = Models_upsampled[:, water_bottom:]
        # scaling
        output_data_real_amplitudes = F_smooth(
            Models_upsampled, sigma_val=int(100/dx))-Models_init
        output_data, scaler_t = scaling_data_01(
            output_data_real_amplitudes, preconditioning=False)
        output_data = output_data.squeeze()
        input_data, scaler_x = scaling_data_01(
            input_data, preconditioning=False)
        input_data = input_data.squeeze()
        Plot_image(input_data.T,        Show_flag=0, Save_flag=1,
                   Title='input_data_scaled', Aspect='equal', Save_pictures_path=opt.save_path)
        Plot_image(imresize(input_data.squeeze(), [256, 256]).T, Show_flag=0, Save_flag=1,
                   Title='input_data_resized', Aspect='equal', Save_pictures_path=opt.save_path)
        ss = 1
        # prediction
        fake_B = generator(Variable(torch.tensor(
            imresize(input_data.squeeze(), [1, 1, 256, 256])).cuda()))
        Predicted_update = fake_B.detach().cpu().numpy().squeeze()
        Plot_image(Predicted_update.T, Show_flag=0, Save_flag=1,
                   Title='prediction', Aspect='equal', Save_pictures_path=opt.save_path)

        Predicted_update = imresize(
            Predicted_update, input_data.squeeze().shape)
        Predicted_update_real_amplitudes = transforming_data_inverse(
            Predicted_update, scaler_t).squeeze()
        # append water for better visualization
        water = np.ones((Nx, water_bottom))*1500
        ideal_init_model = Models_init+output_data_real_amplitudes
        testing_model = Models_init+Predicted_update_real_amplitudes
        testing_model = np.concatenate([water, testing_model], axis=1)
        ideal_init_model = np.concatenate([water, ideal_init_model], axis=1)
        ###################
        pics_6 = [input_data, output_data, Predicted_update,
                  Models_init_orig, testing_model, ideal_init_model]
        Prediction_accuracy = F_r2(Predicted_update, output_data)
        R2val = F_r2(testing_model, Models_upsampled_)
        picture_name = opt.save_path+'/'+'log' + \
            str(opt.log_save_const)+Name+'_epoch_'+str(epoch) + \
            '_'+numstr(Prediction_accuracy)+'.png'
        titles = ['Input', 'Target', 'Prediction, R2(prediction, target) = ' + numstr(Prediction_accuracy),
                  'initial model, R2(initial,true)=' +
                                     numstr(
                                         F_r2(Models_init_orig, Models_upsampled_)),
                  'Predicted initial model for fwi, R2(predicted initial,true)='+numstr(R2val), 'Ideal initial model']
        PLOT_ML_Result_adaptive_colorbar(pics_6, numstr(R2val), history_flag=0,
                                         history=None, Boundaries=[], save_file_path=picture_name,
                                         dx=dx, dy=dz, Title=titles, Title2='', Save_flag=1, adaptive_colorbar=3)
        a = 1
    return None

def calculate_accuracy_on_dataset(dataloader, generator, opt, epoch, flag_show_scaled_data=1):
    cuda = True if torch.cuda.is_available() else False
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    scores = []
    names = []
    for i, batch in enumerate(dataloader):
        NAME = dataloader.dataset.files[i]
        tmp = dataloader.dataset.files[i].split('/')[-1]
        Name = tmp.split('.npz')[0]
        real_A = Variable(batch["A"].type(Tensor))
        real_B = Variable(batch["B"].type(Tensor))
        # Adversarial ground truths
        valid = Variable(
            Tensor(np.ones((real_A.size(0), *opt.patch))), requires_grad=False)
        fake = Variable(
            Tensor(np.zeros((real_A.size(0), *opt.patch))), requires_grad=False)
        fake_B = generator(real_A)
        with open(NAME, 'rb') as f:
            data = np.load(f, allow_pickle=True)
            M0 = data['models'][0, :, :, 0]
            dz = data['dz']
            dx = data['dx']
            M1 = data['input_data']
            M2 = data['output_data']
            Nx = M1.shape[1]
            Nz = M1.shape[2]
            Minit = data['models_init'][0, :, :, 0]
            if 'scaler_x' in data.keys():
                scaler_type = '_individual_scaling'
                scaler_x = data['scaler_x']
                scaler_t = data['scaler_t']
            else:
                from joblib import dump, load
                scaler_type = '_1_scaler'
                scaler_x = load(data_path+'/scaler_x.bin')
                scaler_t = load(data_path+'/scaler_t.bin')
            data.close()
        M1 = imresize(real_A.cpu().data.numpy().squeeze(), (Nx, Nz))
        M1 = np.expand_dims(M1, axis=[0, -1])
        M2 = imresize(real_B.cpu().data.numpy().squeeze(), (Nx, Nz))
        M2 = np.expand_dims(M2, axis=[0, -1])
        M3 = imresize(fake_B.cpu().data.numpy().squeeze(), (Nx, Nz))
        M3 = np.expand_dims(M3, axis=0)
        M3 = np.expand_dims(M3, axis=-1)
        if flag_show_scaled_data == 0:
            M1 = transforming_data_inverse(M1, scaler_x)
            M2 = transforming_data_inverse(M2, scaler_t)
            M3 = transforming_data_inverse(M3, scaler_t)
            Predicted_update = imresize(
                M3[0, :, :, 0], [M1.shape[1], M1.shape[2]])
            True_update = M2
        else:
            tmp = transforming_data_inverse(M3, scaler_t)
            Predicted_update = tmp[0, :, :, 0]
            True_update = transforming_data_inverse(M2, scaler_t)
        True_update = True_update[0, :, :, 0]
        M1 = M1[0, :, :, 0]
        M2 = M2[0, :, :, 0]
        M3 = M3[0, :, :, 0]
        Models_init = Minit
        if Models_init.shape != Predicted_update.shape:
            Predicted_update = imresize(Predicted_update, Models_init.shape)
        if Models_init.shape != True_update.shape:
            True_update = imresize(True_update, Models_init.shape)
        testing_model = Models_init+Predicted_update
        ideal_init_model = Models_init+True_update
        M0_show = M0
        # Crop testing models for better visualization
        water = np.ones((M0_show.shape[0], 18))*1500
        M0_show = np.concatenate([water, M0_show], axis=1)
        testing_model = np.concatenate([water, testing_model], axis=1)
        ideal_init_model = np.concatenate([water, ideal_init_model], axis=1)
        inp_orig_sizes = [M1, M2, M3, testing_model, M0_show]
        pics_6 = [M1, M2, M3, M3-M2, testing_model, ideal_init_model]
        # inp_orig_sizes=[M1,M2,M3,testing_model,ideal_init_model]
        saving_name = NAME.split('augmented_marmousi_10_it')[-1]
        # Plot_image(testing_model.T,Show_flag=1,Save_flag=1,Title='testing_model1'+saving_name,Aspect='equal',Save_pictures_path=Save_pictures_path)
        ####
        Prediction_accuracy = F_r2(M3, M2)
        R2val = F_r2(testing_model, M0_show)
        # R2val=F_r2(pics_6[-2],pics_6[-1])
        # R2val2=F_r2(testing_model,ideal_init_model)
        # tmp=NAME.split('augmented_marmousi_10_it')[-1]
        title = 'Prediction, R2(prediction, target) = ' + \
            numstr(Prediction_accuracy)
        Name = opt.save_path+'/'+'log' + \
            str(opt.log_save_const)+Name+'_epoch_'+str(epoch) + \
            '_'+numstr(Prediction_accuracy)+'.png'
        scores.append(Prediction_accuracy)
        names.append(Name)
        PLOT_ML_Result_adaptive_colorbar(pics_6, numstr(R2val), history_flag=0,
                                         history=None, Boundaries=[], save_file_path=Name,
                                         dx=dx, dy=dz, Title=title, Title2='', Save_flag=1, adaptive_colorbar=3)
        qq = 1
    return names, scores

def comparison_initial_models_with_fwi_m(paths,fname,log_path='./',save_path='./'):
    d=[]
    for path in paths:
        d.append(parse_denise_folder(path,denise_root='./') )
    ################################
    labels_Fontsize=18
    text_Fontsize=30
    labelsize=14
    Fontsize=32    
    textFontsize=20
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 12.4/2*3
    fig_size[1] = 8.0   #height
    plt.rcParams["figure.figsize"] = fig_size
    plt.rcParams["font.size"] = 14
    n_row=3;    n_col=3
    gs = gridspec.GridSpec(n_row,n_col)
    gs.update(left=0.0, right=0.92, wspace=0.0, hspace=0.2)
    # gs=GridSpec(n_row,n_col+1, width_ratios=[1,1,0.1], height_ratios=[1,1,1])
    # gs.update(wspace=0.0,hspace=0.0)
    ax=[None]*9
    labels=['a','a','b','b','c','c','a','b','c','j','k','l','m','n','o']
    a=1500;b=4500

    fig=plt.figure()
    j=0;D=d[j]
    nx_orig=D.NX-320
    ind=D.fwi_model_names.index(os.path.join(D.save_folder,'model','modelTest_vp.bin'))
    ind=D.fwi_model_names.index( fnmatch.filter(D.fwi_model_names,'*vp_stage*')[-1] )
    fwi_res=D.models[ind]
    x = np.arange(nx_orig) * D.DH / 1000
    y = np.arange(D.NY) * D.DH / 1000
    extent = np.array([x.min(), x.max(), y.min(), y.max()])
    i=0; row=0; col=0
    ax[i]=fig.add_subplot(gs[row,col])
    ax[i].axes.xaxis.set_visible(False)
    # ax[i].axes.yaxis.set_visible(False)
    ax[i].text(0.4,0.7, labels[i], fontsize=Fontsize,color = "white",weight="bold")
    score=numstr_3(F_r2(D.model_init.vp[:,0:nx_orig],D.model.vp[:,0:nx_orig]))
    ax[i].set_title('R2(initial,true)='+score,fontsize=textFontsize)
    ax[i].imshow(D.model_init.vp[:,0:nx_orig],extent=extent,vmin=a,vmax=b,aspect='auto');ax[i].invert_yaxis()
    ax[i].set_ylabel('Depth, km',fontsize=labels_Fontsize)
    ax[i].tick_params(labelsize=labelsize)
    i=1; row=0; col=1
    ax[i]=fig.add_subplot(gs[row,col]); ax[i].axes.xaxis.set_visible(False);  ax[i].axes.yaxis.set_visible(False)
    ax[i].text(0.4,0.7, labels[i], fontsize=Fontsize,color = "white",weight="bold")
    score=numstr_3(F_r2(fwi_res[:,0:nx_orig],D.model.vp[:,0:nx_orig]))
    ax[i].set_title('R2(FWI result,true)='+score,fontsize=textFontsize)
    ax[i].imshow(fwi_res[:,0:nx_orig],extent=extent,vmin=a,vmax=b,aspect='auto');ax[i].invert_yaxis()
    ax[i].tick_params(labelsize=labelsize)
    ##
    j=1;D=d[j]
    ind=D.fwi_model_names.index(os.path.join(D.save_folder,'model','modelTest_vp.bin'))
    ind=D.fwi_model_names.index( fnmatch.filter(D.fwi_model_names,'*vp_stage*')[-1] )
    fwi_res=D.models[ind]
    extent = np.array([x.min(), x.max(), y.min(), y.max()])
    i=2; row=1; col=0
    ax[i]=fig.add_subplot(gs[row,col]); 
    ax[i].axes.xaxis.set_visible(False);  
    # ax[i].axes.yaxis.set_visible(False)
    ax[i].text(0.4,0.7, labels[i], fontsize=Fontsize,color = "white",weight="bold")
    score=numstr_3(F_r2(D.model_init.vp[:,0:nx_orig],D.model.vp[:,0:nx_orig]))
    ax[i].set_title('R2(initial,true)='+score,fontsize=textFontsize)
    ax[i].imshow(D.model_init.vp[:,0:nx_orig],extent=extent,vmin=a,vmax=b,aspect='auto');ax[i].invert_yaxis()
    ax[i].tick_params(labelsize=labelsize)
    ax[i].set_ylabel('Depth, km',fontsize=labels_Fontsize)
    i=3; row=1; col=1
    ax[i]=fig.add_subplot(gs[row,col]); ax[i].axes.xaxis.set_visible(False);  ax[i].axes.yaxis.set_visible(False)
    ax[i].text(0.4,0.7, labels[i], fontsize=Fontsize,color = "white",weight="bold")
    score=numstr_3(F_r2(fwi_res[:,0:nx_orig],D.model.vp[:,0:nx_orig]))
    ax[i].set_title('R2(FWI result,true)='+score,fontsize=textFontsize)
    ax[i].imshow(fwi_res[:,0:nx_orig],extent=extent,vmin=a,vmax=b,aspect='auto');ax[i].invert_yaxis()
    ax[i].tick_params(labelsize=labelsize)
    ##
    j=2;D=d[j]
    ind=D.fwi_model_names.index(os.path.join(D.save_folder,'model','modelTest_vp.bin'))
    ind=D.fwi_model_names.index( fnmatch.filter(D.fwi_model_names,'*vp_stage*')[-1] )
    fwi_res=D.models[ind]
    extent = np.array([x.min(), x.max(), y.min(), y.max()])
    i=4; row=2; col=0
    ax[i]=fig.add_subplot(gs[row,col]); 
    # ax[i].axes.xaxis.set_visible(False);  
    # ax[i].axes.yaxis.set_visible(False)
    ax[i].text(0.4,0.7, labels[i], fontsize=Fontsize,color = "white",weight="bold")
    score=numstr_3(F_r2(D.model_init.vp[:,0:nx_orig],D.model.vp[:,0:nx_orig]))
    ax[i].set_title('R2(initial,true)='+score,fontsize=textFontsize)
    ax[i].imshow(D.model_init.vp[:,0:nx_orig],extent=extent,vmin=a,vmax=b,aspect='auto');ax[i].invert_yaxis()
    ax[i].set_xlabel('X, km',fontsize=labels_Fontsize)
    ax[i].set_ylabel('Depth, km',fontsize=labels_Fontsize)
    ax[i].tick_params(labelsize=labelsize)

    i=5; row=2; col=1
    ax[i]=fig.add_subplot(gs[row,col]); 
    # ax[i].axes.xaxis.set_visible(False);  
    ax[i].axes.yaxis.set_visible(False)
    ax[i].text(0.4,0.7, labels[i], fontsize=Fontsize,color = "white",weight="bold")
    ax[i].set_xlabel('X, km',fontsize=labels_Fontsize)
    score=numstr_3(F_r2(fwi_res[:,0:nx_orig],D.model.vp[:,0:nx_orig]))
    ax[i].set_title('R2(FWI result,true)='+score,fontsize=textFontsize)
    ax[i].tick_params(labelsize=labelsize)
    last_image=ax[i].imshow(fwi_res[:,0:nx_orig],extent=extent,vmin=a,vmax=b,aspect='auto');ax[i].invert_yaxis()
    ##
    cbar_ax = fig.add_axes([0.93, 0.12, 0.02, 0.75])
    cbar=fig.colorbar(last_image,cax=cbar_ax)
    # cbar.set_label('V (km/sec)',labelpad=-40, y=1.05, rotation=0)   #horizontalalignment='center'
    cbar.ax.set_title('V (m/sec)',fontsize=18,pad=13.3)
    for t in cbar.ax.get_yticklabels():
        t.set_fontsize(16)
    ##############################  misfits
    niter_list=[];misfit_list=[]

    j=0;D=d[j];path=paths[j]
    i=6; row=0; col=2
    m,m_data=plot_fwi_misfit2(os.path.join(path,'fld','seis_fwi_log.dat'),D)
    nstage=m_data['nstage']
    niter_stage=m_data['niter_stage'];
    nstage_trans=m_data['nstage_trans']
    iteration=m_data['iteration']
    niter_list.append(len(iteration))
    misfit_list.append(m)
    ax[i]=fig.add_subplot(gs[row,col])
    ax[i].plot(iteration,m,'b-', linewidth=3, label='Evolution of the misfit function')
    ax[i].tick_params(labelsize=labelsize)
    for ii in range(1, nstage):
        ax[i].plot([nstage_trans[ii-1]+ii,nstage_trans[ii-1]+ii], [np.min(m),np.max(m)],'k--', linewidth=3)
    ax[i].axes.yaxis.set_visible(False)

    j=1;D=d[j];path=paths[j]
    i=7; row=1; col=2
    m,m_data=plot_fwi_misfit2(os.path.join(path,'fld','seis_fwi_log.dat'),D)
    nstage=m_data['nstage'];
    niter_stage=m_data['niter_stage'];
    nstage_trans=m_data['nstage_trans']
    iteration=m_data['iteration']
    niter_list.append(len(iteration))
    misfit_list.append(m)
    ax[i]=fig.add_subplot(gs[row,col])
    ax[i].plot(iteration,m,'b-', linewidth=3, label='Evolution of the misfit function')
    ax[i].tick_params(labelsize=labelsize)
    for ii in range(1, nstage):
        ax[i].plot([nstage_trans[ii-1]+ii,nstage_trans[ii-1]+ii], [np.min(m),np.max(m)],'k--', linewidth=3)
    ax[i].axes.yaxis.set_visible(False)

    j=2;D=d[j];path=paths[j]
    i=8; row=2; col=2
    m,m_data=plot_fwi_misfit2(os.path.join(path,'fld','seis_fwi_log.dat'),D)
    nstage=m_data['nstage'];
    niter_stage=m_data['niter_stage']
    nstage_trans=m_data['nstage_trans']
    iteration=m_data['iteration'];
    niter_list.append(len(iteration))
    misfit_list.append(m)
    ax[i]=fig.add_subplot(gs[row,col])
    ax[i].plot(iteration,m,'b-', linewidth=3, label='Evolution of the misfit function')
    ax[i].tick_params(labelsize=labelsize)
    ax[i].set_xlabel('Iteration number',fontsize=labels_Fontsize)
    for ii in range(1, nstage):
        ax[i].plot([nstage_trans[ii-1]+ii,nstage_trans[ii-1]+ii], [np.min(m),np.max(m)],'k--',linewidth=3)
    ax[i].axes.yaxis.set_visible(False)
    ######  x limits
    niter_list=np.asarray(niter_list)
    misfit_list=np.asarray(misfit_list)
    mf_min=0;mf_max=0;
    for misfit_curve in misfit_list:
        mf_min=np.min([np.min(misfit_curve),mf_min])
        mf_max=np.max([np.max(misfit_curve),mf_max])
    ax[6].set_xlim(left=1,right=np.max(niter_list))
    ax[7].set_xlim(left=1,right=np.max(niter_list))
    ax[8].set_xlim(left=1,right=np.max(niter_list))
    ax[6].set_ylim(bottom=mf_min,top=mf_max)
    ax[7].set_ylim(bottom=mf_min,top=mf_max)
    ax[8].set_ylim(bottom=mf_min,top=mf_max)
    y_pos=mf_min+(mf_max-mf_min)/5*4
    ax[6].text(10,y_pos, labels[6], fontsize=Fontsize,color = "black",weight="bold")
    ax[7].text(10,y_pos, labels[7], fontsize=Fontsize,color = "black",weight="bold")
    ax[8].text(10,y_pos, labels[8], fontsize=Fontsize,color = "black",weight="bold")
    ######  saving
    save_file_path1=os.path.join(save_path,fname)
    save_file_path2=os.path.join(log_path,fname)
    print('Saving ML_result to '+save_file_path1+'  '+save_file_path2)
    plt.savefig(save_file_path1,dpi=300,bbox_inches='tight')
    plt.savefig(save_file_path2,dpi=300,bbox_inches='tight')
    # plt.show()
    plt.close()
    return None

def PLOT_ML_Result_adaptive_colorbar(inp, val1, history_flag=0, history=None,
        Boundaries=0, save_file_path='', dx=1, dy=1, Plot_vertical_lines=0, Title='',
        Title2='', Save_flag=0, Show_flag=0, adaptive_colorbar=1):
    Nm = len(inp)
    dim1 = inp[0].shape[0]
    dim2 = inp[0].shape[1]
    if dim1 >= dim2:
        Nx = dim1
        Nz = dim2
        for i in range(Nm):
            inp[i] = inp[i].swapaxes(0, 1)
    else:
        Nx = dim2
        Nz = dim1
    #   flip z axis
    for i in range(Nm):
        inp[i] = np.flip(inp[i], axis=0)
    input = inp[0]
    output = inp[1]
    pred = inp[2]
    matrix = inp[4]
    true_model=inp[5]
    x = np.arange(Nx) * dx / 1000
    y = np.arange(Nz) * dy / 1000
    extent = np.array([x.min(), x.max(), y.min(), y.max()])
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 12.4
    fig_size[1] = 14.0
    plt.rcParams["figure.figsize"] = fig_size
    plt.rcParams["font.size"] = 14
    fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(nrows=6, ncols=1)
    #  allocate space for colorbar smartly
    divider1 = make_axes_locatable((ax1))
    divider2 = make_axes_locatable((ax2))
    divider3 = make_axes_locatable((ax3))
    divider4 = make_axes_locatable((ax4))
    divider5 = make_axes_locatable((ax5))
    divider6 = make_axes_locatable((ax6))
    cax1 = divider1.append_axes("right", size="2%", pad=0.05)
    cax2 = divider2.append_axes("right", size="2%", pad=0.05)
    cax3 = divider3.append_axes("right", size="2%", pad=0.05)
    cax4 = divider4.append_axes("right", size="2%", pad=0.05)
    cax5 = divider5.append_axes("right", size="2%", pad=0.05)
    cax6 = divider6.append_axes("right", size="2%", pad=0.05)
    plt.set_cmap('RdBu_r')
    MIN = np.min(output)
    MAX = np.max(output)
    orig_min = np.min(inp[5])
    orig_max = np.max(inp[5])
    if adaptive_colorbar == 4:
        a = np.min((output))
        b = np.max((output))
        im1 = ax1.imshow(input, extent=extent, vmin=a, vmax=b, aspect='auto')
        im2 = ax2.imshow(output, extent=extent, vmin=a, vmax=b, aspect='auto')
        im3 = ax3.imshow(pred, extent=extent, vmin=orig_min,
                            vmax=orig_max, aspect='auto')
        im4 = ax4.imshow(inp[3], extent=extent,
                            vmin=orig_min, vmax=orig_max, aspect='auto')
    if adaptive_colorbar == 3:
        a = np.min((output))
        b = np.max((output))
        im1 = ax1.imshow(input, extent=extent, vmin=a, vmax=b, aspect='auto')
        im2 = ax2.imshow(output, extent=extent, vmin=a, vmax=b, aspect='auto')
        im3 = ax3.imshow(pred, extent=extent, vmin=a, vmax=b, aspect='auto')
        im4 = ax4.imshow(inp[3], extent=extent,
                            vmin=orig_min, vmax=orig_max, aspect='auto')
    if adaptive_colorbar == 2:
        val = np.max(abs(input))
        im1 = ax1.imshow(input, extent=extent, aspect='auto')
        val = np.max(abs(output))
        im2 = ax2.imshow(output, extent=extent, vmin=-
                            val,  vmax=val, aspect='auto')
        im3 = ax3.imshow(pred, extent=extent,   vmin=-
                            val,  vmax=val, aspect='auto')
        im4 = ax4.imshow(inp[3], extent=extent, vmin=-
                            val, vmax=val, aspect='auto')
    if adaptive_colorbar == 1:
        val = np.max(abs(input))
        im1 = ax1.imshow(input, extent=extent, vmin=-
                            val/3, vmax=val/3, aspect='auto')
        val = np.max(abs(output))
        im2 = ax2.imshow(output, extent=extent, vmin=-
                            val,  vmax=val, aspect='auto')
        im3 = ax3.imshow(pred, extent=extent,   vmin=-
                            val,  vmax=val, aspect='auto')
        im4 = ax4.imshow(inp[3], extent=extent, vmin=-
                            val/2, vmax=val/2, aspect='auto')
    if adaptive_colorbar == 0:
        min_lim = -1
        max_lim = 1
        im1 = ax1.imshow(input, extent=extent, vmin=min_lim,
                            vmax=max_lim, aspect='auto')
        im2 = ax2.imshow(output, extent=extent, vmin=min_lim,
                            vmax=max_lim, aspect='auto')
        im3 = ax3.imshow(pred, extent=extent,   vmin=min_lim,
                            vmax=max_lim, aspect='auto')
        im4 = ax4.imshow(inp[3], extent=extent,
                            vmin=min_lim/2, vmax=max_lim/2, aspect='auto')
    tmp = matrix
    tmp_extent = np.array([0, tmp.shape[1]*dx/1000, 0, tmp.shape[0]*dy / 1000])
    im5 = ax5.imshow(matrix, extent=tmp_extent, vmin=orig_min,
                        vmax=orig_max, aspect='auto')
    tmp = true_model
    tmp_extent = np.array([0, tmp.shape[1]*dx/1000, 0, tmp.shape[0]*dy / 1000])
    im6 = ax6.imshow(true_model, extent=tmp_extent,
                        vmin=orig_min, vmax=orig_max, aspect='auto')
    x0 = 0.04
    y0 = -0.25
    if Plot_vertical_lines == 1:
        Boundaries2 = Boundaries * dx / 1000
        for i in range(np.shape(Boundaries)[0]):
            ax1.axvline(Boundaries2[i], color='k',
                        linestyle='--', linewidth=2.5)
            ax2.axvline(Boundaries2[i], color='k',
                        linestyle='--', linewidth=2.5)
            ax3.axvline(Boundaries2[i], color='k',
                        linestyle='--', linewidth=2.5)
    z_label_name = 'z, m'
    ax1.set_ylabel(z_label_name)
    ax1.set_xlabel('x (km)')
    ax2.set_ylabel(z_label_name)
    ax2.set_xlabel('x (km)')
    ax3.set_ylabel(z_label_name)
    ax3.set_xlabel('x (km)')
    ax4.set_ylabel(z_label_name)
    ax4.set_xlabel('x (km)')
    ax5.set_ylabel(z_label_name)
    ax5.set_xlabel('x (km)')
    ax6.set_ylabel(z_label_name)
    ax6.set_xlabel('x (km)')
    ax1.invert_yaxis()
    ax2.invert_yaxis()
    ax3.invert_yaxis()
    ax4.invert_yaxis()
    ax5.invert_yaxis()
    ax6.invert_yaxis()

    ax1.xaxis.set_label_coords(x0, y0)
    ax2.xaxis.set_label_coords(x0, y0)
    ax3.xaxis.set_label_coords(x0, y0)
    ax4.xaxis.set_label_coords(x0, y0)
    ax5.xaxis.set_label_coords(x0, y0)
    ax6.xaxis.set_label_coords(x0, y0)

    ax1.set_title(Title[0])
    ax2.set_title(Title[1])
    ax3.set_title(Title[2])
    ax4.set_title(Title[3])
    ax5.set_title(Title[4])
    ax6.set_title(Title[5])

    cbar1 = plt.colorbar(im1, cax=cax1)
    cbar2 = plt.colorbar(im2, cax=cax2)
    cbar3 = plt.colorbar(im3, cax=cax3)
    cbar4 = plt.colorbar(im4, cax=cax4)
    cbar5 = plt.colorbar(im5, cax=cax5)
    cbar6 = plt.colorbar(im6, cax=cax6)
    x0 = -40    
    y0 = 1.16
    cbar1.set_label('V (m/sec)', labelpad=x0, y=y0, rotation=0)
    cbar2.set_label('V (m/sec)', labelpad=x0, y=y0, rotation=0)
    cbar3.set_label('V (m/sec)', labelpad=x0, y=y0, rotation=0)
    cbar4.set_label('V (m/sec)', labelpad=x0, y=y0, rotation=0)
    cbar5.set_label('V (m/sec)', labelpad=x0, y=y0, rotation=0)
    cbar6.set_label('V (m/sec)', labelpad=x0, y=y0, rotation=0)
    plt.subplots_adjust(bottom=0.08, top=0.95,
                        hspace=0.57, right=0.86, wspace=.1)
    if Save_flag == 1:
        plt.savefig(save_file_path, dpi=300)
        print('Saving ML_result to '+save_file_path)
    if Show_flag == 1:
        plt.show()
    else:
        plt.show(block=False)
    plt.close()
    return None

def PLOT_ML_Result_7_pics(inp, val1, history_flag=0, history=None,
        Boundaries=0, save_file_path='', dx=1, dy=1, Plot_vertical_lines=0, Title='',
        Title2='', Save_flag=0, Show_flag=0, adaptive_colorbar=1):
    Nm = len(inp)
    dim1 = inp[0].shape[0]
    dim2 = inp[0].shape[1]
    if dim1 >= dim2:
        Nx = dim1
        Nz = dim2
        for i in range(Nm):
            inp[i] = inp[i].swapaxes(0, 1)
    else:
        Nx = dim2
        Nz = dim1
    #   flip z axis
    for i in range(Nm):
        inp[i] = np.flip(inp[i], axis=0)
    input = inp[0]
    output = inp[1]
    pred = inp[2]
    matrix = inp[4]
    true_model=inp[5]
    x = np.arange(Nx) * dx / 1000
    y = np.arange(Nz) * dy / 1000
    extent = np.array([x.min(), x.max(), y.min(), y.max()])
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 12.4
    fig_size[1] = 14.0
    plt.rcParams["figure.figsize"] = fig_size
    plt.rcParams["font.size"] = 14
    fig, (ax1, ax2, ax3, ax4, ax5, ax6,ax7) = plt.subplots(nrows=7,ncols=1)
    #  allocate space for colorbar smartly
    divider1 = make_axes_locatable((ax1))
    divider2 = make_axes_locatable((ax2))
    divider3 = make_axes_locatable((ax3))
    divider4 = make_axes_locatable((ax4))
    divider5 = make_axes_locatable((ax5))
    divider6 = make_axes_locatable((ax6))
    divider7 = make_axes_locatable((ax7))
    
    cax1 = divider1.append_axes("right", size="2%", pad=0.05)
    cax2 = divider2.append_axes("right", size="2%", pad=0.05)
    cax3 = divider3.append_axes("right", size="2%", pad=0.05)
    cax4 = divider4.append_axes("right", size="2%", pad=0.05)
    cax5 = divider5.append_axes("right", size="2%", pad=0.05)
    cax6 = divider6.append_axes("right", size="2%", pad=0.05)
    cax7 = divider7.append_axes("right", size="2%", pad=0.05)
    plt.set_cmap('RdBu_r')
    MIN = np.min(output)
    MAX = np.max(output)
    orig_min = np.min(inp[5])
    orig_max = np.max(inp[5])
    if adaptive_colorbar == 4:
        a = np.min((output))
        b = np.max((output))
        im1 = ax1.imshow(input, extent=extent, vmin=a, vmax=b, aspect='auto')
        im2 = ax2.imshow(output, extent=extent, vmin=a, vmax=b, aspect='auto')
        im3 = ax3.imshow(pred, extent=extent, vmin=orig_min,
                            vmax=orig_max, aspect='auto')
        im4 = ax4.imshow(inp[3], extent=extent,
                            vmin=orig_min, vmax=orig_max, aspect='auto')
    if adaptive_colorbar == 3:
        a = np.min((output))
        b = np.max((output))
        im1 = ax1.imshow(input, extent=extent, vmin=a, vmax=b, aspect='auto')
        im2 = ax2.imshow(output, extent=extent, vmin=a, vmax=b, aspect='auto')
        im3 = ax3.imshow(pred, extent=extent, vmin=a, vmax=b, aspect='auto')
        im4 = ax4.imshow(inp[3], extent=extent,vmin=orig_min, vmax=orig_max, aspect='auto')
    if adaptive_colorbar == 2:
        val = np.max(abs(input))
        im1 = ax1.imshow(input, extent=extent, aspect='auto')
        val = np.max(abs(output))
        im2 = ax2.imshow(output, extent=extent, vmin=-
                            val,  vmax=val, aspect='auto')
        im3 = ax3.imshow(pred, extent=extent,   vmin=-
                            val,  vmax=val, aspect='auto')
        im4 = ax4.imshow(inp[3], extent=extent, vmin=-
                            val, vmax=val, aspect='auto')
    if adaptive_colorbar == 1:
        val = np.max(abs(input))
        im1 = ax1.imshow(input, extent=extent, vmin=-
                            val/3, vmax=val/3, aspect='auto')
        val = np.max(abs(output))
        im2 = ax2.imshow(output, extent=extent, vmin=-
                            val,  vmax=val, aspect='auto')
        im3 = ax3.imshow(pred, extent=extent,   vmin=-
                            val,  vmax=val, aspect='auto')
        im4 = ax4.imshow(inp[3], extent=extent, vmin=-
                            val/2, vmax=val/2, aspect='auto')
    if adaptive_colorbar == 0:
        min_lim = -1
        max_lim = 1
        im1 = ax1.imshow(input, extent=extent, vmin=min_lim,
                            vmax=max_lim, aspect='auto')
        im2 = ax2.imshow(output, extent=extent, vmin=min_lim,
                            vmax=max_lim, aspect='auto')
        im3 = ax3.imshow(pred, extent=extent,   vmin=min_lim,
                            vmax=max_lim, aspect='auto')
        im4 = ax4.imshow(inp[3], extent=extent,
                            vmin=min_lim/2, vmax=max_lim/2, aspect='auto')
    tmp = inp[4]
    tmp_extent = np.array([0, tmp.shape[1]*dx/1000, 0, tmp.shape[0]*dy / 1000])
    im5 = ax5.imshow(inp[4], extent=tmp_extent, vmin=orig_min,vmax=orig_max, aspect='auto')
    tmp = inp[5]
    tmp_extent = np.array([0, tmp.shape[1]*dx/1000, 0, tmp.shape[0]*dy / 1000])
    im6 = ax6.imshow(inp[5], extent=tmp_extent,vmin=orig_min, vmax=orig_max, aspect='auto')
    tmp = inp[6]
    tmp_extent = np.array([0, tmp.shape[1]*dx/1000, 0, tmp.shape[0]*dy / 1000])
    im7 = ax7.imshow(inp[6], extent=tmp_extent,vmin=orig_min, vmax=orig_max, aspect='auto')
    #########
    x0 = 0.04
    y0 = -0.25
    if Plot_vertical_lines == 1:
        Boundaries2 = Boundaries * dx / 1000
        for i in range(np.shape(Boundaries)[0]):
            ax1.axvline(Boundaries2[i], color='k',
                        linestyle='--', linewidth=2.5)
            ax2.axvline(Boundaries2[i], color='k',
                        linestyle='--', linewidth=2.5)
            ax3.axvline(Boundaries2[i], color='k',
                        linestyle='--', linewidth=2.5)
    z_label_name = 'z, m'
    ax1.set_ylabel(z_label_name)
    ax1.set_xlabel('x (km)')
    ax2.set_ylabel(z_label_name)
    ax2.set_xlabel('x (km)')
    ax3.set_ylabel(z_label_name)
    ax3.set_xlabel('x (km)')
    ax4.set_ylabel(z_label_name)
    ax4.set_xlabel('x (km)')
    ax5.set_ylabel(z_label_name)
    ax5.set_xlabel('x (km)')
    ax6.set_ylabel(z_label_name)
    ax6.set_xlabel('x (km)')
    ax7.set_ylabel(z_label_name)
    ax7.set_xlabel('x (km)')
    ax1.invert_yaxis()
    ax2.invert_yaxis()
    ax3.invert_yaxis()
    ax4.invert_yaxis()
    ax5.invert_yaxis()
    ax6.invert_yaxis()
    ax7.invert_yaxis()

    ax1.xaxis.set_label_coords(x0, y0)
    ax2.xaxis.set_label_coords(x0, y0)
    ax3.xaxis.set_label_coords(x0, y0)
    ax4.xaxis.set_label_coords(x0, y0)
    ax5.xaxis.set_label_coords(x0, y0)
    ax6.xaxis.set_label_coords(x0, y0)
    ax7.xaxis.set_label_coords(x0, y0)

    ax1.set_title(Title[0])
    ax2.set_title(Title[1])
    ax3.set_title(Title[2])
    ax4.set_title(Title[3])
    ax5.set_title(Title[4])
    ax6.set_title(Title[5])
    ax7.set_title(Title[6])

    cbar1 = plt.colorbar(im1, cax=cax1)
    cbar2 = plt.colorbar(im2, cax=cax2)
    cbar3 = plt.colorbar(im3, cax=cax3)
    cbar4 = plt.colorbar(im4, cax=cax4)
    cbar5 = plt.colorbar(im5, cax=cax5)
    cbar6 = plt.colorbar(im6, cax=cax6)
    cbar7 = plt.colorbar(im7, cax=cax7)
    x0 = -40    
    y0 = 1.16
    cbar1.set_label('V (m/sec)', labelpad=x0, y=y0, rotation=0)
    cbar2.set_label('V (m/sec)', labelpad=x0, y=y0, rotation=0)
    cbar3.set_label('V (m/sec)', labelpad=x0, y=y0, rotation=0)
    cbar4.set_label('V (m/sec)', labelpad=x0, y=y0, rotation=0)
    cbar5.set_label('V (m/sec)', labelpad=x0, y=y0, rotation=0)
    cbar6.set_label('V (m/sec)', labelpad=x0, y=y0, rotation=0)
    cbar7.set_label('V (m/sec)', labelpad=x0, y=y0, rotation=0)
    plt.subplots_adjust(bottom=0.08,top=0.95,hspace=0.57, right=0.86, wspace=.1)
    if Save_flag == 1:
        plt.savefig(save_file_path, dpi=300)
        print('Saving ML_result to '+save_file_path)
    if Show_flag == 1:
        plt.show()
    else:
        plt.show(block=False)
    plt.close()
    return None

def PLOT_ML_Result_all_inputs(inp_orig, val1, history_flag=0, history=None,
        Boundaries=0, save_file_path='', dx=1, dy=1, Plot_vertical_lines=0, Title='',
        Title2='', Save_flag=0, Show_flag=0, adaptive_colorbar=1):
    inp=inp_orig
    Nm = len(inp)
    dim1 = inp[0].shape[0]
    dim2 = inp[0].shape[1]
    if dim1 >= dim2:
        Nx = dim1
        Nz = dim2
        for i in range(Nm):
            if inp[i].ndim==2:
                inp[i] = inp[i].swapaxes(0, 1)
            else:
                inp[i] = inp[i].swapaxes(1,2)
    else:
        Nx = dim2
        Nz = dim1
    #   flip z axis
    for i in range(Nm):
        if inp[i].ndim==2:  inp[i] = np.flip(inp[i], axis=0)
        else:   inp[i] = np.flip(inp[i], axis=1)
    input = inp[0]
    output = inp[1]
    pred = inp[2]
    predicted_initial = inp[4];    
    ideal_initial=inp[5]
    fwi_result=inp[6]
    true_model=inp[7]
    features=inp[8]
    x = np.arange(Nx) * dx / 1000
    y = np.arange(Nz) * dy / 1000
    extent = np.array([x.min(), x.max(), y.min(), y.max()])
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 12.4
    fig_size[1] = 14.0
    plt.rcParams["figure.figsize"] = fig_size
    plt.rcParams["font.size"] = 14
    ################################
    features.shape
    a = np.min((output));   b = np.max((output))
    #### dm_i_1init_1sign_1taper
    fig=plt.figure()
    n_row=5;    n_col=5
    gs=GridSpec(n_row,n_col) # 2 rows, 3 columns
    gs.update(wspace=0.0,hspace=0.0)
    ax=[None]*15
    labels=['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q']
    for i in range(features.shape[0]):
        row=int(np.floor(i/n_row))
        col=i-row*n_row
        # print(i,row,col)
        ax[i]=fig.add_subplot(gs[row,col]); ax[i].axes.xaxis.set_visible(False);  ax[i].axes.yaxis.set_visible(False)
        ax[i].text(0.4,0.7, labels[i], fontsize=35)
        if i<10:
            ax[i].imshow(features[i,::], extent=extent, vmin=a,vmax=b, aspect='auto');
        elif i==10: 
            ax[i].imshow(features[i,::], extent=extent,aspect='auto');
        elif i==11: 
            ax[i].imshow(features[i,::], extent=extent, vmin=-1,vmax=1, aspect='auto');
        elif i==12: 
            ax[i].imshow(features[i,::], extent=extent, vmin=0,vmax=1, aspect='auto');
        ax[i].invert_yaxis()
    col=col+1; i=i+1
    ax[i]=fig.add_subplot(gs[row,col]); ax[i].axes.xaxis.set_visible(False);  ax[i].axes.yaxis.set_visible(False)
    ax[i].text(0.4,0.7, labels[i], fontsize=35)
    ax[i].imshow(output, extent=extent, vmin=a,vmax=b, aspect='auto')
    ax[i].invert_yaxis()
    ####
    i=i+1
    ax_3=fig.add_subplot(gs[3,:])
    im3=ax_3.imshow(predicted_initial, extent=extent, vmin=1500,vmax=4500, aspect='auto');  ax_3.invert_yaxis()
    ax_3.text(0.4,0.7, labels[i], fontsize=35)
    i=i+1
    ax_4=fig.add_subplot(gs[4,:])
    im4=ax_4.imshow(ideal_initial,extent=extent, vmin=1500,vmax=4500, aspect='auto');  ax_4.invert_yaxis()
    ax_4.text(0.4,0.7, labels[i], fontsize=35)
    # fig.savefig('gridspec.png')
    # dd=1
    save_file_path2=save_file_path.split('.png')[0]+'_all_channels.png'
    if Save_flag == 1:
        plt.savefig(save_file_path2, dpi=300,bbox_inches='tight')
        print('Saving ML_result to '+save_file_path2)
    if Show_flag == 1:
        plt.show()
    else:
        plt.show(block=False)
    plt.close()
    return None

def PLOT_ML_Result_all_inputs2(inp_orig, val1, history_flag=0, history=None,
        Boundaries=0, save_file_path='', dx=1, dy=1, Plot_vertical_lines=0, Title='',
        Title2='', Save_flag=0, Show_flag=0, adaptive_colorbar=1):
    inp=inp_orig
    Nm = len(inp)
    dim1 = inp[0].shape[0]
    dim2 = inp[0].shape[1]
    if dim1 >= dim2:
        Nx = dim1
        Nz = dim2
        for i in range(Nm):
            if inp[i].ndim==2:
                inp[i] = inp[i].swapaxes(0, 1)
            else:
                inp[i] = inp[i].swapaxes(1,2)
    else:
        Nx = dim2
        Nz = dim1
    #   flip z axis
    for i in range(Nm):
        if inp[i].ndim==2:  
            inp[i] = np.flip(inp[i], axis=0)
        else:   
            inp[i] = np.flip(inp[i], axis=1)
    input = inp[0]
    output = inp[1]
    pred = inp[2]
    predicted_initial = inp[4];    
    ideal_initial=inp[5]
    fwi_result=inp[6]
    true_model=inp[7]
    features=inp[8]

    x = np.arange(Nx) * dx / 1000
    y = np.arange(Nz) * dy / 1000
    extent = np.array([x.min(), x.max(), y.min(), y.max()])
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 12.4
    # fig_size[1] = 14.0
    fig_size[1] = 3.0
    plt.rcParams["figure.figsize"] = fig_size
    plt.rcParams["font.size"] = 14
    ################################
    output.shape
    features2=features[[0,9,10,12],::]
    features2.shape;    
    features2=np.concatenate((features2,np.expand_dims(output,axis=0)),axis=0)
    features2.shape
    a = np.min((output));   b = np.max((output))
    #### dm_i_1init_1sign_1taper
    fig=plt.figure()
    n_row=1;    n_col=5
    gs=GridSpec(n_row,n_col) # 2 rows, 3 columns
    gs.update(wspace=0.0,hspace=0.0)
    ax=[None]*15
    labels=['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q']
    for i in range(features2.shape[0]):
        row=int(np.floor(i/n_col))
        col=i-row*n_row
        print(i,row,col)
        ax[i]=fig.add_subplot(gs[row,col]); ax[i].axes.xaxis.set_visible(False);  ax[i].axes.yaxis.set_visible(False)
        ax[i].text(0.4,0.7, labels[i], fontsize=35,fontweight='bold')
        if i<2:     # gradients
            ax[i].imshow(features2[i,::], extent=extent, vmin=a,vmax=b, aspect='auto')
        elif i==2: #    initial model
            ax[i].imshow(features2[i,::], extent=extent,aspect='auto')
        # elif i==3: #  gradient signum
        #     ax[i].imshow(features[i,::], extent=extent, vmin=-1,vmax=1, aspect='auto');
        elif i==3: #  water taper
            ax[i].imshow(features2[i,::], extent=extent, vmin=0,vmax=1, aspect='auto')
        elif i==4: #  output
            ax[i].imshow(output, extent=extent, vmin=a,vmax=b, aspect='auto')
        ax[i].invert_yaxis()
    ####
    save_file_path2=save_file_path.split('.png')[0]+'_all_channels_short.png'
    if Save_flag == 1:
        plt.savefig(save_file_path2, dpi=300,bbox_inches='tight')
        print('Saving ML_result to '+save_file_path2)
    if Show_flag == 1:
        plt.show()
    else:
        plt.show(block=False)
    plt.close()
    return None

def PLOT_channels_separately(inp_orig, val1,opt,history_flag=0, history=None,
        Boundaries=0,save_file_path='',model_name='model',dx=1, dy=1, Plot_vertical_lines=0, Title='',
        Title2='', Save_flag=0, Show_flag=0, adaptive_colorbar=1):
    inp=inp_orig
    Nm = len(inp)
    dim1 = inp[0].shape[0]
    dim2 = inp[0].shape[1]
    if dim1 >= dim2:
        Nx = dim1
        Nz = dim2
        for i in range(Nm):
            if inp[i].ndim==2:
                inp[i] = inp[i].swapaxes(0, 1)
            else:
                inp[i] = inp[i].swapaxes(1,2)
    else:
        Nx = dim2
        Nz = dim1
    #   flip z axis
    for i in range(Nm):
        if inp[i].ndim==2:  
            inp[i] = np.flip(inp[i], axis=0)
        else:   
            inp[i] = np.flip(inp[i], axis=1)
    input = inp[0]
    output = inp[1]
    pred = inp[2]
    apriori_initial = inp[3];    
    predicted_initial = inp[4];
    ideal_initial=inp[5]
    fwi_result=inp[6]
    true_model=inp[7]
    features=inp[8]
    Nx=true_model.shape[1]
    Nz=true_model.shape[0]
    x = np.arange(Nx) * dx / 1000
    y = np.arange(Nz) * dy / 1000
    extent = np.array([x.min(), x.max(), y.min(), y.max()])
    ################################
    os.makedirs(os.path.join(opt.save_path,model_name),exist_ok=True)
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 6.0
    fig_size[1] = 6.0
    plt.rcParams["figure.figsize"] = fig_size
    plt.rcParams["font.size"] = 14
    ################################    plot input
    # features2=features[[0,9,10,11,12],::]
    features2=features
    # features2=np.concatenate((features2,np.expand_dims(output,axis=0)),axis=0)
    plot_all_features=0
    plot_all_features2=1
    if plot_all_features==1:
        a = np.min((features2[0,:]));   b = np.max((features2[0,:]))
        for i in range(features2.shape[0]):
            save_file_path2=os.path.join(opt.save_path,model_name,'channel_'+str(i)+'.png')
            plt.figure()
            data=features2[i,::]
            if i<=9:
                plt.imshow(data,extent=extent,aspect='auto',vmin=a,vmax=b)
            elif i==10:
                plt.imshow(data,extent=extent,aspect='auto')
            else:
                plt.imshow(data,extent=extent,aspect='auto')
            plt.colorbar()
            plt.gca().axes.xaxis.set_visible(False)
            plt.gca().axes.yaxis.set_visible(False)
            print('saving to ',save_file_path2)
            plt.savefig(save_file_path2, dpi=300,bbox_inches='tight')
            plt.close()
    if plot_all_features2==1:
model_on_1st_iteration=features2[10,:]+features2[0,:]
a = np.min(model_on_1st_iteration)
b = np.max(model_on_1st_iteration)
for i in range(features2.shape[0]):
    save_file_path2=os.path.join(opt.save_path,model_name,'m_'+str(i)+'.png')
    plt.figure()
    if i<=9:
        data=features2[i,::]+features2[10,:]
    else:
        data=features2[i,::]
    if i<=9:
        plt.imshow(np.flipud(data),extent=extent,aspect='auto',vmin=a,vmax=b)
    else:
        plt.imshow(np.flipud(data),extent=extent,aspect='auto')
    plt.colorbar()
    plt.gca().axes.xaxis.set_visible(False)
    plt.gca().axes.yaxis.set_visible(False)
    print('saving to ',save_file_path2)
    plt.savefig(save_file_path2, dpi=300,bbox_inches='tight')
    plt.close()
    ################################
    labels_Fontsize=18
    labels_Fontsize=22
    lim1=[1.5,4.5]
    parameters = {  'axes.labelsize': labels_Fontsize,
                    'axes.titlesize': labels_Fontsize,
                    'font.size':labels_Fontsize}
    plt.rcParams.update(parameters)
    
    Aspect=5 # Marmousi
    plotting_fraction=0.046
    plotting_pad=0.04
    
    Aspect=5.5 # Overthrust
    plotting_fraction=0.043
    plotting_pad=0.04
    clb_title_pad=15
    ################################    plot label in real amplitudes
    lim2=[-0.5,0.5]
    data=ideal_initial-apriori_initial
    fig = plt.figure()
    fig.dpi = 330
    im=plt.imshow((data/1000),extent=extent,aspect=Aspect,vmin=lim2[0],vmax=lim2[1])
    plt.gca().invert_yaxis()
    plt.xlabel('X, km',fontsize=labels_Fontsize)
    plt.ylabel('Z, km',fontsize=labels_Fontsize)
    # plt.title('Low-wavenumber model update,\n real amplitude')
    plt.title('Label in \nreal amplitudes')
    clb = plt.colorbar(im,fraction=plotting_fraction, pad=plotting_pad)
    clb.ax.set_title('$\mathit{V}$$_{p}$ (km/sec)',fontsize=labels_Fontsize-2,pad=clb_title_pad)
    save_file_path2=os.path.join(opt.save_path,model_name,'label_real_amplitude.png')
    print('saving to ',save_file_path2)
    plt.savefig(save_file_path2,dpi=300,bbox_inches='tight')
    plt.close()
    ################################    plot label
    a = np.min((features2[0,:]));   b = np.max((features2[0,:]))
    lim2=[a,b]
    data=output
    fig = plt.figure()
    fig.dpi = 330
    im=plt.imshow(data,extent=extent,aspect=Aspect,vmin=lim2[0],vmax=lim2[1])
    plt.gca().invert_yaxis()
    plt.xlabel('X, km',fontsize=labels_Fontsize)
    plt.ylabel('Z, km',fontsize=labels_Fontsize)
    # plt.title('Low-wavenumber model update,\n standardized')
    plt.title('Label scaled')
    clb = plt.colorbar(im,fraction=plotting_fraction,pad=plotting_pad)
    clb.ax.set_title('Amplitude',fontsize=labels_Fontsize-2,pad=clb_title_pad)
    # clb.ax.set_title('$\mathit{V}$$_{p}$ (km/sec)',fontsize=16,pad=11.0)
    save_file_path2=os.path.join(opt.save_path,model_name,'label_scaled.png')
    print('saving to ',save_file_path2)
    plt.savefig(save_file_path2,dpi=300,bbox_inches='tight')
    plt.close()
    ################################    plot apriori_initial model
    save_file_path2=os.path.join(opt.save_path,model_name,'apriori_initial.png')
    plt.figure()
    data=apriori_initial
    im=plt.imshow(data/1000,extent=extent,aspect=Aspect,vmin=lim1[0],vmax=lim1[1])
    plt.gca().invert_yaxis()
    plt.xlabel('X, km',fontsize=labels_Fontsize)
    plt.ylabel('Z, km',fontsize=labels_Fontsize)
    plt.title('A priori\n initial model')
    clb = plt.colorbar(im,fraction=plotting_fraction, pad=plotting_pad)
    clb.ax.set_title('$\mathit{V}$$_{p}$ (km/sec)',fontsize=labels_Fontsize-2,pad=clb_title_pad)
    print('saving to ',save_file_path2)
    plt.savefig(save_file_path2, dpi=300,bbox_inches='tight')
    plt.close()
    ################################    plot ideal_initial model
    save_file_path2=os.path.join(opt.save_path,model_name,'ideal_initial.png')
    plt.figure()
    data=ideal_initial
    im=plt.imshow(data/1000,extent=extent,aspect=Aspect,vmin=lim1[0],vmax=lim1[1])
    plt.gca().invert_yaxis()
    plt.xlabel('X, km',fontsize=labels_Fontsize)
    plt.ylabel('Z, km',fontsize=labels_Fontsize)
    plt.title('Smoothed\n ground-truth \n initial model')
    clb = plt.colorbar(im,fraction=plotting_fraction, pad=plotting_pad)
    clb.ax.set_title('$\mathit{V}$$_{p}$ (km/sec)',fontsize=labels_Fontsize-2,pad=clb_title_pad)
    print('saving to ',save_file_path2)
    plt.savefig(save_file_path2, dpi=300,bbox_inches='tight')
    plt.close()
    ################################    plot true model
    save_file_path2=os.path.join(opt.save_path,model_name,'ground_truth.png')
    plt.figure()
    data=true_model
    im=plt.imshow(data/1000,extent=extent,aspect=Aspect,vmin=lim1[0],vmax=lim1[1])
    plt.gca().invert_yaxis()
    plt.xlabel('X, km',fontsize=labels_Fontsize)
    plt.ylabel('Z, km',fontsize=labels_Fontsize)
    plt.title('Ground-truth')
    clb = plt.colorbar(im,fraction=plotting_fraction, pad=plotting_pad)
    clb.ax.set_title('$\mathit{V}$$_{p}$ (km/sec)',fontsize=labels_Fontsize-2,pad=clb_title_pad)
    print('saving to ',save_file_path2)
    plt.savefig(save_file_path2, dpi=300,bbox_inches='tight')
    plt.close()
    ################################    plot prediction
    save_file_path2=os.path.join(opt.save_path,model_name,'predicted_update.png')
    plt.figure()
    data=pred
    # data=np.flipud(data)
    plt.imshow(data,extent=extent,aspect='auto',vmin=a,vmax=b)
    plt.gca().axes.xaxis.set_visible(False)
    plt.gca().axes.yaxis.set_visible(False)
    print('saving to ',save_file_path2)
    plt.savefig(save_file_path2, dpi=300,bbox_inches='tight')
    plt.close()
    ################################    plot predicted_initial model
    save_file_path2=os.path.join(opt.save_path,model_name,'predicted_initial.png')
    plt.figure()
    data=predicted_initial
    # data=np.flipud(data)
    plt.imshow(data,extent=extent,aspect='auto',vmin=1500,vmax=4500)
    plt.gca().axes.xaxis.set_visible(False)
    plt.gca().axes.yaxis.set_visible(False)
    print('saving to ',save_file_path2)
    plt.savefig(save_file_path2, dpi=300,bbox_inches='tight')
    plt.close()
    return None

def PLOT_ML_Result_geophysics(inp, val1, history_flag=0, history=None,
        Boundaries=0, save_file_path='', dx=1, dy=1, Plot_vertical_lines=0, Title='',
        Title2='', Save_flag=0, Show_flag=0, adaptive_colorbar=1):
    # Title=[
    #     'Input',
    #     'Target',
    #     'Prediction, '+'R2(prediction,target)='+numstr(F_r2(M3,M2))+metric_name+numstr(chosen_metric(M3,M2)),
    #     'initial model, '+'R2(initial,ideal_init_model)='+numstr(F_r2(init_model,ideal_init_model))+metric_name+numstr(chosen_metric(init_model,ideal_init_model)),
    #     'Predicted initial model,'+'R2(_-||-,ideal_init_model)='+numstr(F_r2(predicted_init_model,ideal_init_model))+metric_name+numstr(chosen_metric(predicted_init_model,ideal_init_model)),
    #     'Ideal initial model, '+'R2(initial model,true)='+numstr(F_r2(ideal_init_model,true_model))+metric_name+numstr(chosen_metric(ideal_init_model,true_model)),
    #     'FWI result,'+'R2(FWI result,ideal_init_model)='+numstr(F_r2(fwi_result,ideal_init_model))+metric_name+numstr(chosen_metric(fwi_result,ideal_init_model)),
    #     'True model']
    # Title[0]='a) '+Title[0]
    # Title[1]='b) '+Title[1]
    # Title[2]='c) '+Title[2]
    # Title[3]='d) '+Title[3]
    # Title[4]='e) '+Title[4]
    Nm = len(inp)
    dim1 = inp[0].shape[0]
    dim2 = inp[0].shape[1]
    if dim1 >= dim2:
        Nx = dim1
        Nz = dim2
        for i in range(Nm):
            if inp[i].ndim==2:
                inp[i] = inp[i].swapaxes(0, 1)
            else:
                inp[i] = inp[i].swapaxes(1,2)
    else:
        Nx = dim2
        Nz = dim1
    #   flip z axis
    for i in range(Nm):
        if inp[i].ndim==2:  
            inp[i] = np.flip(inp[i], axis=0)
        else:   
            inp[i] = np.flip(inp[i], axis=1)
    input = inp[0]
    output = inp[1]
    pred = inp[2]
    predicted_initial = inp[4];    
    ideal_initial=inp[5]
    fwi_result=inp[6]
    true_model=inp[7]
    features=inp[8]

    x = np.arange(Nx) * dx / 1000
    y = np.arange(Nz) * dy / 1000
    extent = np.array([x.min(), x.max(), y.min(), y.max()])
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 12.4
    fig_size[1] = 14.0  # 8 images
    fig_size[1] = 8.75  # 5 images

    plt.rcParams["figure.figsize"] = fig_size
    plt.rcParams["font.size"] = 14
    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(nrows=5,ncols=1)
    #  allocate space for colorbar smartly
    divider1 = make_axes_locatable((ax1))
    divider2 = make_axes_locatable((ax2))
    divider3 = make_axes_locatable((ax3))
    divider4 = make_axes_locatable((ax4))
    divider5 = make_axes_locatable((ax5))
    cax1 = divider1.append_axes("right", size="2%", pad=0.05)
    cax2 = divider2.append_axes("right", size="2%", pad=0.05)
    cax3 = divider3.append_axes("right", size="2%", pad=0.05)
    cax4 = divider4.append_axes("right", size="2%", pad=0.05)
    cax5 = divider5.append_axes("right", size="2%", pad=0.05)
    plt.set_cmap('RdBu_r')
    MIN = np.min(output)
    MAX = np.max(output)
    orig_min = np.min(ideal_initial)
    orig_max = np.max(ideal_initial)
    if adaptive_colorbar == 4:
        a = np.min((output));   b = np.max((output))
        im1 = ax1.imshow(input, extent=extent, vmin=a, vmax=b, aspect='auto')
        im2 = ax2.imshow(output, extent=extent, vmin=a, vmax=b, aspect='auto')
        im3 = ax3.imshow(pred, extent=extent, vmin=orig_min,vmax=orig_max, aspect='auto')
        im4 = ax4.imshow(inp[3], extent=extent,vmin=orig_min, vmax=orig_max, aspect='auto')
    if adaptive_colorbar == 3:
        a = np.min((output));   b = np.max((output))
        im1 = ax1.imshow(features[9,::], extent=extent, vmin=a, vmax=b, aspect='auto')
        im2 = ax2.imshow(output,extent=extent, vmin=a, vmax=b, aspect='auto')
        im3 = ax3.imshow(pred,  extent=extent, vmin=a, vmax=b, aspect='auto')
    if adaptive_colorbar == 2:
        im1 = ax1.imshow(input, extent=extent, aspect='auto')
        im2 = ax2.imshow(output, extent=extent, aspect='auto')
        im3 = ax3.imshow(pred, extent=extent, aspect='auto')
        im4 = ax4.imshow(inp[3], extent=extent,aspect='auto')
    if adaptive_colorbar == 1:
        val = np.max(abs(input))
        im1 = ax1.imshow(input, extent=extent, vmin=-
                            val/3, vmax=val/3, aspect='auto')
        val = np.max(abs(output))
        im2 = ax2.imshow(output, extent=extent, vmin=-
                            val,  vmax=val, aspect='auto')
        im3 = ax3.imshow(pred, extent=extent,   vmin=-
                            val,  vmax=val, aspect='auto')
        im4 = ax4.imshow(inp[3], extent=extent, vmin=-
                            val/2, vmax=val/2, aspect='auto')
    if adaptive_colorbar == 0:
        min_lim = -1
        max_lim = 1
        im1 = ax1.imshow(input, extent=extent, vmin=min_lim,vmax=max_lim, aspect='auto')
        im2 = ax2.imshow(output, extent=extent, vmin=min_lim,vmax=max_lim, aspect='auto')
        im3 = ax3.imshow(pred, extent=extent,   vmin=min_lim,vmax=max_lim, aspect='auto')
        im4 = ax4.imshow(inp[3], extent=extent,vmin=min_lim/2, vmax=max_lim/2, aspect='auto')
    tmp = inp[4]
    im4 = ax4.imshow(predicted_initial, extent=extent, vmin=orig_min,vmax=orig_max, aspect='auto')
    tmp = inp[5]
    im5 = ax5.imshow(ideal_initial, extent=extent,vmin=orig_min, vmax=orig_max, aspect='auto')
    #########
    x0 = 0.03
    y0 = -0.25
    if Plot_vertical_lines == 1:
        Boundaries2 = Boundaries * dx / 1000
        for i in range(np.shape(Boundaries)[0]):
            ax1.axvline(Boundaries2[i], color='k',
                        linestyle='--', linewidth=2.5)
            ax2.axvline(Boundaries2[i], color='k',
                        linestyle='--', linewidth=2.5)
            ax3.axvline(Boundaries2[i], color='k',
                        linestyle='--', linewidth=2.5)
    z_label_name = 'z, m'
    ax1.set_ylabel(z_label_name)
    ax1.set_xlabel('x (km)')
    ax2.set_ylabel(z_label_name)
    ax2.set_xlabel('x (km)')
    ax3.set_ylabel(z_label_name)
    ax3.set_xlabel('x (km)')
    ax4.set_ylabel(z_label_name)
    ax4.set_xlabel('x (km)')
    ax5.set_ylabel(z_label_name)
    ax5.set_xlabel('x (km)')
    ax1.invert_yaxis()
    ax2.invert_yaxis()
    ax3.invert_yaxis()
    ax4.invert_yaxis()
    ax5.invert_yaxis()
    ax1.xaxis.set_label_coords(x0, y0)
    ax2.xaxis.set_label_coords(x0, y0)
    ax3.xaxis.set_label_coords(x0, y0)
    ax4.xaxis.set_label_coords(x0, y0)
    ax5.xaxis.set_label_coords(x0, y0)
    ax1.set_title('a) '+Title[0])
    ax2.set_title('b) '+Title[1])
    ax3.set_title('c) '+Title[2])
    ax4.set_title('d) '+Title[4])
    ax5.set_title('e) '+Title[5])
    cbar1 = plt.colorbar(im1, cax=cax1)
    cbar2 = plt.colorbar(im2, cax=cax2)
    cbar3 = plt.colorbar(im3, cax=cax3)
    cbar4 = plt.colorbar(im4, cax=cax4)
    cbar5 = plt.colorbar(im5, cax=cax5)
    x0 = -40    
    y0 = 1.16
    cbar1.set_label('V (m/sec)', labelpad=x0, y=y0, rotation=0)
    cbar2.set_label('V (m/sec)', labelpad=x0, y=y0, rotation=0)
    cbar3.set_label('V (m/sec)', labelpad=x0, y=y0, rotation=0)
    cbar4.set_label('V (m/sec)', labelpad=x0, y=y0, rotation=0)
    cbar5.set_label('V (m/sec)', labelpad=x0, y=y0, rotation=0)
    plt.subplots_adjust(bottom=0.08,top=0.95,hspace=0.57, right=0.86, wspace=.1)
    if Save_flag == 1:
        plt.savefig(save_file_path, dpi=300,bbox_inches='tight')
        print('Saving ML_result to '+save_file_path)
    # if Show_flag == 1:
    #     plt.show()
    # else:
    #     plt.show(block=False)
    plt.close()
    return None

def PLOT_ML_Result_geophysics_field_data(inp, val1, history_flag=0, history=None,
        Boundaries=0, save_file_path='', dx=1, dy=1, Plot_vertical_lines=0, Title='',
        Title2='', Save_flag=0, Show_flag=0, adaptive_colorbar=1):
    Nm = len(inp)
    dim1 = inp[0].shape[0]
    dim2 = inp[0].shape[1]
    if dim1 >= dim2:
        Nx = dim1
        Nz = dim2
        for i in range(Nm):
            if inp[i].ndim==2:
                inp[i] = inp[i].swapaxes(0, 1)
            else:
                inp[i] = inp[i].swapaxes(1,2)
    else:
        Nx = dim2
        Nz = dim1
    #   flip z axis
    for i in range(Nm):
        if inp[i].ndim==2:  
            inp[i] = np.flip(inp[i], axis=0)
        else:   
            inp[i] = np.flip(inp[i], axis=1)
    input = inp[0]
    output = inp[1]
    pred = inp[2]
    predicted_initial = inp[4];    
    ideal_initial=inp[5]
    fwi_result=inp[6]
    true_model=inp[7]
    features=inp[8]

    x = np.arange(Nx) * dx / 1000
    y = np.arange(Nz) * dy / 1000
    extent = np.array([x.min(), x.max(), y.min(), y.max()])
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 12.4
    fig_size[1] = 14.0  # 8 images
    fig_size[1] = 8.75  # 5 images
    plt.rcParams["figure.figsize"] = fig_size
    plt.rcParams["font.size"] = 14

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4,ncols=1)
    #  allocate space for colorbar smartly
    divider1 = make_axes_locatable((ax1))
    divider2 = make_axes_locatable((ax2))
    divider3 = make_axes_locatable((ax3))
    divider4 = make_axes_locatable((ax4))
    cax1 = divider1.append_axes("right", size="2%", pad=0.05)
    cax2 = divider2.append_axes("right", size="2%", pad=0.05)
    cax3 = divider3.append_axes("right", size="2%", pad=0.05)
    cax4 = divider4.append_axes("right", size="2%", pad=0.05)
    plt.set_cmap('RdBu_r')
    MIN = np.min(output)
    MAX = np.max(output)
    predicted_initial
    orig_min = np.min(ideal_initial)
    orig_max = np.max(ideal_initial)

    a = np.min((output));   b = np.max((output))
    im1 = ax1.imshow(features[9,::], extent=extent, vmin=a, vmax=b, aspect='auto')
    im2 = ax2.imshow(inp[3],extent=extent,vmin=orig_min,vmax=orig_max, aspect='auto')
    im3 = ax3.imshow(pred,  extent=extent,vmin=a, vmax=b, aspect='auto')
    im4 = ax4.imshow(predicted_initial,extent=extent,vmin=orig_min,vmax=orig_max,aspect='auto')
    #########
    x0 = 0.04
    y0 = -0.25
    if Plot_vertical_lines == 1:
        Boundaries2 = Boundaries * dx / 1000
        for i in range(np.shape(Boundaries)[0]):
            ax1.axvline(Boundaries2[i], color='k',
                        linestyle='--', linewidth=2.5)
            ax2.axvline(Boundaries2[i], color='k',
                        linestyle='--', linewidth=2.5)
            ax3.axvline(Boundaries2[i], color='k',
                        linestyle='--', linewidth=2.5)
    z_label_name = 'z, m'
    ax1.set_ylabel(z_label_name)
    ax1.set_xlabel('x (km)')
    ax2.set_ylabel(z_label_name)
    ax2.set_xlabel('x (km)')
    ax3.set_ylabel(z_label_name)
    ax3.set_xlabel('x (km)')
    ax4.set_ylabel(z_label_name)
    ax4.set_xlabel('x (km)')
    ax1.invert_yaxis()
    ax2.invert_yaxis()
    ax3.invert_yaxis()
    ax4.invert_yaxis()
    ax1.xaxis.set_label_coords(x0, y0)
    ax2.xaxis.set_label_coords(x0, y0)
    ax3.xaxis.set_label_coords(x0, y0)
    ax4.xaxis.set_label_coords(x0, y0)
    ax1.set_title('a) '+Title[0])
    ax2.set_title('b) '+Title[3])
    ax3.set_title('c) '+Title[2])
    ax4.set_title('d) '+Title[4])
    cbar1 = plt.colorbar(im1, cax=cax1)
    cbar2 = plt.colorbar(im2, cax=cax2)
    cbar3 = plt.colorbar(im3, cax=cax3)
    cbar4 = plt.colorbar(im4, cax=cax4)
    x0 = -40    
    y0 = 1.16
    cbar1.set_label('V (m/sec)', labelpad=x0, y=y0, rotation=0)
    cbar2.set_label('V (m/sec)', labelpad=x0, y=y0, rotation=0)
    cbar3.set_label('V (m/sec)', labelpad=x0, y=y0, rotation=0)
    cbar4.set_label('V (m/sec)', labelpad=x0, y=y0, rotation=0)
    plt.subplots_adjust(bottom=0.08,top=0.95,hspace=0.57, right=0.86, wspace=.1)
    if Save_flag == 1:
        plt.savefig(save_file_path, dpi=300,bbox_inches='tight')
        print('Saving ML_result to '+save_file_path)
    # if Show_flag == 1:
    #     plt.show()
    # else:
    #     plt.show(block=False)
    plt.close()
    return None

def PLOT_ML_Result_geophysics2(inp, val1, history_flag=0, history=None,
        Boundaries=0, save_file_path='', dx=1, dy=1, Plot_vertical_lines=0, Title='',
        Title2='', Save_flag=0, Show_flag=0, adaptive_colorbar=1):
    # Title=[
    #     'Input',
    #     'Target',
    #     'Prediction, '+'R2(prediction,target)='+numstr(F_r2(M3,M2))+metric_name+numstr(chosen_metric(M3,M2)),
    #     'initial model, '+'R2(initial,ideal_init_model)='+numstr(F_r2(init_model,ideal_init_model))+metric_name+numstr(chosen_metric(init_model,ideal_init_model)),
    #     'Predicted initial model,'+'R2(_-||-,ideal_init_model)='+numstr(F_r2(predicted_init_model,ideal_init_model))+metric_name+numstr(chosen_metric(predicted_init_model,ideal_init_model)),
    #     'Ideal initial model, '+'R2(initial model,true)='+numstr(F_r2(ideal_init_model,true_model))+metric_name+numstr(chosen_metric(ideal_init_model,true_model)),
    #     'FWI result,'+'R2(FWI result,ideal_init_model)='+numstr(F_r2(fwi_result,ideal_init_model))+metric_name+numstr(chosen_metric(fwi_result,ideal_init_model)),
    #     'True model']
    # Title[0]='a) '+Title[0]
    # Title[1]='b) '+Title[1]
    # Title[2]='c) '+Title[2]
    # Title[3]='d) '+Title[3]
    # Title[4]='e) '+Title[4]
    Nm = len(inp)
    dim1 = inp[0].shape[0]
    dim2 = inp[0].shape[1]
    if dim1 >= dim2:
        Nx = dim1
        Nz = dim2
        for i in range(Nm):
            if inp[i].ndim==2:
                inp[i] = inp[i].swapaxes(0, 1)
            else:
                inp[i] = inp[i].swapaxes(1,2)
    else:
        Nx = dim2
        Nz = dim1
    #   flip z axis
    for i in range(Nm):
        if inp[i].ndim==2:  
            inp[i] = np.flip(inp[i], axis=0)
        else:   
            inp[i] = np.flip(inp[i], axis=1)
    input = inp[0]
    output = inp[1]
    pred = inp[2]
    initial = inp[3];    
    predicted_initial = inp[4];    
    ideal_initial=inp[5]
    fwi_result=inp[6]
    true_model=inp[7]
    features=inp[8]

    x = np.arange(Nx) * dx / 1000
    y = np.arange(Nz) * dy / 1000
    extent = np.array([x.min(), x.max(), y.min(), y.max()])
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 12.4
    fig_size[1] = 14.0  # 8 images
    fig_size[1] = 6*14/8  # 6 images

    plt.rcParams["figure.figsize"] = fig_size
    plt.rcParams["font.size"] = 14
    fig, (ax1, ax2, ax3,ax4, ax5,ax6) = plt.subplots(nrows=6,ncols=1)
    #  allocate space for colorbar smartly
    divider1 = make_axes_locatable((ax1))
    divider2 = make_axes_locatable((ax2))
    divider3 = make_axes_locatable((ax3))
    divider4 = make_axes_locatable((ax4))
    divider5 = make_axes_locatable((ax5))
    divider6 = make_axes_locatable((ax6))
    cax1 = divider1.append_axes("right", size="2%", pad=0.05)
    cax2 = divider2.append_axes("right", size="2%", pad=0.05)
    cax3 = divider3.append_axes("right", size="2%", pad=0.05)
    cax4 = divider4.append_axes("right", size="2%", pad=0.05)
    cax5 = divider5.append_axes("right", size="2%", pad=0.05)
    cax6 = divider6.append_axes("right", size="2%", pad=0.05)
    plt.set_cmap('RdBu_r')
    MIN = np.min(output)
    MAX = np.max(output)
    orig_min = np.min(ideal_initial)
    orig_max = np.max(ideal_initial)
    if adaptive_colorbar == 4:
        a = np.min((output));   b = np.max((output))
        im1 = ax1.imshow(input, extent=extent, vmin=a, vmax=b, aspect='auto')
        im2 = ax2.imshow(output, extent=extent, vmin=a, vmax=b, aspect='auto')
        im3 = ax3.imshow(pred, extent=extent, vmin=orig_min,vmax=orig_max, aspect='auto')
        im4 = ax4.imshow(inp[3], extent=extent,vmin=orig_min, vmax=orig_max, aspect='auto')
    if adaptive_colorbar == 3:
        a = np.min((output));   b = np.max((output))
        im1 = ax1.imshow(features[9,::], extent=extent, vmin=a, vmax=b, aspect='auto')
        im2 = ax2.imshow(output,extent=extent, vmin=a, vmax=b, aspect='auto')
        im3 = ax3.imshow(pred,  extent=extent, vmin=a, vmax=b, aspect='auto')
    if adaptive_colorbar == 2:
        im1 = ax1.imshow(input, extent=extent, aspect='auto')
        im2 = ax2.imshow(output, extent=extent, aspect='auto')
        im3 = ax3.imshow(pred, extent=extent, aspect='auto')
        im4 = ax4.imshow(inp[3], extent=extent,aspect='auto')
    if adaptive_colorbar == 1:
        val = np.max(abs(input))
        im1 = ax1.imshow(input, extent=extent, vmin=-val/3, vmax=val/3, aspect='auto')
        val = np.max(abs(output))
        im2 = ax2.imshow(output, extent=extent, vmin=-val,  vmax=val, aspect='auto')
        im3 = ax3.imshow(pred, extent=extent,   vmin=-val,  vmax=val, aspect='auto')
        im4 = ax4.imshow(inp[3], extent=extent, vmin=-val/2, vmax=val/2, aspect='auto')
    if adaptive_colorbar == 0:
        min_lim = -1
        max_lim = 1
        im1 = ax1.imshow(input, extent=extent, vmin=min_lim,vmax=max_lim, aspect='auto')
        im2 = ax2.imshow(output, extent=extent, vmin=min_lim,vmax=max_lim, aspect='auto')
        im3 = ax3.imshow(pred, extent=extent,   vmin=min_lim,vmax=max_lim, aspect='auto')
        im4 = ax4.imshow(inp[3], extent=extent,vmin=min_lim/2, vmax=max_lim/2, aspect='auto')
    im4 = ax4.imshow(initial, extent=extent, vmin=orig_min,vmax=orig_max, aspect='auto')
    im5 = ax5.imshow(predicted_initial, extent=extent, vmin=orig_min,vmax=orig_max, aspect='auto')
    im6 = ax6.imshow(ideal_initial, extent=extent,vmin=orig_min, vmax=orig_max, aspect='auto')
    #########
    x0 = 0.03
    y0 = -0.25
    if Plot_vertical_lines == 1:
        Boundaries2 = Boundaries * dx / 1000
        for i in range(np.shape(Boundaries)[0]):
            ax1.axvline(Boundaries2[i], color='k',
                        linestyle='--', linewidth=2.5)
            ax2.axvline(Boundaries2[i], color='k',
                        linestyle='--', linewidth=2.5)
            ax3.axvline(Boundaries2[i], color='k',
                        linestyle='--', linewidth=2.5)
    z_label_name = 'z, m'
    ax1.set_ylabel(z_label_name)
    ax1.set_xlabel('x (km)')
    ax2.set_ylabel(z_label_name)
    ax2.set_xlabel('x (km)')
    ax3.set_ylabel(z_label_name)
    ax3.set_xlabel('x (km)')
    ax4.set_ylabel(z_label_name)
    ax4.set_xlabel('x (km)')
    ax5.set_ylabel(z_label_name)
    ax5.set_xlabel('x (km)')
    ax6.set_ylabel(z_label_name)
    ax6.set_xlabel('x (km)')
    ax1.invert_yaxis()
    ax2.invert_yaxis()
    ax3.invert_yaxis()
    ax4.invert_yaxis()
    ax5.invert_yaxis()
    ax6.invert_yaxis()
    ax1.xaxis.set_label_coords(x0, y0)
    ax2.xaxis.set_label_coords(x0, y0)
    ax3.xaxis.set_label_coords(x0, y0)
    ax4.xaxis.set_label_coords(x0, y0)
    ax5.xaxis.set_label_coords(x0, y0)
    ax6.xaxis.set_label_coords(x0, y0)
    ax1.set_title('a) '+Title[0])
    ax2.set_title('b) '+Title[1])
    ax3.set_title('c) '+Title[2])
    ax4.set_title('d) '+Title[3])
    ax5.set_title('e) '+Title[4])
    ax6.set_title('f) '+Title[5])
    cbar1 = plt.colorbar(im1, cax=cax1)
    cbar2 = plt.colorbar(im2, cax=cax2)
    cbar3 = plt.colorbar(im3, cax=cax3)
    cbar4 = plt.colorbar(im4, cax=cax4)
    cbar5 = plt.colorbar(im5, cax=cax5)
    cbar6 = plt.colorbar(im6, cax=cax6)
    x0 = -40    
    y0 = 1.16
    cbar1.set_label('V (m/sec)', labelpad=x0, y=y0, rotation=0)
    cbar2.set_label('V (m/sec)', labelpad=x0, y=y0, rotation=0)
    cbar3.set_label('V (m/sec)', labelpad=x0, y=y0, rotation=0)
    cbar4.set_label('V (m/sec)', labelpad=x0, y=y0, rotation=0)
    cbar5.set_label('V (m/sec)', labelpad=x0, y=y0, rotation=0)
    cbar6.set_label('V (m/sec)', labelpad=x0, y=y0, rotation=0)
    plt.subplots_adjust(bottom=0.08,top=0.95,hspace=0.57, right=0.86, wspace=.1)
    if Save_flag == 1:
        plt.savefig(save_file_path, dpi=300,bbox_inches='tight')
        print('Saving ML_result to '+save_file_path)
    # if Show_flag == 1:
    #     plt.show()
    # else:
    #     plt.show(block=False)
    plt.close()
    return None

def PLOT_ML_Result_geophysics3(inp, val1, history_flag=0, history=None,
        Boundaries=0, save_file_path='', dx=1, dy=1, Plot_vertical_lines=0, Title='',
        Title2='', Save_flag=0, Show_flag=0, adaptive_colorbar=1):
    Nm = len(inp)
    dim1 = inp[0].shape[0]
    dim2 = inp[0].shape[1]
    if dim1 >= dim2:
        Nx = dim1
        Nz = dim2
        for i in range(Nm):
            if inp[i].ndim==2:
                inp[i] = inp[i].swapaxes(0, 1)
            else:
                inp[i] = inp[i].swapaxes(1,2)
    else:
        Nx = dim2
        Nz = dim1
    #   flip z axis
    for i in range(Nm):
        if inp[i].ndim==2:  
            inp[i] = np.flip(inp[i], axis=0)
        else:   
            inp[i] = np.flip(inp[i], axis=1)
    input = inp[0]
    output = inp[1]
    pred = inp[2]
    initial = inp[3];    
    predicted_initial = inp[4];    
    ideal_initial=inp[5]
    fwi_result=inp[6]
    true_model=inp[7]
    features=inp[8]

    x = np.arange(Nx) * dx / 1000
    y = np.arange(Nz) * dy / 1000
    extent = np.array([x.min(), x.max(), y.min(), y.max()])
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 12.4
    fig_size[1] = 14.0  # 8 images
    fig_size[1] = 6*14/8  # 6 images

    plt.rcParams["figure.figsize"] = fig_size
    plt.rcParams["font.size"] = 14
    fig, (ax1, ax2, ax3,ax4, ax5,ax6) = plt.subplots(nrows=6,ncols=1)
    #  allocate space for colorbar smartly
    divider1 = make_axes_locatable((ax1))
    divider2 = make_axes_locatable((ax2))
    divider3 = make_axes_locatable((ax3))
    divider4 = make_axes_locatable((ax4))
    divider5 = make_axes_locatable((ax5))
    divider6 = make_axes_locatable((ax6))
    cax1 = divider1.append_axes("right", size="2%", pad=0.05)
    cax2 = divider2.append_axes("right", size="2%", pad=0.05)
    cax3 = divider3.append_axes("right", size="2%", pad=0.05)
    cax4 = divider4.append_axes("right", size="2%", pad=0.05)
    cax5 = divider5.append_axes("right", size="2%", pad=0.05)
    cax6 = divider6.append_axes("right", size="2%", pad=0.05)
    plt.set_cmap('RdBu_r')
    MIN = np.min(output)
    MAX = np.max(output)
    orig_min = np.min(ideal_initial)
    orig_max = np.max(ideal_initial)
    if adaptive_colorbar == 4:
        a = np.min((output));   b = np.max((output))
        im1 = ax1.imshow(input, extent=extent, vmin=a, vmax=b, aspect='auto')
        im2 = ax2.imshow(output, extent=extent, vmin=a, vmax=b, aspect='auto')
        im3 = ax3.imshow(pred, extent=extent, vmin=orig_min,vmax=orig_max, aspect='auto')
        im4 = ax4.imshow(inp[3], extent=extent,vmin=orig_min, vmax=orig_max, aspect='auto')
    if adaptive_colorbar == 3:
        a = np.min((output));   b = np.max((output))
        im1 = ax1.imshow(features[9,::], extent=extent, vmin=a, vmax=b, aspect='auto')
        im2 = ax2.imshow(output,extent=extent, vmin=a, vmax=b, aspect='auto')
        im3 = ax3.imshow(pred,  extent=extent, vmin=a, vmax=b, aspect='auto')
    if adaptive_colorbar == 2:
        im1 = ax1.imshow(input, extent=extent, aspect='auto')
        im2 = ax2.imshow(output, extent=extent, aspect='auto')
        im3 = ax3.imshow(pred, extent=extent, aspect='auto')
        im4 = ax4.imshow(inp[3], extent=extent,aspect='auto')
    if adaptive_colorbar == 1:
        val = np.max(abs(input))
        im1 = ax1.imshow(input, extent=extent, vmin=-val/3, vmax=val/3, aspect='auto')
        val = np.max(abs(output))
        im2 = ax2.imshow(output, extent=extent, vmin=-val,  vmax=val, aspect='auto')
        im3 = ax3.imshow(pred, extent=extent,   vmin=-val,  vmax=val, aspect='auto')
        im4 = ax4.imshow(inp[3], extent=extent, vmin=-val/2, vmax=val/2, aspect='auto')
    if adaptive_colorbar == 0:
        min_lim = -1
        max_lim = 1
        im1 = ax1.imshow(input, extent=extent, vmin=min_lim,vmax=max_lim, aspect='auto')
        im2 = ax2.imshow(output, extent=extent, vmin=min_lim,vmax=max_lim, aspect='auto')
        im3 = ax3.imshow(pred, extent=extent,   vmin=min_lim,vmax=max_lim, aspect='auto')
        im4 = ax4.imshow(inp[3], extent=extent,vmin=min_lim/2, vmax=max_lim/2, aspect='auto')
    im4 = ax4.imshow(initial, extent=extent, vmin=orig_min,vmax=orig_max, aspect='auto')
    im5 = ax5.imshow(predicted_initial, extent=extent, vmin=orig_min,vmax=orig_max, aspect='auto')
    im6 = ax6.imshow(ideal_initial, extent=extent,vmin=orig_min, vmax=orig_max, aspect='auto')
    #########
    x0 = 0.03
    y0 = -0.25
    if Plot_vertical_lines == 1:
        Boundaries2 = Boundaries * dx / 1000
        for i in range(np.shape(Boundaries)[0]):
            ax1.axvline(Boundaries2[i], color='k',
                        linestyle='--', linewidth=2.5)
            ax2.axvline(Boundaries2[i], color='k',
                        linestyle='--', linewidth=2.5)
            ax3.axvline(Boundaries2[i], color='k',
                        linestyle='--', linewidth=2.5)
    z_label_name = 'z, m'
    ax1.set_ylabel(z_label_name)
    ax1.set_xlabel('x (km)')
    ax2.set_ylabel(z_label_name)
    ax2.set_xlabel('x (km)')
    ax3.set_ylabel(z_label_name)
    ax3.set_xlabel('x (km)')
    ax4.set_ylabel(z_label_name)
    ax4.set_xlabel('x (km)')
    ax5.set_ylabel(z_label_name)
    ax5.set_xlabel('x (km)')
    ax6.set_ylabel(z_label_name)
    ax6.set_xlabel('x (km)')
    ax1.invert_yaxis()
    ax2.invert_yaxis()
    ax3.invert_yaxis()
    ax4.invert_yaxis()
    ax5.invert_yaxis()
    ax6.invert_yaxis()
    ax1.xaxis.set_label_coords(x0, y0)
    ax2.xaxis.set_label_coords(x0, y0)
    ax3.xaxis.set_label_coords(x0, y0)
    ax4.xaxis.set_label_coords(x0, y0)
    ax5.xaxis.set_label_coords(x0, y0)
    ax6.xaxis.set_label_coords(x0, y0)
    ax1.set_title('a) '+Title[0])
    ax2.set_title('b) '+Title[1])
    ax3.set_title('c) '+Title[2])
    ax4.set_title('d) '+Title[3])
    ax5.set_title('e) '+Title[4])
    ax6.set_title('f) '+Title[5])
    cbar1 = plt.colorbar(im1, cax=cax1)
    cbar2 = plt.colorbar(im2, cax=cax2)
    cbar3 = plt.colorbar(im3, cax=cax3)
    cbar4 = plt.colorbar(im4, cax=cax4)
    cbar5 = plt.colorbar(im5, cax=cax5)
    cbar6 = plt.colorbar(im6, cax=cax6)
    x0 = -40    
    y0 = 1.16
    cbar1.set_label('V (m/sec)', labelpad=x0, y=y0, rotation=0)
    cbar2.set_label('V (m/sec)', labelpad=x0, y=y0, rotation=0)
    cbar3.set_label('V (m/sec)', labelpad=x0, y=y0, rotation=0)
    cbar4.set_label('V (m/sec)', labelpad=x0, y=y0, rotation=0)
    cbar5.set_label('V (m/sec)', labelpad=x0, y=y0, rotation=0)
    cbar6.set_label('V (m/sec)', labelpad=x0, y=y0, rotation=0)
    plt.subplots_adjust(bottom=0.08,top=0.95,hspace=0.57, right=0.86, wspace=.1)
    if Save_flag == 1:
        plt.savefig(save_file_path, dpi=300,bbox_inches='tight')
        print('Saving ML_result to '+save_file_path)
    plt.close()
    return None

def PLOT_ML_Result_geophysics4(inp, val1, history_flag=0, history=None,
        Boundaries=0, save_file_path='', dx=1, dy=1, Plot_vertical_lines=0, Title='',
        Title2='', Save_flag=0, Show_flag=0, adaptive_colorbar=1):
    Nm = len(inp)
    dim1 = inp[0].shape[0]
    dim2 = inp[0].shape[1]
    if dim1 >= dim2:
        Nx = dim1
        Nz = dim2
        for i in range(Nm):
            if inp[i].ndim==2:
                inp[i] = inp[i].swapaxes(0, 1)
            else:
                inp[i] = inp[i].swapaxes(1,2)
    else:
        Nx = dim2
        Nz = dim1
    #   flip z axis
    for i in range(Nm):
        if inp[i].ndim==2:  
            inp[i] = np.flip(inp[i], axis=0)
        else:   
            inp[i] = np.flip(inp[i], axis=1)
    input = inp[0]
    output = inp[1]
    pred = inp[2]
    initial = inp[3];    
    predicted_initial = inp[4];    
    ideal_initial=inp[5]
    fwi_result=inp[6]
    true_model=inp[7]
    features=inp[8]
    grad=features[9,::]
    x = np.arange(Nx) * dx / 1000
    y = np.arange(Nz) * dy / 1000
    ##########################################
    extent = np.array([x.min(), x.max(), y.min(), y.max()])
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 12.4
    fig_size[1] = 14.0  # 8 images
    fig_size[1] = 4.5*14/8  # 6 images
    plt.rcParams["figure.figsize"] = fig_size
    plt.rcParams["font.size"] = 14
    fig, (ax1, ax2, ax3,ax4) = plt.subplots(nrows=4,ncols=1)
    #  allocate space for colorbar smartly
    divider1 = make_axes_locatable((ax1))
    divider2 = make_axes_locatable((ax2))
    divider3 = make_axes_locatable((ax3))
    divider4 = make_axes_locatable((ax4))
    cax1 = divider1.append_axes("right", size="2%", pad=0.05)
    cax2 = divider2.append_axes("right", size="2%", pad=0.05)
    cax3 = divider3.append_axes("right", size="2%", pad=0.05)
    cax4 = divider4.append_axes("right", size="2%", pad=0.05)
    plt.set_cmap('RdBu_r')
    MIN_ = np.min(grad)
    MAX_ = np.max(grad)
    if np.abs(MIN_)<=np.abs(MAX_):
        MIN=-np.abs(MIN_);  MAX=np.abs(MIN_);
    else:
        MIN=-np.abs(MAX_);  MAX=np.abs(MAX_);
    orig_min = np.min(ideal_initial)
    orig_max = np.max(ideal_initial)
    im1 = ax1.imshow(initial, extent=extent, vmin=orig_min, vmax=orig_max, aspect='auto')
    im2 = ax2.imshow(grad, extent=extent, vmin=MIN, vmax=MAX, aspect='auto')
    im3 = ax3.imshow(predicted_initial, extent=extent,  vmin=orig_min ,vmax=orig_max, aspect='auto')
    im4 = ax4.imshow(ideal_initial,     extent=extent,  vmin=orig_min, vmax=orig_max, aspect='auto')
    #########
    x0 = 0.03;  y0 = -0.25
    z_label_name = 'z, m'
    ax1.set_ylabel(z_label_name)
    ax1.set_xlabel('x (km)')
    ax2.set_ylabel(z_label_name)
    ax2.set_xlabel('x (km)')
    ax3.set_ylabel(z_label_name)
    ax3.set_xlabel('x (km)')
    ax4.set_ylabel(z_label_name)
    ax4.set_xlabel('x (km)')
    ax1.invert_yaxis()
    ax2.invert_yaxis()
    ax3.invert_yaxis()
    ax4.invert_yaxis()
    ax1.xaxis.set_label_coords(x0, y0)
    ax2.xaxis.set_label_coords(x0, y0)
    ax3.xaxis.set_label_coords(x0, y0)
    ax4.xaxis.set_label_coords(x0, y0)
    ax1.set_title('a) '+Title[3])
    ax2.set_title('b) '+Title[0])
    ax3.set_title('c) '+Title[4])
    ax4.set_title('d) '+Title[5])
    cbar1 = plt.colorbar(im1, cax=cax1)
    cbar2 = plt.colorbar(im2, cax=cax2)
    cbar3 = plt.colorbar(im3, cax=cax3)
    cbar4 = plt.colorbar(im4, cax=cax4)
    x0 = -40;        y0 = 1.16
    cbar1.set_label('V (m/sec)', labelpad=x0, y=y0, rotation=0)
    cbar2.set_label('V (m/sec)', labelpad=x0, y=y0, rotation=0)
    cbar3.set_label('V (m/sec)', labelpad=x0, y=y0, rotation=0)
    cbar4.set_label('V (m/sec)', labelpad=x0, y=y0, rotation=0)
    plt.subplots_adjust(bottom=0.08,top=0.95,hspace=0.57, right=0.86, wspace=.1)
    if Save_flag == 1:
        plt.savefig(save_file_path, dpi=300,bbox_inches='tight')
        print('Saving ML_result to '+save_file_path)
    plt.close()

    return None

def PLOT_ML_Result_geophysics_field_data2(inp, val1, history_flag=0, history=None,
        Boundaries=0, save_file_path='', dx=1, dy=1, Plot_vertical_lines=0, Title='',
        Title2='', Save_flag=0, Show_flag=0, adaptive_colorbar=1):
    Nm = len(inp)
    dim1 = inp[0].shape[0]
    dim2 = inp[0].shape[1]
    if dim1 >= dim2:
        Nx = dim1
        Nz = dim2
        for i in range(Nm):
            if inp[i].ndim==2:
                inp[i] = inp[i].swapaxes(0, 1)
            else:
                inp[i] = inp[i].swapaxes(1,2)
    else:
        Nx = dim2
        Nz = dim1
    #   flip z axis
    for i in range(Nm):
        if inp[i].ndim==2:  
            inp[i] = np.flip(inp[i], axis=0)
        else:   
            inp[i] = np.flip(inp[i], axis=1)
    input = inp[0]
    output = inp[1]
    pred = inp[2]
    initial = inp[3]
    predicted_initial = inp[4];    
    ideal_initial=inp[5]
    fwi_result=inp[6]
    true_model=inp[7]
    features=inp[8]

    x = np.arange(Nx) * dx / 1000
    y = np.arange(Nz) * dy / 1000
    extent = np.array([x.min(), x.max(), y.min(), y.max()])
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 12.4
    fig_size[1] = 14.0  # 8 images
    fig_size[1] = 5*14/8
    plt.rcParams["figure.figsize"] = fig_size
    plt.rcParams["font.size"] = 14

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4,ncols=1)
    #  allocate space for colorbar smartly
    divider1 = make_axes_locatable((ax1))
    divider2 = make_axes_locatable((ax2))
    divider3 = make_axes_locatable((ax3))
    divider4 = make_axes_locatable((ax4))
    cax1 = divider1.append_axes("right", size="2%", pad=0.05)
    cax2 = divider2.append_axes("right", size="2%", pad=0.05)
    cax3 = divider3.append_axes("right", size="2%", pad=0.05)
    cax4 = divider4.append_axes("right", size="2%", pad=0.05)
    plt.set_cmap('RdBu_r')
    MIN = np.min(output)
    MAX = np.max(output)
    predicted_initial
    orig_min = np.min(ideal_initial)
    orig_max = np.max(ideal_initial)

    a = np.min((output));   b = np.max((output))
    im1 = ax1.imshow(features[9,::], extent=extent, vmin=a, vmax=b, aspect='auto')
    im2 = ax2.imshow(inp[3],extent=extent,vmin=orig_min,vmax=orig_max, aspect='auto')
    im3 = ax3.imshow(pred,  extent=extent,vmin=a, vmax=b, aspect='auto')
    im4 = ax4.imshow(predicted_initial,extent=extent,vmin=orig_min,vmax=orig_max,aspect='auto')
    #########
    x0 = 0.04
    y0 = -0.25
    if Plot_vertical_lines == 1:
        Boundaries2 = Boundaries * dx / 1000
        for i in range(np.shape(Boundaries)[0]):
            ax1.axvline(Boundaries2[i], color='k',
                        linestyle='--', linewidth=2.5)
            ax2.axvline(Boundaries2[i], color='k',
                        linestyle='--', linewidth=2.5)
            ax3.axvline(Boundaries2[i], color='k',
                        linestyle='--', linewidth=2.5)
    z_label_name = 'z, m'
    ax1.set_ylabel(z_label_name)
    ax1.set_xlabel('x (km)')
    ax2.set_ylabel(z_label_name)
    ax2.set_xlabel('x (km)')
    ax3.set_ylabel(z_label_name)
    ax3.set_xlabel('x (km)')
    ax4.set_ylabel(z_label_name)
    ax4.set_xlabel('x (km)')
    ax1.invert_yaxis()
    ax2.invert_yaxis()
    ax3.invert_yaxis()
    ax4.invert_yaxis()
    ax1.xaxis.set_label_coords(x0, y0)
    ax2.xaxis.set_label_coords(x0, y0)
    ax3.xaxis.set_label_coords(x0, y0)
    ax4.xaxis.set_label_coords(x0, y0)
    ax1.set_title('a) '+Title[0])
    ax2.set_title('b) '+Title[3])
    ax3.set_title('c) '+Title[2])
    ax4.set_title('d) '+Title[4])
    cbar1 = plt.colorbar(im1, cax=cax1)
    cbar2 = plt.colorbar(im2, cax=cax2)
    cbar3 = plt.colorbar(im3, cax=cax3)
    cbar4 = plt.colorbar(im4, cax=cax4)
    x0 = -40    
    y0 = 1.16
    cbar1.set_label('V (m/sec)', labelpad=x0, y=y0, rotation=0)
    cbar2.set_label('V (m/sec)', labelpad=x0, y=y0, rotation=0)
    cbar3.set_label('V (m/sec)', labelpad=x0, y=y0, rotation=0)
    cbar4.set_label('V (m/sec)', labelpad=x0, y=y0, rotation=0)
    plt.subplots_adjust(bottom=0.08,top=0.95,hspace=0.57, right=0.86, wspace=.1)
    if Save_flag == 1:
        plt.savefig(save_file_path, dpi=300,bbox_inches='tight')
        print('Saving ML_result to '+save_file_path)
    # if Show_flag == 1:
    #     plt.show()
    # else:
    #     plt.show(block=False)
    plt.close()
    return None

def PLOT_cgg_field_data(inp, val1, history_flag=0, history=None,
        Boundaries=0, save_file_path='', dx=1, dy=1, Plot_vertical_lines=0, Title='',
        Title2='', Save_flag=0, Show_flag=0, adaptive_colorbar=1):
    """  26.11.21   Plot CGG CNN field data, its prediction, comparison with well log.    """
    ############################
    Nm = len(inp)
    dim1 = inp[0].shape[0]
    dim2 = inp[0].shape[1]
    if dim1 >= dim2:
        Nx = dim1
        Nz = dim2
        for i in range(Nm):
            if inp[i].ndim==2:
                inp[i] = inp[i].swapaxes(0, 1)
            else:
                inp[i] = inp[i].swapaxes(1,2)
    else:
        Nx = dim2
        Nz = dim1
    #   flip z axis
    for i in range(Nm):
        if inp[i].ndim==2:  
            inp[i] = np.flip(inp[i], axis=0)
        else:   
            inp[i] = np.flip(inp[i], axis=1)
    input = inp[0]
    output = inp[1]
    pred = inp[2]
    initial = inp[3]
    predicted_initial = inp[4]
    ideal_initial=inp[5]
    fwi_result=inp[6]
    true_model=inp[7]
    features=inp[8]
    #########
    x = np.arange(Nx) * dx / 1000
    y = np.arange(Nz) * dy / 1000
    extent = np.array([x.min(), x.max(), y.min(), y.max()])
    #########
    fig_size = plt.rcParams["figure.figsize"]
    # fig_size[1] = 14.0  # 8 images
    # fig_size[0] = 12.4
    # fig_size[1] = 6*14/8
    fig_size[0] = 11
    fig_size[1] = 8
    plt.rcParams["figure.figsize"] = fig_size
    plt.rcParams["font.size"] = 14
    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(nrows=5,ncols=1)
    #  allocate space for colorbar smartly
    divider1 = make_axes_locatable((ax1))
    divider2 = make_axes_locatable((ax2))
    divider3 = make_axes_locatable((ax3))
    divider4 = make_axes_locatable((ax4))
    divider5 = make_axes_locatable((ax5))
    cax1 = divider1.append_axes("right", size="2%", pad=0.05)
    cax2 = divider2.append_axes("right", size="2%", pad=0.05)
    cax3 = divider3.append_axes("right", size="2%", pad=0.05)
    cax4 = divider4.append_axes("right", size="2%", pad=0.05)
    plt.set_cmap('RdBu_r')
    MIN = np.min(output)
    MAX = np.max(output)
    orig_min = np.min(ideal_initial)
    orig_max = np.max(ideal_initial)
    #########
    a = np.min((output));   b = np.max((output))
    im1 = ax1.imshow(np.flipud(features[9,::]), extent=extent, vmin=a, vmax=b, aspect='auto')
    im2 = ax2.imshow(np.flipud(inp[3]),extent=extent,vmin=orig_min,vmax=orig_max, aspect='auto')
    im3 = ax3.imshow(np.flipud(pred),  extent=extent,vmin=a, vmax=b, aspect='auto')
    im4 = ax4.imshow(np.flipud(predicted_initial),extent=extent,vmin=orig_min,vmax=orig_max,aspect='auto')
    #########
    info_file=os.path.join('acq_data_parameters_cgg.pkl')
    with open(info_file,'rb') as input:
        acq_data=pickle.load(input)
    log_dict=acq_data['log_dict']
    log_loc=log_dict['loc']
    log=log_dict['data']
    lvp=log[::-1]
    idx = int(log_loc / 25)
    vh = log_loc * np.ones_like(log)/1000
    well_log_depth=np.arange(len(lvp)) * 25
    inverted_log_depth=np.arange(Nz) * 25
    wl_n=len(lvp)
    idx1=34
    idx1=0
    idx2=wl_n
    label1='Well log'
    label2='Apriori 1d initial model log, PCC(-||-,well log)='+ numstr3(PCC(initial[idx1:idx2,idx],lvp[idx1:idx2]))
    # label2_2='Apriori 1d initial model log, PCC(-||-,well log)='+numstr(PCC(initial[0:wl_n,idx],np.roll(lvp,33)))
    label3='Predicted initial model log, PCC(-||-,well log)='+  numstr3(PCC(predicted_initial[idx1:idx2,idx],lvp[idx1:idx2]))
    label4='Apriori 1d initial model log, PCC(-||-,well log)='+ numstr3(PCC(lvp,lvp))
    label5='Predicted initial model log, PCC(-||-,well log)='+  numstr3(PCC(-lvp,lvp))
    label6='Predicted initial model log, PCC(-||-,well log)='+  numstr3(PCC(np.random.random(10000),np.random.random(10000)))
    ax5.plot(well_log_depth,lvp,'b--',label=label1)
    # ax5.plot(np.roll(lvp,33),lvp,'b--',label=label2_2)
    ax5.plot(inverted_log_depth,initial[:,idx],label=label2,color='red')
    ax5.plot(inverted_log_depth,predicted_initial[:,idx],label=label3,color='green')
    #########
    x0 = 0.04
    y0 = -0.25
    z_label_name = 'z, m'
    ax1.set_ylabel(z_label_name)
    ax1.set_xlabel('x (km)')
    ax2.set_ylabel(z_label_name)
    ax2.set_xlabel('x (km)')
    ax3.set_ylabel(z_label_name)
    ax3.set_xlabel('x (km)')
    ax4.set_ylabel(z_label_name)
    ax4.set_xlabel('x (km)')
    ax5.set_ylabel(z_label_name)
    ax5.set_xlabel('x (km)')
    ax1.invert_yaxis()
    ax2.invert_yaxis()
    ax3.invert_yaxis()
    ax4.invert_yaxis()
    ax1.xaxis.set_label_coords(x0, y0)
    ax2.xaxis.set_label_coords(x0, y0)
    ax3.xaxis.set_label_coords(x0, y0)
    ax4.xaxis.set_label_coords(x0, y0)
    # ax5.xaxis.set_label_coords(x0, y0)
    ax1.set_title('a) '+Title[0])
    ax2.set_title('b) '+Title[3])
    ax3.set_title('c) '+Title[2])
    ax4.set_title('d) '+Title[4])
    ax5.set_title('e) Velocity logs')
    ax5.set_ylim(bottom=1500,top=3100)
    ax5.set_xlim(left=0,right=inverted_log_depth[-1])
    ax5.grid()
    ax5.legend(bbox_to_anchor=(0.5, -0.99, 0.44, 0.5),fontsize=16)
    cbar1 = plt.colorbar(im1, cax=cax1)
    cbar2 = plt.colorbar(im2, cax=cax2)
    cbar3 = plt.colorbar(im3, cax=cax3)
    cbar4 = plt.colorbar(im4, cax=cax4)
    x0 = -40    
    y0 = 1.16
    cbar1.set_label('V (m/sec)', labelpad=x0, y=y0, rotation=0)
    cbar2.set_label('V (m/sec)', labelpad=x0, y=y0, rotation=0)
    cbar3.set_label('V (m/sec)', labelpad=x0, y=y0, rotation=0)
    cbar4.set_label('V (m/sec)', labelpad=x0, y=y0, rotation=0)
    plt.subplots_adjust(bottom=0.08,top=0.95,hspace=0.57, right=0.82, wspace=.1)
    if Save_flag == 1:
        plt.savefig(save_file_path, dpi=300,bbox_inches='tight')
        print('Saving ML_result to '+save_file_path)
    plt.close()

    return None

def PLOT_ML_Result_8_pics(inp, val1, history_flag=0, history=None,
        Boundaries=0, save_file_path='', dx=1, dy=1, Plot_vertical_lines=0, Title='',
        Title2='', Save_flag=0, Show_flag=0, adaptive_colorbar=1):
    Nm = len(inp)
    dim1 = inp[0].shape[0]
    dim2 = inp[0].shape[1]
    if dim1 >= dim2:
        Nx = dim1
        Nz = dim2
        for i in range(Nm):
            inp[i] = inp[i].swapaxes(0, 1)
    else:
        Nx = dim2
        Nz = dim1
    #   flip z axis
    for i in range(Nm):
        inp[i] = np.flip(inp[i], axis=0)
    input = inp[0]
    output = inp[1]
    pred = inp[2]
    matrix = inp[4]
    true_model=inp[5]
    x = np.arange(Nx) * dx / 1000
    y = np.arange(Nz) * dy / 1000
    extent = np.array([x.min(), x.max(), y.min(), y.max()])
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 12.4
    fig_size[1] = 14.0
    plt.rcParams["figure.figsize"] = fig_size
    plt.rcParams["font.size"] = 14
    fig, (ax1, ax2, ax3, ax4, ax5, ax6,ax7,ax8) = plt.subplots(nrows=8,ncols=1)
    #  allocate space for colorbar smartly
    divider1 = make_axes_locatable((ax1))
    divider2 = make_axes_locatable((ax2))
    divider3 = make_axes_locatable((ax3))
    divider4 = make_axes_locatable((ax4))
    divider5 = make_axes_locatable((ax5))
    divider6 = make_axes_locatable((ax6))
    divider7 = make_axes_locatable((ax7))
    divider8 = make_axes_locatable((ax8))
    
    cax1 = divider1.append_axes("right", size="2%", pad=0.05)
    cax2 = divider2.append_axes("right", size="2%", pad=0.05)
    cax3 = divider3.append_axes("right", size="2%", pad=0.05)
    cax4 = divider4.append_axes("right", size="2%", pad=0.05)
    cax5 = divider5.append_axes("right", size="2%", pad=0.05)
    cax6 = divider6.append_axes("right", size="2%", pad=0.05)
    cax7 = divider7.append_axes("right", size="2%", pad=0.05)
    cax8 = divider8.append_axes("right", size="2%", pad=0.05)
    plt.set_cmap('RdBu_r')
    MIN = np.min(output)
    MAX = np.max(output)
    orig_min = np.min(inp[5])
    orig_max = np.max(inp[5])
    if adaptive_colorbar == 4:
        a = np.min((output));   b = np.max((output))
        im1 = ax1.imshow(input, extent=extent, vmin=a, vmax=b, aspect='auto')
        im2 = ax2.imshow(output, extent=extent, vmin=a, vmax=b, aspect='auto')
        im3 = ax3.imshow(pred, extent=extent, vmin=orig_min,vmax=orig_max, aspect='auto')
        im4 = ax4.imshow(inp[3], extent=extent,vmin=orig_min, vmax=orig_max, aspect='auto')
    if adaptive_colorbar == 3:
        # a = -1; b = 1
        a = np.min((output));   b = np.max((output))
        im1 = ax1.imshow(input, extent=extent, vmin=a, vmax=b, aspect='auto')
        im2 = ax2.imshow(output,extent=extent, vmin=a, vmax=b, aspect='auto')
        im3 = ax3.imshow(pred,  extent=extent, vmin=a, vmax=b, aspect='auto')
        # im3 = ax3.imshow(pred, extent=extent,aspect='auto')
        im4 = ax4.imshow(inp[3], extent=extent,vmin=orig_min, vmax=orig_max, aspect='auto')
        # im4 = ax4.imshow(inp[3], extent=extent,aspect='auto')
    if adaptive_colorbar == 2:
        im1 = ax1.imshow(input, extent=extent, aspect='auto')
        im2 = ax2.imshow(output, extent=extent, aspect='auto')
        im3 = ax3.imshow(pred, extent=extent, aspect='auto')
        im4 = ax4.imshow(inp[3], extent=extent,aspect='auto')
    if adaptive_colorbar == 1:
        val = np.max(abs(input))
        im1 = ax1.imshow(input, extent=extent, vmin=-
                            val/3, vmax=val/3, aspect='auto')
        val = np.max(abs(output))
        im2 = ax2.imshow(output, extent=extent, vmin=-
                            val,  vmax=val, aspect='auto')
        im3 = ax3.imshow(pred, extent=extent,   vmin=-
                            val,  vmax=val, aspect='auto')
        im4 = ax4.imshow(inp[3], extent=extent, vmin=-
                            val/2, vmax=val/2, aspect='auto')
    if adaptive_colorbar == 0:
        min_lim = -1
        max_lim = 1
        im1 = ax1.imshow(input, extent=extent, vmin=min_lim,vmax=max_lim, aspect='auto')
        im2 = ax2.imshow(output, extent=extent, vmin=min_lim,vmax=max_lim, aspect='auto')
        im3 = ax3.imshow(pred, extent=extent,   vmin=min_lim,vmax=max_lim, aspect='auto')
        im4 = ax4.imshow(inp[3], extent=extent,vmin=min_lim/2, vmax=max_lim/2, aspect='auto')
    tmp = inp[4]
    tmp_extent = np.array([0, tmp.shape[1]*dx/1000, 0, tmp.shape[0]*dy / 1000])
    im5 = ax5.imshow(inp[4], extent=tmp_extent, vmin=orig_min,vmax=orig_max, aspect='auto')
    # im5 = ax5.imshow(inp[4], extent=tmp_extent,aspect='auto')
    tmp = inp[5]
    tmp_extent = np.array([0, tmp.shape[1]*dx/1000, 0, tmp.shape[0]*dy / 1000])
    im6 = ax6.imshow(inp[5], extent=tmp_extent,vmin=orig_min, vmax=orig_max, aspect='auto')
    tmp = inp[6]
    tmp_extent = np.array([0, tmp.shape[1]*dx/1000, 0, tmp.shape[0]*dy / 1000])
    im7 = ax7.imshow(inp[6], extent=tmp_extent,vmin=orig_min, vmax=orig_max, aspect='auto')
    tmp = inp[7]
    tmp_extent = np.array([0, tmp.shape[1]*dx/1000, 0, tmp.shape[0]*dy / 1000])
    im8 = ax8.imshow(inp[7], extent=tmp_extent,vmin=orig_min, vmax=orig_max, aspect='auto')
    #########
    x0 = 0.04
    y0 = -0.25
    if Plot_vertical_lines == 1:
        Boundaries2 = Boundaries * dx / 1000
        for i in range(np.shape(Boundaries)[0]):
            ax1.axvline(Boundaries2[i], color='k',
                        linestyle='--', linewidth=2.5)
            ax2.axvline(Boundaries2[i], color='k',
                        linestyle='--', linewidth=2.5)
            ax3.axvline(Boundaries2[i], color='k',
                        linestyle='--', linewidth=2.5)
    z_label_name = 'z, m'
    ax1.set_ylabel(z_label_name)
    ax1.set_xlabel('x (km)')
    ax2.set_ylabel(z_label_name)
    ax2.set_xlabel('x (km)')
    ax3.set_ylabel(z_label_name)
    ax3.set_xlabel('x (km)')
    ax4.set_ylabel(z_label_name)
    ax4.set_xlabel('x (km)')
    ax5.set_ylabel(z_label_name)
    ax5.set_xlabel('x (km)')
    ax6.set_ylabel(z_label_name)
    ax6.set_xlabel('x (km)')
    ax7.set_ylabel(z_label_name)
    ax7.set_xlabel('x (km)')
    ax8.set_ylabel(z_label_name)
    ax8.set_xlabel('x (km)')
    ax1.invert_yaxis()
    ax2.invert_yaxis()
    ax3.invert_yaxis()
    ax4.invert_yaxis()
    ax5.invert_yaxis()
    ax6.invert_yaxis()
    ax7.invert_yaxis()
    ax8.invert_yaxis()

    ax1.xaxis.set_label_coords(x0, y0)
    ax2.xaxis.set_label_coords(x0, y0)
    ax3.xaxis.set_label_coords(x0, y0)
    ax4.xaxis.set_label_coords(x0, y0)
    ax5.xaxis.set_label_coords(x0, y0)
    ax6.xaxis.set_label_coords(x0, y0)
    ax7.xaxis.set_label_coords(x0, y0)
    ax8.xaxis.set_label_coords(x0, y0)

    ax1.set_title(Title[0])
    ax2.set_title(Title[1])
    ax3.set_title(Title[2])
    ax4.set_title(Title[3])
    ax5.set_title(Title[4])
    ax6.set_title(Title[5])
    ax7.set_title(Title[6])
    ax8.set_title(Title[7])

    cbar1 = plt.colorbar(im1, cax=cax1)
    cbar2 = plt.colorbar(im2, cax=cax2)
    cbar3 = plt.colorbar(im3, cax=cax3)
    cbar4 = plt.colorbar(im4, cax=cax4)
    cbar5 = plt.colorbar(im5, cax=cax5)
    cbar6 = plt.colorbar(im6, cax=cax6)
    cbar7 = plt.colorbar(im7, cax=cax7)
    cbar8 = plt.colorbar(im8, cax=cax8)
    x0 = -40    
    y0 = 1.16
    cbar1.set_label('V (m/sec)', labelpad=x0, y=y0, rotation=0)
    cbar2.set_label('V (m/sec)', labelpad=x0, y=y0, rotation=0)
    cbar3.set_label('V (m/sec)', labelpad=x0, y=y0, rotation=0)
    cbar4.set_label('V (m/sec)', labelpad=x0, y=y0, rotation=0)
    cbar5.set_label('V (m/sec)', labelpad=x0, y=y0, rotation=0)
    cbar6.set_label('V (m/sec)', labelpad=x0, y=y0, rotation=0)
    cbar7.set_label('V (m/sec)', labelpad=x0, y=y0, rotation=0)
    cbar8.set_label('V (m/sec)', labelpad=x0, y=y0, rotation=0)
    plt.subplots_adjust(bottom=0.08,top=0.95,hspace=0.57, right=0.86, wspace=.1)
    if Save_flag == 1:
        plt.savefig(save_file_path, dpi=300,bbox_inches='tight')
        print('Saving ML_result to '+save_file_path)
    if Show_flag == 1:
        plt.show()
    else:
        plt.show(block=False)
    plt.close()
    return None

def Plot_image(Data, Title='Title', c_lim='', x='', x_label='', y='', y_label='',
               dx='', dy='', Save_flag=0, Save_pictures_path='./Pictures',
               Reverse_axis=1, Curve='', Show_flag=1, Aspect='equal', write_fig_title=1):
    # aspect - 'auto'
    if c_lim == '':
        c_lim = [np.min(Data), np.max(Data)]
    if x == '':
        x = (np.arange(np.shape(Data)[1]))
    if y == '':
        y = (np.arange(np.shape(Data)[0]))
    if dx != '':
        x = (np.arange(np.shape(Data)[1]))*dx
    if dy != '':
        y = (np.arange(np.shape(Data)[0]))*dy
    extent = [x.min(), x.max(), y.min(), y.max()]
    # if Save_flag==1:
    #    plt.ion()
    fig = plt.figure()
    fig.dpi = 330
    # fig_size = plt.rcParams["figure.figsize"]
    # fig_size[0] = 10.4
    # fig_size[1] = 8.0
    # plt.rcParams["figure.figsize"] = fig_size
    plt.set_cmap('RdBu_r')
    # plt.axis(extent, Aspect)
    # plt.axis(extent, 'auto')
    if write_fig_title == 1:
        plt.title(Title)
    if Reverse_axis == 1:
        plt.imshow(np.flipud(Data), extent=extent,interpolation='nearest', aspect=Aspect)
        plt.gca().invert_yaxis()
    else:
        plt.imshow((Data), extent=extent,
                   interpolation='nearest', aspect=Aspect)
    if Curve != '':
        # if len(np.shape(Curve)) == 2:
        #     Curve=Curve[0,:]
        plt.plot(x, Curve, color='white', linewidth=1.2, linestyle='--')

    ax = plt.gca()
    divider1 = make_axes_locatable((ax))
    cax1 = divider1.append_axes("right", size="2%", pad=0.05)
    cbar = plt.colorbar(cax=cax1)
    cbar.set_label("(m/s)")
    plt.clim(c_lim)
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    # plt.axis('equal')
    # plt.axis('tight')
    # tight_figure(fig)
    if Save_flag == 1:
        if not os.path.exists(Save_pictures_path):
            os.mkdir(Save_pictures_path)
        name = Save_pictures_path + '/' + Title + '.png'
        print(name)
        # plt.show()
        # plt.show(block=True)
        # plt.show(block=False)
        plt.savefig(name, bbox_inches='tight')
    if Show_flag == 0:
        plt.show(block=False)
        # plt.show(block=True)
    else:
        if Show_flag == 2:
            a = 1
        else:
            plt.show()
    plt.close()
    return None
