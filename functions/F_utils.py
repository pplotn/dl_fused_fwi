# from F_models import *
from functions.F_modules import *
from functions.F_plotting import *
def F_r2(mat,mat_true):
    # r2 = 1 - (np.std(mat_true.flatten() - mat.flatten()) / np.std(mat_true.flatten())) ** 2
    v1 = mat.flatten()
    v2 = mat_true.flatten()
    if np.isnan(v1).any()==True:
        r2_2=math.nan
    else:
        r2_2 = r2_score(v1,v2)
    return r2_2
def RepresentsInt(s):
    try: 
        int(s)
        return True
    except ValueError:
        return False
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

def F_resize(Mat,dx=1,dz=1,dx_new=4,dz_new=4,flag_plotting=0):
    if dx != dx_new or dz != dz_new:
        sz_old = np.array([Mat.shape[0], Mat.shape[1]], dtype=int)
        F = np.asarray([dx_new / dx, dz_new / dz])
        sz_new = np.array([sz_old - 1] / F, dtype=int)
        A=sz_new[0][0]
        B=sz_new[0][1]
        if dx_new==dx:
            A=Mat.shape[0]
        if dz_new==dz:
            B=Mat.shape[1]
        Mat2 = imresize(Mat, (A,B), anti_aliasing=True)
        print('Size before imresizing', np.shape(Mat))
        print('Size after imresizing', np.shape(Mat2))
        if flag_plotting==1:
            Plot_image(Mat.T,Show_flag=0,Save_flag=1,dx=dx,dy=dz,
                Title="_",Save_pictures_path='./Pictures')
            Plot_image(Mat2.T,Show_flag=0,Save_flag=1,dx=dx_new,dy=dz_new,
                Title="_",Save_pictures_path='./Pictures')
    else:
        Mat2 = Mat
        print('Size before imresizing', np.shape(Mat))
        print('Size after imresizing', np.shape(Mat2))
    return Mat2

def F_calculate_log_number(path, Word, type='.png'):
    Const = len(fnmatch.filter(os.listdir(path), Word + '*'))
    Name = path + '/' + Word + str(Const) + type
    while os.path.exists(Name):
        Const = Const + 1
        Name = path + '/' + Word + str(Const) + type
    return Const

def load_file(filename, variable_name):
    f = h5py.File(filename, 'r')
    dat = np.array(f.get(variable_name))
    return dat

def PLOT_ML_Result(M1, M2, M3, ind_train, ind_test, Train_on1_model_and_test_on_other=0, Train_models=None,
                   Test_models=None, Valid_models=None, Nl=0, history_flag=0, history=None, Boundaries=0, Name='', dx=1,
                   dy=1, Plot_vertical_lines=0,
                   Title='', Save_flag=0, Show_flag=0, COEFF=0.15):
    if Train_on1_model_and_test_on_other == 1:
        Test_models = np.arange(0, 0);
        Train_models = np.arange(0, 1)
    misfit_stats, misfits = F_calculate_misfits(add_dim_forth(M2.T), add_dim_forth(M3.T), ind_train, ind_test,
                                                Train_on1_model_and_test_on_other, Train_models, Test_models,
                                                Valid_models, print_flag=0)
    Title1 = 'Input' + ', ' + str(Nl) + ' models'
    Title2 = 'Target'
    # Title3 = 'Output,R2=' + str('{0:.2f}'.format(F_r2(M2, M3)))
    Title3 = 'Output'
    z_label_name = 'z, m'
    Mat = M1
    # COEFF = 0.6;
    MIN1 = COEFF * np.min(Mat)
    MAX1 = COEFF * np.max(Mat)
    Mat = np.concatenate((M2, M3), axis=0)
    # %
    x = np.arange(np.shape(M1)[1]) * dx / 1000
    y = np.arange(np.shape(M1)[0]) * dy / 1000
    extent = np.array([x.min(), x.max(), y.min(), y.max()])
    # matplotlib.rcParams.update({'font.size': 15})
    M1 = np.flipud(M1)
    M2 = np.flipud(M2)
    M3 = np.flipud(M3)
    ##############  Calculate relative error
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 6.4
    fig_size[1] = 8.0
    plt.rcParams["figure.figsize"] = fig_size
    plt.rcParams["font.size"] = 13
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4, ncols=1)
    #  allocate space for colorbar smartly
    divider1 = make_axes_locatable((ax1))
    divider2 = make_axes_locatable((ax2))
    divider3 = make_axes_locatable((ax3))
    cax1 = divider1.append_axes("right", size="2%", pad=0.05)
    cax2 = divider2.append_axes("right", size="2%", pad=0.05)
    cax3 = divider3.append_axes("right", size="2%", pad=0.05)
    plt.set_cmap('RdBu_r')
    im1 = ax1.imshow(M1, extent=extent, vmin=MIN1, vmax=MAX1, aspect='auto')
    im2 = ax2.imshow(M2, extent=extent, vmin=MIN1, vmax=MAX1, aspect='auto')
    im3 = ax3.imshow(M3, extent=extent, vmin=MIN1, vmax=MAX1, aspect='auto')
    x0 = 0.11;
    y0 = -0.25
    if history_flag == 1:
        ax4.plot(history['loss'], linewidth=2)
        ax4.plot(history['val_loss'], linewidth=2)
        ax4.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax4.autoscale(enable=True, axis='y', tight=True)
        ax4.autoscale(enable=True, axis='x', tight=True)
        bottom, top = ax4.get_ylim()
        # ax4.set_ylim(top=1.3*history['val_loss'][1])
        ax4.legend(['Training', 'Validation'], loc='upper right')
    if Plot_vertical_lines == 1:
        Boundaries2 = Boundaries * dx / 1000
        for i in range(np.shape(Boundaries)[0]):
            ax1.axvline(Boundaries2[i], color='y', linestyle='--', linewidth=2.5)
            ax2.axvline(Boundaries2[i], color='y', linestyle='--', linewidth=2.5)
            ax3.axvline(Boundaries2[i], color='y', linestyle='--', linewidth=2.5)
    ax1.set_ylabel(z_label_name)
    ax1.set_xlabel('x (km)')
    ax2.set_ylabel(z_label_name)
    ax2.set_xlabel('x (km)')
    ax3.set_ylabel(z_label_name)
    ax3.set_xlabel('x (km)')
    ax4.set_ylabel('Loss function')
    ax4.set_xlabel('Epoch')

    ax1.invert_yaxis()
    ax2.invert_yaxis()
    ax3.invert_yaxis()

    ax1.xaxis.set_label_coords(x0, y0)
    ax2.xaxis.set_label_coords(x0, y0)
    ax3.xaxis.set_label_coords(x0, y0)
    # ax4.xaxis.set_label_coords(x0 + 0.03, y0 - 0.045)

    ax1.set_title(Title1)
    ax2.set_title(Title2)
    ax3.set_title(Title3)
    ax4.set_title(Title3)
    # ax5.set_title(Title4)
    # ax6.set_title(Title5)
    cbar1 = plt.colorbar(im1, cax=cax1)
    cbar2 = plt.colorbar(im2, cax=cax2)
    cbar3 = plt.colorbar(im3, cax=cax3)
    x0 = -50;
    y0 = 1.16
    cbar1.set_label('V (m/sec)', labelpad=x0, y=y0, rotation=0)
    cbar2.set_label('V (m/sec)', labelpad=x0, y=y0, rotation=0)
    cbar3.set_label('V (m/sec)', labelpad=x0, y=y0, rotation=0)

    plt.subplots_adjust(bottom=0.08, top=0.95, hspace=.49, right=0.86, wspace=.5)
    if Save_flag == 1:
        plt.savefig(Name, dpi=300)
        print('Saving ML_result to ' + Name)
    if Show_flag == 1:
        plt.show()
    else:
        plt.show(block=False)
    plt.close()
    return None

def PLOT_ML_Result2(inp, history_flag=0, history=None,
                    Boundaries=0, save_file_path='', dx=1, dy=1, Plot_vertical_lines=0, Title='',
                    Title2='', Save_flag=0, Show_flag=0):
    Nm = inp.shape[0];
    dim1 = inp.shape[1]
    dim2 = inp.shape[2]
    if dim1 >= dim2:
        Nx = dim1
        Nz = dim2
        d1 = 'Nx'
        d2 = 'Nz'
        inp = inp.swapaxes(1, 2)
    else:
        Nx = dim2
        Nz = dim1
        d1 = 'Nz'
        d2 = 'Nx'
    inp = np.flip(inp, axis=1)  # revert z axis
    input = inp[0, :, :]
    true = inp[1, :, :];
    pred = inp[2, :, :];
    pred_smoothed = inp[3, :, :]

    Title3 = 'Predicted, ' + Title
    Title4 = 'Predicted smoothed, ' + Title2
    x = np.arange(Nx) * dx / 1000
    y = np.arange(Nz) * dy / 1000
    extent = np.array([x.min(), x.max(), y.min(), y.max()])
    # matplotlib.rcParams.update({'font.size': 15})

    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 10.4
    fig_size[1] = 8.0
    plt.rcParams["figure.figsize"] = fig_size
    plt.rcParams["font.size"] = 10
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4, ncols=1)
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
    MIN1 = np.min(input)
    MAX1 = np.max(input)
    MIN2 = np.min(true);
    MAX2 = np.max(true)
    # MIN1 = clim[0]; MAX1 = clim[1]
    im1 = ax1.imshow(input, extent=extent, vmin=MIN2, vmax=MAX2, aspect='auto')
    im2 = ax2.imshow(true, extent=extent, vmin=MIN2, vmax=MAX2, aspect='auto')
    im3 = ax3.imshow(pred, extent=extent, vmin=MIN2, vmax=MAX2, aspect='auto')
    im4 = ax4.imshow(pred_smoothed, extent=extent, vmin=MIN2, vmax=MAX2, aspect='auto')
    x0 = 0.11;
    y0 = -0.25
    if history_flag == 1:
        ax4.plot(history['loss'], linewidth=2)
        ax4.plot(history['val_loss'], linewidth=2)
        ax4.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax4.autoscale(enable=True, axis='y', tight=True)
        ax4.autoscale(enable=True, axis='x', tight=True)
        bottom, top = ax4.get_ylim()
        ax4.legend(['Training', 'Validation'], loc='upper right')
    if Plot_vertical_lines == 1:
        Boundaries2 = Boundaries * dx / 1000
        for i in range(np.shape(Boundaries)[0]):
            ax1.axvline(Boundaries2[i], color='k', linestyle='--', linewidth=2.5)
            ax2.axvline(Boundaries2[i], color='k', linestyle='--', linewidth=2.5)
            ax3.axvline(Boundaries2[i], color='k', linestyle='--', linewidth=2.5)
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
    ax1.set_title('Input')
    ax2.set_title('True')
    ax3.set_title(Title3)
    ax4.set_title(Title4)
    cbar1 = plt.colorbar(im1, cax=cax1)
    cbar2 = plt.colorbar(im2, cax=cax2)
    cbar3 = plt.colorbar(im3, cax=cax3)
    cbar4 = plt.colorbar(im4, cax=cax4)
    x0 = -50;
    y0 = 1.16
    cbar1.set_label('V (m/sec)', labelpad=x0, y=y0, rotation=0)
    cbar2.set_label('V (m/sec)', labelpad=x0, y=y0, rotation=0)
    cbar3.set_label('V (m/sec)', labelpad=x0, y=y0, rotation=0)
    cbar4.set_label('V (m/sec)', labelpad=x0, y=y0, rotation=0)
    plt.subplots_adjust(bottom=0.08, top=0.95, hspace=0.57, right=0.86, wspace=.1)
    if Save_flag == 1:
        plt.savefig(save_file_path, dpi=400)
        print('Saving ML_result to ' + save_file_path)
    if Show_flag == 1:
        plt.show()
    else:
        plt.show(block=False)
    plt.close()
    return None

def PLOT_ML_Result4(inp,R2val,history_flag=0, history=None,
                    Boundaries=0, save_file_path='', dx=1, dy=1, Plot_vertical_lines=0, Title='',
                    Title2='', Save_flag=0, Show_flag=0):
    Nm = len(inp);
    dim1 = inp[0].shape[0]
    dim2 = inp[0].shape[1]
    if dim1 >= dim2:
        Nx = dim1
        Nz = dim2
        for i in range(Nm):
            inp[i] = inp[i].swapaxes(0,1)
    else:
        Nx = dim2
        Nz = dim1
    #   flip z axis
    for i in range(Nm):
        inp[i] = np.flip(inp[i], axis=0)
    input = inp[0]
    output = inp[1]
    pred = inp[2]
    matrix = inp[3]
    true_model=inp[4]

    Title3 = Title
    x = np.arange(Nx) * dx / 1000
    y = np.arange(Nz) * dy / 1000
    extent = np.array([x.min(), x.max(), y.min(), y.max()])
    # matplotlib.rcParams.update({'font.size': 15})

    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 10.4
    fig_size[1] = 8.0
    plt.rcParams["figure.figsize"] = fig_size
    plt.rcParams["font.size"] = 10
    fig, (ax1, ax2, ax3,ax5, ax6) = plt.subplots(nrows=5, ncols=1)
    #  allocate space for colorbar smartly
    divider1 = make_axes_locatable((ax1))
    divider2 = make_axes_locatable((ax2))
    divider3 = make_axes_locatable((ax3))
    divider5 = make_axes_locatable((ax5))
    divider6 = make_axes_locatable((ax6))

    cax1 = divider1.append_axes("right", size="2%", pad=0.05)
    cax2 = divider2.append_axes("right", size="2%", pad=0.05)
    cax3 = divider3.append_axes("right", size="2%", pad=0.05)
    cax5 = divider5.append_axes("right", size="2%", pad=0.05)
    cax6 = divider6.append_axes("right", size="2%", pad=0.05)
    plt.set_cmap('RdBu_r')
    MIN =np.min(output)
    MAX = np.max(output)
    orig_min=np.min(true_model)
    orig_max=np.max(true_model)

    # im1 = ax1.imshow(input, extent=extent, vmin=-2,vmax=2,aspect='auto')
    # im2 = ax2.imshow(output, extent=extent,vmin=-2,vmax=2, aspect='auto')
    # im3 = ax3.imshow(pred,extent=extent,   vmin=-2, vmax=2, aspect='auto')

    im1 = ax1.imshow(input, extent=extent,aspect='auto')
    im2 = ax2.imshow(output, extent=extent,vmin=MIN,vmax=MAX, aspect='auto')
    im3 = ax3.imshow(pred,extent=extent,  vmin=MIN, vmax=MAX, aspect='auto')
    
    tmp=matrix
    tmp_extent = np.array([0,tmp.shape[1]*dx/1000,0,tmp.shape[0]*dy / 1000])
    im5 = ax5.imshow(matrix, extent=tmp_extent,vmin=orig_min,vmax=orig_max,aspect='auto')

    tmp=true_model
    tmp_extent = np.array([0,tmp.shape[1]*dx/1000,0,tmp.shape[0]*dy / 1000])
    im6 = ax6.imshow(true_model, extent=tmp_extent,vmin=orig_min,vmax=orig_max,aspect='auto')

    x0 = 0.11;
    y0 = -0.25
    if Plot_vertical_lines == 1:
        Boundaries2 = Boundaries * dx / 1000
        for i in range(np.shape(Boundaries)[0]):
            ax1.axvline(Boundaries2[i], color='k', linestyle='--', linewidth=2.5)
            ax2.axvline(Boundaries2[i], color='k', linestyle='--', linewidth=2.5)
            ax3.axvline(Boundaries2[i], color='k', linestyle='--', linewidth=2.5)
    z_label_name = 'z, m'
    ax1.set_ylabel(z_label_name)
    ax1.set_xlabel('x (km)')
    ax2.set_ylabel(z_label_name)
    ax2.set_xlabel('x (km)')
    ax3.set_ylabel(z_label_name)
    ax3.set_xlabel('x (km)')
    ax5.set_ylabel(z_label_name)
    ax5.set_xlabel('x (km)')
    ax6.set_ylabel(z_label_name)
    ax6.set_xlabel('x (km)')

    ax1.invert_yaxis()
    ax2.invert_yaxis()
    ax3.invert_yaxis()
    ax5.invert_yaxis()
    ax6.invert_yaxis()

    ax1.xaxis.set_label_coords(x0, y0)
    ax2.xaxis.set_label_coords(x0, y0)
    ax3.xaxis.set_label_coords(x0, y0)
    ax5.xaxis.set_label_coords(x0, y0)
    ax6.xaxis.set_label_coords(x0, y0)

    ax1.set_title('Input'+Title2)
    # print(output.shape)
    # print(true_model.shape)
    ax2.set_title('Target')
    # ax2.set_title('True'+numstr(F_r2(output,true_model)))
    # ax2.set_title('True'+numstr(F_r2(true_model,output)))
    ax3.set_title(Title3)
    # ax5.set_title('difference (predicted,true)')
    ax5.set_title('Predicted initial model for fwi, R2(predicted initial,true)='+R2val)
    # ax6.set_title('Ideal initial model')
    ax6.set_title('True model')

    cbar1 = plt.colorbar(im1, cax=cax1)
    cbar2 = plt.colorbar(im2, cax=cax2)
    cbar3 = plt.colorbar(im3, cax=cax3)
    cbar5 = plt.colorbar(im5, cax=cax5)
    cbar6 = plt.colorbar(im6, cax=cax6)
    x0 = -50;
    y0 = 1.16
    cbar1.set_label('V (m/sec)', labelpad=x0, y=y0, rotation=0)
    cbar2.set_label('V (m/sec)', labelpad=x0, y=y0, rotation=0)
    cbar3.set_label('V (m/sec)', labelpad=x0, y=y0, rotation=0)
    cbar5.set_label('V (m/sec)', labelpad=x0, y=y0, rotation=0)
    cbar6.set_label('V (m/sec)', labelpad=x0, y=y0, rotation=0)
    plt.subplots_adjust(bottom=0.08, top=0.95, hspace=0.57, right=0.86, wspace=.1)
    if Save_flag == 1:
        plt.savefig(save_file_path,dpi=400)
        print('Saving ML_result to '+save_file_path)
    if Show_flag == 1:
        plt.show()
    else:
        plt.show(block=False)
    plt.close()
    return None

def PLOT_ML_Result_adaptive_colorbar(inp,R2val,history_flag=0, history=None,
    Boundaries=0, save_file_path='', dx=1, dy=1, Plot_vertical_lines=0, Title='',
    Title2='', Save_flag=0, Show_flag=0,adaptive_colorbar=1):
    Nm = len(inp);
    dim1 = inp[0].shape[0]
    dim2 = inp[0].shape[1]
    if dim1 >= dim2:
        Nx = dim1;Nz = dim2
        for i in range(Nm):
            inp[i] = inp[i].swapaxes(0,1)
    else:
        Nx = dim2;Nz = dim1
    #   flip z axis
    for i in range(Nm):
        inp[i] = np.flip(inp[i], axis=0)
    input = inp[0]
    output = inp[1]
    pred = inp[2]
    matrix = inp[4]
    true_model=inp[5]
    Title3 = Title
    x = np.arange(Nx) * dx / 1000
    y = np.arange(Nz) * dy / 1000
    extent = np.array([x.min(), x.max(), y.min(), y.max()])
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 10.4
    fig_size[1] = 8.0
    plt.rcParams["figure.figsize"] = fig_size
    plt.rcParams["font.size"] = 10
    fig, (ax1,ax2,ax3,ax4,ax5,ax6) = plt.subplots(nrows=6, ncols=1)
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
    MIN =np.min(output)
    MAX = np.max(output)
    orig_min=np.min(true_model)
    orig_max=np.max(true_model)
    if adaptive_colorbar==3:
        im1 = ax1.imshow(input, extent=extent,aspect='auto')
        a=np.min((output)); b=np.max((output))
        im2 = ax2.imshow(output, extent=extent,vmin=a,vmax=b, aspect='auto')
        im3 = ax3.imshow(pred,extent=extent,   vmin=a,vmax=b, aspect='auto')
        im4 = ax4.imshow(inp[3],extent=extent,aspect='auto')
    if adaptive_colorbar==2:
        val=np.max(abs(input))
        im1 = ax1.imshow(input, extent=extent,aspect='auto')
        val=np.max(abs(output))
        im2 = ax2.imshow(output, extent=extent,vmin=-val,  vmax=val, aspect='auto')
        im3 = ax3.imshow(pred,extent=extent,   vmin=-val,  vmax=val, aspect='auto')
        im4 = ax4.imshow(inp[3],extent=extent,vmin= -val,vmax=val,aspect='auto')
    if adaptive_colorbar==1:
        val=np.max(abs(input))
        im1 = ax1.imshow(input, extent=extent,vmin=-val/3,vmax=val/3,aspect='auto')
        val=np.max(abs(output))
        im2 = ax2.imshow(output, extent=extent,vmin=-val,  vmax=val, aspect='auto')
        im3 = ax3.imshow(pred,extent=extent,   vmin=-val,  vmax=val, aspect='auto')
        im4 = ax4.imshow(inp[3],extent=extent,vmin= -val/2,vmax=val/2,aspect='auto')
    if adaptive_colorbar==0:
        min_lim=-1;  max_lim=1;
        im1 = ax1.imshow(input, extent=extent, vmin=min_lim, vmax=max_lim,aspect='auto')
        im2 = ax2.imshow(output, extent=extent,vmin=min_lim, vmax=max_lim, aspect='auto')
        im3 = ax3.imshow(pred,extent=extent,   vmin=min_lim, vmax=max_lim, aspect='auto')
        im4 = ax4.imshow(inp[3],extent=extent, vmin=min_lim/2, vmax=max_lim/2,aspect='auto')
    tmp=matrix; tmp_extent = np.array([0,tmp.shape[1]*dx/1000,0,tmp.shape[0]*dy / 1000])
    im5 = ax5.imshow(matrix, extent=tmp_extent,vmin=orig_min,vmax=orig_max,aspect='auto')
    tmp=true_model
    tmp_extent = np.array([0,tmp.shape[1]*dx/1000,0,tmp.shape[0]*dy / 1000])
    im6 = ax6.imshow(true_model, extent=tmp_extent,vmin=orig_min,vmax=orig_max,aspect='auto')
    x0=0.11;y0=-0.25
    if Plot_vertical_lines == 1:
        Boundaries2 = Boundaries * dx / 1000
        for i in range(np.shape(Boundaries)[0]):
            ax1.axvline(Boundaries2[i], color='k', linestyle='--', linewidth=2.5)
            ax2.axvline(Boundaries2[i], color='k', linestyle='--', linewidth=2.5)
            ax3.axvline(Boundaries2[i], color='k', linestyle='--', linewidth=2.5)
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

    ax1.set_title('Input'+Title2)
    ax2.set_title('Target')
    ax3.set_title(Title3)
    ax4.set_title('difference (prediction,target)')
    # ax5.set_title('Predicted initial model for fwi, R2(predicted initial,true)='+R2val)
    ax5.set_title('Predicted initial model for fwi, R2(predicted initial,ideal initial)='+R2val)
    # ax6.set_title('True model') # ('Ideal initial model')
    ax6.set_title('Ideal initial model')

    cbar1 = plt.colorbar(im1, cax=cax1)
    cbar2 = plt.colorbar(im2, cax=cax2)
    cbar3 = plt.colorbar(im3, cax=cax3)
    cbar4 = plt.colorbar(im4, cax=cax4)
    cbar5 = plt.colorbar(im5, cax=cax5)
    cbar6 = plt.colorbar(im6, cax=cax6)
    x0 = -50;
    y0 = 1.16
    cbar1.set_label('V (m/sec)', labelpad=x0, y=y0, rotation=0)
    cbar2.set_label('V (m/sec)', labelpad=x0, y=y0, rotation=0)
    cbar3.set_label('V (m/sec)', labelpad=x0, y=y0, rotation=0)
    cbar4.set_label('V (m/sec)', labelpad=x0, y=y0, rotation=0)
    cbar5.set_label('V (m/sec)', labelpad=x0, y=y0, rotation=0)
    cbar6.set_label('V (m/sec)', labelpad=x0, y=y0, rotation=0)
    plt.subplots_adjust(bottom=0.08, top=0.95, hspace=0.57, right=0.86, wspace=.1)
    if Save_flag == 1:
        plt.savefig(save_file_path,dpi=400)
        print('Saving ML_result to '+save_file_path)
    if Show_flag == 1:
        plt.show()
    else:
        plt.show(block=False)
    plt.close()
    return None

def PLOT_ML_Result_experimental(inp,R2val,history_flag=0, history=None,
    Boundaries=0, save_file_path='', dx=1, dy=1, Plot_vertical_lines=0, Title='',
    Title2='', Save_flag=0, Show_flag=0,adaptive_colorbar=1):
    Nm = len(inp);
    dim1 = inp[0].shape[0]
    dim2 = inp[0].shape[1]
    if dim1 >= dim2:
        Nx = dim1;Nz = dim2
        for i in range(Nm):
            inp[i] = inp[i].swapaxes(0,1)
    else:
        Nx = dim2;Nz = dim1
    #   flip z axis
    for i in range(Nm):
        inp[i] = np.flip(inp[i], axis=0)
    Title3 = Title
    x = np.arange(Nx) * dx / 1000
    y = np.arange(Nz) * dy / 1000
    extent = np.array([x.min(), x.max(), y.min(), y.max()])
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 10.4
    fig_size[1] = 8.0
    plt.rcParams["figure.figsize"] = fig_size
    plt.rcParams["font.size"] = 10
    fig, (ax1,ax2,ax3,ax4,ax5,ax6) = plt.subplots(nrows=6, ncols=1)
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
    mat=inp[5];MIN =np.min(mat);MAX=np.max(mat)
    im1 = ax1.imshow(inp[0], extent=extent,aspect='auto')
    im2 = ax2.imshow(inp[1], extent=extent,aspect='auto')
    im3 = ax3.imshow(inp[2], extent=extent,aspect='auto')
    im4 = ax4.imshow(inp[3], extent=extent,vmin=MIN,vmax=MAX,aspect='auto')
    im5 = ax5.imshow(inp[4], extent=extent,vmin=MIN,vmax=MAX,aspect='auto')
    im6 = ax6.imshow(inp[5], extent=extent,vmin=MIN,vmax=MAX,aspect='auto')
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

    x0=0.11;y0=-0.25
    ax1.xaxis.set_label_coords(x0, y0)
    ax2.xaxis.set_label_coords(x0, y0)
    ax3.xaxis.set_label_coords(x0, y0)
    ax4.xaxis.set_label_coords(x0, y0)
    ax5.xaxis.set_label_coords(x0, y0)
    ax6.xaxis.set_label_coords(x0, y0)

    if Title2=='':   Title2=['','','','','','']
    ax1.set_title(Title2[0])
    ax2.set_title(Title2[1])
    ax3.set_title(Title2[2])
    ax4.set_title(Title2[3])
    ax5.set_title(Title2[4])
    ax6.set_title(Title2[5])
    # ax5.set_title('Predicted initial model for fwi, R2(predicted initial,ideal initial)='+R2val)
    # ax6.set_title('True model') # ('Ideal initial model')

    cbar1 = plt.colorbar(im1, cax=cax1)
    cbar2 = plt.colorbar(im2, cax=cax2)
    cbar3 = plt.colorbar(im3, cax=cax3)
    cbar4 = plt.colorbar(im4, cax=cax4)
    cbar5 = plt.colorbar(im5, cax=cax5)
    cbar6 = plt.colorbar(im6, cax=cax6)
    x0 = -50;y0 = 1.16
    cbar1.set_label('V (m/sec)', labelpad=x0, y=y0, rotation=0)
    cbar2.set_label('V (m/sec)', labelpad=x0, y=y0, rotation=0)
    cbar3.set_label('V (m/sec)', labelpad=x0, y=y0, rotation=0)
    cbar4.set_label('V (m/sec)', labelpad=x0, y=y0, rotation=0)
    cbar5.set_label('V (m/sec)', labelpad=x0, y=y0, rotation=0)
    cbar6.set_label('V (m/sec)', labelpad=x0, y=y0, rotation=0)
    plt.subplots_adjust(bottom=0.08, top=0.95, hspace=0.57, right=0.86, wspace=.1)
    if Save_flag == 1:
        plt.savefig(save_file_path,dpi=400)
        print('Saving ML_result to '+save_file_path)
    if Show_flag == 1:
        plt.show()
    else:
        plt.show(block=False)
    plt.close()
    return None

def PLOT_ML_Result_fixed_colorbar(inp,R2val,history_flag=0, history=None,
                    Boundaries=0, save_file_path='', dx=1, dy=1, Plot_vertical_lines=0, Title='',
                    Title2='', Save_flag=0, Show_flag=0):
    Nm = len(inp);
    dim1 = inp[0].shape[0]
    dim2 = inp[0].shape[1]
    if dim1 >= dim2:
        Nx = dim1;Nz = dim2
        for i in range(Nm):
            inp[i] = inp[i].swapaxes(0,1)
    else:
        Nx = dim2;Nz = dim1
    #   flip z axis
    for i in range(Nm):
        inp[i] = np.flip(inp[i], axis=0)
    input = inp[0]
    output = inp[1]
    pred = inp[2]
    matrix = inp[4]
    true_model=inp[5]
    Title3 = Title
    x = np.arange(Nx) * dx / 1000
    y = np.arange(Nz) * dy / 1000
    extent = np.array([x.min(), x.max(), y.min(), y.max()])
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 10.4
    fig_size[1] = 8.0
    plt.rcParams["figure.figsize"] = fig_size
    plt.rcParams["font.size"] = 10
    fig, (ax1,ax2,ax3,ax4,ax5,ax6) = plt.subplots(nrows=6, ncols=1)
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
    MIN =np.min(output)
    MAX = np.max(output)
    orig_min=np.min(true_model)
    orig_max=np.max(true_model)
    # im1 = ax1.imshow(input, extent=extent,aspect='auto')
    # im2 = ax2.imshow(output, extent=extent,vmin=MIN,vmax=MAX, aspect='auto')
    # im3 = ax3.imshow(pred,extent=extent,  vmin=MIN, vmax=MAX, aspect='auto')
    im1 = ax1.imshow(input, extent=extent, vmin=-1, vmax=1,aspect='auto')
    im2 = ax2.imshow(output, extent=extent,vmin=-1, vmax=1, aspect='auto')
    im3 = ax3.imshow(pred,extent=extent,   vmin=-1, vmax=1, aspect='auto')
    im4 = ax4.imshow(inp[3],extent=extent,vmin=-1, vmax=1,aspect='auto')
    tmp=matrix; tmp_extent = np.array([0,tmp.shape[1]*dx/1000,0,tmp.shape[0]*dy / 1000])
    im5 = ax5.imshow(matrix, extent=tmp_extent,vmin=orig_min,vmax=orig_max,aspect='auto')
    tmp=true_model
    tmp_extent = np.array([0,tmp.shape[1]*dx/1000,0,tmp.shape[0]*dy / 1000])
    im6 = ax6.imshow(true_model, extent=tmp_extent,vmin=orig_min,vmax=orig_max,aspect='auto')
    x0=0.11;y0=-0.25
    if Plot_vertical_lines == 1:
        Boundaries2 = Boundaries * dx / 1000
        for i in range(np.shape(Boundaries)[0]):
            ax1.axvline(Boundaries2[i], color='k', linestyle='--', linewidth=2.5)
            ax2.axvline(Boundaries2[i], color='k', linestyle='--', linewidth=2.5)
            ax3.axvline(Boundaries2[i], color='k', linestyle='--', linewidth=2.5)
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

    ax1.set_title('Input'+Title2)
    ax2.set_title('Target')
    ax3.set_title(Title3)
    ax4.set_title('difference (prediction,target)')
    ax5.set_title('Predicted initial model for fwi, R2(predicted initial,true)='+R2val)
    ax6.set_title('True model') # ('Ideal initial model')

    cbar1 = plt.colorbar(im1, cax=cax1)
    cbar2 = plt.colorbar(im2, cax=cax2)
    cbar3 = plt.colorbar(im3, cax=cax3)
    cbar4 = plt.colorbar(im4, cax=cax4)
    cbar5 = plt.colorbar(im5, cax=cax5)
    cbar6 = plt.colorbar(im6, cax=cax6)
    x0 = -50;
    y0 = 1.16
    cbar1.set_label('V (m/sec)', labelpad=x0, y=y0, rotation=0)
    cbar2.set_label('V (m/sec)', labelpad=x0, y=y0, rotation=0)
    cbar3.set_label('V (m/sec)', labelpad=x0, y=y0, rotation=0)
    cbar4.set_label('V (m/sec)', labelpad=x0, y=y0, rotation=0)
    cbar5.set_label('V (m/sec)', labelpad=x0, y=y0, rotation=0)
    cbar6.set_label('V (m/sec)', labelpad=x0, y=y0, rotation=0)
    plt.subplots_adjust(bottom=0.08, top=0.95, hspace=0.57, right=0.86, wspace=.1)
    if Save_flag == 1:
        plt.savefig(save_file_path,dpi=400)
        print('Saving ML_result to '+save_file_path)
    if Show_flag == 1:
        plt.show()
    else:
        plt.show(block=False)
    plt.close()
    return None

def PLOT_ML_Result3(inp, ind_train, ind_test, Train_on1_model_and_test_on_other=0, Train_models=None, Test_models=None,
                    Valid_models=None, Nl=0, history_flag=0, history=None, Boundaries=0, Name='', dx=1, dy=1,
                    Plot_vertical_lines=1, Title='',
                    Save_flag=0, Show_flag=0, COEFF=0.15, clim=[-1, 1]):
    if Train_on1_model_and_test_on_other == 1:
        Test_models = np.arange(0, 1);
        Train_models = np.arange(0, 1)
    # %%
    z_label_name = 'z, m'
    Nm = inp.shape[0];
    Nx = inp.shape[1];
    Nz = inp.shape[2]
    MIN1 = -0.5;
    MAX1 = 0.5
    true = inp[3];
    pred = inp[4];
    true_smooth = inp[5]
    Title4 = 'Predicted, ' + Title
    Title5 = '4 Hz smoothed'
    inp = inp.swapaxes(1, 2)
    x = np.arange(Nx) * dx / 1000
    y = np.arange(Nz) * dy / 1000
    extent = np.array([x.min(), x.max(), y.min(), y.max()])
    # matplotlib.rcParams.update({'font.size': 15})
    # % Calculate relative error
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 10.4
    fig_size[1] = 8.0
    plt.rcParams["figure.figsize"] = fig_size
    plt.rcParams["font.size"] = 10
    fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(nrows=Nm, ncols=1)
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
    dat = inp[0];
    MIN1 = np.min(dat);
    MAX1 = np.max(dat)
    im1 = ax1.imshow(inp[0], extent=extent, vmin=MIN1, vmax=MAX1, aspect='auto')
    im2 = ax2.imshow(inp[1], extent=extent, vmin=MIN1, vmax=MAX1, aspect='auto')
    im3 = ax3.imshow(inp[2], extent=extent, vmin=MIN1, vmax=MAX1, aspect='auto')
    im4 = ax4.imshow(inp[3], extent=extent, vmin=MIN1, vmax=MAX1, aspect='auto')

    im5 = ax5.imshow(inp[4], extent=extent, aspect='auto')
    im6 = ax6.imshow(inp[5], extent=extent, aspect='auto')
    x0 = 0.11;
    y0 = -0.25
    if history_flag == 1:
        ax4.plot(history['loss'], linewidth=2)
        ax4.plot(history['val_loss'], linewidth=2)
        ax4.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax4.autoscale(enable=True, axis='y', tight=True)
        ax4.autoscale(enable=True, axis='x', tight=True)
        bottom, top = ax4.get_ylim()
        # ax4.set_ylim(top=1.3*history['val_loss'][1])
        ax4.legend(['Training', 'Validation'], loc='upper right')
    if Plot_vertical_lines == 1:
        Boundaries2 = Boundaries * dx / 1000
        for i in range(np.shape(Boundaries)[0]):
            ax1.axvline(Boundaries2[i], color='k', linestyle='--', linewidth=2.5)
            ax2.axvline(Boundaries2[i], color='k', linestyle='--', linewidth=2.5)
            ax3.axvline(Boundaries2[i], color='k', linestyle='--', linewidth=2.5)
            ax4.axvline(Boundaries2[i], color='k', linestyle='--', linewidth=2.5)
            ax5.axvline(Boundaries2[i], color='k', linestyle='--', linewidth=2.5)
            ax6.axvline(Boundaries2[i], color='k', linestyle='--', linewidth=2.5)
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
    # ax4.set_ylabel('Loss function')
    # ax4.set_xlabel('Epoch')
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
    # ax4.xaxis.set_label_coords(x0 + 0.03, y0 - 0.045)
    # Plot_image(Input[i,:,:,0].T, Show_flag=1, dx=dx, dy=dz,Title='high_wavenumber_update_1_iter_input_cnn' + str(i),
    #     Save_flag=1, Aspect='equal',Save_pictures_path=Save_pictures_path,c_lim=[-10,10])
    # Plot_image(Output[i, :, :, 0].T, Show_flag=1, dx=dx, dy=dz, Title='low_wavenumber_true_update_output_cnn'+str(i),
    #     Save_flag=1, Aspect='equal', Save_pictures_path=Save_pictures_path)
    # Plot_image(low_wavenumber,Show_flag=1, dx=dx, dy=dz,c_lim=[a,b],
    #     Title='low_wavenumber_true_model_' + str(i), Save_flag=1,Aspect='equal', Save_pictures_path=Save_pictures_path)
    # Plot_image(high_wavenumber, Show_flag=1, dx=dx, dy=dz,c_lim=[a,b],
    #     Title='high_wavenumber_model_' + str(i), Save_flag=1, Aspect='equal', Save_pictures_path=Save_pictures_path)
    # Plot_image(high_wavenumber-true.T,Show_flag=1, dx=dx, dy=dz, c_lim=[a, b],
    #     Title='high_wavenumber_model_' + str(i), Save_flag=1, Aspect='equal', Save_pictures_path=Save_pi
    ax1.set_title('true_model' + Title)
    ax2.set_title('background_model')
    ax3.set_title('CNN_INPUT, 30th iteration of FWI, starting from background model')
    ax4.set_title('CNN_OUTPUT, smoothed_true_model')
    ax5.set_title('low_wavenumbers=smoothed_true_model-background_model')
    ax6.set_title('high_wavenumbers=true_model-smoothed_true_model')
    cbar1 = plt.colorbar(im1, cax=cax1)
    cbar2 = plt.colorbar(im2, cax=cax2)
    cbar3 = plt.colorbar(im3, cax=cax3)
    cbar4 = plt.colorbar(im4, cax=cax4)
    cbar5 = plt.colorbar(im5, cax=cax5)
    cbar6 = plt.colorbar(im6, cax=cax6)
    x0 = -50;
    y0 = 1.16
    cbar1.set_label('V (m/sec)', labelpad=x0, y=y0, rotation=0)
    cbar2.set_label('V (m/sec)', labelpad=x0, y=y0, rotation=0)
    cbar3.set_label('V (m/sec)', labelpad=x0, y=y0, rotation=0)
    cbar4.set_label('V (m/sec)', labelpad=x0, y=y0, rotation=0)
    cbar5.set_label('V (m/sec)', labelpad=x0, y=y0, rotation=0)
    cbar6.set_label('V (m/sec)', labelpad=x0, y=y0, rotation=0)
    plt.subplots_adjust(bottom=0.08, top=0.95, hspace=0.57, right=0.86, wspace=.1)
    if Save_flag == 1:
        plt.savefig(Name, dpi=400)
        print('Saving ML_result to ' + Name)
    if Show_flag == 1:
        plt.show()
    else:
        plt.show(block=False)
    plt.close()
    return None

def PLOT_ML_Result_err_analyze(inp,R2val,history_flag=0, history=None,
                    Boundaries=0, save_file_path='', dx=1, dy=1, Plot_vertical_lines=0, Title='',
                    Title2='', Save_flag=0, Show_flag=0):
    Nm = len(inp);
    dim1 = inp[0].shape[0]
    dim2 = inp[0].shape[1]
    if dim1 >= dim2:
        Nx = dim1
        Nz = dim2
        for i in range(Nm):
            inp[i] = inp[i].swapaxes(0,1)
    else:
        Nx = dim2
        Nz = dim1
    #   flip z axis
    for i in range(Nm):
        inp[i] = np.flip(inp[i], axis=0)
    input = inp[0]
    output = inp[1]
    pred = inp[2]
    matrix = inp[3]
    true_model=inp[4]

    Title3 = Title
    # x = np.arange(Nx) * dx / 1000
    # y = np.arange(Nz) * dy / 1000
    x = np.arange(Nx)
    y = np.arange(Nz)
    extent = np.array([x.min(), x.max(), y.min(), y.max()])
    # matplotlib.rcParams.update({'font.size': 15})

    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 10.4
    fig_size[1] = 8.0
    plt.rcParams["figure.figsize"] = fig_size
    plt.rcParams["font.size"] = 10
    fig, (ax1, ax2, ax3,ax5, ax6) = plt.subplots(nrows=5, ncols=1)
    #  allocate space for colorbar smartly
    divider1 = make_axes_locatable((ax1))
    divider2 = make_axes_locatable((ax2))
    divider3 = make_axes_locatable((ax3))
    divider5 = make_axes_locatable((ax5))
    divider6 = make_axes_locatable((ax6))

    cax1 = divider1.append_axes("right", size="2%", pad=0.05)
    cax2 = divider2.append_axes("right", size="2%", pad=0.05)
    cax3 = divider3.append_axes("right", size="2%", pad=0.05)
    cax5 = divider5.append_axes("right", size="2%", pad=0.05)
    cax6 = divider6.append_axes("right", size="2%", pad=0.05)
    plt.set_cmap('RdBu_r')
    orig_min=np.min(true_model)
    orig_max=np.max(true_model)

    MIN =np.min(inp[2])
    MAX = np.max(inp[2])
    # im1 = ax1.imshow(inp[0], extent=extent, aspect='auto')
    # im2 = ax2.imshow(inp[1], extent=extent, aspect='auto')
    # im3 = ax3.imshow(inp[2], extent=extent,vmin=MIN,vmax=MAX, aspect='auto')
    # im5 = ax5.imshow(inp[3], extent=extent,vmin=MIN,vmax=MAX,aspect='auto')
    # im6 = ax6.imshow(inp[4], extent=extent,aspect='auto')

    im1 = ax1.imshow(inp[0], aspect='auto')
    im2 = ax2.imshow(inp[1], aspect='auto')
    im3 = ax3.imshow(inp[2],    vmin=MIN,vmax=MAX, aspect='auto')
    im5 = ax5.imshow(inp[3],    vmin=MIN,vmax=MAX,aspect='auto')
    im6 = ax6.imshow(inp[4],vmin=-500,vmax=500, aspect='auto')

    x0 = 0.11
    y0 = -0.25
    if Plot_vertical_lines == 1:
        Boundaries2 = Boundaries * dx / 1000
        for i in range(np.shape(Boundaries)[0]):
            ax1.axvline(Boundaries2[i], color='k', linestyle='--', linewidth=2.5)
            ax2.axvline(Boundaries2[i], color='k', linestyle='--', linewidth=2.5)
            ax3.axvline(Boundaries2[i], color='k', linestyle='--', linewidth=2.5)
    z_label_name = 'z, m'
    ax1.set_ylabel(z_label_name)
    ax1.set_xlabel('x (km)')
    ax2.set_ylabel(z_label_name)
    ax2.set_xlabel('x (km)')
    ax3.set_ylabel(z_label_name)
    ax3.set_xlabel('x (km)')
    ax5.set_ylabel(z_label_name)
    ax5.set_xlabel('x (km)')
    ax6.set_ylabel(z_label_name)
    ax6.set_xlabel('x (km)')

    ax1.invert_yaxis()
    ax2.invert_yaxis()
    ax3.invert_yaxis()
    ax5.invert_yaxis()
    ax6.invert_yaxis()

    ax1.xaxis.set_label_coords(x0, y0)
    ax2.xaxis.set_label_coords(x0, y0)
    ax3.xaxis.set_label_coords(x0, y0)
    ax5.xaxis.set_label_coords(x0, y0)
    ax6.xaxis.set_label_coords(x0, y0)

    ax1.set_title(Title[0])
    ax2.set_title(Title[1])
    ax3.set_title(Title[2])
    ax5.set_title(Title[3])
    ax6.set_title(Title[4])

    cbar1 = plt.colorbar(im1, cax=cax1)
    cbar2 = plt.colorbar(im2, cax=cax2)
    cbar3 = plt.colorbar(im3, cax=cax3)
    cbar5 = plt.colorbar(im5, cax=cax5)
    cbar6 = plt.colorbar(im6, cax=cax6)
    x0 = -50;
    y0 = 1.16
    cbar1.set_label('V (m/sec)', labelpad=x0, y=y0, rotation=0)
    cbar2.set_label('V (m/sec)', labelpad=x0, y=y0, rotation=0)
    cbar3.set_label('V (m/sec)', labelpad=x0, y=y0, rotation=0)
    cbar5.set_label('V (m/sec)', labelpad=x0, y=y0, rotation=0)
    cbar6.set_label('V (m/sec)', labelpad=x0, y=y0, rotation=0)
    plt.subplots_adjust(bottom=0.08, top=0.95, hspace=0.57, right=0.86, wspace=.1)
    if Save_flag == 1:
        plt.savefig(save_file_path,dpi=400)
        print('Saving ML_result to '+save_file_path)
    if Show_flag == 1:
        plt.show()
    else:
        plt.show(block=False)
    plt.close()
    return None

def cmd(command):
    """Run command and pipe what you would see in terminal into the output cell"""
    process = subprocess.Popen(command, stderr=subprocess.PIPE, stdout=subprocess.PIPE, shell=True)
    while True:
        output = process.stderr.readline().decode('utf-8')
        if output == '' and process.poll() is not None:
            # this prints the stdout in the end
            output2 = process.stdout.read().decode('utf-8')
            print(output2.strip())
            break
        if output:
            print(output.strip())
    rc = process.poll()
    return rc

##################################################################
# VELOCITY generator 
##################################################################
####  elastic transform for generator 1

def F_smooth_initial(Models,log_idx,dz,water_sz):
    n1 = Models.shape[0];n2 = Models.shape[1]
    if n1<n2:
        nx=n2
        nz=n1
    else:
        nx=n1
        nz=n2
        Models=Models.T
    true_log=Models[:,log_idx]
    smoothed_log=np.copy(true_log)
    smoothed_log=scipy.ndimage.filters.gaussian_filter1d(smoothed_log,15)
    
    final_log=np.copy(true_log)
    final_log[(water_sz):]=smoothed_log[(water_sz):]
    final_log[0:water_sz]=1500
    init=np.tile(final_log, (nx,1))
    if n1<n2:
        init=init.T
    return init

def F_smooth_initial2(Models,log_idx,dz,water_sz):
    initial_model_lin_grad=F_initial_vz_model_custom(Models,dz,water_sz)
    initial_model_lin_grad=initial_model_lin_grad.T
    n1 = Models.shape[0];n2 = Models.shape[1]
    if n1<n2:
        nx=n2
        nz=n1
    else:
        nx=n1
        nz=n2
        Models=Models.T
    true_log_original=Models[:,log_idx]
    Models_smoothed=scipy.ndimage.filters.gaussian_filter1d(Models,1000,axis=1)
    true_log_smoothed=Models_smoothed[:,log_idx]

    smoothed_log=np.copy(true_log_original)
    smoothed_log=scipy.ndimage.filters.gaussian_filter1d(smoothed_log,15)
    final_log=np.copy(true_log_original)
    final_log[(water_sz):]=smoothed_log[(water_sz):]
    final_log[0:water_sz]=1500
    init=np.tile(final_log, (nx,1))
    init=init.T

    smoothed_log2=np.copy(true_log_smoothed)
    smoothed_log2=scipy.ndimage.filters.gaussian_filter1d(smoothed_log2,6)
    final_log2=np.copy(true_log_smoothed)
    final_log2[(water_sz):]=smoothed_log2[(water_sz):]
    final_log2[0:water_sz]=1500
    init2=np.tile(final_log2, (nx,1))
    init2=init2.T
    return init2.T


def elastic_transform_1(image, alpha_x, alpha_z, sigma_x, sigma_z, seed=None, mode='mirror'):
    """Elastic deformation of images as described in Simard, Steinkraus and Platt, "Best Practices for Convolutional Neural Networks applied to Visual Document Analysis", in Proc. of the International Conference on Document Analysis and Recognition, 2003.
    .. Vladimir Kazei, 2019; Oleg Ovcharenko, 2019
        mode : str, optional
        Points outside the boundaries of the input are filled according
        to the given mode ('constant', 'nearest', 'reflect', 'mirror' or 'wrap').
        Default is 'mirror'.
    cval : scalar, optional
        Value used for points outside the boundaries of the input if
        ``mode='constant'``. Default is 0.0
    """
    image = to_3D(image)
    if seed:
        random_state_number = int(seed)
    else:
        random_state_number = np.random.randint(1, 1000)
    geo_before = GeologyScaler(image)
    random_state = np.random.RandomState(random_state_number)
    shape = image.shape
    # with our velocities dx is vertical shift
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), (sigma_x, sigma_z, 1), mode="constant", cval=0) * alpha_x
    # with our velocities dy is horizontal shift
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), (sigma_x, sigma_z, 1), mode="constant", cval=0) * alpha_z
    dz = np.zeros_like(dx)

    dx[..., 1] = dx[..., 0]
    dx[..., 2] = dx[..., 0]
    dy[..., 1] = dy[..., 0]
    dy[..., 2] = dy[..., 0]

    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)), np.reshape(z, (-1, 1))

    distorted_image = map_coordinates(image, indices, order=1, mode=mode, prefilter=False)
    distorted_image = distorted_image.reshape(image.shape)

    geo = GeologyScaler(distorted_image)
    distorted_image = geo.g2r(distorted_image)
    distorted_image = geo_before.r2g(distorted_image)
    return distorted_image
def F_initial_vz_model(Models, dz):
    nx = Models.shape[0];nz = Models.shape[1]
    zz = np.arange(nz) * dz
    zz = zz - 340
    zz = np.tile(zz, (nx, 1))
    zz[zz < 0] = 0
    init = 1.5 + 0.9e-3 * zz
    return init
def F_initial_vz_model_custom(Models,dz,water_sz,vel_grad=0.9):
    nx = Models.shape[0];nz = Models.shape[1]
    n1 = Models.shape[0];n2 = Models.shape[1]
    if n1<n2:
        nx=n2;  nz=n1
    else:
        nx=n1;  nz=n2
    zz = np.arange(nz-water_sz) * dz
    zz = np.tile(zz, (nx, 1))
    init = 1500 + vel_grad* zz
    water=np.ones((nx,water_sz))*1500
    init=np.concatenate([water,init],axis=1)
    return init
def F_initial_overthrust(Models,dz,water_sz,vel_grad=0.9):
    nx = Models.shape[0];nz = Models.shape[1]
    n1 = Models.shape[0];n2 = Models.shape[1]
    if n1<n2:   nx=n2;  nz=n1
    else:   nx=n1;  nz=n2
    ######################
    log_idx=100
    fig, ax = plt.subplots(1,1)
    ax_depth2=np.arange(nz)*dz/1000
    zz = np.arange(nz-water_sz) * dz
    init = 1800 + 0.6* zz
    water=np.ones((water_sz))*1500
    init=np.concatenate([water,init])
    ax.plot(ax_depth2, Models[log_idx,:] / 1000,label='true1')
    ax.plot(ax_depth2, Models[log_idx+200,:] / 1000,label='true2')
    ax.plot(ax_depth2, Models[log_idx+400,:] / 1000,label='true3')
    ax.plot(ax_depth2, Models[log_idx+550,:] / 1000,label='true4')
    ax.plot(ax_depth2, init/ 1000,label='init')
    ax.set_ylabel('Velocity, km/s')
    ax.set_xlabel('Depth, km')
    ax.set_title('Initial models construction by Pavel')
    ax.grid(True)
    ax.legend()
    save_name='./pictures_for_check/init_model_logs_x_'+str(log_idx*dz)+'m.png'
    print('save to ',save_name)
    fig.savefig(save_name)
    ######################
    init = np.tile(init, (nx, 1))
    return init
def F_initial_random_linear_trend(Models,dz,water_sz):
    nx = Models.shape[0];nz = Models.shape[1]
    n1 = Models.shape[0];n2 = Models.shape[1]
    if n1<n2:
        nx=n2
        nz=n1
    else:
        nx=n1
        nz=n2
    zz = np.arange(nz-water_sz) * dz
    zz = np.tile(zz, (nx, 1))
    c1=0.33
    c2=(c1+(0.8-c1)*np.random.rand())
    init=1500+c2*zz
    water=np.ones((nx,water_sz))*1500
    init=np.concatenate([water,init],axis=1)
    return init
def calculate_water_taper(generated_model,min_water_velocity=1490.001):
    n1,n2=generated_model.shape
    if n1<n2:
        generated_model=generated_model.T
    ind=np.where(generated_model<=min_water_velocity)
    if ind[0].size==0:
        min_water_velocity=1500.000000001     #1506
        ind=np.where(generated_model<=min_water_velocity)
    taper=np.zeros_like(generated_model)
    taper[ind[0],ind[1]]=1
    water_taper=np.copy(taper)
    taper_shifted=np.roll(taper,1,axis=1)
    taper_diff=taper_shifted-taper
    taper_diff[taper_diff<0]=0
    water_boundary_=[]
    for ii in range(taper_diff.shape[0]):
        aa=np.where(taper_diff[ii,:]==1)
        water_boundary=aa[0][0]
        water_boundary_.append(water_boundary)
    ########### smooth water boundary
    water_boundary_1=np.array(water_boundary_)
    curve3=water_boundary_1
    ###########
    for ii in range(taper_diff.shape[0]):
        water_taper[ii,int(curve3[ii]):]=0
    ###########
    if n1<n2:
        water_taper=water_taper.T
    return water_taper
class GeologyScaler(object):
    def __init__(self, img, lim=[0, 1]):
        if len(img.shape) == 3:
            # print('Apply GeologyScaler to 3 channel data')
            vmax = np.array([img[:, :, i].max() for i in range(3)])
            vmin = np.array([img[:, :, i].min() for i in range(3)])
            self.vmax = vmax.reshape(1, 1, 3)
            self.vmin = vmin.reshape(1, 1, 3)
        else:
            self.vmax = img.max()
            self.vmin = img.min()
        self.lmin, self.lmax = lim
    # From geological scale to reference scale
    def g2r(self, img):
        return (img - self.vmin) * (self.lmax - self.lmin) / (self.vmax - self.vmin) + self.lmin
    # From reference scale to geological scale
    def r2g(self, img):
        return (img - self.lmin) * (self.vmax - self.vmin) / (self.lmax - self.lmin) + self.vmin
def get_reflectivity(h,k=0.85):
    r = -1 + 2 * np.random.random(h)
    mask = np.random.random(h)
    mask[mask < k] = 0
    return  np.multiply(0.25 * r, mask)
def ref2vel(r, v0):
    # Assuming rho=1, then R = (v2 - v1) / (v2 + v1)
    vel = np.zeros(len(r) + 1)
    vel[0] = v0
    for i in range(len(vel) - 1):
        vel[i+1] = vel[i] * (r[i] + 1) / (1 - r[i])
    return vel[:-1]
def get_trend(h, v0, v1):
    idx = np.arange(h)
    return v0 + idx * (v1 - v0) / h
def get_2d_layered_model(h, w, vmin_trend=1, vmax_trend=1, dv0=1, vmin=0., vmax=1.):#, alpha_x, alpha_z, beta_x, beta_z):
    """
    Args:
        more_layers (float): 0..1, makes layers finer
    """
    r = get_reflectivity(h)
    vel = ref2vel(r, dv0)
    trend = get_trend(h, vmin_trend, vmax_trend)
    v = trend + vel
    vrand = np.repeat(np.expand_dims(v,1), w, 1) / np.max(v)
    return vrand
def to_3D(img):
    if len(img.shape) < 3:
        img = np.expand_dims(img, -1)
        return np.concatenate((img, img, img), axis=2)
    else:
        return img
def velocity_generator(gen_mode,model_name='model__Marmousi',filename='1.hdf5',dh=20,out_shape=[1200,168],water_height=400,floor_height=500,flag_record=0,program_root='./'):
    root_models=os.path.join('./data_generation','original_models')
    if gen_mode=='generator1': 
        """ 
        create Vp velocity models with Oleg Ovcharenko's approach (curved layered velocity models generator) from
        https://www.earthdoc.org/content/papers/10.3997/2214-4609.202112949 """
        flag_plotting=0
        ############    generate velocity prototype model
        dx_new=dh;  dz_new=dh
        water_thickness=int(water_height/dz_new)
        nx,nz=out_shape
        model = get_2d_layered_model(nz,nx)
        ############    we need to separate first layer of the model, from other layers properly, in order to calculate water with no errors
        first_layer_val=np.min(model)-0.1*np.min(model)
        ############    
        model[:int( water_thickness + 3*np.random.rand() ), :]=first_layer_val    #   0.05*np.random.rand()*nz
        alpha_x = 1e3 + 1e3 * np.random.rand()
        sigma_x = 20 + 80 * np.random.rand()
        sigma_z = 50 + 30 * np.random.rand()
        alpha_z = 1*1e4 + 5*1e3 * np.random.rand()      #   my parameters 1
        ############    elastic transform
        model_ = elastic_transform_1(model, alpha_x, alpha_z, sigma_x, sigma_z, mode='nearest')
        model_ = model_[..., 0]
        ############    flipud model
        orig_coef=np.copy(model_)
        ############    geology scaler
        model_3=GeologyScaler(orig_coef,[1500, 3000 + 2200 * np.random.rand()]).g2r(orig_coef)
        model_3[:int( water_thickness  ), :]=1500
        ############    
        vel=model_3.T
        generated_model1=vel
        original_water_taper=calculate_water_taper(generated_model1)
        ############    construct initial model with different strategies. We use here random linear trends with different slopes.
        initial_model=F_initial_random_linear_trend(generated_model1,dz_new,water_thickness)
        #initial_model=F_initial_vz_model_custom(generated_model1,dz_new,water_thickness)
        #initial_model=F_smooth_initial(generated_model1,log_idx,dz_new,int(water_height/dz_new))       
        #initial_model2=F_smooth_initial2(generated_model1,log_idx,dz_new,int(water_height/dz_new))
        ############    delete mean
        MEAN=np.repeat(np.expand_dims(np.mean(generated_model1,axis=0), 1), nx, axis=-1)
        MEAN=MEAN.T
        perturbations=generated_model1-MEAN
        generated_model2=initial_model+perturbations
        tmp=np.copy(generated_model2)
        ############    lower box condition
        zz = np.arange(nz)*dz_new
        zz = zz - water_height      # water_height/dz_new
        zz = np.tile(zz, (nx, 1))
        zz[zz < 0] = 0
        box_c_low = 1500+0.15*zz
        tmp=np.where(tmp>box_c_low,tmp,box_c_low)
        ############
        generated_model3=tmp  
        generated_model3[original_water_taper==1]=1500   
        initial_model[original_water_taper==1]=1500
        ############    upper box condition
        box_max=4700
        generated_model3=np.where(generated_model3<box_max,generated_model3,box_max)
        if flag_plotting==1:
            Plot_image(MEAN.T,Show_flag=0,Save_flag=1,Title=model_name+'_MEAN_',Save_pictures_path='./pictures_for_check',c_lim=[1500,4500])
            Plot_image(perturbations.T,Show_flag=0,Save_flag=1,Title=model_name+'_perturbations',Save_pictures_path='./pictures_for_check')
            Plot_image(generated_model1.T,Show_flag=0,Save_flag=1,Title=model_name+'_generated_model1_',Save_pictures_path='./pictures_for_check',c_lim=[1500,4500])
            Plot_image(generated_model2.T,Show_flag=0,Save_flag=1,Title=model_name+'_generated_model2_',Save_pictures_path='./pictures_for_check',c_lim=[1500,4500])
            Plot_image(generated_model3.T,Show_flag=0,Save_flag=1,Title=model_name+'_generated_model3_',Save_pictures_path='./pictures_for_check',c_lim=[1500,4500])
            Plot_image(initial_model.T,Show_flag=0,Save_flag=1,Title=model_name+'_initial_model',Save_pictures_path='./pictures_for_check',c_lim=[1500,4500])
            Plot_image(original_water_taper.T,Show_flag=0,Save_flag=1,Title=model_name+'_original_water_taper',Save_pictures_path='./pictures_for_check')
        ############
        generated_model=generated_model3
    elif gen_mode=='test':
        """create one of these models: 'Marmousi','Overthrust','Seam','Seam2' and its variants"""
        dx_new=dh;dz_new=dh
        ######  load data
        if 'Marmousi' in model_name:
            vel = np.loadtxt(os.path.join(root_models,'marm2_10.dat'),delimiter=',').T*1000
            dx = 10;     dz = dx
            ss=np.where(vel==1500)
            water_sz=np.max(ss[1])+1
            vel=vel[:,water_sz:]
        elif 'Overthrust' in model_name:
            name=os.path.join(root_models,'overthrust_test_2D_2.hdf5')
            dz=load_file(name,'d1')
            dx=load_file(name,'d2')
            vel=load_file(name,'vp')
            ######################
            name=os.path.join(root_models,'overthrust_data.npz')
            with open(name,'rb') as f:
                data = np.load(f, allow_pickle=True)
                vel = data['vel']
                dz = data['dz']
                dx = data['dx']
                data.close()
        elif 'model__Seam'==model_name:
            name=os.path.join(root_models,'seam_i_sediments.hdf5')
            dz=load_file(name,'d1')
            dx=load_file(name,'d2')
            vel=load_file(name,'vp')
        elif model_name=='model__Seam2' or model=='model__Seam2_full':
            name=os.path.join(root_models,'vpb2d_.hdf5')
            dz=load_file(name,'d1')
            dx=load_file(name,'d2')
            vel=load_file(name,'vp')
            water_sz=(np.where(vel[0,:]==1500))[0].size;
            vel=vel[:,water_sz:]
        ######  resize data
        vel=F_resize(vel,dx=dx,dz=dz,dx_new=dx_new,dz_new=dz_new)
        ######  crop data
        nx,nz=vel.shape;    print(vel.shape)
        ############    add water and ocean floor from above
        water=np.ones((nx,int(water_height/dz_new)))*1500
        generated_model=np.concatenate([water,vel],axis=1)
        ############    construct initial model
        Plot_image(generated_model.T,Show_flag=0,Save_flag=1,Title='generator_result_'+model_name,Save_pictures_path='./pictures_for_check',dx=dx_new,dy=dz_new,c_lim=[1500,4500])
        if 'Marmousi' in model_name:
            log_idx=int(7500/dx_new)
        elif 'Overthrust' in model_name:
            log_idx=int(2500/dx_new)
        elif 'Seam' in model_name:
            log_idx=int(7500/dx_new)
        elif model_name=='model__Seam2' or model=='model__Seam2_full':
            log_idx=int(7500/dx_new)
        ####################    choose initial model
        if model_name=='model__Marmousi':
            initial_model=F_smooth_initial2(generated_model,log_idx,dz_new,int(water_height/dz_new))   
        elif model_name=='model__Marmousi_linear_initial':
            initial_model=F_smooth_initial(generated_model,log_idx,dz_new,int(water_height/dz_new))
        elif 'model__Marmousi_1d_lin' in model_name:
            initial_model=F_initial_vz_model_custom(generated_model,dz_new,int(water_height/dz_new),vel_grad=0.78)
        elif model_name=='model__Overthrust':
            initial_model=F_smooth_initial2(generated_model,log_idx,dz_new,int(water_height/dz_new))   
        elif model_name=='model__Overthrust_linear_initial':
            initial_model=F_smooth_initial(generated_model,log_idx,dz_new,int(water_height/dz_new))
        elif 'model__Overthrust_1d_lin' in model_name:
            initial_model=F_initial_overthrust(generated_model,dz_new,int(water_height/dz_new),vel_grad=0.7)
        else:
            tmp=F_smooth_initial2(generated_model,log_idx,dz_new,int(water_height/dz_new)) #Marmousi
            initial_model=np.copy( tmp )   
        ############    create water taper
        original_water_taper=np.ones_like(generated_model)
        original_water_taper [:,0:int(water_height/dz_new)]=0
    elif gen_mode=='test_real_data':
        with open(os.path.join('./for_pasha','acq_data_parameters_cgg.pkl'),'rb') as input:
            acq_data=pickle.load(input)
        print(acq_data)
        if model_name=='model__cgg_tomo_long1':
            initial_model,original_water_taper=acq_data[model_name.split('model__')[-1] ]
            generated_model=initial_model
        elif model_name=='model__cgg_tomo_long2':
            initial_model,original_water_taper=acq_data[model_name.split('model__')[-1] ]
            generated_model=initial_model
        elif model_name=='model__cgg_lin_vp_long':
            initial_model,original_water_taper=acq_data[model_name.split('model__')[-1] ]
            generated_model=initial_model
        #########   calculate water taper
        original_water_taper=calculate_water_taper(np.rot90(generated_model,3) )
        #########
        generated_model=np.rot90(generated_model,3)
        initial_model=np.rot90(initial_model,3)
        nx,nz=generated_model.shape
        generated_model=generated_model[0:(nx-320),:]
        initial_model=initial_model[0:(nx-320),:]
        original_water_taper=original_water_taper[0:(nx-320),:]
    return generated_model,initial_model,original_water_taper