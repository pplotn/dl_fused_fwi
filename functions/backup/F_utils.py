# from F_models import *
from functions.F_modules import *
from functions.F_plotting import *
def F_r2(mat, mat_true):
    # r2 = 1 - (np.std(mat_true.flatten() - mat.flatten()) / np.std(mat_true.flatten())) ** 2
    v1 = mat.flatten()
    v2 = mat_true.flatten()
    if np.isnan(v1).any()==True:
        r2_2=math.nan
    else:
        r2_2 = r2_score(v1, v2)
    return r2_2
def RepresentsInt(s):
    try: 
        int(s)
        return True
    except ValueError:
        return False
def plt_nb_T(vel, fname="Velocity", title="",
             ylabel="Depth (km)", xlabel="Distance (km)",
             cbar_label="(km/s)",
             vmin=None, vmax=None,
             split_line=False,
             dx=25, dz=25, no_labels=False, origin_in_middle=False):
    plt.figure(figsize=(16, 9))
    plt.set_cmap('RdBu_r')
    vel_image = vel[:, :].T
    extent = (0, dx * vel.shape[0] * 1e-3, dz * vel.shape[1] * 1e-3, 0)
    if origin_in_middle:
        extent = (-dx * vel.shape[0] * .5e-3, dx * vel.shape[0] * .5e-3, dz * vel.shape[1] * 1e-3, 0)
    plt.imshow(vel_image * 1e-3, origin='upper', extent=extent)
    plt.axis("tight")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.clim(vmin, vmax)
    cbar = plt.colorbar()
    cbar.ax.set_ylabel(cbar_label)
    if split_line:
        plt.axvline(x=extent[1] / 2, color='black', linewidth=10, linestyle='-')

    if no_labels:
        plt.axis('off')
        plt.xlabel()
    plt.savefig(fname, bbox_inches='tight')
    plt.show()
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

def scaling_data_01_backup(data,preconditioning=True,visualize_scaling_results=0,save_pictures_path=''):
    data_processing_history=[];
    n_channels=data.ndim
    data_=np.squeeze(data);  
    Nx=data_.shape[0];  Nz=data_.shape[1]
    data_processing_history.append(data_)
    const1=np.abs(np.min(data_));  data_=data_+const1;  data_processing_history.append(data_)
    const2=np.max(np.abs(data_));  data_=data_/const2;  data_processing_history.append(data_)
    if preconditioning==True:
        coefs=(1,0.05)
        coefs=(1,0.020)
        geom_spreading_matrix=(coefs[0]+coefs[1]*np.repeat(np.expand_dims(np.arange(0,Nz),axis=0),Nx,axis=0))
        data_=data_*geom_spreading_matrix;      data_processing_history.append(data_)
        const3=np.max(np.abs(data_));  data_=data_/const3;  data_processing_history.append(data_)
        scaler=[coefs,const1,const2,const3]
    else:
        scaler=[const1,const2]
    if visualize_scaling_results==1 and save_pictures_path!='':
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
    if data_.ndim==2:
        data_=np.expand_dims(data_,axis=0); data_=np.expand_dims(data_,axis=-1);
    return data_,scaler
def scaling_data_01_back_backup(data,scaler):
    data_=np.squeeze(data);  Nx=data_.shape[0];  Nz=data_.shape[1]
    if len(scaler)==2:
        preconditioning=False
        const1=scaler[0];const2=scaler[1]
    else:
        preconditioning=True
        coefs=scaler[0];    const1=scaler[1];const2=scaler[2];const3=scaler[3]
        geom_spreading_matrix=(coefs[0]+coefs[1]*np.repeat(np.expand_dims(np.arange(0,Nz),axis=0),Nx,axis=0))
        data_=np.squeeze(data_)*const3
        data_=data_/geom_spreading_matrix
    data_=data_*const2
    data_=data_-const1
    data_=np.expand_dims(data_,axis=0); data_=np.expand_dims(data_,axis=-1)
    return data_

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

def F_init_resize_models(Mat, dx=1, dz=1, Reshape_dataset_flag=1, sz_new='None'):
    # resizing according certain size
    # dx,dz - old spacing
    Nl = Mat.shape[0]
    print('Size before imresizing', np.shape(Mat))
    sz_old = np.array([Mat.shape[1], Mat.shape[2]], dtype=int)
    Nch = Mat.shape[-1]
    if Reshape_dataset_flag == 0:
        Mat2 = np.zeros([Nl, sz_old[0], sz_old[1], Nch], dtype=float)
        for i in range(Nl):
            Mat2[i, :, :, :] = Mat[i, :, :, :]
    else:
        if sz_new == 'None':
            sz_new = np.array([sz_old[0], 320], dtype=int)
        F = (sz_old - 1) / (sz_new - 1)
        dx = dx * F[0]
        dz = dz * F[1]
        Mat2 = np.zeros([Nl, sz_new[0], sz_new[1], Nch], dtype=float)
        for i in range(Nl):
            for j in range(Nch):
                Mat2[i, :, :, j] = imresize(Mat[i, :, :, j], sz_new, anti_aliasing=True)
    print('Shape after resizing', np.shape(Mat2))
    return Mat2, dx, dz

def F_init_resize_models2(Mat, dx2=1, dz2=1, dx=1, dz=1):
    # resizing according to spacing
    Nl = Mat.shape[0]
    if len(Mat.shape) == 2:
        Mat = Mat.reshape((1,) + Mat.shape)
    sz_old = np.asarray(np.shape(Mat)[1:3])
    Nch = Mat.shape[-1]
    print('Size before imresizing', np.shape(Mat))
    F = np.asarray([dx2 / dx, dz2 / dz])
    sz_new = np.array([sz_old - 1] / F, dtype=int)
    sz_new2 = [sz_new[0, 0], sz_new[0, 1]]
    Mat2 = np.zeros([Nl, sz_new2[0], sz_new2[1], Nch], dtype=float)
    print('Shape after resizing', np.shape(Mat2))
    for i in range(Nl):
        for j in range(Nch):
            Mat2[i, :, :, j] = imresize(Mat[i, :, :, j], sz_new2)
    return Mat2, dx2, dz2

def F_check_for_stride(patch_sz_x, patch_sz_z, strides_x, strides_z, Nx, Nz):
    tmp_x = Nx - patch_sz_x
    tmp_z = Nz - patch_sz_z
    if (tmp_x % strides_x == 0) and (tmp_z % strides_z == 0):
        print('Size is OK')
        recommended_size = [Nx, Nz]
    else:
        print('Size is NOT OK')
        Nx2 = tmp_x + patch_sz_x - tmp_x % strides_x
        Nz2 = tmp_z + patch_sz_z - tmp_z % strides_z
        recommended_size = [Nx2, Nz2]
    return recommended_size

def F_check_for_patch_sz(patch_sz_x, patch_sz_z, strides_x, strides_z, Nx, Nz):
    print('check_for_patch_sz and stride')
    old_size = [Nx, Nz]
    print('old size', old_size)
    ind_x = mov_window0(np.arange(Nx), patch_sz_x, strides_x)
    ind_z = mov_window0(np.arange(Nz), patch_sz_z, strides_z)
    if None in ind_x[-1]:
        Nx2 = ind_x[ind_x.shape[0] - 2][-1] + 1
        ind_x = np.delete(ind_x, -1, axis=0)
    else:
        Nx2 = Nx
    if None in ind_z[-1]:
        Nz2 = ind_z[ind_z.shape[0] - 2][-1] + 1
        ind_z = np.delete(ind_z, -1, axis=0)
    else:
        Nz2 = Nz
    recommended_size = [Nx2, Nz2]
    print('new size', recommended_size)
    ind_x0 = mov_window0(np.arange(Nx), patch_sz_x, strides_x)
    ind_z0 = mov_window0(np.arange(Nz), patch_sz_z, strides_z)
    ind_x1 = mov_window0(np.arange(Nx2), patch_sz_x, strides_x)
    ind_z1 = mov_window0(np.arange(Nz2), patch_sz_z, strides_z)
    return recommended_size

def F_init_models(Nl):
    # Train_models = np.array([0]);Test_models = np.setdiff1d(np.arange(0, Nl, 1), Train_models)
    # Train_models=np.arange(0,15);     Test_models=np.setdiff1d(np.arange(0, Nl, 1),Train_models)
    Train_models = np.arange(0, 20);
    Test_models = np.setdiff1d(np.arange(0, Nl, 1), Train_models)
    # Train_models = np.arange(6,7);    Test_models = np.setdiff1d(np.arange(0, Nl, 1), Train_models)
    Test_models = np.arange(0, 1);
    Train_models = np.setdiff1d(np.arange(0, Nl, 1), Test_models)
    Train_models = np.arange(1, Nl);
    Test_models = np.setdiff1d(np.arange(0, Nl, 1), Train_models)
    Train_models = np.arange(0, 2);
    Test_models = np.setdiff1d(np.arange(0, Nl, 1), Train_models)
    Train_models = np.arange(0, Nl - 30);
    Test_models = np.setdiff1d(np.arange(0, Nl, 1), Train_models)
    Train_models = np.arange(0, Nl - 10);
    Test_models = np.setdiff1d(np.arange(0, Nl, 1), Train_models)
    # Train_models = np.arange(0,Nl-1);   Test_models = np.setdiff1d(np.arange(0, Nl, 1), Train_models)
    # Train_models = np.arange(1, 3);     Test_models = np.arange(0, 1)
    # Train_models = np.array([1]);     Test_models = np.array([0])
    # Test_models = np.arange(0,2);  Train_models = np.setdiff1d(np.arange(0, Nl, 1), Test_models)
    # Train_models = np.arange(1,20);    Test_models = np.setdiff1d(np.arange(0, Nl, 1), Train_models)
    train_frac = 0.5
    val_frac = 0.4
    test_frac = 0.1
    test_frac / (val_frac + test_frac) * Nl

    Train_models = np.arange(0,Nl * train_frac, dtype=int)
    Valid_models = np.setdiff1d(np.arange(0, Nl, 1), Train_models)
    Test_models = Valid_models[ 0:  int ( test_frac / (val_frac + test_frac) * len(Valid_models)  ) ]
    Valid_models = np.setdiff1d(Valid_models, Test_models)

    len(Train_models) + len(Test_models) + len(Valid_models)
    Test_models = np.array([0]);
    Valid_models = np.array([2])
    Train_models = np.array([3, 4, 5]);

    Train_models = np.array([1])
    Test_models = np.array([1])
    Valid_models = np.array([0])
    return Train_models, Test_models, Valid_models

def F_init_train_test(Input, Patch_sz, Train_on1_model_and_test_on_other):
    # Plot_image(Input[0, :, :].T, Show_flag=1)
    Patch_num = 10
    Train_patches = np.asarray([4, 5, 6]);
    Test_patches = np.setdiff1d(np.arange(0, Patch_num, 1), Train_patches)
    Patch_num = 2
    Train_patches = np.asarray([0, 1]);
    Test_patches = np.setdiff1d(np.arange(0, Patch_num, 1), Train_patches)
    Patch_num = 100
    Train_patches = np.arange(10, 24, 1);
    Test_patches = np.setdiff1d(np.arange(0, Patch_num, 1), Train_patches)
    # Train_patches = np.arange(10,30, 1) #+np.arange(70,85, 1);
    # Test_patches = np.setdiff1d(np.arange(0, Patch_num, 1), Train_patches)
    Nx = Input.shape[1];
    Nz = Input.shape[2]
    ind_patch = list(window2(range(Nx), Patch_sz))
    ########################
    if Train_on1_model_and_test_on_other == 1:
        ind_train = list(range(0, Nx))
        ind_test = list(range(0, 0))
    else:
        [ind_train, ind_test] = F_split(Nx, Patch_num, Test_patches)
    # ind_train = list((np.arange(50, 120, dtype=int)))
    # ind_train=list( (np.arange(0,170, dtype=int)) )
    # ind_train=list( (np.arange(170,341, dtype=int)) )
    ind_train = list((np.arange(0, 180, dtype=int))) + list((np.arange(230, 341, dtype=int)))
    ind_train = list((np.arange(0, 50, dtype=int))) + list((np.arange(250, Nx, dtype=int)))
    ind_train = list((np.arange(250, Nx, dtype=int)))
    ind_train = list((np.arange(0, Nx / 2, dtype=int)))
    # ind_train = list((np.arange(100,250, dtype=int)))
    ind_test = list((np.setxor1d(np.arange(Nx, dtype=int), ind_train)))
    ########################
    # # ind_all=list(np.arange(0,Nx,1))
    # # ind_train= ind_patch[70] + ind_patch[200]+ ind_patch[275]
    # # patch_list=list(range(250,275))+list(range(650,700)) #+list(range(830,836))  #850,180
    # # patch_list = list(range(150, 175)) + list(range(350, 400))  # +list(range(830,836))  #850,180
    # # patch_list = list(range(150, 175)) #+ list(range(350, 400))  # +list(range(830,836))  #850,180
    # # patch_list = [80,170]
    # patch_list = [80,90]
    # ind_train=[]
    # for i in range(len(patch_list)):
    #     ind_train =ind_train+list(ind_patch[patch_list[i]])
    # ind_train= list( np.unique(ind_train) )
    # ind_test=list ( (np.setxor1d( np.arange(Nx,dtype=int),ind_train  )) )
    ########################
    # if Patch_num>Nx:
    #     raise Exception('set Nx of your data bigger than Patch_num')
    if len(ind_train) < len(ind_patch[0]) and Train_on1_model_and_test_on_other == 0:
        raise Exception('set Patch_sz smaller than ind_train')
    myarray = np.array(ind_train)
    lowerBounds = (myarray + 1)[:-1]
    upperBounds = (myarray - 1)[1:]
    mask = lowerBounds <= upperBounds
    upperBounds, lowerBounds = upperBounds[mask], lowerBounds[mask]
    Boundaries = np.concatenate((lowerBounds, upperBounds))
    Boundaries = np.append(Boundaries, ind_train[0])
    Boundaries = np.append(Boundaries, ind_train[-1])
    Boundaries = np.sort(Boundaries)
    print('Train patch Nx size putted into the network', [len(ind_train), Nz])
    return ind_train, ind_test, Boundaries

def F_init_CNN_model():
    Model_to_load_const = 527  # FWI_gradients_Marm_50      fitting all data
    Model_to_load_const = 554  # FWI_gradients_Marm_50      training on right left parts
    Model_to_load_const = 561  # FWI_gradients_3_1Hz_it_best300      training on right left parts
    Model_to_load_const = 90
    Model_to_load_const = 106
    Model_to_load_const = 172
    Model_to_load_const = 229
    Model_to_load_const = 288
    Model_to_load_const = 15
    Model_to_load_const = 9
    Model_to_load_const = 39
    Model_to_load_const = 127
    Model_to_load_const = 128
    Model_to_load_const = 129
    Model_to_load_const = 166
    Model_to_load_const = 67
    Model_to_load_const = 130
    Model_to_load_const = 42
    Model_to_load_const = 205
    Model_to_load_const = 234
    Model_to_load_const = 758
    Model_to_load_const = 817
    return Model_to_load_const

def F_create_folder(folder):
    os.makedirs(folder,exist_ok=True)
    return None

def F_create_dataset(name, Input, Output, dx, dz):
    f = h5py.File(name, 'w')
    f.create_dataset('DV_High2', data=Input)
    f.create_dataset('DV_Low2', data=Output)
    f.create_dataset('dx', data=dx)
    f.create_dataset('dz', data=dz)
    f.close()
    return None

def F_count_files_old_version(path, Word, type='.png'):
    Const = len(fnmatch.filter(os.listdir(path), Word + '*'))
    Name = path + '/' + Word + str(Const) + type
    while os.path.isfile(Name):
        Const = Const + 1
        Name = path + '/' + Word + str(Const) + type
    return Const

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

def save_file2(filename, variable=1, variable_name=1):
    f = h5py.File(filename, 'w')
    f.create_dataset(variable_name, data=variable)
    f.close()
    return None

def add_dim_back(dat):
    dat = np.expand_dims(dat, axis=-1);
    return dat

def add_dim_forth(dat):
    dat = np.expand_dims(dat, axis=0);
    return dat

def F_reduce_zero_dim(mat):
    SZ = np.shape(mat)
    New_SZ = tuple([SZ[0] * SZ[1]]) + SZ[2:]
    mat = np.reshape(mat, New_SZ)
    return mat

def F_reduce_last_dim(mat):
    SZ = np.shape(mat)
    New_SZ = SZ[0: -1: 1]
    mat = np.reshape(mat, New_SZ)
    return mat

def window(a, w=4, o=2, copy=False):
    sh = (a.size - w + 1, w)
    st = a.strides * 2
    view = np.lib.stride_tricks.as_strided(a, strides=st, shape=sh)[0::o]
    if copy:
        return view.copy()
    else:
        return view

def mov_window0(a, width, step):
    N = len(a)
    width = int(width)
    step = int(step)
    ind = (list(more_itertools.windowed(a, n=width, step=step, fillvalue=None)))
    ind = np.asarray(ind)
    return ind

def mov_window(a, width, step):
    N = len(a)
    width = int(width)
    step = int(step)
    ind = (list(more_itertools.windowed(a, n=width, step=step, fillvalue=None)))
    if None in ind[-1]:
        print('Nx or Nz does not divides by patch size and stride')
        ind[-1] = tuple(list(range(N - width, N)))
        ind = np.asarray(ind)
    else:
        ind = np.asarray(ind)
    return ind

def factors(n):
    return np.array(list(set(reduce(list.__add__,
                                    ([i, n // i] for i in range(1, int(n ** 0.5) + 1) if n % i == 0)))))

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

# @njit(parallel=True)
def mov_window2(a, width, step):
    N = len(a)
    width = int(width)
    step = int(step)
    ind = (list(more_itertools.windowed(a, n=width, step=step, fillvalue=None)))
    if None in ind[-1]:
        # if ind[-1][-1]!=a[-1]:
        ind[-1] = [i for i in ind[-1] if i != None]
        ind[-1] = tuple(ind[-1])
    return ind

def slidingWindow2(sequence, winSize, step=1):
    # Verify the inputs
    if not ((type(winSize) == type(0)) and (type(step) == type(0))):
        raise Exception("**ERROR** type(winSize) and type(step) must be int.")
    if step > winSize:
        raise Exception("**ERROR** step must not be larger than winSize.")
    if winSize > len(sequence):
        raise Exception("**ERROR** winSize must not be larger than sequence length.")
    # Pre-compute number of chunks to emit
    numOfChunks = ((len(sequence) - winSize) / step) + 1
    # Do the work
    for i in range(0, numOfChunks * step, step):
        yield sequence[i:i + winSize]

def F_scale_forth(data,scaler,fit_flag=1):
    transformed2 = np.zeros_like(data)
    Nl = data.shape[0];
    Nch = data.shape[-1];
    Nx = data.shape[1];
    Nz = data.shape[2]
    data_orig = np.zeros((Nx, Nl * Nch * Nz))
    it = 0
    for i in range(Nl):
        for j in range(Nch):
            data_orig[:, ((it) * Nz):((it + 1) * Nz)] = data[i, :, :, j]
            it = it + 1
    ascolumns = data_orig.reshape(-1, 1)
    if fit_flag == 1:
        ascolumns2 = scaler.fit_transform(ascolumns)
    else:
        ascolumns2 = scaler.transform(ascolumns)
    transformed = ascolumns2.reshape(data_orig.shape)
    for i in range(Nl):
        for j in range(Nch):
            it = i * Nch + j
            transformed2[i, :, :, j] = transformed[:, ((it) * Nz):((it + 1) * Nz)]
    return transformed2, scaler

def F_scale_back(data, scaler, Scaleback_flag=1):
    transformed2 = np.zeros_like(data)
    Nl = data.shape[0];
    Nch = data.shape[-1];
    Nx = data.shape[1];
    Nz = data.shape[2]
    data_orig = np.zeros((Nx, Nl * Nch * Nz))
    it = 0
    for i in range(Nl):
        for j in range(Nch):
            data_orig[:, ((it) * Nz):((it + 1) * Nz)] = data[i, :, :, j]
            it = it + 1
    ascolumns = data_orig.reshape(-1, 1)
    ascolumns2 = scaler.inverse_transform(ascolumns)
    transformed = ascolumns2.reshape(data_orig.shape)
    for i in range(Nl):
        for j in range(Nch):
            it = i * Nch + j
            transformed2[i, :, :, j] = transformed[:, ((it) * Nz):((it + 1) * Nz)]
    return transformed2

def extract_patches(x, A, B, stridesA, stridesB):
    # https: // stackoverflow.com / questions / 44047753 / reconstructing - an - image - after - using - extract - image - patches
    a = (tf.__version__)
    ind = a.split('.')
    ver = float(ind[0] + '.' + ind[1])
    if ver == 1.12:
        return tf.extract_image_patches(
            # return tf.image.extract_patches(
            x, 
            (1, A, B, 1), 
            (1, stridesA, stridesB, 1),
            (1, 1, 1, 1),
            padding="VALID")
            # padding = "SAME")
    else:
        # return tf.compat.v1.extract_image_patches(
        return tf.image.extract_patches(
            x, 
            (1, A, B, 1), 
            (1, stridesA, stridesB, 1),
            (1, 1, 1, 1), 
            padding="VALID")
            # padding="SAME")
        # ss=tf.image.extract_patches(
        #     x, 
        #     (1, A, B, 1), 
        #     (1, stridesA, stridesB, 1),
        #     (1, 1, 1, 1), 
        #     # padding="VALID")
        #     padding="SAME")
        # ss2=tf.image.extract_patches(
        #     x, 
        #     (1, A, B, 1), 
        #     (1, stridesA, stridesB, 1),
        #     (1, 1, 1, 1), 
        #     padding="VALID")
        #     # padding="SAME")

# @njit(parallel=True)
def extract_patches_inverse(x_in,y_in,A,B,strides_x,strides_z):
    a = (tf.__version__)
    ind = a.split('.')
    ver = float(ind[0] + '.' + ind[1])
    # x = np.repeat(x_in, 1, axis=0)
    # y = np.repeat(y_in, 1, axis=0)
    # x=x_in
    # y=y_in
    _x = tf.zeros_like(x_in)
    _y = extract_patches(_x, A, B, strides_x,strides_z)
    grad = tf.gradients(_y, _x)[0]

    # with tf.device('/cpu:0'):
    #     # images_reconstructed = tf.gradients(_y*y,_x)[0]/grad
    #     images_reconstructed =tf.gradients(_y, _x, grad_ys=y_in)[0]/grad
    #     if ver == 1.12:
    #         sess = tf.Session(config=tf.ConfigProto())
    #     else:
    #         sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=False))
    #     images_r=images_reconstructed.eval(session=sess)
    #     sess.close()
    ####################
    # images_reconstructed = tf.gradients(_y*y,_x)[0]/grad
    images_reconstructed =tf.gradients(_y, _x, grad_ys=y_in)[0]/grad
    if ver == 1.12:
        sess = tf.Session(config=tf.ConfigProto())
    else:
        sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=False))
    images_r=images_reconstructed.eval(session=sess)
    # sess.close()
    return images_r

def extract_patches_inverse_try(x_in,y_in,A,B,strides_x,strides_z):
    a = (tf.__version__)
    ind = a.split('.')
    ver = float(ind[0] + '.' + ind[1])
    x = np.repeat(x_in, 1, axis=0)
    y = np.repeat(y_in, 1, axis=0)
    _x = tf.zeros_like(x)
    _y = extract_patches(_x, A,B,1,1)
    grad = tf.gradients(_y, _x)[0]
    with tf.device('/cpu:0'):
        images_reconstructed = tf.gradients(_y*y,_x)[0]/grad
        if ver == 1.12:
            sess = tf.Session(config=tf.ConfigProto())
        else:
            sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=False))
        images_r = images_reconstructed.eval(session=sess)
        sess.close()
    return images_r

def F_extract_patches(x, Patch_sz_x, Patch_sz_z, strides_x, strides_z, Nm_to_invert=100):
    tmp = extract_patches(x[0:1, :, :, 0:1], Patch_sz_x, Patch_sz_z, strides_x, strides_z)
    Npx = tmp.shape[1];
    Npz = tmp.shape[2]
    Nch = x.shape[-1]
    Nl = x.shape[0]
    Nx = x.shape[1]
    Nz = x.shape[2]
    patches_all = np.empty((Nl, Npx, Npz, Patch_sz_x, Patch_sz_z, Nch))
    a = (tf.__version__)
    ind = a.split('.')
    ver = float(ind[0] + '.' + ind[1])
    # %%
    # tf.compat.v1.disable_eager_execution()
    if Nl <= Nm_to_invert:
        with tf.device('/cpu:0'):
            # sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))
            if ver == 1.12:
                sess = tf.Session(config=tf.ConfigProto())
            else:
                sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=False))
            patches1 = extract_patches(x, Patch_sz_x, Patch_sz_z, strides_x, strides_z)
            tmp = patches1.eval(session=sess)
            tf.keras.backend.clear_session()
            sess.close()
            patches_all = np.reshape(tmp, (tmp.shape[0], tmp.shape[1], tmp.shape[2], Patch_sz_x, Patch_sz_z, Nch))
    else:
        ind = mov_window2(np.arange(Nl), Nm_to_invert, Nm_to_invert)
        for i in range(len(ind)):
            print(ind[i])
            with tf.device('/cpu:0'):
                # sess = tf.compat.v1.keras.backend.get_session()
                # sess = tf.Session(config=tf.ConfigProto())
                if ver == 1.12:
                    sess = tf.Session(config=tf.ConfigProto())
                else:
                    sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=False))
                patches1 = extract_patches(x[ind[i], ::], Patch_sz_x, Patch_sz_z, strides_x, strides_z)
                tmp = patches1.eval(session=sess)
                patches_all[ind[i], ::] = np.reshape(tmp, (
                tmp.shape[0], tmp.shape[1], tmp.shape[2], Patch_sz_x, Patch_sz_z, Nch))
                tf.keras.backend.clear_session()
                sess.close()
    # %%
    return patches_all

def F_extract_patches2(x, patch_sz_x, patch_sz_z):
    Nm = np.shape(x)[0]
    Nz = np.shape(x)[2]
    Nx = np.shape(x)[1]
    ind_patchx = list(window2(range(Nx), patch_sz_x))
    ind_patchz = list(window2(range(Nz), patch_sz_z))
    Npx = len(ind_patchx);
    Npz = len(ind_patchz);
    patches_norm = np.zeros((Nm, Npx, Npz, patch_sz_x, patch_sz_z))
    for i in range(Nm):
        for j in range(Npx):
            for k in range(Npz):
                patches_norm[i, j, k, :, :] = x[i, ind_patchx[j], ind_patchz[k]]
    return patches_norm

def F_extract_patches_inverse(patches, Patch_sz, Patch_sz2, Nx, Nz, Nl, Npx, Npz, strides_x, strides_z):
    Np = int(patches.shape[0] / Nl)
    array = np.zeros((Nl, Nx, Nz, 1))
    patches_tmp2 = np.reshape(patches, (Nl, Npx, Npz, Patch_sz * Patch_sz2))
    images_r = extract_patches_inverse(array, patches_tmp2, Patch_sz, Patch_sz2, strides_x, strides_z)
    return images_r

# @njit(nopython=True, parallel=True)
def F_extract_patches_inverse3(data, Patch_sz, Patch_sz2, Nx, Nz, Nl, Npx, Npz, strides_x, strides_z, Nm_to_invert=15):
    # Nm_to_invert - number of models to invert
    # Nmodels - number of models for one calculation
    Nch = data.shape[-1]
    Np = Npx * Npz
    data = np.reshape(data, (Nl, Npx, Npz, Patch_sz * Patch_sz2 * Nch))
    # Nl=1000;data=np.repeat(data,1000,axis=0);Nmodels=30     #Testing function
    images_r = np.zeros((Nl, Nx, Nz, Nch))
    if Nm_to_invert > Nl:
        Nm_to_invert = Nl
    models = mov_window2(np.arange(Nl), Nm_to_invert, Nm_to_invert)
    # @jit
    # @njit(parallel=True)
    # for i in range(len(models)):
    for i in range(len(models)):
        print(models[i], '!!!!!!!!!!!!!!')
        tmp = data[models[i], ::]
        array = np.zeros((tmp.shape[0], Nx, Nz, Nch))
        images_r[models[i], ::] = extract_patches_inverse(array, tmp, Patch_sz, Patch_sz2, strides_x, strides_z)
    return images_r

def F_break_into_patches(x, Patch_sz=128, Patch_sz2=128, strides_x=1, strides_z=1):
    Nm = np.shape(x)[0]
    x_all2 = F_extract_patches(x, Patch_sz, Patch_sz2, strides_x, strides_z)
    sz = x_all2.shape
    Npz = sz[2];
    Npx = sz[1]
    Np = Npx * Npz
    x_all = np.reshape(x_all2, (Nm * Np, sz[3::][0], sz[3::][1], sz[3::][2]))
    return x_all, Npx, Npz

def F_merge_123_dim(mat):
    sz = np.shape(mat)
    New_sz = tuple([sz[0] * sz[1] * sz[2]]) + sz[3::]
    mat = np.reshape(mat, New_sz)
    return mat

def F_data_splitting(x, t, ind_train, ind_test,
                     Train_on1_model_and_test_on_other, Train_models, Test_models, pars_h):
    ##  train/test splitting
    if Train_on1_model_and_test_on_other == 0:
        x_train = x[:, ind_train, :, :]
        t_train = t[:, ind_train, :, :]
        x_test_orig = x[:, ind_test, :, :]
        t_test_orig = t[:, ind_test, :, :]
    elif Train_on1_model_and_test_on_other == 1:
        t_train = t[Train_models, :, :, :]
        x_train = x[Train_models, :, :, :]
        t_test_orig = t[Test_models, :, :, :]
        x_test_orig = x[Test_models, :, :, :]
        if Test_models.size == 0:
            x_test_orig = x_train;
            t_test_orig = t_train
    ##  test/validation splitting for testing data
    Nl = x_test_orig.shape[0]
    Nch_in = x_test_orig.shape[-1];
    Nch_out = t_test_orig.shape[-1]
    Nx = x_test_orig.shape[1];
    Nz = x_test_orig.shape[2]
    test_val_frac = 0.8
    SHUFFLE = pars_h['val_shuffle']
    ind_x = np.arange(Nx);
    ind_z = np.arange(Nz)
    ind1 = int(Nx * test_val_frac)
    ind2 = Nx - ind1
    if SHUFFLE == False:
        ind_x2 = np.arange(ind1)
        ind_x3 = np.arange(ind1, Nx)
        x_test = x_test_orig[:, ind_x2, :, :]
        x_valid = x_test_orig[:, ind_x3, :, :]
        t_test = t_test_orig[:, ind_x2, :, :]
        t_valid = t_test_orig[:, ind_x3, :, :]
    else:
        x_test = np.empty((Nl, ind1, Nz, Nch_in))
        t_test = np.empty((Nl, ind1, Nz, Nch_out))
        x_valid = np.empty((Nl, ind2, Nz, Nch_in))
        t_valid = np.empty((Nl, ind2, Nz, Nch_out))
        for i in range(Nl):
            new_array = random.sample(list(range(Nx)), Nx)
            ind_x2 = new_array[0:ind1]
            ind_x3 = new_array[ind1:Nx]
            x_test[i, :, :, :] = x_test_orig[i, ind_x2, :, :]
            x_valid[i, :, :, :] = x_test_orig[i, ind_x3, :, :]
            t_test[i, :, :, :] = t_test_orig[i, ind_x2, :, :]
            t_valid[i, :, :, :] = t_test_orig[i, ind_x3, :, :]
    return x_train, t_train, x_test, t_test, x_valid, t_valid

def F_data_splitting2(x, t, ind_train, ind_test,
                      Train_on1_model_and_test_on_other, Train_models, Test_models, pars_h):
    ##  train/test splitting
    if Train_on1_model_and_test_on_other == 0:
        x_train = x[:, ind_train, :, :]
        t_train = t[:, ind_train, :, :]
        x_test = x[:, ind_test, :, :]
        t_test = t[:, ind_test, :, :]
    elif Train_on1_model_and_test_on_other == 1:
        t_train = t[Train_models, :, :, :]
        x_train = x[Train_models, :, :, :]
        t_test = t[Test_models, :, :, :]
        x_test = x[Test_models, :, :, :]
        if Test_models.size == 0:
            x_test = x_train;
            t_test = t_train
    ##  test/validation splitting for testing data
    Nl = x_test.shape[0]
    Nch_in = x_test.shape[-1];
    Nch_out = t_test.shape[-1]
    Nx = x_test.shape[1];
    Nz = x_test.shape[2]
    x_valid = None;
    t_valid = None;
    return x_train, t_train, x_test, t_test, x_valid, t_valid

def F_data_splitting3(x, t):
    train_frac = 0.7
    val_frac = 0.2
    test_frac = 0.1
    x_train, x_test, t_train, t_test = train_test_split(x, t, test_size=val_frac + test_frac, shuffle=False)
    x_valid, x_test, t_valid, t_test = train_test_split(x_test, t_test, test_size=test_frac / (val_frac + test_frac),
                                                        shuffle=False)
    return x_train, t_train, x_test, t_test, x_valid, t_valid

def F_data_splitting4(x,t,Train_models, Test_models, Valid_models):
    t_train = t[Train_models, :, :, :]
    x_train = x[Train_models, :, :, :]
    t_test = t[Test_models, :, :, :]
    x_test = x[Test_models, :, :, :]
    t_valid = t[Valid_models, :, :, :]
    x_valid = x[Valid_models, :, :, :]
    if len(x_train.shape) == 3:
        x_train = np.expand_dims(x_train, axis=0)
        t_train = np.expand_dims(t_train, axis=0)
    if len(x_test.shape) == 3:
        x_test = np.expand_dims(x_test, axis=0)
        t_test = np.expand_dims(t_test, axis=0)
    if len(x_valid.shape) == 3:
        x_valid = np.expand_dims(x_valid, axis=0)
        t_valid = np.expand_dims(t_valid, axis=0)
    return x_train, t_train, x_test, t_test, x_valid, t_valid

def F_preprocess_forth2D_mov_window5(x, t, ind_train, ind_test,
                                     Train_on1_model_and_test_on_other, Train_models, Test_models, Patch_sz=128,
                                     Patch_sz2=128, strides_x=1, strides_z=1):
    Nx = np.shape(x)[1]
    Nmodels = np.shape(x)[0]
    x_all2 = F_extract_patches(x, Patch_sz, Patch_sz2, strides_x, strides_z)
    t_all2 = F_extract_patches(t, Patch_sz, Patch_sz2, strides_x, strides_z)
    sz = x_all2.shape
    Npz = sz[2];
    Npx = sz[1];
    Np = Npx * Npz;
    # Plot_image(x[0,:, :].T)
    # Plot_image(x_all2[0, 55,33,:,:])
    # Plot_image(x_all2[0, 0, 1, :, :])
    ###################################  Make splitting
    if Train_on1_model_and_test_on_other == 0:
        x_train = F_extract_patches(x[:, ind_train, :], Patch_sz, Patch_sz2, strides_x, strides_z)
        t_train = F_extract_patches(t[:, ind_train, :], Patch_sz, Patch_sz2, strides_x, strides_z)
        x_test = F_extract_patches(x[:, ind_test, :], Patch_sz, Patch_sz2, strides_x, strides_z)
        t_test = F_extract_patches(t[:, ind_test, :], Patch_sz, Patch_sz2, strides_x, strides_z)
    elif Train_on1_model_and_test_on_other == 1:
        t_train = t_all2[Train_models, :, :, :, :]
        x_train = x_all2[Train_models, :, :, :, :]
        t_test = t_all2[Test_models, :, :, :, :]
        x_test = x_all2[Test_models, :, :, :, :]
        if Test_models.size == 0:
            x_test = x_train;
            t_test = t_train
    #####################################################
    # splitting 1st way
    x_test = np.swapaxes(x_test, 0, 1)
    t_test = np.swapaxes(t_test, 0, 1)
    x_test, x_valid, t_test, t_valid = train_test_split(
        x_test, t_test, test_size=0.2, shuffle=False)
    # splitting 2nd way
    x_all = F_merge_123_dim(x_all2)
    t_all = F_merge_123_dim(t_all2)
    x_train = F_merge_123_dim(x_train)
    x_test = F_merge_123_dim(x_test)
    t_train = F_merge_123_dim(t_train)
    t_test = F_merge_123_dim(t_test)
    x_valid = F_merge_123_dim(x_valid)
    t_valid = F_merge_123_dim(t_valid)
    ##################################################### Perform data augmentation, flip 2d patches before feeding to CNN
    # x_train=np.concatenate((x_train,   x_train[:,::-1,:]    ),axis=0)
    # t_train = np.concatenate((t_train, t_train[:, ::-1, :]), axis=0)
    # print('Training DATA augmented')
    # x_all=np.concatenate((x_all,   x_all[:,::-1,:]    ),axis=0)
    # t_all = np.concatenate((t_all, t_all[:, ::-1, :]), axis=0)
    # print('Training DATA augmented')
    #####################################################
    x_train = add_dim_back(x_train)
    t_train = add_dim_back(t_train)
    x_test = add_dim_back(x_test)
    t_test = add_dim_back(t_test)
    x_all = add_dim_back(x_all)
    t_all = add_dim_back(t_all)
    x_valid = add_dim_back(x_valid)
    t_valid = add_dim_back(t_valid)
    print('Train data size', np.shape(x_train))
    print('Test data size', np.shape(x_test))
    print('Validation data size', np.shape(x_valid))
    return x_train, t_train, x_test, t_test, x_all, t_all, x_valid, t_valid, Npx, Npz

def F_preprocess_back2D_mov_window2(pred_data_long, Patch_sz, Patch_sz2, Nl, Nx, Nz, Npx, Npz, strides_x, strides_z):
    ###########    Scaling back   and Assembling dataset into full size sections
    pred_data_long = np.asarray(pred_data_long[:, :, :, 0])
    if (pred_data_long.size == 0):
        x_2 = pred_data_long
    else:
        x1 = pred_data_long
        x_2 = F_extract_patches_inverse(x1, Patch_sz, Patch_sz2, Nx, Nz, Nl, Npx, Npz, strides_x, strides_z)
        # Plot_image(x_2[0,:,:].T)
    return x_2

def F_split(L, Patch_num, Test_patches):
    Train_patches = np.setdiff1d(np.arange(0, Patch_num, 1), Test_patches)
    Array = np.arange(0, L)
    Ind = np.array_split(Array, Patch_num)
    ###########################
    Patch = np.zeros((Patch_num, 2), dtype=int)
    for i in range(len(Ind)):
        Patch[i, :] = [Ind[i][0], Ind[i][-1]]
    ind_train = []
    for i in Train_patches:
        ind_train.append(list(range(Patch[i, 0], Patch[i, 1] + 1, 1)))
    ind_train = list(itertools.chain.from_iterable(ind_train))
    ind_test = []
    for i in Test_patches:
        ind_test.append(list(range(Patch[i, 0], Patch[i, 1] + 1, 1)))
    ind_test = list(itertools.chain.from_iterable(ind_test))
    return ind_train, ind_test

def F_find_min_max(Data, COEFF=1):
    MIN = COEFF * np.min(Data)
    MAX = COEFF * np.max(Data)
    return MIN, MAX

def F_gen_random_numbers(N, dz, a1=200, a2=1900, b=300, c=1500):
    window_length = np.zeros((2, N));
    for i_realization in range(N):
        b1 = np.ceil(b * random.uniform(0, 1));
        b2 = np.ceil(c * random.uniform(0, 1));
        L1 = a1 + b1;
        L2 = a2 + b2;
        window_length[0, i_realization] = np.ceil(L1 / dz) + 1;
        window_length[1, i_realization] = np.ceil(L2 / dz) + 1;
    return window_length

def F_smoothing_not_used(Data, Nl, window_length, dz, Angles, filter, Filt_plotting_flag, Approach, COEFF=0.1):
    # Generate data with random smoothing and then filter this data
    #  Random smoothing  Smooth Vp log 2 times with random moving average windows and then Subtract 1 smoothed log from another
    # Approach1: #Input - High wavenumbers, Output - Low wavenumbers
    # Approach2: Input - High wavenumbers, Output - All wavenumbers(High + Low)
    Perturbations = np.zeros((Nl, np.shape(Data)[0], np.shape(Data)[1]))
    DV_High = np.zeros((Nl, np.shape(Data)[0], np.shape(Data)[1]))
    DV_Low = np.zeros((Nl, np.shape(Data)[0], np.shape(Data)[1]))
    Background = np.zeros((np.shape(Perturbations)))
    Smoothed = np.zeros((np.shape(Perturbations)))
    smoothed = np.zeros((1, np.shape(Perturbations)[1]))
    background = np.zeros((1, np.shape(Perturbations)[1]))
    pert = np.zeros((1, np.shape(Perturbations)[1]))
    for i_x in range(Data.shape[0]):
        print('i_x=', i_x)
        log = Data[i_x, :];
        ################################################################
        for i_realization in range(Nl):
            ##################  Smoothing
            df = pd.DataFrame(log)
            tmp = df.rolling(int(window_length[0, i_realization]), center=True, min_periods=1).mean()
            smoothed = tmp.values
            tmp = df.rolling(int(window_length[1, i_realization]), center=True, min_periods=1).mean()
            background = tmp.values
            pert = background - smoothed
            dv = pert;
            Vo = background;
            if len(dv.shape) == 2: dv = dv[:, 0]
            if len(Vo.shape) == 2: Vo = Vo[:, 0]
            #########   Calculate Boundary between low wavenumbers and middle wavenumbers
            v_water = 1500;
            fs = 1 / dz;
            half_offset = filter.max_offset / 2;
            cosin = (np.cos(Angles / 180 * np.pi));
            k1 = 2 * filter.fMin * (cosin) / Vo / 1000;
            # plt.figure()
            # plt.plot(k1[0,:])
            # plt.show()
            # plt.close()
            #########   Calculate spectrum
            n_perseg = np.shape(dv)[0] - 1
            n_overlap = n_perseg - 1
            t_true = np.arange(np.shape(dv)[0]) * dz
            f, t, Spectrum = signal.stft(dv, fs, nperseg=n_perseg, noverlap=n_overlap, boundary='even')
            #########   Allocate additional matrices
            K1 = np.repeat(k1, np.shape(Spectrum)[0], axis=0)
            F = np.repeat(np.expand_dims(f, 1), np.shape(Spectrum)[1], axis=1)
            Min = np.min(abs(Spectrum));
            Max = COEFF * np.max(abs(Spectrum))
            #########   Filter high frequencies
            Spectrum_high_f = np.where(F > K1, Spectrum, 0)
            # #Plot_image(np.flipud(abs(Spectrum_high_f)), 'Spectrum_high_f', [Min, Max], x_label='Z', y_label='Frequency')
            # Min = np.min(np.real(Spectrum));
            # Max = COEF_clim * np.max(np.real(Spectrum))
            # #Plot_image(np.flipud(abs(Spectrum)), 'Spectrum', x_label='Z', y_label='Frequency')
            # #Plot_image(np.flipud(np.real(Spectrum)), 'Spectrum', x_label='Z', y_label='Frequency')
            _, dv_High = signal.istft(Spectrum_high_f, fs, nperseg=n_perseg, noverlap=n_overlap, boundary='even')
            Spectrum_low_f = np.where(F <= K1, Spectrum, 0)
            _, dv_Low = signal.istft(Spectrum_low_f, fs, nperseg=n_perseg, noverlap=n_overlap, boundary='even')
            ###################   STFT,ISTFT functions testing  #######################################
            # f2, t2, Spectrum_high_f2= signal.stft(dv_High, fs, nperseg=n_perseg, noverlap=n_overlap, boundary='even')
            # f2, t2, Spectrum_low_f2 = signal.stft(dv_Low, fs, nperseg=n_perseg, noverlap=n_overlap, boundary='even')
            # _, dv_High2 = signal.istft(Spectrum_high_f2, fs, nperseg=n_perseg, noverlap=n_overlap, boundary='even')
            # _, dv_Low2 = signal.istft(Spectrum_low_f2, fs, nperseg=n_perseg, noverlap=n_overlap, boundary='even')
            # plt.figure()
            # plt.plot(dv_High)
            # plt.plot(dv_High2)
            # plt.plot(dv_High-dv_High2)
            # plt.plot(dv_Low)
            # plt.plot(dv_Low2)
            # # plt.plot(dv_High + dv_Low)
            # # plt.plot(dv)
            # plt.legend(['1', '2', '3', '4', '5'])
            # # plt.legend(['dv_High','dv_High2', 'dv_Low', 'dv_High+dv_Low', 'dv'])
            # plt.show()
            # plt.close()
            # Colorbar_max = 0.1
            # Plot_image2(np.flipud(abs(Spectrum)), '', 'Spectrum_f', x_label='Z, m',
            #             y_label='$K_{z}$ - vertical wavenumber', x=t,
            #             y=f, Reverse_axis=0, c_lim=[0, Colorbar_max], Save_flag=0)
            # Plot_image2(np.flipud(abs(Spectrum_high_f)), '', 'Spectrum_high_f', x_label='Z, m',
            #             y_label='$K_{z}$ - vertical wavenumber', x=t,
            #             y=f, Reverse_axis=0, c_lim=[0, Colorbar_max], Save_flag=0)
            # Plot_image2(np.flipud(abs(Spectrum_high_f2)), '', 'Spectrum_high_f2', x_label='Z, m',
            #             y_label='$K_{z}$ - vertical wavenumber', x=t,
            #             y=f, Reverse_axis=0, c_lim=[0, Colorbar_max], Save_flag=0)
            # Plot_image2(np.flipud(abs(Spectrum_low_f)), '', 'Spectrum_low_f', x_label='Z, m',
            #             y_label='$K_{z}$ - vertical wavenumber', x=t,
            #             y=f, Reverse_axis=0, c_lim=[0, Colorbar_max], Save_flag=0)
            # Plot_image2(np.flipud(abs(Spectrum_low_f2)), '', 'Spectrum_low_f2', x_label='Z, m',
            #             y_label='$K_{z}$ - vertical wavenumber', x=t,
            #             y=f, Reverse_axis=0, c_lim=[0, Colorbar_max], Save_flag=0)
            # Plot_image2(np.flipud(abs(Spectrum_low_f - Spectrum_low_f2)), '', 'delta Low', x_label='Z, m',
            #             y_label='$K_{z}$ - vertical wavenumber', x=t,
            #             y=f, Reverse_axis=0, c_lim='', Save_flag=0)
            # Plot_image2(np.flipud(abs(Spectrum_high_f - Spectrum_high_f2)), '', 'delta High', x_label='Z, m',
            #             y_label='$K_{z}$ - vertical wavenumber', x=t,
            #             y=f, Reverse_axis=0, c_lim='', Save_flag=0)
            #
            # Plot_image2(np.flipud(abs((Spectrum_low_f+Spectrum_high_f)-(Spectrum_low_f2+Spectrum_high_f2))), '', 'delta High', x_label='Z, m',
            #             y_label='$K_{z}$ - vertical wavenumber', x=t,
            #             y=f, Reverse_axis=0, c_lim='', Save_flag=0)
            # Plot_image2(np.flipud(abs((Spectrum) - (Spectrum_low_f2 + Spectrum_high_f2))), '',
            #             'delta High', x_label='Z, m',
            #             y_label='$K_{z}$ - vertical wavenumber', x=t,
            #             y=f, Reverse_axis=0, c_lim='', Save_flag=0)
            #
            # Plot_image2(np.flipud(abs(Spectrum_high_f2)), '', 'Spectrum_high_f2', x_label='Z, m',
            #             y_label='$K_{z}$ - vertical wavenumber', x=t,
            #             y=f, Reverse_axis=0, c_lim=[0, Colorbar_max], Save_flag=0)
            # Plot_image2(np.flipud(abs(Spectrum_high_f - Spectrum_high_f2)), '', 'delta High', x_label='Z, m',
            #             y_label='$K_{z}$ - vertical wavenumber', x=t,
            #             y=f, Reverse_axis=0, c_lim='', Save_flag=0)
            # Plot_image2(np.flipud(abs(Spectrum_high_f2 - Spectrum_high_f)), '', 'delta2', x_label='Z, m',
            #             y_label='$K_{z}$ - vertical wavenumber', x=t,
            #             y=f, Reverse_axis=0, c_lim='', Save_flag=0)

            #########   Filter low frequencies
            # Spectrum_low_f = np.where(np.repeat(np.expand_dims(f, 1), np.shape(Spectrum)[1], axis=1) <= 0.005, Spectrum, 0)
            ########################################################################
            Perturbations[i_realization, i_x, :] = pert.T;
            if Approach == 1:
                DV_High[i_realization, i_x, :] = dv_High;
                DV_Low[i_realization, i_x, :] = dv_Low;
            if Approach == 2:
                DV_High[i_realization, i_x, :] = dv_High;
                DV_Low[i_realization, i_x, :] = dv_High + dv_Low;
            dv2 = dv_High + dv_Low
            if Filt_plotting_flag == 1:
                Colorbar_max = 0.1
                Plot_image2(np.flipud(abs(Spectrum)), k1, 'STFT Spectrum', x_label='Z, m',
                            y_label='$K_{z}$ - vertical wavenumber', x=t,
                            y=f, Reverse_axis=0, c_lim=[0, Colorbar_max], Save_flag=0)
                Plot_image2(np.flipud(abs(Spectrum_high_f)), k1, 'Spectrum_high_f', x_label='Z, m',
                            y_label='$K_{z}$ - vertical wavenumber', x=t,
                            y=f, Reverse_axis=0, c_lim=[0, Colorbar_max], Save_flag=0)
                # Plot_image2(np.flipud(abs(Spectrum_low_f)), k1, 'Spectrum_low_f', x_label='Z, m',
                #             y_label='$K_{z}$ - vertical wavenumber', x=t,
                #             y=f, Reverse_axis=0, c_lim=[0, Colorbar_max], Save_flag=0)
                plt.figure()
                plt.plot(dv_High)
                plt.plot(dv_Low, '*')
                plt.plot(dv_High + dv_Low)
                plt.plot(dv)
                plt.legend(['dv_High', 'dv_Low', 'dv_High+dv_Low', 'dv'])
                plt.show()
                plt.close()
                M1_spectrum, df = F_STFT_spectrum(dv_High, 'Input, dv_High', dz, Plotting_flag=1,
                                                  Colorbar_max=Colorbar_max)
                M2_spectrum, df = F_STFT_spectrum(dv_Low, 'Target, dv_Low', dz, Plotting_flag=1,
                                                  Colorbar_max=Colorbar_max)
                M3_spectrum, df = F_STFT_spectrum(dv_High + dv_Low, 'dv_High+dv_Low', dz, Plotting_flag=1,
                                                  Colorbar_max=Colorbar_max)
                # PLOT_Compare_spectrums(M1_spectrum, M2_spectrum, M3_spectrum,
                #     dz, df,
                #     Title='', Save_flag=1, COEFF=1,Colorbar_max=Colorbar_max)
                eer = 1
    return Perturbations, DV_High, DV_Low, Smoothed, Background

def F_smooth(data, plot_flag=0, sigma_val=8):
    data2 = gaussian_filter(data, sigma=sigma_val, order=0, output=None,
                            mode="reflect", cval=0.0, truncate=4.0)
    if plot_flag == 1:
        a = np.min(data);
        b = np.max(data)
        Plot_image(data.T, Show_flag=1, c_lim=[a, b])
        Plot_image(data2.T, Show_flag=1, c_lim=[a, b])
    return data2

def F_filtering(Data, Perturbations, Nl, window_length, dz, Angles, filter, Filt_plotting_flag=1,
                Result_plotting_flag=0, Save_pictures_path=''):
    # Filter some matrix
    DV_High = np.zeros((Nl, np.shape(Data)[0], np.shape(Data)[1]))
    DV_Low = np.zeros((Nl, np.shape(Data)[0], np.shape(Data)[1]))
    Background = np.zeros((np.shape(Perturbations)))
    for i_x in range(Data.shape[0]):
        print('i_x=', i_x)
        pert = Perturbations[i_x, :];
        log = Data[i_x, :];
        ################################################################
        for i_realization in range(Nl):
            df = pd.DataFrame(log)
            tmp = df.rolling(int(window_length[1, i_realization]), center=True, min_periods=1).mean()
            background = tmp.values
            dv = pert;
            Vo = background;
            if len(dv.shape) == 2: dv = dv[:, 0]
            if len(Vo.shape) == 2: Vo = Vo[:, 0]
            COEF_clim = 0.2;
            #########   Calculate Boundary between low wavenumbers and middle wavenumbers
            v_water = 1500;
            fs = 1 / dz;
            Frequency_limits = [0, filter['fMax'] / v_water];
            half_offset = filter.max_offset / 2;
            cosin = (np.cos(Angles / 180 * np.pi));
            tmp = cosin / Vo / 1000
            k1 = 2 * filter.fMin * (cosin) / Vo / 1000;
            #########   Calculate spectrum
            n_perseg = np.shape(dv)[0] - 1
            n_overlap = n_perseg - 1
            t_true = np.arange(np.shape(dv)[0]) * dz
            # f, t, Spectrum = signal.stft(dv, fs, nperseg=n_perseg, noverlap=n_overlap, boundary='even')
            f, t, Spectrum = signal.stft(dv, fs, nperseg=n_perseg, noverlap=n_overlap, boundary='zeros')
            #########   Allocate additional matrices
            K1 = np.repeat(k1, np.shape(Spectrum)[0], axis=0)
            F = np.repeat(np.expand_dims(f, 1), np.shape(Spectrum)[1], axis=1)
            Min = np.min(abs(Spectrum));
            Max = COEF_clim * np.max(abs(Spectrum))
            #########   Filter high frequencies
            Spectrum_high_f = np.where(F >= K1, Spectrum, 0)
            _, dv_High = signal.istft(Spectrum_high_f, fs, nperseg=n_perseg, noverlap=n_overlap, boundary='even')
            # Min = np.min(np.real(Spectrum));
            # Max = COEF_clim * np.max(np.real(Spectrum))
            ########   Filter low frequencies
            # Spectrum_low_f = np.where(np.repeat(np.expand_dims(f, 1), np.shape(Spectrum)[1], axis=1) <= 0.005, Spectrum, 0)
            Spectrum_low_f = np.where(F <= K1, Spectrum, 0)
            _, dv_Low = signal.istft(Spectrum_low_f, fs, nperseg=n_perseg, noverlap=n_overlap, boundary='even')
            ########################################################################
            DV_High[i_realization, i_x, :] = dv_High;
            DV_Low[i_realization, i_x, :] = dv_Low;
            if Filt_plotting_flag == 1:
                [Min, Max] = F_find_min_max(Spectrum, COEFF=1)
                # Plot_image(np.flipud(abs(Spectrum)), 'Spectrum', x_label='Z', y_label='Frequency')
                # Plot_image(np.flipud(np.real(Spectrum)), 'Spectrum', x_label='Z', y_label='Frequency')
                Plot_image(np.flipud(abs(Spectrum)), Title='Spectrum_high_f', x_label='Z', y_label='Frequency')
                Plot_image(np.flipud(abs(Spectrum_high_f)), 'Spectrum_high_f', x_label='Z', y_label='Frequency')
                Plot_image(np.flipud(abs(Spectrum_low_f)), 'Spectrum_low_f', x_label='Z', y_label='Frequency')
    if Result_plotting_flag == 1:
        for i_realization in range(Nl):
            PLOT_ML_Result(DV_High[i_realization, :, :].T, DV_Low[i_realization, :, :].T, DV_Low[i_realization, :, :].T,
                           Plot_vertical_lines=0, Title='')
            # Plot_image(DV_High[i_realization,:,:].T, 'Spectrum_high_wavenumbers', x_label='Z', y_label='Frequency',Save_flag=1)
            # Plot_image(DV_Low[i_realization, :, :].T, 'Spectrum_low_wavenumbers', x_label='Z', y_label='Frequency',Save_flag=1)
    return DV_High, DV_Low, Background

def F_STFT_spectrum(dv, Title, dz, Plotting_flag=1, COEFF=1, Colorbar_max=''):
    if len(dv.shape) == 2:
        dv = dv[:, 0]
    fs = 1 / dz
    #########   Calculate spectrum
    n_perseg = np.shape(dv)[0] - 1
    n_overlap = n_perseg - 1
    t_true = np.arange(np.shape(dv)[0]) * dz
    # f, z, spectrum = signal.stft(dv, fs, nperseg=n_perseg, noverlap=n_overlap, boundary=None)
    f, z, spectrum = signal.stft(dv, fs, nperseg=n_perseg, noverlap=n_overlap, boundary='even')
    # sp = np.fft.fft(dv)
    # f2 = np.fft.fftfreq(dv.shape[-1],d=dz)
    df = f[1] - f[0]
    # df=f[0][1]-f[0][0]
    Spectrum = abs(spectrum)
    #########   Allocate additional matrices
    F = np.repeat(np.expand_dims(f, 1), np.shape(spectrum)[1], axis=1)
    MIN = np.min(abs(spectrum))
    if Colorbar_max == '':
        MAX = np.max(abs(spectrum))
    else:
        MAX = Colorbar_max
    # %
    if Plotting_flag == 1:
        Plot_image(np.flipud(abs(spectrum)), Title, x_label='Z, m',
                   y_label='K_{z}$ - vertical wavenumber', x=z, y=f, Reverse_axis=0, c_lim=[MIN, COEFF * MAX],
                   Save_flag=0)
    # Plot_image2(np.flipud(abs(Spectrum)), k1, 'STFT Spectrum', x_label='Z, m',
    #             y_label='$K_{z}$ - vertical wavenumber', x=z,
    #             y=f, Reverse_axis=0, c_lim=[Min, Max], Save_flag=1)
    return spectrum, df

def F_crosscorr(F1, F2, Nx):
    Norm = np.sqrt((np.power(F1, 2)).sum(axis=0) * (np.power(F2, 2)).sum(axis=0))
    Autocorr = np.sum(F1 * F2 / np.repeat(Norm[np.newaxis, :, :], Nx, axis=0), axis=0)
    return Autocorr, Norm

def F_calc_aij(Mi,xi,ti,NAME):
    with open(NAME,'rb') as f:
        data=np.load(f)
        Mj=data['models'][0,:,:,0]
        xj=data['input_data'][0,:,:,0]
        tj=data['output_data'][0,:,:,0]
    data.close()
    Aij=F_r2(xi,xj)
    Bij=F_r2(ti,tj)
    Dij=F_r2(Mi,Mj)
    A2ij=1-F_r2(xi,xj)
    B2ij=1-F_r2(ti,tj)
    return Aij,Bij,Dij,A2ij,B2ij

def F_calc_dij(Mi,NAME):
    with open(NAME,'rb') as f:
        data=np.load(f)
        Mj=data['models'][0,:,:,0]
    data.close()
    Dij=F_r2(Mi,Mj)
    return Dij

def F_calculate_misfits(t, t_pred,ind_train, ind_test, Train_on1_model_and_test_on_other, Train_models, Test_models,
                        Valid_models, print_flag=0):
    # %%   Calculate model misfits (R2 score) on test, validation, training datasets separately
    Nl = t.shape[0]
    misfit_stats = np.zeros((4, 2))
    # %%     Allocate    data
    if Train_on1_model_and_test_on_other == 0:
        dataset_predicted_train =t_pred[:, ind_train, :]
        dataset_predicted_test = t_pred[:, ind_test, :]
        dataset_true_train = t[:, ind_train, :]
        dataset_true_test = t[:, ind_test, :]
        dataset_predicted_all = t_pred
        dataset_true_all = t
        misfit_stats2 = np.zeros((3, Nl))
        for i_x in range(Nl):
            misfit_stats2[0, i_x] = F_r2(dataset_true_train[i_x, :, :],dataset_predicted_train[i_x, :, :])
            misfit_stats2[1, i_x] = F_r2(dataset_true_test[i_x, :, :],dataset_predicted_test[i_x, :, :])
            misfit_stats2[2, i_x] = F_r2(dataset_true_all[i_x, :, :],dataset_predicted_all[i_x, :, :])
        for i_x in range(3):
            misfit_stats[i_x, 0] = (misfit_stats2[i_x, :]).min()
            misfit_stats[i_x, 1] = (misfit_stats2[i_x, :]).max()
        ######
        dataset_predicted_train = F_reduce_zero_dim(dataset_predicted_train)
        dataset_predicted_test = F_reduce_zero_dim(dataset_predicted_test)
        dataset_predicted_all = F_reduce_zero_dim(dataset_predicted_all)
        dataset_true_train = F_reduce_zero_dim(dataset_true_train)
        dataset_true_test = F_reduce_zero_dim(dataset_true_test)
        dataset_true_all = F_reduce_zero_dim(dataset_true_all)
        misfits = np.empty(3)
        misfits[0] = F_r2(dataset_true_train,dataset_predicted_train)
        misfits[2] = F_r2(dataset_true_all,dataset_predicted_all)
        misfits[1] = F_r2(dataset_true_test,dataset_predicted_test)
    elif Train_on1_model_and_test_on_other == 1:
        dataset_predicted_train = t_pred[Train_models, :, :]
        dataset_predicted_test = t_pred[Test_models, :, :]
        dataset_predicted_valid = t_pred[Valid_models, :, :]
        dataset_predicted_all = t_pred
        dataset_true_train = t[Train_models, :, :]
        dataset_true_test = t[Test_models, :, :]
        dataset_true_valid = t[Valid_models, :, :]
        dataset_true_all = t
        tmp = np.zeros((dataset_predicted_train.shape[0]))
        for i_x in range(tmp.shape[0]):
          tmp[i_x] = F_r2(dataset_true_train[i_x, :, :],dataset_predicted_train[i_x, :, :])
        misfit_stats[0, 0] = tmp.min()
        misfit_stats[0, 1] = tmp.max()
        idx_train_best=np.where(tmp==tmp.max())
        idx_train_worst=np.where(tmp==tmp.min())
        idx_train_best=int(idx_train_best[0])
        idx_train_worst=int(idx_train_worst[0])

        tmp = np.zeros((dataset_predicted_test.shape[0]))
        for i_x in range(tmp.shape[0]):
            tmp[i_x] = F_r2(dataset_true_test[i_x, :, :],dataset_predicted_test[i_x, :, :])
        misfit_stats[1, 0] = tmp.min()
        misfit_stats[1, 1] = tmp.max()

        tmp = np.zeros((dataset_predicted_all.shape[0]))
        for i_x in range(tmp.shape[0]):
            tmp[i_x] = F_r2(dataset_true_all[i_x, :, :],dataset_predicted_all[i_x, :, :])
        misfit_stats[2, 0] = tmp.min()
        misfit_stats[2, 1] = tmp.max()

        tmp = np.zeros((dataset_predicted_valid.shape[0]))
        for i_x in range(tmp.shape[0]):
            tmp[i_x] = F_r2(dataset_true_valid[i_x, :, :],dataset_predicted_valid[i_x, :, :])
        misfit_stats[3, 0] = tmp.min()
        misfit_stats[3, 1] = tmp.max()
        ######
        dataset_predicted_train = F_reduce_zero_dim(dataset_predicted_train)
        dataset_predicted_test = F_reduce_zero_dim(dataset_predicted_test)
        dataset_predicted_all = F_reduce_zero_dim(dataset_predicted_all)
        dataset_predicted_valid = F_reduce_zero_dim(dataset_predicted_valid)

        dataset_true_train = F_reduce_zero_dim(dataset_true_train)
        dataset_true_test = F_reduce_zero_dim(dataset_true_test)
        dataset_true_all = F_reduce_zero_dim(dataset_true_all)
        dataset_true_valid = F_reduce_zero_dim(dataset_true_valid)
        misfits = np.zeros(4)
        misfits[0] = F_r2(dataset_true_train,dataset_predicted_train)
        misfits[1] = F_r2(dataset_true_valid,dataset_predicted_valid)
        misfits[2] = F_r2(dataset_true_test,dataset_predicted_test)
        misfits[3] = F_r2(dataset_true_all,dataset_predicted_all)
    # %%    Print output
    if print_flag == 1:
        print('Min R2 score across all models/Max R2 score across all models')
        print('In train data', numstr(misfit_stats[0, 0]) + '/' + numstr(misfit_stats[0, 1]))
        print('In validation data', numstr(misfit_stats[3, 0]) + '/' + numstr(misfit_stats[3, 1]))
        print('In test data', numstr(misfit_stats[1, 0]) + '/' + numstr(misfit_stats[1, 1]))
        print('Average score for ALL models: Train R2', numstr(misfits[0]), ',Valid R2', numstr(misfits[3]), ',Test R2',
              numstr(misfits[1]),
              ',All R2', numstr(misfits[2]))
        print('In (train+tested) data', numstr(misfit_stats[2, 0]) + '/' + numstr(misfit_stats[2, 1]))
    return misfit_stats, misfits,(idx_train_best),(idx_train_worst)

def F_calculate_misfits_across_files(list_all,list_test,list_train,list_valid,
    dataset_predicted_train,dataset_predicted_valid,dataset_predicted_test,
    Train_on1_model_and_test_on_other, Train_models, Test_models,
    Valid_models, print_flag=0,test_status=0):
    # %%   Calculate model misfits (R2 score) on test, validation, training datasets separately
    misfit_stats = np.zeros((4, 2))
    # %%     Allocate    data
    if Train_on1_model_and_test_on_other == 1:
        dataset_predicted_all=np.concatenate((dataset_predicted_train, \
             dataset_predicted_valid,dataset_predicted_test),axis=0)
        Nl = dataset_predicted_all.shape[0]
        print('Train')
        tmp = np.zeros(len(Train_models))
        for i_x in range(len(Train_models)):
            NAME=list_train[i_x]
            true=load_true_data(NAME,test_status)
            A=true[0,:,:,0]
            B=dataset_predicted_train[i_x, :, :,0]
            tmp[i_x] = F_r2(B,A)
        
        train_misfits=tmp
        misfit_stats[0, 0] = tmp.min()
        misfit_stats[0, 1] = tmp.max()

        idx_train_best=np.where(tmp==tmp.max())
        idx_train_worst=np.where(tmp==tmp.min())
        idx_train_best=idx_train_best[0]
        idx_train_worst=idx_train_worst[0]
        train_best_val=tmp.max()
        train_worst_val=tmp.min()
        train_best_file= list_train[int(idx_train_best [0])]
        train_worst_file=list_train[int(idx_train_worst[0])]

        print('Test')
        tmp = np.zeros(len(Test_models))
        for i_x in range(tmp.shape[0]):
            NAME=list_test[i_x]
            true=load_true_data(NAME,test_status)
            A=true[0,:,:,0]
            B=dataset_predicted_test[i_x, :, :,0]
            tmp[i_x] = F_r2(B,A)
            
        misfit_stats[1, 0] = tmp.min()
        misfit_stats[1, 1] = tmp.max()
        test_misfits=tmp

        tmp = np.zeros(Nl)
        for i_x in range(Nl):
            NAME=list_all[i_x]
            true=load_true_data(NAME,test_status)
            A=true[0,:,:,0]
            B=dataset_predicted_all[i_x, :, :,0]
            tmp[i_x] = F_r2(B,A)
        misfit_stats[2, 0] = tmp.min()
        misfit_stats[2, 1] = tmp.max()

        tmp = np.zeros(len(Valid_models))
        for i_x in range(tmp.shape[0]):
            NAME=list_valid[i_x]
            true=load_true_data(NAME,test_status)
            A=true[0,:,:,0]
            B=dataset_predicted_valid[i_x, :, :,0]
            tmp[i_x] = F_r2(B,A)
        misfit_stats[3, 0] = tmp.min()
        misfit_stats[3, 1] = tmp.max()
        valid_misfits=tmp
    # %%    Print output
    if print_flag == 1:
        print('Min R2 score across all models/Max R2 score across all models:')
        print('In train data', numstr(misfit_stats[0, 0]) + '/' + numstr(misfit_stats[0, 1]))
        print('In validation data', numstr(misfit_stats[3, 0]) + '/' + numstr(misfit_stats[3, 1]))
        print('In test data', numstr(misfit_stats[1, 0]) + '/' + numstr(misfit_stats[1, 1]))

        print('idx_train_best ', idx_train_best)
        print('train_best_file ',train_best_file)
        print('train_best_val ',train_best_val)
        
        print('idx_train_worst ',idx_train_worst)
        print('train_worst_file',train_worst_file)
        print('train_worst_val ',train_worst_val)

        # print('Average score for ALL models: Train R2', numstr(misfits[0]), ',Valid R2', numstr(misfits[3]), ',Test R2',
        #       numstr(misfits[1]),
        #       ',All R2', numstr(misfits[2]))
        # print('In (train+tested) data', numstr(misfit_stats[2, 0]) + '/' + numstr(misfit_stats[2, 1]))
        ss=str(train_misfits.tolist())
        # print('Train misfits='+str(train_misfits.tolist()) )
        # print('Validation misfits='+str(valid_misfits.tolist()))
        # print('Train misfits='+str(test_misfits.tolist()))
    return misfit_stats,(idx_train_best),(idx_train_worst),train_misfits,valid_misfits,test_misfits,  \
    train_best_val,train_worst_val,train_best_file,train_worst_file

def F_WSST_Filtering(dv, Vo, dz,Angles, filter, Plotting=0):
    if len(dv.shape) == 2: dv = dv[:, 0]
    if len(Vo.shape) == 2: Vo = Vo[:, 0]
    COEF_clim = 0.2;
    #########   Calculate Boundary between low wavenumbers and middle wavenumbers
    v_water = 1500;
    fs = 1 / dz;
    Frequency_limits = [0, filter['fMax'] / v_water];
    half_offset = filter.max_offset / 2;
    cosin = (np.cos(Angles / 180 * np.pi));
    tmp = cosin / Vo / 1000
    k1 = 2 * filter.fMin * (cosin) / Vo / 1000;
    #########   Calculate spectrum
    n_perseg = np.shape(dv)[0] - 1
    n_overlap = n_perseg - 1
    t_true = np.arange(np.shape(dv)[0]) * dz
    # if
    f, t, Spectrum = signal.stft(dv, fs, nperseg=n_perseg, noverlap=n_overlap, boundary='even')
    #########   Allocate additional matrices
    K1 = np.repeat(k1, np.shape(Spectrum)[0], axis=0)
    F = np.repeat(np.expand_dims(f, 1), np.shape(Spectrum)[1], axis=1)
    Min = np.min(abs(Spectrum));
    Max = COEF_clim * np.max(abs(Spectrum))
    #########   Filter high frequencies
    Spectrum_high_f = np.where(F >= K1, Spectrum, 0)
    # Plot_image(np.flipud(abs(Spectrum_high_f)), 'Spectrum_high_f', [Min, Max], x_label='Z', y_label='Frequency')
    # Min = np.min(np.real(Spectrum));
    # Max = COEF_clim * np.max(np.real(Spectrum))
    # Plot_image(np.flipud(abs(Spectrum)), 'Spectrum', x_label='Z', y_label='Frequency')
    # Plot_image(np.flipud(np.real(Spectrum)), 'Spectrum', x_label='Z', y_label='Frequency')
    _, dv_High = signal.istft(Spectrum_high_f, fs, nperseg=n_perseg, noverlap=n_overlap, boundary='even')
    #########   Filter low frequencies
    # Spectrum_low_f = np.where(np.repeat(np.expand_dims(f, 1), np.shape(Spectrum)[1], axis=1) <= 0.005, Spectrum, 0)
    Spectrum_low_f = np.where(F <= K1, Spectrum, 0)
    _, dv_Low = signal.istft(Spectrum_low_f, fs, nperseg=n_perseg, noverlap=n_overlap, boundary='even')
    #########
    if Plotting == 1:
        Plot_image(np.flipud(abs(Spectrum)), 'Full Spectrum', [Min, Max], x_label='Z', y_label='Frequency', x=t, y=f,
                   Reverse_axis=0)
        Plot_image(np.flipud(abs(Spectrum_high_f)), 'Spectrum_high_f', [Min, Max], x_label='Z', y_label='Frequency',
                   x=t, y=f, Reverse_axis=0)
        Plot_image(np.flipud(abs(Spectrum_low_f)), 'Spectrum_low_f', [Min, Max], x_label='Z', y_label='Frequency', x=t,
                   y=f, Reverse_axis=0)
        # np.flipud
        plt.figure()
        plt.plot(dv)
        plt.plot(dv_High)
        plt.plot(dv_Low)
        plt.legend(['dv, full spectrum', 'dv, high frequencies', 'dv, low frequencies'])
        plt.savefig('LOGS.png')
        plt.show()
    return (dv_High, dv_Low, Spectrum)

def load_data_files(model_name, res_dir, models_path):
    filename = 'model' + str(model_name)
    print('loading file ', model_name)
    res_path = res_dir + '/' + model_name

    Models = rsf_to_np(res_path + '/model.rsf')
    Models = np.expand_dims(Models, axis=-1)

    Models_init = rsf_to_np(res_path + '/init_model.rsf')
    Models_init = np.expand_dims(Models_init, axis=-1)

    input_data = rsf_to_np(res_path + '/vsnaps.rsf')
    if len(input_data.shape) == 2:
        input_data = np.expand_dims(input_data, axis=0)
    input_data = np.moveaxis(input_data, 0, -1)

    input_data=input_data[:,:,:,-1]
    input_data = np.expand_dims(input_data, axis=-1)

    print(Models.shape)
    print(Models_init.shape)
    print(input_data.shape)
    return Models, Models_init, input_data

def save_data_file(model_name,  res_dir,savepath):
    a = model_name.split('model')
    model_number=a[1]

    tmp = res_dir + '/' + model_name + '/model_res' + model_number + '.hdf5'
    dx = load_file(tmp, 'dx');dx = float(dx)
    dz = load_file(tmp, 'dz');dz = float(dz)

    res_path = res_dir + '/' + model_name
    Models = rsf_to_np(res_path + '/model.rsf')
    Models = np.expand_dims(Models, axis=(0,-1))

    Models_init = rsf_to_np(res_path + '/init_model.rsf')
    Models_init = np.expand_dims(Models_init,axis=(0,-1))

    input_data = rsf_to_np(res_path + '/vsnaps.rsf')
    if len(input_data.shape) == 2:
        input_data = np.expand_dims(input_data, axis=(0,-1))
        Nch=1
    else:
        input_data=np.moveaxis(input_data, 0, -1)
        Nch=input_data.shape[-1]
        input_data = np.expand_dims(input_data, axis=0)

    #   Choose channel
    input_data=input_data[:,:,:,-1]
    input_data=np.expand_dims(input_data,axis=-1)

    FILENAME='augmented_marmousi_'+str(Nch)+'_it_' + model_number + '.hdf5'
    path = savepath +'/'+ FILENAME
    print('Saving file' + path)
    if os.path.exists(path):
        os.remove(path)
    f = h5py.File(path, 'a')
    f.create_dataset('models', data=Models)
    f.create_dataset('models_init', data=Models_init)
    f.create_dataset('input_data', data=input_data)
    f.create_dataset('dx', data=dx)
    f.create_dataset('dz', data=dz)
    f.close()
    return None

def return_CNN_target_size(res_dir,test_status,model_name):
    a=model_name.split('model')
    model_number=a[1]
    res_path = res_dir + '/' + model_name
    tmp = res_path + '/model_res' + model_number + '.hdf5'
    input_data=load_file(tmp,'result')
    input_data = np.expand_dims(input_data, axis=0)
    input_data = np.expand_dims(input_data, axis=-1)
    dx = load_file(tmp, 'dx');dx = float(dx)
    dz = load_file(tmp, 'dz');dz = float(dz)
    #########################
    #   cut certain region from data
    Nx=input_data.shape[1]
    Nz=input_data.shape[2]
    print('Nx=',Nx)
    print('Nz=',Nz)
    val=2
    sz_x = (548//val)*val
    sz_z = (60//val)*val
    # edges=math.floor((Nx-sz_x)/2)
    edges=math.floor((Nx-sz_x)/2)
    #   Crop edges with acquisition effects
    ix1=0;  ix2=sz_x
    #   Crop water out
    iz1=18
    iz2=iz1+sz_z
    if test_status==6:
        ix1=0;  ix2=sz_x
        iz1=0;  iz2=Nz
    input_data=input_data[:,ix1:ix2,iz1:iz2,:]
    #   fit to network Nz
    tmp=Resize_data_deep2(input_data,k_width=3)
    Nx_out=tmp.shape[1]
    Nz_out=tmp.shape[2]
    crop_pars={'output':[Nx_out,Nz_out],'ix1':ix1,'ix2':ix2
        ,'iz1':iz1,'iz2':iz2}
    return crop_pars

def save_data_file2(crop_pars,res_dir,test_status,savepath,rewrite_file,model_name):
    # test_status=1       #input – same,output-fixed number       Ok
    # test_status=2       # input – same,output- smoothed input
    # test_status=3     #input – output imresized, output-same.Encoder
    # test_status=4       # amplify inputs with depth coefficient(somewhat, gradient tapering)
    # test_status=5       # cut the data and square patches with gradient tapering
    # test_status=6       #   put all data with water
    out_shape=crop_pars['output']
    ix1=crop_pars['ix1']
    ix2=crop_pars['ix2']
    iz1=crop_pars['iz1']
    iz2=crop_pars['iz2']
    if test_status!=5:
        a=model_name.split('model')
        model_number=a[1]
        FILENAME='augmented_marmousi_10_it_' + model_number + '.npz'
        path = savepath +'/'+ FILENAME
        print('Saving file' + path)
        if os.path.exists(path) and rewrite_file==1:
            os.remove(path)
        if os.path.exists(path) and rewrite_file==0:
            print('skipped')
            return
        res_path = res_dir + '/' + model_name
        tmp = res_path + '/model_res' + model_number + '.hdf5'
        input_data=load_file(tmp,'result')
        input_data = np.expand_dims(input_data, axis=0)
        input_data = np.expand_dims(input_data, axis=-1)
        dx = load_file(tmp, 'dx');dx = float(dx)
        dz = load_file(tmp, 'dz');dz = float(dz)
        Models=rsf_to_np(res_path+'/model.rsf')     #!!!!!!!!!!!!!!!!!
        Models=np.expand_dims(Models,axis=0)
        Models=np.expand_dims(Models,axis=-1)
        Models_init = rsf_to_np(res_path + '/init_model.rsf')   #!!!!!!!!!!!!!1
        Models_init=np.expand_dims(Models_init,axis=0)
        Models_init=np.expand_dims(Models_init,axis=-1)
        output_data = np.zeros_like(Models)
        #   we create output for cnn with smoothing of the true model
        output_data[0,:,:,0]= F_smooth(Models[0,:,:,0], sigma_val=int(100 / dx))    #Vladimir recomendation
        # output_data[0,:,:,0]= F_smooth(Models[0,:,:,0], sigma_val=int(600 / dx))    #big smoothing
        output_data[0,:,:,0]= output_data[0,:,:,0]-Models_init[0,:,:,0]
        input_data [0,:,:,0]= input_data[0,:,:,0]-Models_init[0,:,:,0]
        #########################
        if test_status==0:
            hi=1
        if test_status==2:
            Nx_out=out_shape.shape[0]
            Nz_out=out_shape.shape[1]
            x=input_data[0,:,:,0]
            x=F_smooth(x, sigma_val=int(400/20))
            x=imresize(x,[Nx_out,Nz_out])
            x=np.expand_dims(x,axis=0)
            x=np.expand_dims(x,axis=-1)
            output_data_subsampled=x
        if test_status==3:
            Nx_out=out_shape.shape[0]
            Nz_out=out_shape.shape[1]
            output_data_subsampled=imresize(output_data[0,:,:,0],[Nx_out,Nz_out])
            output_data_subsampled=np.expand_dims(output_data_subsampled,axis=(0,-1))
        if test_status==4:
            Nx_out=out_shape.shape[0]
            Nz_out=out_shape.shape[1]
            sz_new=np.array([Nx_out,Nz_out], dtype=int)
            output_data_subsampled=imresize(output_data[0,:,:,0],sz_new,anti_aliasing=True)
            output_data_subsampled=np.expand_dims(output_data_subsampled,axis=0)
            output_data_subsampled=np.expand_dims(output_data_subsampled,axis=-1)
            xi=input_data[0,:,:,0]
            nx = xi.shape[0]
            nz = xi.shape[1]
            zz = np.arange(nz)*dz/1000
            zz = np.tile(zz,(nx, 1))          
            xi=xi*zz
            input_data=np.expand_dims(xi,axis=0)
            input_data=np.expand_dims(xi,axis=-1)
        if test_status==6:
            tmp=Resize_data_deep2(input_data)
            Nx_out=tmp.shape[1];    Nz_out=tmp.shape[2]
            sz_new=np.array([Nx_out,Nz_out], dtype=int)
            output_data_subsampled=imresize(output_data[0,:,:,0],sz_new,anti_aliasing=True)
            output_data_subsampled=np.expand_dims(output_data_subsampled,axis=0 )
            output_data_subsampled=np.expand_dims(output_data_subsampled,axis=-1)
            return None
        ##########################
        input_data=input_data[:,ix1:ix2,iz1:iz2,:]
        output_data=output_data[:,ix1:ix2,iz1:iz2,:]
        Models=Models[:,ix1:ix2,iz1:iz2,:]
        Models_init=Models_init[:,ix1:ix2,iz1:iz2,:]
        Nx_out=out_shape[0];  Nz_out=out_shape[1]
        sz_new=np.array([Nx_out,Nz_out], dtype=int)
        ##########################  Downsample output data
        # output_data_subsampled=imresize(output_data[0,:,:,0],sz_new,anti_aliasing=True)
        # output_data_subsampled=np.expand_dims(output_data_subsampled,axis=0 )
        # output_data_subsampled=np.expand_dims(output_data_subsampled,axis=-1)
        ##########################
        # print('Input data=', input_data.shape)
        # print('Output data=',output_data_subsampled.shape)
        np.savez(path,input_data=input_data,output_data=output_data,
        models_init=Models_init,models=Models,dx=dx,dz=dz)
    else:
        if model_number=='_Marmousi':
            FILENAME='augmented_marmousi_10_it_'+model_number+'_'+str(i)+'.npz'
        elif model_number=='_Overthrust':
            FILENAME='augmented_marmousi_10_it_'+model_number+'_'+str(i)+'.npz'
        else:
            FILENAME='augmented_marmousi_10_it_'+model_number+'_'+str(i)+'.npz'
            # FILENAME='augmented_marmousi_'+str(Nch)+'_it_' + model_number + '.npz'
        
        path = savepath +'/'+ FILENAME
        if os.path.exists(path) and append_new_files_flag==0:
            os.remove(path)
        if os.path.exists(path) and append_new_files_flag==1:
            return

        print(model_name)
        a=model_name.split('model')
        model_number=a[1]

        res_path = res_dir + '/' + model_name
        tmp = res_path + '/model_res' + model_number + '.hdf5'
        input_data = rsf_to_np(res_path + '/vsnaps.rsf')
        if len(input_data.shape) == 2:
            input_data = np.expand_dims(input_data, axis=(0,-1))
            Nch=1
        else:
            input_data=np.moveaxis(input_data, 0, -1)
            Nch=input_data.shape[-1]
            input_data = np.expand_dims(input_data, axis=0)

        #   Choose channel
        input_data=input_data[:,:,:,-1]
        input_data=np.expand_dims(input_data,axis=-1)

        dx = load_file(tmp, 'dx');dx = float(dx)
        dz = load_file(tmp, 'dz');dz = float(dz)
        Models=rsf_to_np(res_path+'/model.rsf')
        Models=np.expand_dims(Models,axis=(0,-1))
        Models_init = rsf_to_np(res_path + '/init_model.rsf')
        Models_init = np.expand_dims(Models_init,axis=(0,-1))
        #########################
        output_data = np.zeros_like(Models)
        #   we create output for cnn with smoothing of the true model
        output_data[0,:,:,0]= F_smooth(Models[0,:,:,0], sigma_val=int(140 / dx))
        output_data[0,:,:,0]= output_data[0,:,:,0]-Models_init[0,:,:,0]
        input_data [0,:,:,0]= input_data[0,:,:,0]- Models_init[0,:,:,0]
        #   cut certain region from data
        Nx=Models.shape[1]
        Nz=Models.shape[2]
        print('Nx=',Nx)
        print('Nz=',Nz)
        sz_x =int(np.floor((Nx+1)/2)*2)
        sz_z = 142
        
        #   Crop edges with acquisition effects
        ix1=0
        ix2=sz_x
        input_data=input_data[:,ix1:ix2,:,:]
        output_data=output_data[:,ix1:ix2,:,:]
        Models=Models[:,ix1:ix2,:,:]
        Models_init=Models_init[:,ix1:ix2,:,:]
        print('Stage 1=',Models.shape)

        #   Crop water out
        iz1=18
        iz2=Nz
        input_data=input_data[:,:,iz1:iz2, :]
        output_data=output_data[:,:,iz1:iz2, :]
        Models=Models[:,:,iz1:iz2,:]
        Models_init=Models_init[:,:,iz1:iz2,:]
        print('Stage 2=',Models.shape)
        
        #   fit to network Nz
        iz1=0
        iz2=sz_z
        input_data=input_data[:,:,iz1:iz2, :]
        output_data=output_data[:,:,iz1:iz2, :]
        Models=Models[:,:,iz1:iz2,:]
        Models_init=Models_init[:,:,iz1:iz2,:]
        print('Stage 3=',Models.shape)
        Nx=Models.shape[1]
        Nz=Models.shape[2]
        
        Np=int(np.floor(Nx/Nz))
        input_shape=(Nz,Nz,1)
        [Nx_out,Nz_out]=Get_cnn_output_shape(input_shape)
        sz_new=np.array([Nx_out,Nz_out], dtype=int)
        zz = np.arange(Nz)*dz
        zz = np.tile(zz,(Nz, 1))
        zz =np.expand_dims(zz,axis=0)
        zz =np.expand_dims(zz,axis=-1)
        for i in range(Np):
            ix1=i*Nz;   ix2=(i+1)*Nz;
            Patch_x=input_data[:,ix1:ix2,:,:]
            Patch_t=output_data[:,ix1:ix2,:,:]
            Patch_Models_init=Models_init[:,ix1:ix2,:,:]
            Patch_Models=Models[:,ix1:ix2,:,:]
            Patch_t=imresize(Patch_t[0,:,:,0],sz_new,anti_aliasing=True)
            Patch_t=np.expand_dims(Patch_t,axis=(0,-1))
            ##########################
            Patch_x=Patch_x*zz
            ##########################
            if model_number=='_Marmousi':
                FILENAME='augmented_marmousi_10_it_'+model_number+'_'+str(i)+'.npz'
            elif model_number=='_Overthrust':
                FILENAME='augmented_marmousi_10_it_'+model_number+'_'+str(i)+'.npz'
            else:
                FILENAME='augmented_marmousi_10_it_'+model_number+'_'+str(i)+'.npz'
            # FILENAME='augmented_marmousi_'+str(Nch)+'_it_' + model_number + '.npz'
            path = savepath +'/'+ FILENAME
            print('Saving file' + path)
            if os.path.exists(path) and append_new_files_flag==0:
                os.remove(path)
            if os.path.exists(path) and append_new_files_flag==1:
                return
            
            print('Patch_x',Patch_x.shape)
            print('Patch_t',Patch_t.shape)
            print('Patch_Models_init',Patch_Models_init.shape)
            print('Patch_Models',Patch_Models.shape)
            np.savez(path,input_data=Patch_x,output_data=Patch_t,
            models_init=Patch_Models_init,models=Patch_Models,dx=dx,dz=dz)
    return None

def save_data_file3(crop_pars,res_dir,savepath,model_name,flag_what_to_record='dv_input',
        rewrite_file=0,init_model_type='individual',path_to_save_pictures='./pictures2/'):
    # out_shape=crop_pars['output']
    # ix1=crop_pars['ix1'];   ix2=crop_pars['ix2']
    # iz1=crop_pars['iz1'];   iz2=crop_pars['iz2']
    save_full_file=1
    model_number=model_name.split('model')[1]
    model_number=model_number.split('/')[0]
    FILENAME='augmented_marmousi_10_it_' + model_number
    path = savepath +'/'+ FILENAME+ '.npz'
    if save_full_file==1:
        if os.path.exists(path) and rewrite_file==1:
            print('file '+path+' exists, rewriting')
            os.remove(path)
        if os.path.exists(path) and rewrite_file==0:
            print('file '+path+' exists, no rewriting, skipped processing')
            return
    ##########################
    res_path=res_dir+'/'+model_name
    if init_model_type=='individual':
        path_to_result_file=res_path+'model_res_ideal.hdf5'
        path_to_init_file=res_path+'models_init_ideal.rsf'
        path_to_result_file=res_path+'models_res_ideal_better_guess.hdf5'
        path_to_init_file=res_path+'models_init_ideal_better_guess.rsf'
    elif init_model_type=='same':
        path_to_result_file=res_path+'model_res' + model_number + '.hdf5'
        path_to_init_file=res_path+'init_model.rsf'
    elif init_model_type=='same_new_params':
        path_to_result_file=res_path+'model_res_new_params.hdf5'
        path_to_init_file=res_path+'init_model.rsf'
    elif init_model_type=='same_new_params_50_iter':
        path_to_result_file=res_path+'model_res_iter_50.hdf5'
        path_to_init_file=res_path+'init_model.rsf'
    input_data=load_file(path_to_result_file,'result')
    # Plot_image(input_data.T,Show_flag=0,Save_flag=1,Save_pictures_path='./',Title='input_data')
    input_data = np.expand_dims(input_data, axis=0)
    input_data = np.expand_dims(input_data, axis=-1)
    dx = load_file(path_to_result_file, 'dx');dx = float(dx)
    dz = load_file(path_to_result_file, 'dz');dz = float(dz)
    if dx==50:
        Models=rsf_to_np(res_path+'vel.rsf')
        Models_init = rsf_to_np(path_to_init_file)
    elif dx==25:
        Models_init = load_file(path_to_result_file,'models_init')
        Models=load_file(path_to_result_file,'models')
    iz1=18; iz2=-1; ix1=0;  ix2=-1;
    Models=np.expand_dims(Models,axis=0);Models=np.expand_dims(Models,axis=-1)
    Models_init=np.expand_dims(Models_init,axis=0);Models_init=np.expand_dims(Models_init,axis=-1)
    output_data = np.zeros_like(Models);output_data_smooth_100=np.zeros_like(Models)
    #   we create output for cnn with smoothing of the true model
    output_data[0,:,:,0]= F_smooth(Models[0,:,:,0], sigma_val=int(300 / dx))    #Vladimir recomendation
    output_data[0,:,:,0]= output_data[0,:,:,0]-Models_init[0,:,:,0]
    output_data_smooth_100[0,:,:,0]= F_smooth(Models[0,:,:,0], sigma_val=int(100 / dx))    #Vladimir recomendation
    output_data_smooth_100[0,:,:,0]= output_data_smooth_100[0,:,:,0]-Models_init[0,:,:,0]
    input_data [0,:,:,0]= input_data[0,:,:,0]-Models_init[0,:,:,0]
    input_data=input_data[:,ix1:ix2,    iz1:iz2,:]
    output_data=output_data[:,ix1:ix2,  iz1:iz2,:]
    output_data_smooth_100=output_data_smooth_100[:,ix1:ix2,  iz1:iz2,:]
    Models=Models[:,ix1:ix2,            iz1:iz2,:]
    Models_init=Models_init[:,ix1:ix2,  iz1:iz2,:]
    output_data_smooth_100,scaler_t_smooth_100=scaling_data_01(output_data_smooth_100,preconditioning=False)
    if os.path.exists(os.path.join(res_path,'grads.rsf')):
        grads=rsf_to_np(os.path.join(res_path,'grads.rsf'));grads=grads[:,ix1:ix2,iz1:iz2]
    ##########################  scale,save file
    if flag_what_to_record==0:
        np.savez(path,input_data=input_data,output_data=output_data,
            models_init=Models_init,models=Models,dx=dx,dz=dz)
    elif flag_what_to_record=='dv_input':
        input_data_real_amplitudes=input_data
        output_data_real_amplitudes=output_data
        input_data,scaler_x=scaling_data_01(input_data,preconditioning=False,
            visualize_scaling_results=0,save_pictures_path='./pictures/'+FILENAME+'.png')
        output_data,scaler_t=scaling_data_01(output_data,preconditioning=False)
        np.savez(path,input_data=input_data,output_data=output_data,
            models_init=Models_init,models=Models,dx=dx,dz=dz,
            input_data_real_amplitudes=input_data_real_amplitudes,
            output_data_real_amplitudes=output_data_real_amplitudes,
            scaler_x=scaler_x,scaler_t=scaler_t)
    elif flag_what_to_record=='multichannel_input':
        data=[];    titles=[]
        for i in range(len(grads)):
            titles.append(FILENAME+' ,grad'+str(i))
            data.append(grads[i,:,:].squeeze())
        data.append(input_data.squeeze())
        titles.append(FILENAME+' ,dv')
        # data.append(output_data.squeeze())
        # titles.append(FILENAME+' ,t')
        ##########################  check gradients channel
        flag_plot_data=0;   
        Preconditioning_flag=False
        cnn_input_data=np.empty((len(data),data[0].shape[0],data[0].shape[1])); cnn_input_data_scaled=np.empty_like(cnn_input_data)
        if Preconditioning_flag==False:
            scaler_x=np.empty((len(data),2))
        else:
            scaler_x=np.empty((len(data),4))
        for k in range(len(data)):
            cnn_input_data[k,:,:]=data[k]
            cnn_input_data_scaled[k,:,:],scaler_x[k,:]=scaling_data_01(data[k],preconditioning=Preconditioning_flag,
                visualize_scaling_results=0)
        if flag_plot_data==1:
            fig=plt.figure(); # fig.suptitle(FILENAME)
            # fig.set_size_inches(10,19); 
            fig.set_size_inches(10,120); 
            gs = fig.add_gridspec(nrows=cnn_input_data.shape[0],ncols=1,hspace=0.2,wspace=0)
            axs = gs.subplots(sharex=True,sharey=True)
            for i in range(cnn_input_data.shape[0]):
                ax=axs[i];    pt=ax.imshow(cnn_input_data[i,::].T)
                divider = make_axes_locatable(ax);  cax = divider.append_axes("right", size="5%", pad=0.05)
                fig.colorbar(pt,cax=cax);     
                ax.title.set_text(titles[i])
            os.makedirs(path_to_save_pictures,exist_ok=True)
            save_pictures_path=os.path.join(path_to_save_pictures,FILENAME+'_not_scaled.png')
            print('saving picture to '+save_pictures_path)
            plt.savefig(save_pictures_path,bbox_inches='tight')
            plt.show()  # plt.show(block=False)
            plt.close()
            ####
            fig=plt.figure(); # # fig.suptitle(FILENAME)
            # fig.set_size_inches(10,19); 
            fig.set_size_inches(10,120); 
            gs = fig.add_gridspec(nrows=cnn_input_data.shape[0],ncols=1,hspace=0.2,wspace=0)
            axs = gs.subplots(sharex=True,sharey=True)
            for i in range(cnn_input_data.shape[0]):
                ax=axs[i];    pt=ax.imshow(cnn_input_data_scaled[i,::].T)
                divider = make_axes_locatable(ax);  cax = divider.append_axes("right", size="5%", pad=0.05)
                fig.colorbar(pt,cax=cax);     
                ax.title.set_text(titles[i])
            os.makedirs(path_to_save_pictures,exist_ok=True)
            save_pictures_path=os.path.join(path_to_save_pictures,FILENAME+'_scaled.png')
            print('saving picture to '+save_pictures_path)
            plt.savefig(save_pictures_path,bbox_inches='tight')
            plt.show()  # plt.show(block=False)
            plt.close()
        ########
        if flag_plot_data==0 and save_full_file==1:
            print('Saving file' + path)
            if cnn_input_data_scaled.ndim==3:
                cnn_input_data_scaled=np.expand_dims(cnn_input_data_scaled,axis=-1)
            output_data_scaled,scaler_t=scaling_data_01(output_data,preconditioning=False)
            np.savez(path,input_data=cnn_input_data_scaled,output_data=output_data_scaled,
                models_init=Models_init,models=Models,dx=dx,dz=dz,
                scaler_x=scaler_x,scaler_t=scaler_t,titles=titles)
        elif flag_plot_data==0 and save_full_file==0:
            with open(path,'rb') as f:
                data=np.load(f,allow_pickle=True)
                input_data=data['input_data'];   #print('data[input_data]=',data['input_data'].shape)
                output_data=data['output_data']
                models_init=data['models_init']
                models=data['models']
                scaler_x=data['scaler_x']
                scaler_t=data['scaler_t']
                titles=data['titles']
                dz=data['dz'];  dx=data['dx'];
                data.close()
            np.savez(path,input_data=input_data,output_data=output_data_smooth_100,
                models_init=Models_init,models=Models,dx=dx,dz=dz,
                scaler_x=scaler_x,scaler_t=scaler_t,titles=titles,
                output_data_smooth_100=output_data_smooth_100,scaler_t_smooth_100=scaler_t_smooth_100)
            # with open(path,'rb') as f:
            #     data=np.load(f,allow_pickle=True)
            #     input_data=data['input_data'];   #print('data[input_data]=',data['input_data'].shape)
            #     output_data=data['output_data']
            #     models_init=data['models_init']
            #     models=data['models']
            #     scaler_x=data['scaler_x']
            #     scaler_t=data['scaler_t']
            #     titles=data['titles']
            #     dz=data['dz'];  dx=data['dx'];
            #     data.close()
    return None

def save_data_file_2_ch(crop_pars,res_dir,savepath,append_new_files_flag,full_path_to_file):
    model_name=full_path_to_file[0].split('/')[-1]
    out_shape=crop_pars['output']
    ix1=crop_pars['ix1']
    ix2=crop_pars['ix2']
    iz1=crop_pars['iz1']
    iz2=crop_pars['iz2']
    a=model_name.split('model')
    model_number=a[1]
    FILENAME='augmented_marmousi_10_it_' + model_number + '.npz'
    path = savepath +'/'+ FILENAME
    print('Saving file' + path)
    if os.path.exists(path) and append_new_files_flag==0:
        os.remove(path)
    if os.path.exists(path) and append_new_files_flag==1:
        print('skipped')
        return
    res_path = res_dir + '/' + model_name
    tmp = res_path + '/model_res' + model_number + '.hdf5'
    dx = load_file(tmp, 'dx');dx = float(dx)
    dz = load_file(tmp, 'dz');dz = float(dz)
    print('full_path_to_file',full_path_to_file)
    Models=rsf_to_np(res_path+'/model.rsf')
    Models=np.expand_dims(Models,axis=0)
    Models=np.expand_dims(Models,axis=-1)
    Models_init = rsf_to_np(res_path + '/init_model.rsf')
    Models_init=np.expand_dims(Models_init,axis=0)
    Models_init=np.expand_dims(Models_init,axis=-1)
    output_data = np.zeros_like(Models)
    #   we create output for cnn with smoothing of the true model
    output_data[0,:,:,0]= F_smooth(Models[0,:,:,0], sigma_val=int(100 / dx))    #Vladimir recomendation
    # output_data[0,:,:,0]= F_smooth(Models[0,:,:,0], sigma_val=int(600 / dx))    #big smoothing
    output_data[0,:,:,0]= output_data[0,:,:,0]-Models_init[0,:,:,0]
    #########################
    input_data_1_ch=load_file(tmp,'result')
    input_data_1_ch = np.expand_dims(input_data_1_ch, axis=0)
    input_data_1_ch = np.expand_dims(input_data_1_ch, axis=-1)
    input_data_2_ch=load_file(tmp,'result')
    input_data_2_ch = np.expand_dims(input_data_2_ch, axis=0)
    input_data_2_ch = np.expand_dims(input_data_2_ch, axis=-1)
    input_data_1_ch=input_data_1_ch-Models_init
    input_data_2_ch=input_data_2_ch-Models_init
    ##########################
    input_data=np.concatenate((input_data_1_ch,input_data_2_ch),axis=-1)
    input_data=input_data[:,ix1:ix2,iz1:iz2,:]
    output_data=output_data[:,ix1:ix2,iz1:iz2,:]
    Models=Models[:,ix1:ix2,iz1:iz2,:]
    Models_init=Models_init[:,ix1:ix2,iz1:iz2,:]
    Nx_out=out_shape[0];  Nz_out=out_shape[1]
    sz_new=np.array([Nx_out,Nz_out], dtype=int)
    output_data_subsampled=imresize(output_data[0,:,:,0],sz_new,anti_aliasing=True)
    output_data_subsampled=np.expand_dims(output_data_subsampled,axis=0 )
    output_data_subsampled=np.expand_dims(output_data_subsampled,axis=-1)
    ##########################
    # print('Input data=', input_data.shape)
    # print('Output data=',output_data_subsampled.shape)
    np.savez(path,input_data=input_data,output_data=output_data_subsampled,
        models_init=Models_init,models=Models,dx=dx,dz=dz)
    return None

def save_data_file_in_full_size(crop_pars,res_dir,test_status,savepath,model_name,append_new_files_flag):
    out_shape=crop_pars['output']
    ix1=crop_pars['ix1']
    ix2=crop_pars['ix2']
    iz1=crop_pars['iz1']
    iz2=crop_pars['iz2']
    a=model_name.split('model')
    model_number=a[1]
    FILENAME='augmented_marmousi_10_it_' + model_number + '.npz'
    path = savepath +'/'+ FILENAME
    print('Saving file' + path)
    if os.path.exists(path) and append_new_files_flag==0:
        os.remove(path)
    if os.path.exists(path) and append_new_files_flag==1:
        print('skipped')
        return
    res_path = res_dir + '/' + model_name
    tmp = res_path + '/model_res' + model_number + '.hdf5'
    input_data=load_file(tmp,'result')
    input_data = np.expand_dims(input_data, axis=0)
    input_data = np.expand_dims(input_data, axis=-1)
    dx = load_file(tmp, 'dx');dx = float(dx)
    dz = load_file(tmp, 'dz');dz = float(dz)
    Models=rsf_to_np(res_path+'/model.rsf')
    Models=np.expand_dims(Models,axis=0)
    Models=np.expand_dims(Models,axis=-1)
    Models_init = rsf_to_np(res_path + '/init_model.rsf')
    Models_init=np.expand_dims(Models_init,axis=0)
    Models_init=np.expand_dims(Models_init,axis=-1)
    input_data=input_data[:,ix1:ix2,iz1:,:]
    Models=Models[:,ix1:ix2,iz1:,:]
    Models_init=Models_init[:,ix1:ix2,iz1:,:]
    output_data = np.zeros_like(Models)
    #   we create output for cnn with smoothing of the true model
    output_data[0,:,:,0]= F_smooth(Models[0,:,:,0], sigma_val=int(100 / dx))    #Vladimir recomendation
    # output_data[0,:,:,0]= F_smooth(Models[0,:,:,0], sigma_val=int(600 / dx))    #big smoothing
    output_data[0,:,:,0]= output_data[0,:,:,0]-Models_init[0,:,:,0]
    input_data [0,:,:,0]= input_data[0,:,:,0]-Models_init[0,:,:,0]
    Nx_out=out_shape[0];  Nz_out=out_shape[1]
    tmp=Resize_data_deep2(input_data,k_width=3)
    Nx_out=tmp.shape[1]
    Nz_out=tmp.shape[2]
    sz_new=np.array([Nx_out,Nz_out],dtype=int)
    output_data_subsampled=imresize(output_data[0,:,:,0],sz_new,anti_aliasing=True)
    output_data_subsampled=np.expand_dims(output_data_subsampled,axis=0 )
    output_data_subsampled=np.expand_dims(output_data_subsampled,axis=-1)
    ##########################
    np.savez(path,input_data=input_data,output_data=output_data_subsampled,
    models_init=Models_init,models=Models,dx=dx,dz=dz)
    return None

def check_npz_file(data_path,file):
    file_path=data_path+'/'+file
    # os.stat(file_path).st_size
    if not zipfile.is_zipfile(file_path):
        print('delete '+file_path)
        cmd('rm -r '+file_path)
    # if not os.path.exists(file_path):
    #     print('delete '+file_path)
    #     cmd('rm -r '+file_path)
    return None

def check_folder(res_path,model_name):
    # print(os.getcwd())
    res_name = fnmatch.filter(os.listdir(res_path), 'model_res' + '*')
    # print(res_name[0])
    Output=0
    if res_name:
        Output=1
        makecheck=0
        if makecheck==1:
            a=model_name.split('model')
            model_number=a[1]
            tmp = res_path + '/model_res' + model_number + '.hdf5'
            print(os.getcwd())
            input_data=load_file(tmp,'result')
            dx = load_file(tmp, 'dx');dx = float(dx)
            dz = load_file(tmp, 'dz');dz = float(dz)
            print(res_path+'/model.rsf')
            Models=rsf_to_np(res_path+'/model.rsf')
            Models_init = rsf_to_np(res_path + '/init_model.rsf')
            if Models.shape!=[551,78]:
                cmd('rm -r '+res_path)
            # if input_data.shape!=Models_init.shape:
            #     print('input_data.shape!=Models_init.shape  DELETING '+res_path)
            #     cmd('rm -r '+res_path)
            #     a=1
            # else:
            #     input_data=input_data-Models_init
            #     if input_data.shape!=Models_init.shape:
            #         print('input_data.shape!=Models_init.shape  DELETING '+res_path)
            #         cmd('rm -r '+res_path)
        # return model_name
        # return None
    else:
        print('deleting '+res_path)
        cmd('rm -r '+res_path)
        # return None
    return Output

def check_folder2(savepath_for_dataset,model_name):
    a = model_name.split('model')
    model_number=a[1]
    FILENAME='augmented_marmousi_10_it_'+model_number+'.npz'
    # file_list = fnmatch.filter(os.listdir(savepath_for_dataset),'augmented_marmousi_10_it_' + '*')
    if os.path.isfile(savepath_for_dataset+'/'+FILENAME):
        Output=0
        return None
    else:
        print(model_name)
        Output=1
        return model_name

def check_folder3(res_dir,model_name):
    res_path=res_dir+'/'+model_name
    res_name = fnmatch.filter(os.listdir(res_path), 'model_res' + '*')
    Output=0
    if res_name:
        Output=1
        makecheck=0
        if makecheck==1:
            a=model_name.split('model')
            model_number=a[1]
            tmp = res_path + '/model_res' + model_number + '.hdf5'
            input_data=load_file(tmp,'result')
            # print('input_data.shape=',input_data.shape)
            # dx = load_file(tmp, 'dx');dx = float(dx)
            # dz = load_file(tmp, 'dz');dz = float(dz)
            # print(res_path+'/model.rsf')
            # Models=rsf_to_np(res_path+'/model.rsf')
            # Models_init = rsf_to_np(res_path + '/init_model.rsf')
            if input_data.shape!=(551,78):
                print('deleting '+res_path+'=',input_data.shape)
                cmd('rm -r '+res_path)
                # exit()

            # if input_data.shape!=Models_init.shape:
            #     print('input_data.shape!=Models_init.shape  DELETING '+res_path)
            #     cmd('rm -r '+res_path)
            #     a=1
            # else:
            #     input_data=input_data-Models_init
            #     if input_data.shape!=Models_init.shape:
            #         print('input_data.shape!=Models_init.shape  DELETING '+res_path)
            #         cmd('rm -r '+res_path)
        second_check=1
        if second_check==1:
            res_name = fnmatch.filter(os.listdir(res_path), 'init_model' + '*')
            if (not res_name)==True:
                print('deleting because of lack of init model '+res_path)
                cmd('rm -r '+res_path)        
                return None
        return model_name
        # return None
    else:
        print('deleting '+res_path)
        cmd('rm -r '+res_path)
        return None

def check_folder4(res_dir,model_name,init_model_type='same'):
    res_path=res_dir+'/'+model_name;    print('checking ',res_path)
    # if os.path.exists(res_path+'models_init_ideal.rsf') and os.path.exists(res_path+'model_res_ideal.hdf5'):
    #     return model_name
    if init_model_type=='individual':
        if os.path.exists(res_path+'models_init_ideal_better_guess.rsf') and os.path.exists(res_path+'models_res_ideal_better_guess.hdf5'):
            return model_name
    if init_model_type=='same_new_params':
        res_name = fnmatch.filter(os.listdir(res_path),'model_res_new_params*');    
        if res_name!=[]:
            res_name=res_name[0]
            if os.path.exists(res_path+'init_model.rsf') and os.path.exists(res_path+res_name):
                return model_name
    if init_model_type=='same_new_params_50_iter':
        if os.path.exists(res_path+'init_model.rsf') and os.path.exists(res_path+'model_res_iter_50.hdf5'):
            return model_name
    return None

def clean_folder(res_dir,model_name):
    res_path=res_dir+'/'+model_name;    
    print('checking ',res_path)
    if os.path.exists(res_path+'models_init_ideal.hdf5'):
        print('deleting '+res_path+'models_init_ideal.hdf5')
        cmd((f"rm {res_path+'models_init_ideal.rsf'}"))
    if os.path.exists(res_path+'model_res_ideal.hdf5'):
        print('deleting '+res_path+'model_res_ideal.hdf5')
        cmd((f"rm {res_path+'model_res_ideal.hdf5'}"))
    else:
        res_name = fnmatch.filter(os.listdir(res_path),'model_res_ideal*')
        if not res_name==[]:
            if os.path.exists(res_path+res_name[0]):
                print('deleting '+res_path+res_name[0])
                cmd((f"rm {res_path+res_name[0]}"))
    return None

def check_dataset_file(data_path,file):
    NAME=data_path+'/'+file
    with open(NAME, 'rb') as f:
        # data=np.load(f)
        data=np.load(f,allow_pickle=True)
        x=data['input_data']
        t=data['output_data']
        data.close()
    if x.shape!=(1,548,60,1):
        print('deleting '+NAME+'=',x.shape)
        cmd('rm '+NAME)
    return None

def filter_directories_only(basepath, fname):
    path = os.path.join(basepath, fname)
    if os.path.isdir(path):
        print(fname)
        return fname
    else:
        return

def F_construct_lists_for_multichannel_data_loading(dir_ch_1,dir_ch_2,fld1,file_):
    if file_ in fld1:
        filepath_ch_1=dir_ch_1+'/'+file_
        filepath_ch_2=dir_ch_2+'/'+file_
        res_name = fnmatch.filter(os.listdir(filepath_ch_1), 'model_res' + '*')
        if res_name:
            if os.path.exists(filepath_ch_1+'/model.rsf'):
                if os.path.exists(filepath_ch_1+'/init_model.rsf'):
                    return filepath_ch_1,filepath_ch_2
                else:
                    return
            else:
                return
    else:
        return

def check_results(Save_pictures_path,res_dir,model_name):
    a=model_name.split('model')
    model_number=a[1]
    res_path = res_dir + '/' + model_name
    tmp = res_path + '/model_res' + model_number + '.hdf5'
    print(os.getcwd())
    input_data=load_file(tmp,'result')
    dx = load_file(tmp, 'dx');dx = float(dx)
    dz = load_file(tmp, 'dz');dz = float(dz)
    print(res_path+'/model.rsf')
    Models=rsf_to_np(res_path+'/model.rsf')
    Models_init = rsf_to_np(res_path + '/init_model.rsf')
    input_data=input_data-Models_init

    f_name='augmented_marmousi_10_it_'+model_number
    Plot_image(Models.T,Show_flag=0,Save_flag=1,dx=dz, dy=dz,Save_pictures_path=Save_pictures_path,Title=f_name+'_model',Aspect='equal')
    Plot_image(input_data.T,Show_flag=0,Save_flag=1,dx=dz, dy=dz,Save_pictures_path=Save_pictures_path,Title=f_name+'_inp',Aspect='equal')
    print('model.shape=',Models.shape)
    return Models
    # pts=NAME.split('/')
    # f_name=pts[-1]
    # f_name=f_name[:-4]
    # with open(NAME, 'rb') as f:
    #     data=np.load(f)
    #     M0=data['models'][0,:,:,0]
    #     # M1=data['input_data'][0,:,:,0]
    #     M2=data['output_data'][0,:,:,0]
    #     data.close()
    # Plot_image(M0.T,Show_flag=0,Save_flag=1,dx=dz, dy=dz,Save_pictures_path=Save_pictures_path,Title=f_name,Aspect='equal')
    # Plot_image(M2.T,Show_flag=0,Save_flag=1,dx=dz, dy=dz,Save_pictures_path=Save_pictures_path,Title=f_name+'M2',Aspect='equal')
    # print('model.shape=',M0.shape)
    # print('target.shape=',M2.shape)
    return None

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
    # plt.figure()
    # plt.plot(true_log)
    # plt.plot(smoothed_log)
    # plt.plot(final_log)
    # plt.title('Curve')
    # Name='./pictures_for_check/logs.png'
    # plt.savefig(Name,dpi=300)
    # plt.show(block=False)
    # print('plotting ',Name)
    # plt.close()
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


    # Plot_image(Models,Show_flag=0,Save_flag=1,Title='Models'+'_r2_',
    #         Save_pictures_path='./pictures_for_check',dx=25,dy=25,c_lim=[1500,4500])
    # Plot_image(Models_smoothed,Show_flag=0,Save_flag=1,Title='Models_smoothed'+'_r2_'+str(F_r2(Models_smoothed,Models)),
    #         Save_pictures_path='./pictures_for_check',dx=25,dy=25,c_lim=[1500,4500])
    # Plot_image(init,Show_flag=0,Save_flag=1,Title='init'+'_r2_'+str(F_r2(init,Models)),
    #         Save_pictures_path='./pictures_for_check',dx=25,dy=25,c_lim=[1500,4500])
    # Plot_image(init2,Show_flag=0,Save_flag=1,Title='init2'+'_r2_'+str(F_r2(init2,Models)),
    #         Save_pictures_path='./pictures_for_check',dx=25,dy=25,c_lim=[1500,4500])
    # Plot_image(initial_model_lin_grad,Show_flag=0,Save_flag=1,Title='initial_model_lin_grad'+'_r2_'+str(F_r2(initial_model_lin_grad,Models)),
    #         Save_pictures_path='./pictures_for_check',dx=25,dy=25,c_lim=[1500,4500])
    # Plot_image(Models-Models_smoothed,Show_flag=0,Save_flag=1,Title='Models_diff',
    #         Save_pictures_path='./pictures_for_check',dx=25,dy=25)

    # plt.figure()
    # plt.plot(true_log_original)
    # plt.plot(smoothed_log)
    # plt.plot(final_log)
    # plt.plot(Models_smoothed[:,log_idx],'k--')
    # plt.plot(init2[:,log_idx],'k--')
    # plt.title('Curve')
    # Name='./pictures_for_check/logs.png'
    # plt.savefig(Name,dpi=300)
    # plt.show(block=False)
    # print('plotting ',Name)
    # plt.close()

    a=1
    return init2.T

def F_initial_vz_model2(shape,dz):
    nx = shape[0]
    nz = shape[1]
    zz = np.arange(nz) * dz
    zz = zz - 340
    zz = np.tile(zz, (nx, 1))
    zz[zz < 0] = 0
    init = 1.5 + 0.9e-3 * zz
    #init[:,0:18]=1.5
    return init

def F_initial_vz_model_from_shape(shape,dz):
    nx = shape[0]
    nz = shape[1]
    zz = np.arange(nz) * dz
    zz = np.tile(zz,(nx, 1))
    init = 1500 + 0.9 * zz
    return init

def str_list_to_int_list(str_list):
    n = 0
    while n < len(str_list):
        aa = (str_list[n]).split('model')
        str_list[n] = int(aa[1])
        n += 1
    return (str_list)

def wavenumber_filter(i_m, Nm, Nx, Nz, dz, Angles, filter, perturbation_model, background_model):
    print('i_m=', i_m)
    velocity_model_high = np.zeros_like(perturbation_model)
    velocity_model_low = np.zeros_like(perturbation_model)
    for i_i in range(Nx):
        dv = perturbation_model[i_i, :]
        Vo = background_model[i_i, :]
        nz = len(dv)
        v_water = 1500;
        fs = 1 / dz;
        Frequency_limits = [0, filter['fMax'] / v_water];
        half_offset = filter['max_offset'] / 2;
        cosin = (np.cos(Angles / 180 * np.pi));
        #########   Calculate spectrum
        if (nz % 2) == 1:  # even
            n_perseg = np.shape(dv)[0]
        else:  # odd
            n_perseg = np.shape(dv)[0] - 1
        n_overlap = n_perseg - 1
        t_true = np.arange(np.shape(dv)[0]) * dz
        f, t, Spectrum = signal.stft(dv, fs, nperseg=n_perseg, noverlap=n_overlap, boundary='even')
        #########   Allocate additional matrices
        tmp = cosin / Vo / 1000
        k1 = 2 * filter['fMax'] * (cosin) / Vo / 1000;
        K1 = np.repeat(k1, np.shape(Spectrum)[0], axis=0)
        F = np.repeat(np.expand_dims(f, 1), np.shape(Spectrum)[1], axis=1)
        Min = np.min(abs(Spectrum));
        Max = 0.2 * np.max(abs(Spectrum))
        #########   Filter high frequencies
        Spectrum_high_f = np.where(F >= K1, Spectrum, 0)
        _, dv_High = signal.istft(Spectrum_high_f, fs, nperseg=n_perseg, noverlap=n_overlap, boundary='even')
        Spectrum_low_f = np.where(F <= K1, Spectrum, 0)
        _, dv_Low = signal.istft(Spectrum_low_f, fs, nperseg=n_perseg, noverlap=n_overlap, boundary='even')
        # _, dv_inverted = signal.istft(Spectrum, fs, nperseg=n_perseg, noverlap=n_overlap, boundary='even')

        # plt.plot(dv_High)
        # plt.plot(dv_Low)
        # plt.plot(dv)
        # plt.show()
        # plt.close()

        velocity_model_high[i_i, :] = dv_High + Vo
        velocity_model_low[i_i, :] = dv_Low + Vo
    return velocity_model_high, velocity_model_low

def F_wavenumber_filter(velocity_model, background_model, filter, flag_single_thread=0):
    # this function filters velocity model on low wavenumbers and high wavenumbers
    if len(velocity_model.shape) == 4:
        velocity_model = np.squeeze(velocity_model, axis=-1)
        background_model = np.squeeze(background_model, axis=-1)
    elif len(velocity_model.shape) == 2:
        velocity_model = np.expand_dims(velocity_model, axis=[0])
        background_model = np.expand_dims(background_model, axis=[0])
    # check velocity units
    a = 1
    # Plot_image(velocity_model[0,:,:].T,Show_flag=0,Save_flag=1,Title='check')
    # Plot_image(background_model[0,:,:].T,Show_flag=0,Save_flag=1,Title='check2')
    if (np.mean(velocity_model) > 0) and (np.mean(velocity_model) < 100):  # units m/sec
        velocity_model = velocity_model * 1000
    if (np.mean(background_model) > 0) and (np.mean(background_model) < 100):  # units m/sec
        background_model = background_model * 1000
    # Plot_image(velocity_model[0,:,:].T,Show_flag=0,Save_flag=1,Title='check')
    # Plot_image(background_model[0,:,:].T,Show_flag=0,Save_flag=1,Title='check2')
    perturbation_model = velocity_model - background_model
    velocity_model_high = np.zeros_like(velocity_model)
    velocity_model_low = np.zeros_like(velocity_model)
    Angles = filter['angles'];
    dz = filter['dz']
    Angles = imresize(Angles, [1, velocity_model.shape[2]])
    Nm = velocity_model.shape[0]
    Nx = velocity_model.shape[1];
    Nz = velocity_model.shape[2]
    # %%
    if flag_single_thread == 1:
        print('Hello, I will start single thread filtering now')
        for i_m in range(Nm):
            a = wavenumber_filter(i_m, Nm, Nx, Nz, dz, Angles, filter, perturbation_model[i_m, :, :],
                                  background_model[i_m, :, :])
            velocity_model_high[i_m, :, :] = a[0]
            velocity_model_low[i_m, :, :] = a[1]
    else:
        model_list = list(range(Nm))
        cpu_num = multiprocessing.cpu_count() - 5
        print('Hello, I will start parallel filtering now on cpus=', cpu_num)
        processed_list = Parallel(n_jobs=cpu_num)(delayed(wavenumber_filter)
                                                  (i_m, Nm, Nx, Nz, dz, Angles, filter,
                                                   perturbation_model[i_m, :, :],
                                                   background_model[i_m, :, :]) for i_m in model_list)
        for i_m in range(Nm):
            velocity_model_high[i_m, :, :] = processed_list[i_m][0]
            velocity_model_low[i_m, :, :] = processed_list[i_m][1]
    #   Control the output, output should be in km/sec
    velocity_model_high = velocity_model_high / 1000
    velocity_model_low = velocity_model_low / 1000
    return velocity_model_high, velocity_model_low

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
####  elastic transform for generator 2
def elastic_transform_2(image, alpha, sigma, random_state_number=None, v_dx=20, plot_name=None):
    """Elastic deformation of images as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.
    """
  
    random_state = np.random.RandomState(random_state_number)

    shape = image.shape
    #print(shape)
    
    # with our velocities dx is vertical shift
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), (sigma, sigma/10, 1), mode="constant", cval=0) * 4 * alpha
    
    # with our velocities dy is horizontal
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), (sigma, sigma/10, 1),  mode="constant", cval=0) * alpha
    dz = np.zeros_like(dx)
    
    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1)),  np.reshape(z, (-1, 1))
    
    distorted_image = map_coordinates(image, indices, order=1, mode='reflect', prefilter=False)
    distorted_image = distorted_image.reshape(image.shape)
    
    if plot_name != None:
        plt_nb_T(v_dx * np.squeeze(dx[:,:]), fname=f"VerticalShifts_{alpha}", title="Vertical shifts (km)")
        dq_x = 100
        dq_z = 17
        M = np.hypot(dy.squeeze()[::dq_x,::dq_z].T, dx.squeeze()[::dq_x,::dq_z].T)
        M = dx.squeeze()[::dq_x,::dq_z].T
        M = np.squeeze(image)[::dq_x,::dq_z].T
        if 1:
            fig1, ax1 = plt.subplots(figsize=(16,9))
            ax1.set_title('Guiding model')
            plt.imshow(1e-3*np.squeeze(image.T), extent=(0, v_dx * dx.shape[0] * 1e-3, v_dx * dx.shape[1] *1e-3, 0))
            plt.axis("tight")
            plt.xlabel("Distance (km)")
            plt.ylabel("Depth (km)")
            plt.colorbar()
            Q = ax1.quiver(
            1e-3*v_dx *y.squeeze()[::dq_x,::dq_z].T, 1e-3*v_dx *x.squeeze()[::dq_x,::dq_z].T, 
            np.abs(1e-4*v_dx*dx.squeeze()[::dq_x,::dq_z].T), 1e-3*v_dx*dx.squeeze()[::dq_x,::dq_z].T, 
            scale_units='xy', scale=1, pivot='tip')
            plt.savefig(f"../latex/Fig/shiftsVectors", bbox_inches='tight')
            plt_show_proceed()

        fig1, ax1 = plt.subplots(figsize=(16,9))
        ax1.set_title('Distorted model')
        plt.imshow(1e-3*np.squeeze(distorted_image.T), extent=(0, v_dx * dx.shape[0] * 1e-3, v_dx * dx.shape[1] *1e-3, 0))
        plt.axis("tight")
        plt.xlabel("Distance (km)")
        plt.ylabel("Depth (km)")
        plt.colorbar()
        Q = ax1.quiver(
            1e-3*v_dx *y.squeeze()[::dq_x,::dq_z].T, 1e-3*v_dx *x.squeeze()[::dq_x,::dq_z].T, 
            np.abs(1e-4*v_dx*dx.squeeze()[::dq_x,::dq_z].T), 1e-3*v_dx*dx.squeeze()[::dq_x,::dq_z].T, 
            scale_units='xy', scale=1, pivot='tip')
        plt.savefig(f"../latex/Fig/deformedModel{plot_name}", bbox_inches='tight')
        plt_show_proceed()
    return distorted_image
def generate_model(model_input='./original_models/marmvel.hh', 
                   model_output="./original_models/marm.rsf",
                   dx=20,
                   out_shape=(500,200),
                   stretch_X=1,
                   training_flag=False,
                   random_state_number=314,                  
                   distort_flag=True,
                   distort_flag2=False,
                   crop_flag=True,
                   verbose=False,
                   test_flag=False,
                   show_flag=False):
    """take model without water, duplicate, crop, resize to dx, distort """
    # downscale marmousi
    #def rescale_to_dx(rsf_file_in, rsf_file_out, dx)
    # cmd(('source /home/plotnips/Madagascar/RSFSRC/env.sh'))
    import m8r as sf
    model_orig = sf.Input(model_input)
    Nz=model_orig.int("n1")
    Nx=model_orig.int("n2")
    dz_orig=model_orig.int("d1")
    dx_orig=model_orig.int("d2")
    vel = model_orig.read()
    vel_orig=vel.copy() 
    if test_flag:
        n_cut = int(((const.sxbeg + const.gxbeg) * const.dx) // (model_orig.float("d1")))
        vel = np.concatenate((vel[-n_cut:,:], np.flipud(vel), vel[:n_cut,:]), axis = 0)
    else:
        vel = np.concatenate((vel, np.flipud(vel), vel), axis = 0)
        
    if show_flag:
        np.random.RandomState(random_state_number)
        random.seed(random_state_number)
        np.random.seed(random_state_number)
    # Plot_image(vel_orig.T,Show_flag=0,Save_flag=1,Title='model_orig',Save_pictures_path='./pictures_for_check')
    # Plot_image(vel.T,Show_flag=0,Save_flag=1,Title='base_vel_model',Save_pictures_path='./pictures_for_check')
    if crop_flag:
        vel_log_res = vel
        #vel_log_res = resize(vel_log_res[:,:], (np.shape(vel)[0]//2, np.shape(vel)[1]//2))
        if verbose:
            print(f"Random state number = {random_state_number}")
        #vel = resize(vel_log_res, vel.shape)    
        # while 0!=1:
        l0 = randint(np.shape(vel)[0])
        #print(f"l0={l0}")      
        h0 = min(l0 + np.shape(vel)[0]//4 + randint(np.shape(vel)[0]//2), np.shape(vel)[0])
        l1 = randint(np.shape(vel)[1]//3)
        h1 = min(l1 + np.shape(vel)[1]//3 + randint(np.shape(vel)[1]//2), 
                np.shape(vel)[1])
        if verbose:
            print(l0, l1, h0, h1)
        # vel_log_res2 = vel_log_res[l0:h0, l1:h1]
        # print(vel_log_res2.shape)
        # Plot_image(vel_log_res2.T,Show_flag=0,Save_flag=1,Title='vel_cropped',Save_pictures_path='./pictures_for_check',c_lim=[1500,4500])
        vel_log_res = vel_log_res[l0:h0, l1:h1]
        ### extra step resize picture to desired size
        scale_factor=dx/dx_orig
        out_shape=tuple(scale_factor*np.array(out_shape))
        vel = resize(vel_log_res,out_shape)
        # vel = resize(vel_log_res, vel.shape)
    # we downscale
    if 'scale_factor' not in locals():
        scale_factor = dx / dx_orig
    vel = resize(vel[:,:], (stretch_X*np.shape(vel)[0]//scale_factor, np.shape(vel)[1]//scale_factor))
    
    if verbose:
        print(np.shape(vel))
        print(f"Model downscaled {scale_factor} times to {dx} meter sampling")
        if stretch_X != 1:
            print(f"Model stretched {stretch_X} times to {dx} meter sampling \n")
    
    # we concatenate horizontally, this is confusing because of flipped axis in madagascar
    vel = np.atleast_3d(vel)
    alpha_deform=500
    sigma_deform=50
    if distort_flag:
        # Plot_image(np.squeeze(vel).T,Show_flag=0,Save_flag=1,Title='vel0',Save_pictures_path='./pictures_for_check',c_lim=[1500,5000])
        vel = elastic_transform_2(vel,alpha_deform,sigma_deform,v_dx=dx,random_state_number=random_state_number)
        # Plot_image(np.squeeze(vel).T,Show_flag=0,Save_flag=1,Title='vel1',Save_pictures_path='./pictures_for_check',c_lim=[1500,5000])
    vel = np.squeeze(vel)
    if distort_flag2:
        c_lim=[np.min(vel),np.max(vel)]
        # Plot_image(vel.T,Show_flag=0,Save_flag=1,Title='vel',Save_pictures_path='./pictures_for_check',c_lim=c_lim)
        vel_alpha = (0.8+0.4*resize(np.random.rand(5,10), vel.shape))
        # Plot_image(vel_alpha.T,Show_flag=0,Save_flag=1,Title='vel_alpha',Save_pictures_path='./pictures_for_check')
        vel *= vel_alpha
        # Plot_image(vel.T,Show_flag=0,Save_flag=1,Title='vel2',Save_pictures_path='./pictures_for_check',c_lim=c_lim)
    # add water
    # vel = np.concatenate((1500*np.ones((vel.shape[0], 20)), vel), 
    #                      axis=1)
    #vel = ndimage.median_filter(vel, size=(7,3))
    #vel = 1500 * np.ones_like(vel)
    if verbose:
        print(f"Writing to {model_output}")
    # np_to_rsf(vel, model_output)
    return vel
####  elastic transform for generator 1
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
    if gen_mode=='generator2':
        """ create Vp velocity models with Vladimir Kazei's cropping and skewing approach,
        https://github.com/vkazei/deeplogs """
        T1=datetime.datetime.now();print(T1)
        counter=0
        dx_new=dh;  dz_new=dh
        ############    load Overthrust
        name=os.path.join(root_models,'overthrust_test_2D_2.hdf5')
        dz=load_file(name,'d1')
        dx=load_file(name,'d2')
        V_over=load_file(name,'vp')
        V_over=F_resize(V_over,dx=dx,dz=dz,dx_new=dx_new,dz_new=dz_new)
        ############    check Marmousi
        V_marm=np.loadtxt('./original_models/marm2_10.dat', delimiter=',').T*1000
        dx = 10;     dz = dx
        water_sz=(np.where(V_marm[0,:]==1500))[0].size;
        V_marm=V_marm[:,water_sz:]
        V_marm=F_resize(V_marm,dx=dx,dz=dz,dx_new=dx_new,dz_new=dz_new)
        ############    choose which model to emulate, either Marmousi or Overthrust (random choice).
        value=np.random.uniform()   #   create random value from 0 to 1
        if value<0.2:
            velocity_protype_model='Marmousi'
            matching_value=-0.5
        else:
            velocity_protype_model='Overthrust'
            matching_value=0.4
        while counter==0 or F_r2(generated_model,model_to_emulate)<matching_value:
            ############    generate velocity protype model
            nx=out_shape[0];    
            not_model_thickness=int((water_height+floor_height)/dz_new)
            vel=generate_model(dx=dx_new,out_shape=(out_shape[0],out_shape[1]-not_model_thickness),random_state_number=randint(100000),
                distort_flag=True,distort_flag2=False,crop_flag=True,verbose=False)
            ############    add water and ocean floor from above
            water=np.ones((nx,int(water_height/dx_new)))*1500
            depth_grad=np.linspace(1500,np.mean(vel[:,0]),int(floor_height/dz_new))
            ocean_floor=np.broadcast_to(depth_grad,(nx,int(floor_height/dz_new)))
            generated_model=np.concatenate([water,ocean_floor,vel],axis=1)
            ############    construct initial model
            initial_model=F_initial_vz_model_custom(generated_model,dz_new,int(water_height/dz_new))   #   F_initial_vz_model_custom
            ############    load model to emulate
            if velocity_protype_model=='Marmousi':   #    load Marmousi
                model_to_emulate=V_marm;
            elif velocity_protype_model=='Overthrust':
                model_to_emulate=V_over;    
            nx,nz=model_to_emulate.shape
            crop_width=int( (nx-out_shape[0])/2)
            ix1=crop_width;     ix2=ix1+out_shape[0]
            model_to_emulate=model_to_emulate[ix1:ix2,0:(out_shape[1]-not_model_thickness)]
            depth_grad=np.linspace(1500,np.mean(model_to_emulate[:,0]),int(floor_height/dz_new))
            ocean_floor=np.broadcast_to(depth_grad,(out_shape[0],int(floor_height/dz_new)))
            model_to_emulate=np.concatenate([water,ocean_floor,model_to_emulate],axis=1)
            print('simulate '+velocity_protype_model+', R2 score(generated_model,model_to_emulate)=',F_r2(generated_model,model_to_emulate))
            counter=counter+1
        T2=datetime.datetime.now()
        print('Velocity model generation time=',T2-T1)
    elif gen_mode=='generator1': 
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
            vel = np.loadtxt(os.path.join(program_root,'original_models/marm2_10.dat'),delimiter=',').T*1000
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
            name=os.path.join(program_root,'original_models/overthrust_data.npz')
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
        Plot_image(generated_model.T,Show_flag=0,Save_flag=1,Title='generated_model'+model_name,Save_pictures_path='./pictures_for_check',dx=dx_new,dy=dz_new,c_lim=[1500,4500])
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