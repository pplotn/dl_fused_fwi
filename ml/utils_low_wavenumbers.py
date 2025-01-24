from imports import *
def F_r2(mat, mat_true):
    # r2 = 1 - (np.std(mat_true.flatten() - mat.flatten()) / np.std(mat_true.flatten())) ** 2
    v1 = mat.flatten()
    v2 = mat_true.flatten()
    r2_2 = r2_score(v1, v2)
    return r2_2
def Plot_accuracy2(history,Title='Title',Save_pictures_path='./Pictures',Save_flag=0):
    plt.figure()
    plt.plot(history['g_accuracy_history'])
    plt.plot(history['d_accuracy_history'])
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.axis('tight')
    plt.ylim(0,100)
    string=', R2 accuracy'
    plt.title(Title+string)
    plt.legend(['generator accuracy','discriminator accuracy'], loc='lower right')
    if Save_flag == 1:
        name = Save_pictures_path + '/' + Title + '.png';   print(name)
        plt.savefig(name)
    plt.show(block=False)
    plt.close()
    return None
def Plot_loss(history,Title='Title',Save_pictures_path='./Pictures',Save_flag=0):
    plt.figure()
    plt.plot(history['g_loss_history'])
    plt.plot(history['d_loss_history'])
    plt.yscale('log')
    plt.ylabel('Loss function')
    plt.xlabel('Epochs')
    plt.axis('tight')
    string=', R2 loss'
    plt.title(Title)
    plt.legend(['generator loss','discriminator loss'], loc='upper right')
    if Save_flag == 1:
        name=Save_pictures_path + '/' + Title + '.png'
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
def transforming_data(data,scaler):
    shape_orig=data.shape
    N=shape_orig[0]
    Nx=shape_orig[1]
    Nz=shape_orig[2]
    Nch=shape_orig[3]
    data=np.squeeze(data)
    data=np.reshape(data,(int(data.size/scaler.n_features_in_),scaler.n_features_in_))
    data=scaler.transform(data)
    data=np.reshape(data,shape_orig)
    return data
def transforming_data_inverse(data,scaler):
    if isinstance(scaler, np.ndarray):
        data4=scaling_data_01_back(data,scaler)
    else:
        shape_orig=data.shape
        N=shape_orig[0]
        Nx=shape_orig[1]
        Nz=shape_orig[2]
        Nch=shape_orig[3]
        data=np.squeeze(data)
        data2=np.reshape(data,(N,Nx*Nz*Nch))
        data3 = scaler.inverse_transform(data2)
        data4=np.reshape(data3,shape_orig)
    return data4
def scaling_data_01(data):
    data=np.squeeze(data);  Nx=data.shape[0];  Nz=data.shape[1]
    const1=np.abs(np.min(data));  data_=data + const1;
    const2=np.max(np.abs(data_));  data_=data_/const2
    coefs=(1,0.05)
    geom_spreading_matrix=(coefs[0]+coefs[1]*np.repeat(np.expand_dims(np.arange(0,Nz),axis=0),Nx,axis=0))
    data_=data_*geom_spreading_matrix
    const3=np.max(np.abs(data_));  data_=data_/const3
    scaler=[coefs,const1,const2,const3]
    data_=np.expand_dims(data_,axis=0); data_=np.expand_dims(data_,axis=-1);
    return data_,scaler
def scaling_data_01_back(data,scaler):
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

def numstr(x):
    string = str('{0:.2f}'.format(x))
    return string
def load_numpy_file(name):
    with open(name,'rb') as f:
        data=np.load(f,allow_pickle=True)
        input_data =data['input_data']
        output_data=data['output_data']
        dx=data['dx']
        data.close()
    return input_data,output_data,dx

class Pix2Pix():
    def __init__(self,Save_pictures_path,log_save_const):
        # Input shape
        self.img_rows = 256
        self.img_cols = 256
        self.channels = 1
        self.img_shape = (self.img_rows,self.img_cols,self.channels)
        self.save_path=Save_pictures_path
        self.log_save_const=log_save_const
        self.keras_models_path='./keras_models';os.makedirs(self.keras_models_path,exist_ok=True)
        self.training_proceeded=0
        # Configure data loader
        self.dataset_path='./datasets'
        self.dataset_name = 'facades'
        # self.dataset_path='/ibex/scratch/plotnips/intel/MWE/datasets/'
        self.dataset_path='/ibex/scratch/projects/c2107/MWE/datasets'
        self.dataset_name='dataset_vl_gen_scaled_287145_smooth_450m_scaler_picture_max_abs'
        
        self.dataset_path='/home/plotnips/Dropbox/Log_extrapolation/scripts/MWE/datasets'
        self.dataset_name='dataset_vl_gen_scaled_130_smooth_450m_scaler_picture_max_abs'
        self.dataset_name='dataset_vl_gen_scaled_3000_smooth_450m_scaler_picture_-11'

        self.dataset_path='/ibex/scratch/plotnips/intel/MWE/datasets/'
        # self.dataset_name='dataset_vl_gen_scaled_60000_smooth_450m_scaler_picture_01'
        self.dataset_name='dataset_vl_gen_scaled_3000_smooth_450m_scaler_picture_individual_scaling'
        self.dataset_name='dataset_vl_gen_scaled_287145_smooth_450m_6860_scaler_picture_individual_scaling'
        # self.dataset_name='dataset_vl_gen_scaled_3005_smooth_450m_scaler_picture_individual_scaling_true'
        self.dataset_name='dataset_vl_gen_scaled_3005_smooth_450m_scaler_picture_individual_scaling_false'
        self.dataset_name='dataset_vl_gen_scaled_287145_smooth_450m_scaler_picture_individual_scaling_false'
        
        # self.dataset_name='dataset_vl_gen_scaled_3000_smooth_450m_scaler_picture_01'
        print('saving data to '+self.dataset_path+self.dataset_name)
        path=glob('%s%s/*' % (self.dataset_path,self.dataset_name))
        path=fnmatch.filter(sorted(path),'*.npz')
        path_test=path[-10:]
        path=list(set(path)-set(path_test))
        self.N=len(path)
        self.N=50
        self.train_frac= 0.9
        path=path[0:self.N]
        # path=random.sample(path,len(path))    # randomize??
        path_train=path[0:int(len(path)*self.train_frac)]
        print(path_train[0:10])
        path_valid=list(set(path)-set(path_train))
        self.path_train=path_train;self.path_test=path_test;self.path_valid=path_valid;
        print('Models for training:',  len(path_train))
        print('Models for validation:',len(path_valid))
        print('Models for testing:',   len(path_test))
        self.data_loader = DataLoader(dataset_name=self.dataset_name,
                                      dataset_path=self.dataset_path,
                                      path_train=self.path_train,
                                      path_valid=self.path_valid,
                                      path_test=self.path_test,
                                      img_res=(self.img_rows, self.img_cols),
                                      save_pictures_path=Save_pictures_path)
        input_data,output_data,dx=load_numpy_file(path[0])
        self.orig_input_shape=input_data.shape
        self.orig_output_shape=output_data.shape
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
        #-------------------------
        # Construct Computational
        #   Graph of Generator
        #-------------------------
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
                              loss_weights=[1,100],
                              optimizer=optimizer)
        print('generator.summary()')
        self.generator.summary()
        print('discriminator.summary()')
        self.discriminator.summary()
        # print('combined.summary()')
        # self.combined.summary()
        print(optimizer._hyper)
        print('gf=',self.gf)
        print('df=',self.df)
        aa=1
    def build_generator(self):
        """U-Net Generator"""
        def conv2d(layer_input, filters, f_size=4, bn=True):
            """Layers used during downsampling"""
            d = Conv2D(filters, kernel_size=f_size, strides=2,padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            return d
        def deconv2d(layer_input, skip_input, filters, f_size=4, dropout_rate=0):
            """Layers used during upsampling"""
            u = UpSampling2D(size=2)(layer_input)
            u = Conv2D(filters, kernel_size=f_size, strides=1,padding='same',activation='relu')(u)
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
        output_img = Conv2D(self.channels, kernel_size=4, strides=1, padding='same', activation='relu')(u7)
        return Model(d0, output_img)
    def build_discriminator(self):
        def d_layer(layer_input, filters, f_size=4, bn=True):
            """Discriminator layer"""
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
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
        final=d4
        validity = Conv2D(1, kernel_size=4, strides=1, padding='same')(final)
        return Model([img_A, img_B], validity)
    def train(self, epochs, batch_size=1, sample_interval=1,plotting_interval=1):
        print('epochs=',    epochs)
        print('batch_size=',batch_size)
        print("Training parameters, epochs:%d, batch_size:%d, sample_interval:%d" % (epochs,batch_size,sample_interval))
        start_time = datetime.datetime.now()
        # Adversarial loss ground truths
        valid = np.ones((batch_size,) + self.disc_patch)
        fake = np.zeros((batch_size,) + self.disc_patch)
        # aa=enumerate(self.data_loader.load_batch(batch_size))
        aa=self.data_loader.load_data(batch_size)
        g_loss_history=[];d_loss_history=[]
        g_accuracy_history=[];d_accuracy_history=[]
        for epoch in range(epochs):
            for batch_i, (imgs_A,imgs_B) in enumerate(self.data_loader.load_batch(batch_size)):
                # ---------------------
                #  Train Discriminator
                # ---------------------
                # Condition on B and generate a translated version
                fake_A = self.generator.predict(imgs_B)
                # Train the discriminators (original images = real / generated = Fake)
                d_loss_real = self.discriminator.train_on_batch([imgs_A, imgs_B], valid)
                d_loss_fake = self.discriminator.train_on_batch([fake_A, imgs_B], fake)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
                # -----------------
                #  Train Generator
                # -----------------
                # Train the generators
                g_loss = self.combined.train_on_batch([imgs_A, imgs_B], [valid, imgs_A])
                elapsed_time = datetime.datetime.now() - start_time
                # Plot the progress
                print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %3d%%] [G loss: %f] [G accuracy: %f] time: %s" % (epoch, epochs,
                                                                        batch_i, self.data_loader.n_batches,
                                                                        d_loss[0], 100*d_loss[1],
                                                                        g_loss[0], 100*g_loss[1],
                                                                        elapsed_time))
                # If at save interval => save generated model
            g_loss_history.append(g_loss[0])
            d_loss_history.append(d_loss[0])
            g_accuracy_history.append(100*g_loss[1])
            d_accuracy_history.append(100*d_loss[1])
            if epoch % sample_interval == 0:
                # self.combined.save_weights( self.keras_models_path+'/model_combined'+str(self.log_save_const)+'epoch_'+str(epoch)+'.hdf5')
                self.generator.save_weights(self.keras_models_path+'/generator'+str(self.log_save_const)+'epoch_'+     str(epoch)+'.hdf5')
            if epoch % plotting_interval==0:
                # self.sample_images(0,1,original_plotting=1)
                self.sample_list(epoch,self.path_train[0:1])
        history={'g_loss_history':g_loss_history,'d_loss_history':d_loss_history,
            'g_accuracy_history':g_accuracy_history,'d_accuracy_history':d_accuracy_history}
        Plot_loss(history, Title='log' +      str(self.log_save_const) + 'loss_mse', Save_pictures_path=self.save_path, Save_flag=1)
        Plot_accuracy2(history, Title='log' + str(self.log_save_const) + 'r2accuracy', Save_pictures_path=self.save_path,Save_flag=1)
        print('saving model')
        self.combined.save_weights( self.keras_models_path+'/model_combined'+str(self.log_save_const)+'.hdf5')   #+str()
        self.generator.save_weights(self.keras_models_path+'/generator'+str(self.log_save_const)+'.hdf5')   #+str()
        self.training_proceeded=1
    def sample_images(self,epoch,batch_i,original_plotting=0):
        original_plotting=0
        if original_plotting==1:
            os.makedirs('images/%s' % self.dataset_name, exist_ok=True)
            r, c = 3, 3
            imgs_A,imgs_B=self.data_loader.load_data(batch_size=3,is_testing=True)
            fake_A = self.generator.predict(imgs_B)
            gen_imgs = np.concatenate([imgs_B, fake_A, imgs_A])       
            titles = ['Condition', 'Generated','Original']
            fig, axs = plt.subplots(r, c)
            cnt = 0
            for i in range(r):
                for j in range(c):
                    mat=gen_imgs[cnt,:,:,0].T
                    im=axs[i,j].imshow(mat)
                    axs[i,j].set_title(titles[i])
                    axs[i,j].axis('off')
                    cbar = fig.colorbar(im, extend='both', shrink=0.9, ax=axs[i,j])
                    cnt += 1
            fig.savefig(self.save_path+"/%d_%d.png" % (epoch,batch_i))
            plt.close()
        else:
            flag_show_scaled_data=1
            # list_train=self.path_train[0::];list_valid=self.path_valid[0:1];list_test=self.path_test[0::]
            list_train=self.path_train[0:7];list_valid=self.path_valid[0:2];list_test=self.path_test[0::]
            list_all=list_test+list_train+list_valid
            [x_all_,t_all_]=self.data_loader.record_dataset_spec_ids(list_all)
            t_predicted_all_ = self.generator.predict(x_all_,verbose=1)
            t_predicted_all=np.zeros((len(list_all),self.orig_output_shape[1],
                self.orig_output_shape[2],1))
            t_all=np.zeros((len(list_all),self.orig_output_shape[1],
                self.orig_output_shape[2],1))
            x_all=np.zeros((len(list_all),self.orig_input_shape[1],
                self.orig_input_shape[2],1))
            for i_x,NAME in enumerate(list_all):
                t_predicted_all[i_x,:,:,0]=imresize(t_predicted_all_[i_x,:,:,0],
                    (self.orig_output_shape[1],self.orig_output_shape[2]))
                t_all[i_x,:,:,0]=imresize(t_all_[i_x,:,:,0],
                    (self.orig_output_shape[1],self.orig_output_shape[2]))
                x_all[i_x,:,:,0]=imresize(x_all_[i_x,:,:,0],
                    (self.orig_input_shape[1],self.orig_input_shape[2]))
                with open(NAME,'rb') as f:
                    data=np.load(f,allow_pickle=True)
                    M0=data['models'][0,:,:,0]
                    dz=data['dz']
                    dx=data['dx']
                    Minit=data['models_init'][0,:,:,0]
                    if 'scaler_x' in data.keys():
                        scaler_type='_individual_scaling'
                        scaler_x=data['scaler_x']
                        scaler_t=data['scaler_t']
                    else:
                        from joblib import dump,load
                        scaler_type='_1_scaler'
                        scaler_x=load(data_path+'/scaler_x.bin')
                        scaler_t=load(data_path+'/scaler_t.bin')
                    data.close()
                M1=x_all[i_x,:,:,:];M1=np.expand_dims(M1,axis=0)
                M2=t_all[i_x,:,:,:];M2=np.expand_dims(M2,axis=0)
                M3=t_predicted_all[i_x,:,:,:];M3=np.expand_dims(M3,axis=0)
                if flag_show_scaled_data==0:
                    M1=transforming_data_inverse(M1,scaler_x)
                    M2=transforming_data_inverse(M2,scaler_t)
                    M3=transforming_data_inverse(M3,scaler_t)
                    Predicted_update=imresize(M3[0,:,:,0],[M1.shape[1],M1.shape[2]])
                    True_update=M2
                else:
                    tmp=transforming_data_inverse(M3,scaler_t)
                    Predicted_update=tmp[0,:,:,0]
                    True_update=transforming_data_inverse(M2,scaler_t)
                True_update=True_update[0,:,:,0]
                M1=M1[0,:,:,0];M2=M2[0,:,:,0];M3=M3[0,:,:,0]
                Models_init=Minit
                if Models_init.shape!=Predicted_update.shape:
                    Predicted_update=imresize(Predicted_update,Models_init.shape)
                if Models_init.shape!=True_update.shape:
                    True_update=imresize(True_update,Models_init.shape)
                testing_model=Models_init+Predicted_update
                ideal_init_model=Models_init+True_update
                M0_show=M0
                ################### Crop testing models for better visualization
                water=np.ones((M0_show.shape[0],18))*1500
                M0_show=np.concatenate([water,M0_show],axis=1)
                testing_model=np.concatenate([water,testing_model],axis=1)    
                ideal_init_model=np.concatenate([water,ideal_init_model],axis=1)
                inp_orig_sizes=[M1,M2,M3,testing_model,M0_show]
                pics_6=[M1,M2,M3,M3-M2,testing_model,ideal_init_model]
                # inp_orig_sizes=[M1,M2,M3,testing_model,ideal_init_model]
                saving_name=NAME.split('augmented_marmousi_10_it')[-1]
                saving_name=saving_name.split('.npy')[0]
                # Plot_image(testing_model.T,Show_flag=1,Save_flag=1,Title='testing_model1'+saving_name,Aspect='equal',Save_pictures_path=Save_pictures_path)
                ####
                Prediction_accuracy=F_r2(M3,M2)
                R2val=F_r2(testing_model,M0_show)
                # R2val=F_r2(pics_6[-2],pics_6[-1])
                # R2val2=F_r2(testing_model,ideal_init_model)
                if NAME in list_train:
                    data_type='Train'
                elif NAME in list_test:
                    data_type='Test'
                    tmp2=NAME.split('augmented_marmousi')
                    path =self.save_path +'/'+ tmp2[1][0:-4]+'_weights_'+str(self.log_save_const)
                    np.savez(path,input_data=M1,output_data=M2,
                        models_init=Models_init,models=M0,predicted_update=Predicted_update,dx=dx,dz=dz)
                elif NAME in list_valid:
                    data_type='Valid'
                tmp=NAME.split('augmented_marmousi_10_it')[-1]
                tmp=tmp.split('.npz')[0]
                if NAME in list_test:
                    data_type = '_' + data_type+tmp+'_'+numstr(Prediction_accuracy)
                    title='Prediction, R2(prediction, target) = ' + numstr(Prediction_accuracy)
                else:
                    data_type = '_' + data_type+tmp+'_'+numstr(Prediction_accuracy)
                    title='Prediction, R2(prediction, target) = ' + numstr(Prediction_accuracy)
                Name=self.save_path + '/' + 'log' + str(self.log_save_const) + data_type+'.png'
                Name=self.save_path + '/' + 'log' + str(self.log_save_const) + data_type+'_6pics'+'.png'
                #   PLOT_ML_Result_fixed_colorbar
                PLOT_ML_Result_adaptive_colorbar(pics_6,numstr(R2val),history_flag=0,
                    history=None,Boundaries=[],save_file_path=Name,
                    dx=dx,dy=dz,Title=title,Title2='',Save_flag=1,adaptive_colorbar=3)
                i_x=i_x+1
            ################################# 
            aa=1
            # imgs_A,imgs_B=self.data_loader.load_data(batch_size=3,is_testing=True)
            # fake_A = self.generator.predict(imgs_B)
    def sample_list(self,epoch,sample_list):
        flag_show_scaled_data=1;    list_all=sample_list
        [x_all_,t_all_]=self.data_loader.record_dataset_spec_ids(list_all)
        t_predicted_all_ = self.generator.predict(x_all_,verbose=1)
        t_predicted_all=np.zeros((len(list_all),self.orig_output_shape[1],
            self.orig_output_shape[2],1))
        t_all=np.zeros((len(list_all),self.orig_output_shape[1],
            self.orig_output_shape[2],1))
        x_all=np.zeros((len(list_all),self.orig_input_shape[1],
            self.orig_input_shape[2],1))
        for i_x,NAME in enumerate(list_all):
            t_predicted_all[i_x,:,:,0]=imresize(t_predicted_all_[i_x,:,:,0],
                (self.orig_output_shape[1],self.orig_output_shape[2]))
            t_all[i_x,:,:,0]=imresize(t_all_[i_x,:,:,0],
                (self.orig_output_shape[1],self.orig_output_shape[2]))
            x_all[i_x,:,:,0]=imresize(x_all_[i_x,:,:,0],
                (self.orig_input_shape[1],self.orig_input_shape[2]))
            with open(NAME,'rb') as f:
                data=np.load(f,allow_pickle=True)
                M0=data['models'][0,:,:,0]
                dz=data['dz']
                dx=data['dx']
                Minit=data['models_init'][0,:,:,0]
                if 'scaler_x' in data.keys():
                    scaler_type='_individual_scaling'
                    scaler_x=data['scaler_x']
                    scaler_t=data['scaler_t']
                else:
                    from joblib import dump,load
                    scaler_type='_1_scaler'
                    scaler_x=load(data_path+'/scaler_x.bin')
                    scaler_t=load(data_path+'/scaler_t.bin')
                data.close()
            M1=x_all[i_x,:,:,:];M1=np.expand_dims(M1,axis=0)
            M2=t_all[i_x,:,:,:];M2=np.expand_dims(M2,axis=0)
            M3=t_predicted_all[i_x,:,:,:];M3=np.expand_dims(M3,axis=0)
            if flag_show_scaled_data==0:
                M1=transforming_data_inverse(M1,scaler_x)
                M2=transforming_data_inverse(M2,scaler_t)
                M3=transforming_data_inverse(M3,scaler_t)
                Predicted_update=imresize(M3[0,:,:,0],[M1.shape[1],M1.shape[2]])
                True_update=M2
            else:
                tmp=transforming_data_inverse(M3,scaler_t)
                Predicted_update=tmp[0,:,:,0]
                True_update=transforming_data_inverse(M2,scaler_t)
            True_update=True_update[0,:,:,0]
            M1=M1[0,:,:,0];M2=M2[0,:,:,0];M3=M3[0,:,:,0]
            Models_init=Minit
            if Models_init.shape!=Predicted_update.shape:
                Predicted_update=imresize(Predicted_update,Models_init.shape)
            if Models_init.shape!=True_update.shape:
                True_update=imresize(True_update,Models_init.shape)
            testing_model=Models_init+Predicted_update
            ideal_init_model=Models_init+True_update
            M0_show=M0
            ################### Crop testing models for better visualization
            water=np.ones((M0_show.shape[0],18))*1500
            M0_show=np.concatenate([water,M0_show],axis=1)
            testing_model=np.concatenate([water,testing_model],axis=1)    
            ideal_init_model=np.concatenate([water,ideal_init_model],axis=1)
            inp_orig_sizes=[M1,M2,M3,testing_model,M0_show]
            pics_6=[M1,M2,M3,M3-M2,testing_model,ideal_init_model]
            # inp_orig_sizes=[M1,M2,M3,testing_model,ideal_init_model]
            saving_name=NAME.split('augmented_marmousi_10_it')[-1]
            saving_name=saving_name.split('.npy')[0]
            # Plot_image(testing_model.T,Show_flag=1,Save_flag=1,Title='testing_model1'+saving_name,Aspect='equal',Save_pictures_path=Save_pictures_path)
            ####
            Prediction_accuracy=F_r2(M3,M2)
            R2val=F_r2(testing_model,M0_show)
            # R2val=F_r2(pics_6[-2],pics_6[-1])
            # R2val2=F_r2(testing_model,ideal_init_model)
            tmp=NAME.split('augmented_marmousi_10_it')[-1]
            tmp=tmp.split('.npz')[0]
            data_type=tmp
            title='Prediction, R2(prediction, target) = ' + numstr(Prediction_accuracy)
            Name=self.save_path+'/'+'log'+str(self.log_save_const) + data_type+'_epoch_'+str(epoch)+'_'+numstr(Prediction_accuracy)+'.png'
            PLOT_ML_Result_adaptive_colorbar(pics_6,numstr(R2val),history_flag=0,
                history=None,Boundaries=[],save_file_path=Name,
                dx=dx,dy=dz,Title=title,Title2='',Save_flag=1,adaptive_colorbar=3)
            i_x=i_x+1
    def inference(self,Model_to_load_const,model_name='',batch_size=1):
        if model_name=='':
            model_name='generator'+str(Model_to_load_const)+'.hdf5'
        if self.training_proceeded==1:
            imgs_A,imgs_B=self.data_loader.load_data(batch_size=3,is_testing=True)
            fake_A = self.generator.predict(imgs_B)
            aa=1
        else:
            print('model loaded=',model_name)
            self.generator.load_weights(self.keras_models_path+'/'+model_name,by_name=1)
            # self.combined.load_weights(self.keras_models_path+'/model_combined'+str(Model_to_load_const)+'.hdf5')
        # history = F_load_history_from_file(Keras_models_path,Model_to_load_const)
class DataLoader():
    def __init__(self,dataset_name,path_train,
            path_valid,path_test,dataset_path,img_res=(128, 128),save_pictures_path=''):
        self.dataset_name=dataset_name
        self.dataset_path=dataset_path
        self.path_train=path_train
        self.path_valid=path_valid
        self.path_test=path_test
        self.img_res=img_res
        self.save_pictures_path=save_pictures_path
    def unpack_npz_file(self,path):
        if os.stat(path).st_size == 0:
            print('File is empty'); os.remove(path);   return None
        with open(path,'rb') as f:
            data=np.load(f,allow_pickle=True)
            x=data['input_data']
            t=data['output_data']
            data.close()
        x=np.squeeze(x)
        t=np.squeeze(t)
        # Plot_image(x.T,Show_flag=1,Save_flag=1,Title='x',
        #     Aspect='equal',Save_pictures_path=self.save_pictures_path)
        # Plot_image(t.T,Show_flag=1,Save_flag=1,Title='t',
        #     Aspect='equal',Save_pictures_path=self.save_pictures_path)
        x=imresize(x,[self.img_res[0],self.img_res[1]])
        t=imresize(t,[self.img_res[0],self.img_res[1]])
        # Plot_image(x.T,Show_flag=1,Save_flag=1,Title='x_imresized',
        #     Aspect='equal',Save_pictures_path=self.save_pictures_path)
        # Plot_image(t.T,Show_flag=1,Save_flag=1,Title='t_imresized',
        #     Aspect='equal',Save_pictures_path=self.save_pictures_path)
        x=np.expand_dims(x,axis=0); x=np.expand_dims(x,axis=-1)
        t=np.expand_dims(t,axis=0); t=np.expand_dims(t,axis=-1)
        return x,t
    def load_data(self,batch_size=1,is_testing=False):
        data_type = "train" if not is_testing else "test"
        # path = glob('%s/%s/%s/*' % (self.dataset_path,self.dataset_name, data_type))
        if data_type=='test':
            path=self.path_test
        elif data_type=='train':
            path=self.path_train
        # batch_images = np.random.choice(path, size=batch_size)
        # batch_images2=np.random.choice(np.arange(100), size=batch_size)
        batch_images=random.sample(path,batch_size)
        imgs_A = []
        imgs_B = []
        for img_path in batch_images:
            x,t=self.unpack_npz_file(img_path)
            img_A=t[0,::];img_B=x[0,::]
            imgs_A.append(img_A)
            imgs_B.append(img_B)
        imgs_A = np.array(imgs_A)
        imgs_B = np.array(imgs_B)
        return imgs_A,imgs_B
    def load_batch(self, batch_size=1, is_testing=False):
        #   same as load_data function
        data_type = "train" if not is_testing else "val"
        # path = glob('./datasets/%s/%s/*' % (self.dataset_name, data_type))
        if data_type=='test':
            path=self.path_test
        elif data_type=='train':
            path=self.path_train
        self.n_batches = int(len(path) / batch_size)
        for i in range(self.n_batches-1):
            batch = path[i*batch_size:(i+1)*batch_size]
            imgs_A, imgs_B = [], []
            for img in batch:
                x,t=self.unpack_npz_file(img)
                img_A=t[0,::];img_B=x[0,::];
                imgs_A.append(img_A)
                imgs_B.append(img_B)
            imgs_A = np.array(imgs_A)
            imgs_B = np.array(imgs_B)
            yield imgs_A, imgs_B
    def imread(self,path):
        return scipy.misc.imread(path, mode='RGB').astype(np.float)
    def record_dataset_spec_ids(self,file_names):
        out_x=np.empty((len(file_names),self.img_res[0],self.img_res[1],1))
        out_t=np.empty((len(file_names),self.img_res[0],self.img_res[1],1))
        for count,name in enumerate(file_names):
            x,t=self.unpack_npz_file(name)
            out_x[count,::]=x[0,::]
            out_t[count,::]=t[0,::]
        return out_x,out_t
def PLOT_ML_Result_adaptive_colorbar(inp,val1,history_flag=0, history=None,
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
    ax5.set_title('Predicted initial model for fwi, R2(predicted initial,true)='+val1)
    # ax5.set_title('Predicted initial model for fwi, R2(predicted initial,ideal initial)='+val1)
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
def Plot_image(Data, Title='Title', c_lim='',x='',x_label='',y='',y_label='',
               dx='',dy='',Save_flag=0,Save_pictures_path='./Pictures',
               Reverse_axis=1,Curve='',Show_flag=1,Aspect='equal',write_fig_title=1):
    # aspect - 'auto'
    if c_lim == '':  c_lim =[np.min(Data), np.max(Data)]
    if x == '':  x=(np.arange(np.shape(Data)[1]))
    if y == '':  y=(np.arange(np.shape(Data)[0]))
    if dx != '':  x=(np.arange(np.shape(Data)[1]))*dx
    if dy != '':  y=(np.arange(np.shape(Data)[0]))*dy
    extent = [x.min(), x.max(), y.min(), y.max()]
    #if Save_flag==1:
    #    plt.ion()
    fig=plt.figure()
    fig.dpi=330
    # fig_size = plt.rcParams["figure.figsize"]
    # fig_size[0] = 10.4
    # fig_size[1] = 8.0
    # plt.rcParams["figure.figsize"] = fig_size
    plt.set_cmap('RdBu_r')
    # plt.axis(extent, Aspect)
    # plt.axis(extent, 'auto')
    if write_fig_title==1:
        plt.title(Title)
    if Reverse_axis == 1:
        plt.imshow(np.flipud(Data), extent=extent, interpolation='nearest',aspect=Aspect)
        plt.gca().invert_yaxis()
    else:
        plt.imshow((Data), extent=extent, interpolation='nearest',aspect=Aspect)
    if Curve != '':
        # if len(np.shape(Curve)) == 2:
        #     Curve=Curve[0,:]
        plt.plot(x, Curve, color='white', linewidth=1.2, linestyle='--')
    
    ax = plt.gca()
    divider1 = make_axes_locatable((ax))
    cax1 = divider1.append_axes("right", size="2%", pad=0.05)
    cbar=plt.colorbar(cax=cax1)
    cbar.set_label("(m/s)")
    plt.clim(c_lim)
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    # plt.axis('equal')
    plt.axis('tight')
    # tight_figure(fig)
    if Save_flag == 1:
        if not os.path.exists(Save_pictures_path):
            os.mkdir(Save_pictures_path)
        name=Save_pictures_path + '/' + Title + '.png'
        print(name)
        # plt.show()
        # plt.show(block=True)
        # plt.show(block=False)
        plt.savefig(name,bbox_inches='tight')
    if Show_flag==0:
        plt.show(block=False)
        # plt.show(block=True)
    else:
        if Show_flag == 2:
            a=1
        else:
            plt.show()
    plt.close()
    return None


# class DataLoader2(tf.keras.utils.Sequence):
#     def __init__(self,files_list,batch_size,shuffle=True,
#         to_fit=True,to_predict=False,out_shape='not_equal',
#         smoothing_radius=450):
#         """Initialization
#         :param list_IDs: list of all 'label' ids to use in the generator
#         :param labels: list of image labels (file names)
#         :param to_fit: True to return X and y, False to return X only
#         :param batch_size: batch size at each iteration
#         :param dim: tuple indicating image dimension
#         :param shuffle: True to shuffle label indexes after every epoch"""
#         input_data,output_data,dx=load_numpy_file(files_list[0])
#         self.files=files_list
#         # self.Nx_in=(input_data.shape[1]//8) * 8
#         # self.Nz_in=(input_data.shape[2]//8) * 8
#         self.smoothing_radius=smoothing_radius
#         self.Nx_in=input_data.shape[1]
#         self.Nz_in=input_data.shape[2]
#         self.Nx_out=output_data.shape[1]
#         self.Nz_out=output_data.shape[2]
#         self.in_out_shape='not_equal'
#         if self.in_out_shape=='equal':
#             self.Nx_out=self.Nx_in
#             self.Nz_out=self.Nz_in
#         elif self.in_out_shape=='special':
#             self.Nx_out=68
#             self.Nz_out=60
#         self.Ns=len(files_list)
#         self.Nch_in=input_data.shape[3]
#         self.Nch_out=1
#         self.list_IDs=np.arange(self.Ns)     
#         if to_predict==False:
#             self.batch_size=batch_size
#         else:
#             self.batch_size=batch_size
#         self.shuffle = shuffle
#         self.to_fit=to_fit
#         self.on_epoch_end()
#     def return_self(self):
#         return self
#     def __len__(self):
#         return int(np.floor(self.Ns / self.batch_size))
#     def __getitem__(self, index):
#         # 'Generate one batch of data'
#         indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
#         # Find list of IDs
#         list_IDs_temp=[self.list_IDs[k] for k in indexes]
#         # Generate data
#         if self.to_fit:
#             X= self.__data_generation(list_IDs_temp,'x')
#             y=self.__data_generation(list_IDs_temp,'y')
#             return X, y
#         else:
#             X = self.__data_generation(list_IDs_temp,'x')
#             return X
#     def __getdataset__(self,size=-1):
#         if size==-1:
#             size=self.Ns
#         list_IDs_temp = np.arange(size)
#         if self.to_fit:
#             X= self.__data_generation(list_IDs_temp,'x',use_multi_processing_implementation_flag=1)
#             y=self.__data_generation(list_IDs_temp,'y',use_multi_processing_implementation_flag=1)
#             return X, y
#         else:
#             X = self.__data_generation(list_IDs_temp,'x',use_multi_processing_implementation_flag=1)
#             return X
#     def record_dataset(self,sampling=-1):
#         size=self.Ns
#         list_IDs_temp=np.arange(start=0,stop=size,step=sampling)
#         X= self.__data_generation(list_IDs_temp,'x',use_multi_processing_implementation_flag=1)
#         y=self.__data_generation(list_IDs_temp,'y',use_multi_processing_implementation_flag=1)
#         return X, y
#     def record_dataset_spec_ids(self,file_names):
#         out_x=np.empty((len(file_names),self.Nx_in,self.Nz_in,self.Nch_in))
#         out_t=np.empty((len(file_names),self.Nx_out,self.Nz_out,self.Nch_out))
#         for count,name in enumerate(file_names):
#             x=load_npz_file(name,'x',
#                 in_out_shape=self.in_out_shape,smoothing_radius=self.smoothing_radius)
#             t=load_npz_file(name,'t',
#                 in_out_shape=self.in_out_shape,smoothing_radius=self.smoothing_radius)
#             out_x[count,::]=x
#             out_t[count,::]=t
#         return out_x,out_t
#     def record_dataset_scaled(self,size=-1):
#         list_IDs_temp = np.arange(start=0,stop=size)
#         X= self.__data_generation(list_IDs_temp,'x')
#         y=self.__data_generation(list_IDs_temp,'y')
#         return X, y
#     def on_epoch_end(self):
#         'Updates indexes after each epoch'
#         self.indexes = np.arange(len(self.list_IDs))
#         if self.shuffle == True:
#             np.random.shuffle(self.indexes)
#     def __data_generation(self, list_IDs_temp,data_type,use_multi_processing_implementation_flag=0):
#         # Initialization
#         if data_type=='x':
#             out=np.empty((len(list_IDs_temp),self.Nx_in,self.Nz_in,self.Nch_in))
#         else:
#             out=np.empty((len(list_IDs_temp),self.Nx_out,self.Nz_out,self.Nch_out))
#         if use_multi_processing_implementation_flag==1:
#             from functools import partial
#             print('start multiprocessing loading')
#             pool = multiprocessing.Pool(multiprocessing.cpu_count()-3)
#             temp = partial(parse_npz_data_file,self,data_type)
#             data_list = pool.map(func=temp, iterable=list_IDs_temp)
#             pool.close()
#             pool.join()
#             print('end multiprocessing loading')
#             print('filling out structure')
#             for i in range(len(data_list)):
#                 out[i]=data_list[i]
#         else:
#             for i, id in enumerate(list_IDs_temp):
#                 x=parse_npz_data_file(self,data_type,id)
#                 out[i,:,:,:]=x
#                 # print('data type=', data_type)
#                 # print('data shape=',     out.shape)
#         return out

# img_A=img_A[0,::];img_B=img_B[0,::];
# img_A=imresize(img_A,self.img_res)
# img_B=imresize(img_B,self.img_res)
# img_A=np.expand_dims(img_A,axis=-1)
# img_B=np.expand_dims(img_B,axis=-1)
            
# def unpack_npz_file(path):
#     if os.stat(path).st_size == 0:
#         print('File is empty'); os.remove(path);   return None
#     with open(path,'rb') as f:
#         data=np.load(f,allow_pickle=True)
#         x=data['input_data']
#         t=data['output_data']
#         # t=F_smooth(data['models'],sigma_val=int(450/data['dx']))-data['models_init']
#         data.close()
#         # x=np.repeat(x,3,axis=3);
#         # t=np.repeat(t,3,axis=3);
#         x=np.squeeze(x)
#         t=np.squeeze(t)
#     return x,t
# def load_npz_file(file_path,data_type,in_out_shape='not_equal',smoothing_radius=450):
#     with open(file_path, 'rb') as f:
#         data=np.load(f,allow_pickle=True)
#         if data_type=='x':
#             x=data['input_data']
#         else:
#             if in_out_shape=='equal':
#                 x=F_smooth(data['models'],sigma_val=int(smoothing_radius/data['dx']))-data['models_init']
#                 # x=data['models']-data['models_init']
#             elif in_out_shape=='not_equal':
#                 x=data['output_data']
#             elif in_out_shape=='special':
#                 x=F_smooth(data['models'],sigma_val=int(smoothing_radius/data['dx']))-data['models_init']
#                 x=imresize(x,(1,68,60,1))
#         data.close()
#         x=imresize(x,(1,256,256,1))
#     return x

# Plot_image(x_all_[i_x,:,:,0].T,Show_flag=1,Save_flag=1,Title='x_all_'+str(i_x),
#     Aspect='equal',Save_pictures_path=self.save_path)
# Plot_image(t_all_[i_x,:,:,0].T,Show_flag=1,Save_flag=1,Title='t_all_'+str(i_x),
#     Aspect='equal',Save_pictures_path=self.save_path)
# Plot_image(t_predicted_all[i_x,:,:,0].T,Show_flag=1,Save_flag=1,Title='t_predicted_all'+str(i_x),
#     Aspect='equal',Save_pictures_path=self.save_path)
