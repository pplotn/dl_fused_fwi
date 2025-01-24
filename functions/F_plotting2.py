import os
print(os.getcwd())
from functions.F_modules import *
from functions.F_fwi import *
water_start=0
water_start=33
score_to_use='PCC'
# score_to_use='MSE'
# score_to_use='r2'
#############################   seg abstract scripts 25.09.21
def plot_method_picture():
    root1=os.path.join('./fwi','ws_fwi_35_strategy_l5_0','Overthrust_cnn_w_1401')
    prediction_path = './predictions/predictions_1401'
    prediction_number=prediction_path.split('/')[-1]
    prediction_number=prediction_number.split('predictions_')[-1]
    models = ['lin_vp_long']
    data_types = ['cnn']
    aa=1
    return None
def comparison_initial_models_with_fwi(log_path='./',save_path='./'):
    ################################    Marmousi
    root1=os.path.join('./fwi','ws_fwi_35_strategy_l5_0')
    dirs=next(os.walk(os.path.join(root1)))[1]
    paths=[ os.path.join(root1,'Marmousi_f_z'),
            os.path.join(root1,'Marmousi_cnn_w_1351'),
            os.path.join(root1,'Marmousi_true') ]
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
    fig_size[0] = 12.4
    fig_size[1] = 8.0   #height
    plt.rcParams["figure.figsize"] = fig_size
    plt.rcParams["font.size"] = 14
    n_row=3;    n_col=2
    gs = gridspec.GridSpec(n_row,n_col)
    gs.update(left=0.0, right=0.92, wspace=0.0, hspace=0.2)
    # gs=GridSpec(n_row,n_col+1, width_ratios=[1,1,0.1], height_ratios=[1,1,1])
    # gs.update(wspace=0.0,hspace=0.0)
    ax=[None]*6
    labels=['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o']
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
    save_file_path2=os.path.join(log_path,'marm.png')
    plt.savefig(save_file_path2,dpi=400,bbox_inches='tight')
    plt.savefig(os.path.join(save_path,'marm.png'),dpi=400,bbox_inches='tight')
    print('Saving ML_result to '+save_file_path2);  # plt.show()
    plt.close()
    ################################
    # root1=os.path.join('./fwi','ws_fwi_47_strategy_l5_0')
    # dirs=next(os.walk(os.path.join(root1)))[1]
    # paths=[ os.path.join(root1,dirs[4]),os.path.join(root1,dirs[5]),os.path.join(root1,dirs[2])     ]
    root1=os.path.join('./fwi','ws_fwi_35_strategy_l5_0')
    dirs=next(os.walk(os.path.join(root1)))[1]
    paths=[os.path.join(root1,'Overthrust_f_z'),
        os.path.join(root1,'Overthrust_cnn_w_1351'),
        os.path.join(root1,'Overthrust_true') ]
    d=[]
    for path in paths:
        d.append(parse_denise_folder(path,denise_root='./') )
    ################################
    labels_Fontsize=18
    text_Fontsize=30
    labelsize=14
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 12.4
    fig_size[1] = 8.0   #height
    plt.rcParams["figure.figsize"] = fig_size
    plt.rcParams["font.size"] = 14
    n_row=3;    n_col=2
    gs = gridspec.GridSpec(n_row,n_col)
    gs.update(left=0.0, right=0.92, wspace=0.0, hspace=0.2)
    # gs=GridSpec(n_row,n_col+1, width_ratios=[1,1,0.1], height_ratios=[1,1,1])
    # gs.update(wspace=0.0,hspace=0.0)
    ax=[None]*6
    labels=['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o']
    a=1500;b=4500
    Fontsize=32    
    textFontsize=20
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
    save_file_path2=os.path.join(log_path,'over.png')
    plt.savefig(save_file_path2,dpi=400,bbox_inches='tight')
    plt.savefig(os.path.join(save_path,'over.png'),dpi=400,bbox_inches='tight')
    print('Saving ML_result to '+save_file_path2);  # plt.show()
    plt.close()
    ################################
    return None
def comparison_initial_models_with_fwi_misfits(paths,fname,log_path='./',save_path='./'):
    compare_with='true_model'
    # compare_with='cnn_target_model'
    plot_model_misfit=0
    d=[]
    for path in paths:
        d.append(parse_denise_folder(path,denise_root='./') )
    data_availability=[]
    for data_ in d:
        if hasattr(data_,'fwi_model_names')==False:
            data_availability.append(False)
        else:
            data_availability.append(True)
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
    # labels=['a','a','b','b','c','c','a','b','c','j','k','l','m','n','o']
    labels=['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o']
    a=1500;b=4500

    if data_availability[2]==True and compare_with!='true_model':
        D=d[2]
        nx_orig=D.NX
        cnn_target_model=D.model_init.vp

    fig=plt.figure()
    j=0;D=d[j]
    if data_availability[j]==True:
        nx_orig=D.NX        #-320
        x = np.arange(nx_orig) * D.DH / 1000
        y = np.arange(D.NY) * D.DH / 1000
        extent = np.array([x.min(), x.max(), y.min(), y.max()])
        ind=D.fwi_model_names.index( fnmatch.filter(D.fwi_model_names,'*vp_stage*')[-1] )
        fwi_res=D.models[ind]
        i=0; row=0; col=0
        ax[i]=fig.add_subplot(gs[row,col])
        ax[i].axes.xaxis.set_visible(False)
        # ax[i].axes.yaxis.set_visible(False)
        ax[i].text(0.4,0.7, labels[i], fontsize=Fontsize,color = "white",weight="bold")
        if compare_with=='true_model':
            score=numstr_3(F_r2(D.model_init.vp[:,0:nx_orig],D.model.vp[:,0:nx_orig]))
            ax[i].set_title('R2(initial,true)='+score,fontsize=textFontsize)
        else:
            score2=numstr_3(F_r2(D.model_init.vp[:,0:nx_orig],cnn_target_model[:,0:nx_orig]))
            ax[i].set_title('R2(initial,CNN-target initial)='+score2,fontsize=textFontsize)
        ax[i].imshow(D.model_init.vp[:,0:nx_orig],extent=extent,vmin=a,vmax=b,aspect='auto');ax[i].invert_yaxis()
        ax[i].set_ylabel('Depth, km',fontsize=labels_Fontsize)
        ax[i].tick_params(labelsize=labelsize)
        i=1; row=0; col=1
        ax[i]=fig.add_subplot(gs[row,col]); ax[i].axes.xaxis.set_visible(False);  ax[i].axes.yaxis.set_visible(False)
        ax[i].text(0.4,0.7, labels[i], fontsize=Fontsize,color = "white",weight="bold")
        score=numstr_3(F_r2(fwi_res[:,0:nx_orig],D.model.vp[:,0:nx_orig]))
        ax[i].set_title('R2(FWI result,true)='+score,fontsize=textFontsize)
        ax[i].tick_params(labelsize=labelsize)
        last_image=ax[i].imshow(fwi_res[:,0:nx_orig],extent=extent,vmin=a,vmax=b,aspect='auto');ax[i].invert_yaxis()
    ##
    j=1;D=d[j]
    if data_availability[j]==True:
        nx_orig=D.NX
        x = np.arange(nx_orig) * D.DH / 1000
        y = np.arange(D.NY) * D.DH / 1000
        extent = np.array([x.min(), x.max(), y.min(), y.max()])
        ind=D.fwi_model_names.index( fnmatch.filter(D.fwi_model_names,'*vp_stage*')[-1] )
        fwi_res=D.models[ind]
        i=2; row=1; col=0
        ax[i]=fig.add_subplot(gs[row,col]); 
        ax[i].axes.xaxis.set_visible(False);  
        # ax[i].axes.yaxis.set_visible(False)
        ax[i].text(0.4,0.7, labels[i], fontsize=Fontsize,color = "white",weight="bold")
        if compare_with=='true_model':
            score=numstr_3(F_r2(D.model_init.vp[:,0:nx_orig],D.model.vp[:,0:nx_orig]))
            ax[i].set_title('R2(initial,true)='+score,fontsize=textFontsize)
        else:
            score2=numstr_3(F_r2(D.model_init.vp[:,0:nx_orig],cnn_target_model[:,0:nx_orig]))
            ax[i].set_title('R2(initial,CNN-target initial)='+score2,fontsize=textFontsize)
        ax[i].imshow(D.model_init.vp[:,0:nx_orig],extent=extent,vmin=a,vmax=b,aspect='auto');ax[i].invert_yaxis()
        ax[i].tick_params(labelsize=labelsize)
        ax[i].set_ylabel('Depth, km',fontsize=labels_Fontsize)
        i=3; row=1; col=1
        ax[i]=fig.add_subplot(gs[row,col]); ax[i].axes.xaxis.set_visible(False);  ax[i].axes.yaxis.set_visible(False)
        ax[i].text(0.4,0.7, labels[i], fontsize=Fontsize,color = "white",weight="bold")
        score=numstr_3(F_r2(fwi_res[:,0:nx_orig],D.model.vp[:,0:nx_orig]))
        ax[i].set_title('R2(FWI result,true)='+score,fontsize=textFontsize)
        ax[i].tick_params(labelsize=labelsize)
        last_image=ax[i].imshow(fwi_res[:,0:nx_orig],extent=extent,vmin=a,vmax=b,aspect='auto');ax[i].invert_yaxis()
    ##
    j=2;D=d[j]
    if data_availability[j]==True:
        nx_orig=D.NX
        x = np.arange(nx_orig) * D.DH / 1000
        y = np.arange(D.NY) * D.DH / 1000
        extent = np.array([x.min(), x.max(), y.min(), y.max()])

        ind=D.fwi_model_names.index( fnmatch.filter(D.fwi_model_names,'*vp_stage*')[-1] )
        fwi_res=D.models[ind]
        i=4; row=2; col=0
        ax[i]=fig.add_subplot(gs[row,col]); 
        # ax[i].axes.xaxis.set_visible(False);  
        # ax[i].axes.yaxis.set_visible(False)
        ax[i].text(0.4,0.7, labels[i], fontsize=Fontsize,color = "white",weight="bold")
        if compare_with=='true_model':
            score=numstr_3(F_r2(D.model_init.vp[:,0:nx_orig],D.model.vp[:,0:nx_orig]))
            ax[i].set_title('R2(initial,true)='+score,fontsize=textFontsize)
        else:
            score2=numstr_3(F_r2(D.model_init.vp[:,0:nx_orig],cnn_target_model[:,0:nx_orig]))
            ax[i].set_title('R2(initial,CNN-target initial)='+score2,fontsize=textFontsize)
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
    niter_list=[];
    misfit_list=[]
    model_misfit_list=[]
    j=0;D=d[j];path=paths[j]
    if data_availability[j]==True:
        i=6; row=0; col=2
        fwi_vp_names=fnmatch.filter(D.fwi_model_names,'*vp_stage*')
        model_misfit=np.empty((len(fwi_vp_names)))
        iteration2=np.arange(len(fwi_vp_names))
        for iter,fwi_vp_name in enumerate(fwi_vp_names):
            fwi_res=D.models[D.fwi_model_names.index(fwi_vp_name)]
            # score=numstr_3(F_r2(D.model_init.vp[:,0:nx_orig],D.model.vp[:,0:nx_orig]))
            score=      F_r2(fwi_res,D.model.vp)
            model_misfit[iter]=score
        m,m_data=plot_fwi_misfit2(os.path.join(path,'fld','seis_fwi_log.dat'),D)
        nstage=m_data['nstage']
        niter_stage=m_data['niter_stage'];
        nstage_trans=m_data['nstage_trans']
        iteration=m_data['iteration']
        niter_list.append(len(iteration))
        misfit_list.append(m)
        ax[i]=fig.add_subplot(gs[row,col])
        if plot_model_misfit==1:
            ax_extra=ax[i].twinx()
            ax_extra.plot(iteration2,model_misfit,'g-', linewidth=3, label='R2 model misfit') #Evolution of the misfit function
            ax_extra.axes.yaxis.set_visible(False)
        ax[i].plot(iteration,m,'b-', linewidth=3, label='Evolution of the misfit function')
        ax[i].tick_params(labelsize=labelsize)
        for ii in range(1, nstage):
            ax[i].plot([nstage_trans[ii-1]+ii,nstage_trans[ii-1]+ii], [np.min(m),np.max(m)],'k--', linewidth=3)
        ax[i].axes.yaxis.set_visible(False)
    j=1;D=d[j];path=paths[j]
    if data_availability[j]==True:
        i=7; row=1; col=2
        fwi_vp_names=fnmatch.filter(D.fwi_model_names,'*vp_stage*')
        model_misfit=np.empty((len(fwi_vp_names)))
        iteration2=np.arange(len(fwi_vp_names))
        for iter,fwi_vp_name in enumerate(fwi_vp_names):
            fwi_res=D.models[D.fwi_model_names.index(fwi_vp_name)]
            # score=numstr_3(F_r2(D.model_init.vp[:,0:nx_orig],D.model.vp[:,0:nx_orig]))
            score=      F_r2(fwi_res,D.model.vp)
            model_misfit[iter]=score
        m,m_data=plot_fwi_misfit2(os.path.join(path,'fld','seis_fwi_log.dat'),D)
        nstage=m_data['nstage']
        niter_stage=m_data['niter_stage'];
        nstage_trans=m_data['nstage_trans']
        iteration=m_data['iteration']
        niter_list.append(len(iteration))
        misfit_list.append(m)
        ax[i]=fig.add_subplot(gs[row,col])
        if plot_model_misfit==1:
            ax_extra=ax[i].twinx()
            ax_extra.plot(iteration2,model_misfit,'g-', linewidth=3, label='R2 model misfit') #Evolution of the misfit function
            ax_extra.axes.yaxis.set_visible(False)
        ax[i].plot(iteration,m,'b-', linewidth=3, label='Evolution of the misfit function')
        ax[i].tick_params(labelsize=labelsize)
        for ii in range(1, nstage):
            ax[i].plot([nstage_trans[ii-1]+ii,nstage_trans[ii-1]+ii], [np.min(m),np.max(m)],'k--', linewidth=3)
        ax[i].axes.yaxis.set_visible(False)
    j=2;D=d[j];path=paths[j]
    if data_availability[j]==True:
        i=8; row=2; col=2
        fwi_vp_names=fnmatch.filter(D.fwi_model_names,'*vp_stage*')
        model_misfit=np.empty((len(fwi_vp_names)))
        iteration2=np.arange(len(fwi_vp_names))
        for iter,fwi_vp_name in enumerate(fwi_vp_names):
            fwi_res=D.models[D.fwi_model_names.index(fwi_vp_name)]
            # score=numstr_3(F_r2(D.model_init.vp[:,0:nx_orig],D.model.vp[:,0:nx_orig]))
            score=      F_r2(fwi_res,D.model.vp)
            model_misfit[iter]=score
        m,m_data=plot_fwi_misfit2(os.path.join(path,'fld','seis_fwi_log.dat'),D)
        nstage=m_data['nstage']
        niter_stage=m_data['niter_stage'];
        nstage_trans=m_data['nstage_trans']
        iteration=m_data['iteration']
        niter_list.append(len(iteration))
        misfit_list.append(m)
        ax[i]=fig.add_subplot(gs[row,col])
        if plot_model_misfit==1:
            ax_extra=ax[i].twinx()
            ax_extra.plot(iteration2,model_misfit,'g-', linewidth=3, label='R2 model misfit') #Evolution of the misfit function
            ax_extra.axes.yaxis.set_visible(False)
        ax[i].plot(iteration,m,'b-', linewidth=3, label='Evolution of the misfit function')
        ax[i].tick_params(labelsize=labelsize)
        for ii in range(1, nstage):
            ax[i].plot([nstage_trans[ii-1]+ii,nstage_trans[ii-1]+ii], [np.min(m),np.max(m)],'k--', linewidth=3)
        ax[i].axes.yaxis.set_visible(False)
    ######  x limits
    niter_list=np.asarray(niter_list)
    misfit_list=np.asarray(misfit_list)
    mf_min=0;mf_max=0;
    for misfit_curve in misfit_list:
        mf_min=np.min([np.min(misfit_curve),mf_min])
        mf_max=np.max([np.max(misfit_curve),mf_max])
    y_pos=mf_min+(mf_max-mf_min)/5*4
    if data_availability[0]==True:
        ax[6].set_xlim(left=1,right=np.max(niter_list))
        ax[6].set_ylim(bottom=mf_min,top=mf_max)
        # ax[6].text(10,y_pos, labels[6], fontsize=Fontsize,color = "black",weight="bold")
    if data_availability[1]==True:
        ax[7].set_xlim(left=1,right=np.max(niter_list))
        ax[7].set_ylim(bottom=mf_min,top=mf_max)
        # ax[7].text(10,y_pos, labels[7], fontsize=Fontsize,color = "black",weight="bold")
    if data_availability[2]==True:
        ax[8].set_xlim(left=1,right=np.max(niter_list))
        ax[8].set_ylim(bottom=mf_min,top=mf_max)
        # ax[8].text(10,y_pos, labels[8], fontsize=Fontsize,color = "black",weight="bold")
    ######  saving
    save_file_path1=os.path.join(save_path,fname)
    save_file_path2=os.path.join(log_path,fname)
    print('Saving ML_result to '+save_file_path1+'  '+save_file_path2)
    plt.savefig(save_file_path1,dpi=300,bbox_inches='tight')
    plt.savefig(save_file_path2,dpi=300,bbox_inches='tight')
    # plt.show()
    plt.close()
    return None
def comparison_initial_models_with_fwi_FFT_spectrums(paths,fname,log_path='./',save_path='./',pclip=0.002,fmax=10,kmax=5):
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
    fig_size[1] = 8.0/3*6   #height
    plt.rcParams["figure.figsize"] = fig_size
    plt.rcParams["font.size"] = 14
    n_row=6;    n_col=3
    gs = gridspec.GridSpec(n_row,n_col)
    gs.update(left=0.0, right=0.92, wspace=0.0, hspace=0.2)
    # gs=GridSpec(n_row,n_col+1, width_ratios=[1,1,0.1], height_ratios=[1,1,1])
    # gs.update(wspace=0.0,hspace=0.0)
    ax=[None]*n_row*n_col
    labels=['a','a','b','b','c','c','d','d','e','e','f','g']
    a=1500;b=4500

    fig=plt.figure()
    niter_list=[];misfit_list=[]
    j=0;D=d[j];path=paths[j]
    nx_orig=D.NX-320
    ind=D.fwi_model_names.index(os.path.join(D.save_folder,'model','modelTest_vp.bin'))
    ind=D.fwi_model_names.index( fnmatch.filter(D.fwi_model_names,'*vp_stage*')[-1] )
    fwi_res=D.models[ind]
    x = np.arange(nx_orig) * D.DH / 1000
    y = np.arange(D.NY) * D.DH / 1000
    extent = np.array([x.min(), x.max(), y.min(), y.max()])

    i=0;row=int(np.floor(i/n_col));col=i-row*n_col;
    ax[i]=fig.add_subplot(gs[row,col])
    ax[i].axes.xaxis.set_visible(False)
    # ax[i].axes.yaxis.set_visible(False)
    ax[i].text(0.4,0.7, labels[row], fontsize=Fontsize,color = "white",weight="bold")
    score=numstr_3(F_r2(D.model_init.vp[:,0:nx_orig],D.model.vp[:,0:nx_orig]))
    ax[i].set_title('R2(initial,true)='+score,fontsize=textFontsize)
    ax[i].imshow(D.model_init.vp[:,0:nx_orig],extent=extent,vmin=a,vmax=b,aspect='auto');ax[i].invert_yaxis()
    ax[i].set_ylabel('Depth, km',fontsize=labels_Fontsize)
    ax[i].tick_params(labelsize=labelsize)

    i=1;row=int(np.floor(i/n_col));col=i-row*n_col;
    ax[i]=fig.add_subplot(gs[row,col]); ax[i].axes.xaxis.set_visible(False);  ax[i].axes.yaxis.set_visible(False)
    ax[i].text(0.4,0.7, labels[row], fontsize=Fontsize,color = "white",weight="bold")
    score=numstr_3(F_r2(fwi_res[:,0:nx_orig],D.model.vp[:,0:nx_orig]))
    ax[i].set_title('R2(FWI result,true)='+score,fontsize=textFontsize)
    ax[i].imshow(fwi_res[:,0:nx_orig],extent=extent,vmin=a,vmax=b,aspect='auto');ax[i].invert_yaxis()
    ax[i].tick_params(labelsize=labelsize)

    i=2;row=int(np.floor(i/n_col));col=i-row*n_col;
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

    dt=D.DT;    dx=D.DH

    mat_a=D.model_init.vp[:,0:nx_orig].T
    mat_b=fwi_res[:,0:nx_orig].T
    mat_a_spectrum, fax, kax = get_spectrum2_model(mat_a, dx, dx, fmax, kmax)
    mat_b_spectrum, fax, kax = get_spectrum2_model(mat_b, dx, dx, fmax, kmax)
    spec_min=0;    spec_max=pclip*np.max(np.abs(mat_a_spectrum))
    i=3;row=int(np.floor(i/n_col));col=i-row*n_col;
    ax[i]=fig.add_subplot(gs[row,col]); 
    # ax[i].axes.xaxis.set_visible(False);  ax[i].axes.yaxis.set_visible(False)
    # ax[i].text(0.4,0.7, labels[row], fontsize=Fontsize,color = "white",weight="bold")
    ax[i].imshow(mat_a_spectrum.T, cmap='RdBu_r',vmin=spec_min,vmax=spec_max)
    ax[i].tick_params(labelsize=labelsize)

    i=4;row=int(np.floor(i/n_col));col=i-row*n_col;
    ax[i]=fig.add_subplot(gs[row,col]); 
    ax[i].axes.xaxis.set_visible(False);  ax[i].axes.yaxis.set_visible(False)
    # ax[i].text(0.4,0.7, labels[row], fontsize=Fontsize,color = "white",weight="bold")
    ax[i].imshow(mat_b_spectrum.T, cmap='RdBu_r',vmin=spec_min,vmax=spec_max)
    ax[i].tick_params(labelsize=labelsize)
    ##
    j=1;D=d[j];path=paths[j]
    ind=D.fwi_model_names.index(os.path.join(D.save_folder,'model','modelTest_vp.bin'))
    ind=D.fwi_model_names.index( fnmatch.filter(D.fwi_model_names,'*vp_stage*')[-1] )
    fwi_res=D.models[ind]
    extent = np.array([x.min(), x.max(), y.min(), y.max()])

    i=6;row=int(np.floor(i/n_col));col=i-row*n_col;
    ax[i]=fig.add_subplot(gs[row,col]); 
    ax[i].axes.xaxis.set_visible(False); # ax[i].axes.yaxis.set_visible(False)
    # ax[i].text(0.4,0.7, labels[row], fontsize=Fontsize,color = "white",weight="bold")
    score=numstr_3(F_r2(D.model_init.vp[:,0:nx_orig],D.model.vp[:,0:nx_orig]))
    ax[i].set_title('R2(initial,true)='+score,fontsize=textFontsize)
    ax[i].imshow(D.model_init.vp[:,0:nx_orig],extent=extent,vmin=a,vmax=b,aspect='auto');ax[i].invert_yaxis()
    ax[i].tick_params(labelsize=labelsize)
    ax[i].set_ylabel('Depth, km',fontsize=labels_Fontsize)

    i=7;row=int(np.floor(i/n_col));col=i-row*n_col;
    ax[i]=fig.add_subplot(gs[row,col]); ax[i].axes.xaxis.set_visible(False);  ax[i].axes.yaxis.set_visible(False)
    # ax[i].text(0.4,0.7, labels[row], fontsize=Fontsize,color = "white",weight="bold")
    score=numstr_3(F_r2(fwi_res[:,0:nx_orig],D.model.vp[:,0:nx_orig]))
    ax[i].set_title('R2(FWI result,true)='+score,fontsize=textFontsize)
    ax[i].imshow(fwi_res[:,0:nx_orig],extent=extent,vmin=a,vmax=b,aspect='auto');ax[i].invert_yaxis()
    ax[i].tick_params(labelsize=labelsize)

    i=8;row=int(np.floor(i/n_col));col=i-row*n_col;
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

    mat_a=D.model_init.vp[:,0:nx_orig].T
    mat_b=fwi_res[:,0:nx_orig].T
    mat_a_spectrum, fax, kax = get_spectrum2_model(mat_a, dx, dx, fmax, kmax)
    mat_b_spectrum, fax, kax = get_spectrum2_model(mat_b, dx, dx, fmax, kmax)
    i=9;row=int(np.floor(i/n_col));col=i-row*n_col;
    ax[i]=fig.add_subplot(gs[row,col]); ax[i].axes.xaxis.set_visible(False);  ax[i].axes.yaxis.set_visible(False)
    # ax[i].text(0.4,0.7, labels[row], fontsize=Fontsize,color = "white",weight="bold")
    ax[i].imshow(mat_a_spectrum.T, cmap='RdBu_r',vmin=spec_min,vmax=spec_max)
    ax[i].tick_params(labelsize=labelsize)

    i=10;row=int(np.floor(i/n_col));col=i-row*n_col;
    ax[i]=fig.add_subplot(gs[row,col]); ax[i].axes.xaxis.set_visible(False);  ax[i].axes.yaxis.set_visible(False)
    # ax[i].text(0.4,0.7, labels[row], fontsize=Fontsize,color = "white",weight="bold")
    ax[i].imshow(mat_b_spectrum.T, cmap='RdBu_r',vmin=spec_min,vmax=spec_max)
    ax[i].tick_params(labelsize=labelsize)
    ##
    j=2;D=d[j];path=paths[j]
    ind=D.fwi_model_names.index(os.path.join(D.save_folder,'model','modelTest_vp.bin'))
    ind=D.fwi_model_names.index( fnmatch.filter(D.fwi_model_names,'*vp_stage*')[-1] )
    fwi_res=D.models[ind]
    extent = np.array([x.min(), x.max(), y.min(), y.max()])

    i=12;row=int(np.floor(i/n_col));col=i-row*n_col;
    ax[i]=fig.add_subplot(gs[row,col]); 
    ax[i].text(0.4,0.7, labels[row], fontsize=Fontsize,color = "white",weight="bold")
    score=numstr_3(F_r2(D.model_init.vp[:,0:nx_orig],D.model.vp[:,0:nx_orig]))
    ax[i].set_title('R2(initial,true)='+score,fontsize=textFontsize)
    ax[i].imshow(D.model_init.vp[:,0:nx_orig],extent=extent,vmin=a,vmax=b,aspect='auto');ax[i].invert_yaxis()
    ax[i].set_xlabel('X, km',fontsize=labels_Fontsize)
    ax[i].set_ylabel('Depth, km',fontsize=labels_Fontsize)
    ax[i].tick_params(labelsize=labelsize)

    i=13;row=int(np.floor(i/n_col));col=i-row*n_col;
    ax[i]=fig.add_subplot(gs[row,col]); 
    # ax[i].axes.xaxis.set_visible(False);  
    ax[i].axes.yaxis.set_visible(False)
    ax[i].text(0.4,0.7, labels[row], fontsize=Fontsize,color = "white",weight="bold")
    ax[i].set_xlabel('X, km',fontsize=labels_Fontsize)
    score=numstr_3(F_r2(fwi_res[:,0:nx_orig],D.model.vp[:,0:nx_orig]))
    ax[i].set_title('R2(FWI result,true)='+score,fontsize=textFontsize)
    ax[i].tick_params(labelsize=labelsize)
    last_image=ax[i].imshow(fwi_res[:,0:nx_orig],extent=extent,vmin=a,vmax=b,aspect='auto');ax[i].invert_yaxis()

    i=14;row=int(np.floor(i/n_col));col=i-row*n_col;
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
    #####################
    mat_a=D.model_init.vp[:,0:nx_orig].T
    mat_b=fwi_res[:,0:nx_orig].T
    mat_a_spectrum, fax, kax = get_spectrum2_model(mat_a, dx, dx, fmax, kmax)
    mat_b_spectrum, fax, kax = get_spectrum2_model(mat_b, dx, dx, fmax, kmax)
    i=15;row=int(np.floor(i/n_col));col=i-row*n_col;
    ax[i]=fig.add_subplot(gs[row,col]); ax[i].axes.xaxis.set_visible(False);  ax[i].axes.yaxis.set_visible(False)
    # ax[i].text(0.4,0.7, labels[row], fontsize=Fontsize,color = "white",weight="bold")
    ax[i].imshow(mat_a_spectrum.T, cmap='RdBu_r',vmin=spec_min,vmax=spec_max)
    ax[i].tick_params(labelsize=labelsize)

    i=16;row=int(np.floor(i/n_col));col=i-row*n_col;
    ax[i]=fig.add_subplot(gs[row,col]); ax[i].axes.xaxis.set_visible(False);  ax[i].axes.yaxis.set_visible(False)
    # ax[i].text(0.4,0.7, labels[row], fontsize=Fontsize,color = "white",weight="bold")
    # ax[i].imshow(mat_b_spectrum.T, cmap='RdBu_r',vmin=spec_min,vmax=spec_max,extent=[np.min(kax),np.max(kax),np.min(fax),np.max(fax)])
    ax[i].imshow(mat_b_spectrum.T, cmap='RdBu_r',vmin=spec_min,vmax=spec_max)
    ax[i].tick_params(labelsize=labelsize)
    ##
    cbar_ax = fig.add_axes([0.93, 0.12, 0.02, 0.75])
    cbar=fig.colorbar(last_image,cax=cbar_ax)
    # cbar.set_label('V (km/sec)',labelpad=-40, y=1.05, rotation=0)   #horizontalalignment='center'
    cbar.ax.set_title('V (m/sec)',fontsize=18,pad=13.3)
    for t in cbar.ax.get_yticklabels():
        t.set_fontsize(16)
    ######  x limits for misfits
    # niter_list=np.asarray(niter_list)
    # misfit_list=np.asarray(misfit_list)
    mf_min=0;mf_max=0;
    for misfit_curve in misfit_list:
        print(misfit_curve.shape)
        mf_min=np.min([np.min(misfit_curve),mf_min])
        mf_max=np.max([np.max(misfit_curve),mf_max])
    ax[2].set_xlim(left=1,right=np.max(niter_list))
    ax[8].set_xlim(left=1,right=np.max(niter_list))
    ax[14].set_xlim(left=1,right=np.max(niter_list))
    ax[2].set_ylim(bottom=mf_min,top=mf_max)
    ax[8].set_ylim(bottom=mf_min,top=mf_max)
    ax[14].set_ylim(bottom=mf_min,top=mf_max)
    y_pos=mf_min+(mf_max-mf_min)/5*4
    ax[2].text(10,y_pos, labels[0], fontsize=Fontsize,color = "black",weight="bold")
    ax[8].text(10,y_pos, labels[1], fontsize=Fontsize,color = "black",weight="bold")
    ax[14].text(10,y_pos, labels[2], fontsize=Fontsize,color = "black",weight="bold")
    ######  saving
    save_file_path1=os.path.join(save_path,fname)
    save_file_path2=os.path.join(log_path,fname)
    print('Saving ML_result to '+save_file_path1+'  '+save_file_path2)
    plt.savefig(save_file_path1,dpi=300,bbox_inches='tight')
    plt.savefig(save_file_path2,dpi=300,bbox_inches='tight')
    # plt.show()
    plt.close()
    return None
def comparison_initial_models_with_fwi_cgg_models(log_path='./',save_path='./'):
    """  deprecated """
    root1=os.path.join('./fwi','cgg_real_data','fwi_47_strategy_l5_weight_1351_0')  #cnn
    dirs1=next(os.walk(os.path.join(root1)))[1]
    root2=os.path.join('./fwi','cgg_real_data','fwi_47_strategy_l5_weight_1351_1')  #1d
    dirs2=next(os.walk(os.path.join(root1)))[1]
    # l5
    # root0=os.path.join('./fwi','cgg_real_data');    root0=os.path.join('./fwi')
    # root1=os.path.join(root0,'fwi_35_strategy_l5_weight_1351_1')  #cnn
    # dirs1=next(os.walk(os.path.join(root1)))[1]
    # root2=os.path.join(root0,'fwi_35_strategy_l5_weight_1351_0')  #1d
    # dirs2=next(os.walk(os.path.join(root1)))[1]
    # l2
    # root1=os.path.join('./fwi','cgg_real_data','fwi_35_strategy_l2_weight_1351_2')  #cnn
    # dirs1=next(os.walk(os.path.join(root1)))[1]
    # root2=os.path.join('./fwi','cgg_real_data','fwi_35_strategy_l2_weight_1351_1')  #1d
    # dirs2=next(os.walk(os.path.join(root1)))[1]
    ################################
    paths=[
        os.path.join(root2,dirs2[0]),    #row 0
        os.path.join(root1,dirs1[0]),    #row 1
        ]
    d=[]
    for path in paths:
        d.append(parse_denise_folder(path,denise_root='./') )
    ################################
    dx=25
    info_file=os.path.join('./','for_pasha','acq_data_parameters_cgg.pkl')
    with open(info_file,'rb') as input:
        acq_data=pickle.load(input)
    log_dict=acq_data['log_dict']
    log_loc=log_dict['loc']
    log=log_dict['data']
    log_idx = int(log_loc / 25)
    vh = log_loc * np.ones_like(log) / 1000 
    ################################
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 12.4
    fig_size[1] = 5.0   #height
    plt.rcParams["figure.figsize"] = fig_size
    plt.rcParams["font.size"] = 14
    n_row=2;    n_col=2
    gs = gridspec.GridSpec(n_row,n_col)
    gs.update(left=0.0, right=0.92, wspace=0.0, hspace=0.0)
    # gs=GridSpec(n_row,n_col+1, width_ratios=[1,1,0.1], height_ratios=[1,1,1])
    # gs.update(wspace=0.0,hspace=0.0)
    ax=[None]*6
    labels=['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o']
    a=1500;b=3000
    Fontsize=32
    labels_Fontsize=18
    text_Fontsize=30
    labelsize=14
    Fontsize=32    
    textFontsize=20
    fig=plt.figure()
    j=0;D=d[j]
    nx_orig=600
    ind=D.fwi_model_names.index(os.path.join(D.save_folder,'model','modelTest_vp.bin'))
    ind=D.fwi_model_names.index( fnmatch.filter(D.fwi_model_names,'*vp_stage*')[-1] )
    print(D.fwi_model_names[ind])
    fwi_res=D.models[ind]
    x = np.arange(nx_orig) * D.DH / 1000
    y = np.arange(D.NY) * D.DH / 1000
    extent = np.array([x.min(), x.max(), y.min(), y.max()])
    i=0; row=0; col=0
    ax[i]=fig.add_subplot(gs[row,col]); 
    ax[i].axes.xaxis.set_visible(False)
    ax[i].text(0.4,0.7, labels[i], fontsize=Fontsize,color = "white",weight="bold")
    ax[i].imshow(D.model_init.vp[:,0:nx_orig],extent=extent,vmin=a,vmax=b,aspect='auto');ax[i].invert_yaxis()
    ax[i].plot(vh, np.arange(len(log))*dx/1000, 'k--')
    ax[i].plot(vh + (log[::-1] - min(log)) / 1000, np.arange(len(log))*dx/1000, 'k')
    ax[i].set_ylabel('Depth, km',fontsize=labels_Fontsize)
    ax[i].tick_params(labelsize=labelsize)
    i=1; row=0; col=1
    ax[i]=fig.add_subplot(gs[row,col]); ax[i].axes.xaxis.set_visible(False);  ax[i].axes.yaxis.set_visible(False)
    ax[i].text(0.4,0.7, labels[i], fontsize=Fontsize,color = "white",weight="bold")
    ax[i].imshow(fwi_res[:,0:nx_orig],extent=extent,vmin=a,vmax=b,aspect='auto');ax[i].invert_yaxis()
    ax[i].plot(vh, np.arange(len(log))*dx/1000, 'k--')
    ax[i].plot(vh + (log[::-1] - min(log)) / 1000, np.arange(len(log))*dx/1000, 'k')
    ##
    j=1;D=d[j]
    ind=D.fwi_model_names.index(os.path.join(D.save_folder,'model','modelTest_vp.bin'))
    ind=D.fwi_model_names.index( fnmatch.filter(D.fwi_model_names,'*vp_stage*')[-1] )
    # ind=D.fwi_model_names.index( fnmatch.filter(D.fwi_model_names,'*vp_stage_1_it_12.bin*')[-1] )
    print(D.fwi_model_names[ind])
    fwi_res=D.models[ind]
    extent = np.array([x.min(), x.max(), y.min(), y.max()])
    i=2; row=1; col=0
    ax[i]=fig.add_subplot(gs[row,col]);
    # ax[i].axes.xaxis.set_visible(False);  ax[i].axes.yaxis.set_visible(False)
    ax[i].text(0.4,0.7, labels[i], fontsize=Fontsize,color = "white",weight="bold")
    ax[i].imshow(D.model_init.vp[:,0:nx_orig],extent=extent,vmin=a,vmax=b,aspect='auto');ax[i].invert_yaxis()
    ax[i].plot(vh, np.arange(len(log))*dx/1000, 'k--')
    ax[i].plot(vh + (log[::-1] - min(log)) / 1000, np.arange(len(log))*dx/1000, 'k')
    ax[i].set_xlabel('X, km',fontsize=labels_Fontsize)
    ax[i].set_ylabel('Depth, km',fontsize=labels_Fontsize)
    ax[i].tick_params(labelsize=labelsize)
    i=3; row=1; col=1
    ax[i]=fig.add_subplot(gs[row,col]); 
    ax[i].text(0.4,0.7, labels[i], fontsize=Fontsize,color = "white",weight="bold")
    last_image=ax[i].imshow(fwi_res[:,0:nx_orig],extent=extent,vmin=a,vmax=b,aspect='auto');ax[i].invert_yaxis()
    ax[i].plot(vh, np.arange(len(log))*dx/1000, 'k--')
    ax[i].plot(vh + (log[::-1] - min(log)) / 1000, np.arange(len(log))*dx/1000, 'k')
    ax[i].set_xlabel('X, km',fontsize=labels_Fontsize)
    # ax[i].axes.xaxis.set_visible(False);  
    ax[i].axes.yaxis.set_visible(False)
    ax[i].tick_params(labelsize=labelsize)
    cbar_ax = fig.add_axes([0.93, 0.12, 0.02, 0.75])
    cbar=fig.colorbar(last_image,cax=cbar_ax)
    # cbar.set_label('V (km/sec)',labelpad=-40, y=1.05, rotation=0)   #horizontalalignment='center'
    cbar.ax.set_title('V (m/sec)',fontsize=18,pad=13.3)
    for t in cbar.ax.get_yticklabels():
        t.set_fontsize(16)
    # cbar.set_label(fontsize=22)
    save_file_path2=os.path.join(log_path,'cgg.png')
    # plt.savefig(save_file_path2,dpi=400)
    plt.savefig(save_file_path2,dpi=400,bbox_inches='tight')
    plt.savefig(os.path.join(save_path,'cgg.png'),dpi=400,bbox_inches='tight')
    print('Saving ML_result to '+save_file_path2);  # plt.show()
    plt.close()
    ################################
    return None
def comparison_initial_models_with_fwi_cgg_models_test_function(paths,log_path='./',save_path='./',name='cgg'):
    """  """
    d=[]
    for path in paths:
        d.append(parse_denise_folder(path,denise_root='./') )
    ################################
    dx=25
    info_file=os.path.join('./','for_pasha','acq_data_parameters_cgg.pkl')
    with open(info_file,'rb') as input:
        acq_data=pickle.load(input)
    log_dict=acq_data['log_dict']
    log_loc=log_dict['loc']
    log=log_dict['data']
    log_idx = int(log_loc / 25)
    vh = log_loc * np.ones_like(log) / 1000 
    ################################
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 12.4/2*3
    fig_size[1] = 5.0   #height
    plt.rcParams["figure.figsize"] = fig_size
    plt.rcParams["font.size"] = 14
    n_row=2;    n_col=3
    gs = gridspec.GridSpec(n_row,n_col)
    gs.update(left=0.0, right=0.92, wspace=0.0, hspace=0.0)
    # gs=GridSpec(n_row,n_col+1, width_ratios=[1,1,0.1], height_ratios=[1,1,1])
    # gs.update(wspace=0.0,hspace=0.0)
    ax=[None]*6
    labels=['a','a','b','b','a','b']
    a=1500;b=3000
    Fontsize=32
    labels_Fontsize=18
    text_Fontsize=30
    labelsize=14
    Fontsize=32    
    textFontsize=20
    fig=plt.figure()
    j=0;D=d[j]
    nx_orig=600
    ind=D.fwi_model_names.index(os.path.join(D.save_folder,'model','modelTest_vp.bin'))
    ind=D.fwi_model_names.index( fnmatch.filter(D.fwi_model_names,'*vp_stage*')[-1] )
    print(D.fwi_model_names[ind])
    fwi_res=D.models[ind]
    x = np.arange(nx_orig) * D.DH / 1000
    y = np.arange(D.NY) * D.DH / 1000
    extent = np.array([x.min(), x.max(), y.min(), y.max()])
    i=0; row=0; col=0
    ax[i]=fig.add_subplot(gs[row,col]); 
    ax[i].axes.xaxis.set_visible(False)
    ax[i].text(0.4,0.7, labels[i], fontsize=Fontsize,color = "white",weight="bold")
    ax[i].imshow(D.model_init.vp[:,0:nx_orig],extent=extent,vmin=a,vmax=b,aspect='auto');ax[i].invert_yaxis()
    ax[i].plot(vh, np.arange(len(log))*dx/1000, 'k--')
    ax[i].plot(vh + (log[::-1] - min(log)) / 1000, np.arange(len(log))*dx/1000, 'k')
    ax[i].set_ylabel('Depth, km',fontsize=labels_Fontsize)
    ax[i].tick_params(labelsize=labelsize)
    i=1; row=0; col=1
    ax[i]=fig.add_subplot(gs[row,col]); ax[i].axes.xaxis.set_visible(False);  ax[i].axes.yaxis.set_visible(False)
    ax[i].text(0.4,0.7, labels[i], fontsize=Fontsize,color = "white",weight="bold")
    ax[i].imshow(fwi_res[:,0:nx_orig],extent=extent,vmin=a,vmax=b,aspect='auto');ax[i].invert_yaxis()
    ax[i].plot(vh, np.arange(len(log))*dx/1000, 'k--')
    ax[i].plot(vh + (log[::-1] - min(log)) / 1000, np.arange(len(log))*dx/1000, 'k')
    ##
    j=1;D=d[j]
    ind=D.fwi_model_names.index(os.path.join(D.save_folder,'model','modelTest_vp.bin'))
    ind=D.fwi_model_names.index( fnmatch.filter(D.fwi_model_names,'*vp_stage*')[-1] )
    # ind=D.fwi_model_names.index( fnmatch.filter(D.fwi_model_names,'*vp_stage_1_it_12.bin*')[-1] )
    print(D.fwi_model_names[ind])
    fwi_res=D.models[ind]
    extent = np.array([x.min(), x.max(), y.min(), y.max()])
    i=2; row=1; col=0
    ax[i]=fig.add_subplot(gs[row,col]);
    # ax[i].axes.xaxis.set_visible(False);  ax[i].axes.yaxis.set_visible(False)
    ax[i].text(0.4,0.7, labels[i], fontsize=Fontsize,color = "white",weight="bold")
    ax[i].imshow(D.model_init.vp[:,0:nx_orig],extent=extent,vmin=a,vmax=b,aspect='auto');ax[i].invert_yaxis()
    ax[i].plot(vh, np.arange(len(log))*dx/1000, 'k--')
    ax[i].plot(vh + (log[::-1] - min(log)) / 1000, np.arange(len(log))*dx/1000, 'k')
    ax[i].set_xlabel('X, km',fontsize=labels_Fontsize)
    ax[i].set_ylabel('Depth, km',fontsize=labels_Fontsize)
    ax[i].tick_params(labelsize=labelsize)
    i=3; row=1; col=1
    ax[i]=fig.add_subplot(gs[row,col]); 
    ax[i].text(0.4,0.7, labels[i], fontsize=Fontsize,color = "white",weight="bold")
    last_image=ax[i].imshow(fwi_res[:,0:nx_orig],extent=extent,vmin=a,vmax=b,aspect='auto');ax[i].invert_yaxis()
    ax[i].plot(vh, np.arange(len(log))*dx/1000, 'k--')
    ax[i].plot(vh + (log[::-1] - min(log)) / 1000, np.arange(len(log))*dx/1000, 'k')
    ax[i].set_xlabel('X, km',fontsize=labels_Fontsize)
    # ax[i].axes.xaxis.set_visible(False);  
    ax[i].axes.yaxis.set_visible(False)
    ax[i].tick_params(labelsize=labelsize)
    cbar_ax = fig.add_axes([0.93, 0.12, 0.02, 0.75])
    cbar=fig.colorbar(last_image,cax=cbar_ax)
    # cbar.set_label('V (km/sec)',labelpad=-40, y=1.05, rotation=0)   #horizontalalignment='center'
    cbar.ax.set_title('V (m/sec)',fontsize=18,pad=13.3)
    for t in cbar.ax.get_yticklabels():
        t.set_fontsize(16)
    #############################   misfits
    niter_list=[];misfit_list=[]
    j=0;D=d[j];path=paths[j]
    i=4; row=0; col=2
    m,m_data=plot_fwi_misfit2(os.path.join(path,'fld','seis_fwi_log.dat'),D)
    nstage=m_data['nstage']
    niter_stage=m_data['niter_stage']
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
    i=5; row=1; col=2
    m,m_data=plot_fwi_misfit2(os.path.join(path,'fld','seis_fwi_log.dat'),D)
    nstage=m_data['nstage']
    niter_stage=m_data['niter_stage']
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
    ax[i].set_xlabel('Iteration number',fontsize=labels_Fontsize)
    ######  x limits
    niter_list=np.asarray(niter_list)
    misfit_list=np.asarray(misfit_list)
    mf_min=0;mf_max=0;
    for misfit_curve in misfit_list:
        mf_min=np.min([np.min(misfit_curve),mf_min])
        mf_max=np.max([np.max(misfit_curve),mf_max])
    ax[4].set_xlim(left=1,right=np.max(niter_list))
    ax[4].set_ylim(bottom=mf_min,top=mf_max)
    ax[5].set_xlim(left=1,right=np.max(niter_list))
    ax[5].set_ylim(bottom=mf_min,top=mf_max)
    y_pos=mf_min+(mf_max-mf_min)/5*4
    ax[4].text(10,y_pos, labels[4], fontsize=Fontsize,color = "black",weight="bold")
    ax[5].text(10,y_pos, labels[5], fontsize=Fontsize,color = "black",weight="bold")
    #############################
    # cbar.set_label(fontsize=22)
    save_file_path2=os.path.join(log_path,name+'.png')
    # plt.savefig(save_file_path2,dpi=400)
    plt.savefig(save_file_path2,dpi=400,bbox_inches='tight')
    plt.savefig(os.path.join(save_path,name+'.png'),dpi=400,bbox_inches='tight')
    print('Saving ML_result to '+save_file_path2);  # plt.show()
    plt.close()
    return None
def comparison_initial_models_with_fwi_cgg_logs(log_path='./',save_path='./'):
    """  deprecated """
    root1=os.path.join('./fwi','cgg_real_data','fwi_47_strategy_l5_weight_1351_0')  #cnn
    root1=os.path.join('./fwi','cgg_real_data','fwi_47_strategy_l5_weight_1401_0')  #cnn  old
    dirs1=next(os.walk(os.path.join(root1)))[1]
    root2=os.path.join('./fwi','cgg_real_data','fwi_47_strategy_l5_weight_1351_1')  #1d
    dirs2=next(os.walk(os.path.join(root1)))[1]
    # l5
    # root0=os.path.join('./fwi','cgg_real_data');    root0=os.path.join('./fwi')
    # root1=os.path.join(root0,'fwi_35_strategy_l5_weight_1351_1')  #cnn
    # dirs1=next(os.walk(os.path.join(root1)))[1]
    # root2=os.path.join(root0,'fwi_35_strategy_l5_weight_1351_0')  #1d
    # dirs2=next(os.walk(os.path.join(root1)))[1]
    # l2
    # root1=os.path.join('./fwi','cgg_real_data','fwi_35_strategy_l2_weight_1351_2')  #cnn
    # dirs1=next(os.walk(os.path.join(root1)))[1]
    # root2=os.path.join('./fwi','cgg_real_data','fwi_35_strategy_l2_weight_1351_1')  #1d
    # dirs2=next(os.walk(os.path.join(root1)))[1]
    ################################
    paths=[
        os.path.join(root2,dirs2[0]),    #row 0
        os.path.join(root1,dirs1[0]),    #row 1
        ]
    d=[]
    for path in paths:
        d.append(parse_denise_folder(path,denise_root='./') )
    ################################
    dx=25
    info_file=os.path.join('./','for_pasha','acq_data_parameters_cgg.pkl')
    with open(info_file,'rb') as input:
        acq_data=pickle.load(input)
    log_dict=acq_data['log_dict']
    log_loc=log_dict['loc']
    log=log_dict['data']
    idx = int(log_loc / 25)
    vh = log_loc * np.ones_like(log) / 1000 
    ################################
    lvp=log[::-1]
    print(lvp.shape)
    # shear velocity, [m/s]
    lvs = lvp.copy() / (3 ** 0.5)
    lvs = np.where(lvp < 1.01 * np.min(lvp), 0, lvs)
    # density, [kg/m3] 
    lrho = 1e3*0.3 * lvp.copy()**0.25
    lrho = np.where(lvp < 1.01 * np.min(lvp), 1000, lrho)
    ################################
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 12.4
    fig_size[1] = 5.0   #height
    plt.rcParams["figure.figsize"] = fig_size
    n_row=2;    n_col=2
    gs = gridspec.GridSpec(n_row,n_col)
    gs.update(left=0.0, right=1.0, wspace=0.0, hspace=0.0)
    # gs=GridSpec(n_row,n_col+1, width_ratios=[1,1,0.1], height_ratios=[1,1,1])
    # gs.update(wspace=0.0,hspace=0.0)
    ax=[None]*6
    labels=['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o']
    a=1500;b=3000

    Fontsize=15
    text_Fontsize=30
    labelsize=14
    fig=plt.figure()
    j=0;D=d[j]
    nx_orig=600
    x = np.arange(nx_orig) * D.DH / 1000
    y = np.arange(D.NY) * D.DH / 1
    y2 = np.arange(len(lvp)) * D.DH / 1
    extent = np.array([x.min(), x.max(), y.min(), y.max()])
    ##
    j=0;D=d[j]
    tmp=fnmatch.filter(D.fwi_model_names,'*vp_stage*')[-1] 
    tmp2=tmp.split('_stage_')[-1]
    fwi_res_vp=D.models[D.fwi_model_names.index(os.path.join(D.save_folder,'model','modelTest_vp_stage_'+tmp2))]
    fwi_res_vs=D.models[D.fwi_model_names.index(os.path.join(D.save_folder,'model','modelTest_vs_stage_'+tmp2))]
    fwi_res_rho=D.models[D.fwi_model_names.index(os.path.join(D.save_folder,'model','modelTest_rho_stage_'+tmp2))]
    # fwi_res_vp=D.models[D.fwi_model_names.index(os.path.join(D.save_folder,'model','modelTest_vp.bin'))]
    # fwi_res_vs=D.models[D.fwi_model_names.index(os.path.join(D.save_folder,'model','modelTest_vs.bin'))]
    # fwi_res_rho=D.models[D.fwi_model_names.index(os.path.join(D.save_folder,'model','modelTest_rho.bin'))]
    i=0; row=0; col=0
    ax[i]=fig.add_subplot(gs[row,col]); 
    # ax[i].axes.xaxis.set_visible(False);  
    ax[i].text(0.4,0.7, labels[i], fontsize=text_Fontsize,color = "black",weight="bold")
    ax[i].plot(y2,lvp,'b--',label='well_log')
    ax[i].plot(y2,lvs, color='orange',linestyle='--'); 
    ax[i].plot(y2,lrho,'g--');
    ax[i].plot(y,D.model_init.vp[::-1,idx], 'b',label='vp'); 
    ax[i].plot(y,D.model_init.vs[::-1,idx], 'orange',label='vs')
    ax[i].plot(y,D.model_init.rho[::-1,idx],'g', label='rho'); 
    ax[i].set_ylim(bottom=0,top=3100)
    ax[i].grid()
    ax[i].tick_params(labelsize=labelsize)
    plt.legend(fontsize=Fontsize)
    i=1; row=0; col=1
    ax[i]=fig.add_subplot(gs[row,col]); 
    ax[i].text(0.4,0.7, labels[i], fontsize=text_Fontsize,color = "black",weight="bold")
    ax[i].plot(y2,lvp,'b--',label='well_log')
    ax[i].plot(y2,lvs, color='orange',linestyle='--'); 
    ax[i].plot(y2,lrho,'g--');
    ax[i].plot(y,fwi_res_vp[::-1,idx], 'b',label='vp'); 
    ax[i].plot(y,fwi_res_vs[::-1,idx], 'orange',label='vs')
    ax[i].plot(y,fwi_res_rho[::-1,idx],'g', label='rho'); 
    ax[i].set_ylim(bottom=0,top=3100)
    ax[i].grid()
    ax[i].tick_params(labelsize=labelsize)
    ax[i].yaxis.set_major_formatter(plt.NullFormatter())
    # ax[i].axes.yaxis.set_visible(False)
    plt.legend(fontsize=Fontsize)
    ##
    j=1;D=d[j]
    tmp=fnmatch.filter(D.fwi_model_names,'*vp_stage*')[-1] 
    tmp2=tmp.split('_stage_')[-1]
    tmp2='1_it_12.bin'
    fwi_res_vp=D.models[D.fwi_model_names.index(os.path.join(D.save_folder,'model','modelTest_vp_stage_'+tmp2))]
    fwi_res_vs=D.models[D.fwi_model_names.index(os.path.join(D.save_folder,'model','modelTest_vs_stage_'+tmp2))]
    fwi_res_rho=D.models[D.fwi_model_names.index(os.path.join(D.save_folder,'model','modelTest_rho_stage_'+tmp2))]
    i=2; row=1; col=0
    ax[i]=fig.add_subplot(gs[row,col]); 
    ax[i].text(0.4,0.7, labels[i], fontsize=text_Fontsize,color = "black",weight="bold")
    ax[i].plot(y2,lvp,'b--',label='well_log')
    ax[i].plot(y2,lvs, color='orange',linestyle='--'); 
    ax[i].plot(y2,lrho,'g--');
    ax[i].plot(y,D.model_init.vp[::-1,idx], 'b',label='vp'); 
    ax[i].plot(y,D.model_init.vs[::-1,idx], 'orange',label='vs')
    ax[i].plot(y,D.model_init.rho[::-1,idx],'g', label='rho');
    ax[i].set_ylim(bottom=0,top=3100)
    ax[i].grid(); 
    ax[i].set_xlabel('Depth, m',fontsize=Fontsize)
    ax[i].set_ylabel('Velocity, m/sec',fontsize=Fontsize)
    ax[i].tick_params(labelsize=labelsize)
    plt.legend(fontsize=Fontsize)
    i=3; row=1; col=1
    ax[i]=fig.add_subplot(gs[row,col]); 
    ax[i].text(0.4,0.7, labels[i], fontsize=text_Fontsize,color = "black",weight="bold")
    ax[i].plot(y2,lvp,'b--',label='well_log')
    ax[i].plot(y2,lvs, color='orange',linestyle='--'); 
    ax[i].plot(y2,lrho,'g--');
    ax[i].plot(y,fwi_res_vp[::-1,idx], 'b',label='vp'); 
    ax[i].plot(y,fwi_res_vs[::-1,idx], 'orange',label='vs')
    ax[i].plot(y,fwi_res_rho[::-1,idx],'g', label='rho');
    ax[i].set_ylim(bottom=0,top=3100)
    ax[i].grid(); 
    ax[i].tick_params(labelsize=labelsize)
    ax[i].yaxis.set_major_formatter(plt.NullFormatter())
    plt.legend(fontsize=Fontsize)
    ##
    save_file_path2=os.path.join(log_path,'cgg_logs.png')
    plt.savefig(save_file_path2,dpi=400,bbox_inches='tight')
    plt.savefig(os.path.join(save_path,'cgg_logs.png'),dpi=400,bbox_inches='tight')
    print('Saving ML_result to '+save_file_path2);  # plt.show()
    plt.close()
    ################################
    return None
def comparison_initial_models_with_fwi_cgg_logs_test_function(paths,log_path='./',save_path='./',name='cgg_logs_new.png'):
    """ plot 4 logs in 4v figures """
    d=[]
    for path in paths:
        d.append(parse_denise_folder(path,denise_root='./') )
    ################################
    dx=25
    info_file=os.path.join('./','for_pasha','acq_data_parameters_cgg.pkl')
    with open(info_file,'rb') as input:
        acq_data=pickle.load(input)
    log_dict=acq_data['log_dict']
    log_loc=log_dict['loc']
    log=log_dict['data']
    idx = int(log_loc / 25)
    vh = log_loc * np.ones_like(log) / 1000 
    ################################
    lvp=log;   print(lvp.shape)
    # shear velocity, [m/s]
    lvs = lvp.copy() / (3 ** 0.5)
    lvs = np.where(lvp < 1.01 * np.min(lvp), 0, lvs)
    # density, [kg/m3] 
    lrho = 1e3*0.3 * lvp.copy()**0.25
    lrho = np.where(lvp < 1.01 * np.min(lvp), 1000, lrho)
    ################################
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 12.4
    fig_size[1] = 5.0   #height
    plt.rcParams["figure.figsize"] = fig_size
    n_row=2;    n_col=2
    gs = gridspec.GridSpec(n_row,n_col)
    gs.update(left=0.0, right=1.0, wspace=0.0, hspace=0.0)
    # gs=GridSpec(n_row,n_col+1, width_ratios=[1,1,0.1], height_ratios=[1,1,1])
    # gs.update(wspace=0.0,hspace=0.0)
    ax=[None]*6
    labels=['a','a','b','b']
    a=1500;b=3000

    Fontsize=15
    text_Fontsize=30
    labelsize=14
    fig=plt.figure()
    j=0;D=d[j]
    nx_orig=600
    x = np.arange(nx_orig) * D.DH / 1000
    y = np.arange(D.NY) * D.DH / 1
    y2 = np.arange(len(lvp)) * D.DH / 1
    extent = np.array([x.min(), x.max(), y.min(), y.max()])
    ##
    j=0;D=d[j]
    tmp=fnmatch.filter(D.fwi_model_names,'*vp_stage*')[-1] 
    tmp2=tmp.split('_stage_')[-1]
    print(D.fwi_model_names[D.fwi_model_names.index(os.path.join(D.save_folder,'model','modelTest_vp_stage_'+tmp2))])
    fwi_res_vp=D.models[D.fwi_model_names.index(os.path.join(D.save_folder,'model','modelTest_vp_stage_'+tmp2))]
    fwi_res_vs=D.models[D.fwi_model_names.index(os.path.join(D.save_folder,'model','modelTest_vs_stage_'+tmp2))]
    fwi_res_rho=D.models[D.fwi_model_names.index(os.path.join(D.save_folder,'model','modelTest_rho_stage_'+tmp2))]
    # fwi_res_vp=D.models[D.fwi_model_names.index(os.path.join(D.save_folder,'model','modelTest_vp.bin'))]
    # fwi_res_vs=D.models[D.fwi_model_names.index(os.path.join(D.save_folder,'model','modelTest_vs.bin'))]
    # fwi_res_rho=D.models[D.fwi_model_names.index(os.path.join(D.save_folder,'model','modelTest_rho.bin'))]
    i=0; row=0; col=0
    ax[i]=fig.add_subplot(gs[row,col]); 
    # ax[i].axes.xaxis.set_visible(False);  
    ax[i].text(0.4,0.7, labels[i], fontsize=text_Fontsize,color = "black",weight="bold")
    ax[i].plot(y2,lvp[::-1],'b--',label='well_log')
    ax[i].plot(y2,lvs[::-1], color='orange',linestyle='--'); 
    ax[i].plot(y2,lrho[::-1],'g--');
    ax[i].plot(y,D.model_init.vp[::-1,idx], 'b',label='vp'); 
    ax[i].plot(y,D.model_init.vs[::-1,idx], 'orange',label='vs')
    ax[i].plot(y,D.model_init.rho[::-1,idx],'g', label='rho'); 
    ax[i].set_ylim(bottom=0,top=3100)
    ax[i].grid()
    ax[i].tick_params(labelsize=labelsize)
    plt.legend(fontsize=Fontsize)
    i=1; row=0; col=1
    ax[i]=fig.add_subplot(gs[row,col]); 
    ax[i].text(0.4,0.7, labels[i], fontsize=text_Fontsize,color = "black",weight="bold")
    ax[i].plot(y2,lvp[::-1],'b--',label='well_log')
    ax[i].plot(y2,lvs[::-1], color='orange',linestyle='--'); 
    ax[i].plot(y2,lrho[::-1],'g--');
    ax[i].plot(y,fwi_res_vp[::-1,idx], 'b',label='vp'); 
    ax[i].plot(y,fwi_res_vs[::-1,idx], 'orange',label='vs')
    ax[i].plot(y,fwi_res_rho[::-1,idx],'g', label='rho'); 
    ax[i].set_ylim(bottom=0,top=3100)
    ax[i].grid()
    ax[i].tick_params(labelsize=labelsize)
    ax[i].yaxis.set_major_formatter(plt.NullFormatter())
    # ax[i].axes.yaxis.set_visible(False)
    plt.legend(fontsize=Fontsize)
    ##
    j=1;D=d[j]
    tmp=fnmatch.filter(D.fwi_model_names,'*vp_stage*')[-1]
    tmp2=tmp.split('_stage_')[-1]
    print(D.fwi_model_names[D.fwi_model_names.index(os.path.join(D.save_folder,'model','modelTest_vp_stage_'+tmp2))])
    # tmp2='1_it_12.bin'   # fwi_47_strategy_l5_weight_1351_0
    fwi_res_vp=D.models[D.fwi_model_names.index(os.path.join(D.save_folder,'model','modelTest_vp_stage_'+tmp2))]
    fwi_res_vs=D.models[D.fwi_model_names.index(os.path.join(D.save_folder,'model','modelTest_vs_stage_'+tmp2))]
    fwi_res_rho=D.models[D.fwi_model_names.index(os.path.join(D.save_folder,'model','modelTest_rho_stage_'+tmp2))]
    i=2; row=1; col=0
    ax[i]=fig.add_subplot(gs[row,col]); 
    ax[i].text(0.4,0.7, labels[i], fontsize=text_Fontsize,color = "black",weight="bold")
    ax[i].plot(y2,lvp[::-1],'b--',label='well_log')
    ax[i].plot(y2,lvs[::-1], color='orange',linestyle='--'); 
    ax[i].plot(y2,lrho[::-1],'g--');
    ax[i].plot(y,D.model_init.vp[::-1,idx], 'b',label='vp'); 
    ax[i].plot(y,D.model_init.vs[::-1,idx], 'orange',label='vs')
    ax[i].plot(y,D.model_init.rho[::-1,idx],'g', label='rho');
    ax[i].set_ylim(bottom=0,top=3100)
    ax[i].grid(); 
    ax[i].set_xlabel('Depth, m',fontsize=Fontsize)
    ax[i].set_ylabel('Velocity, m/sec',fontsize=Fontsize)
    ax[i].tick_params(labelsize=labelsize)
    plt.legend(fontsize=Fontsize)
    i=3; row=1; col=1
    ax[i]=fig.add_subplot(gs[row,col]); 
    ax[i].text(0.4,0.7, labels[i], fontsize=text_Fontsize,color = "black",weight="bold")
    ax[i].plot(y2,lvp[::-1],'b--',label='well_log')
    ax[i].plot(y2,lvs[::-1], color='orange',linestyle='--'); 
    ax[i].plot(y2,lrho[::-1],'g--');
    ax[i].plot(y,fwi_res_vp[::-1,idx], 'b',label='vp'); 
    ax[i].plot(y,fwi_res_vs[::-1,idx], 'orange',label='vs')
    ax[i].plot(y,fwi_res_rho[::-1,idx],'g', label='rho');
    ax[i].set_ylim(bottom=0,top=3100)
    ax[i].grid(); 
    ax[i].tick_params(labelsize=labelsize)
    ax[i].yaxis.set_major_formatter(plt.NullFormatter())
    plt.legend(fontsize=Fontsize)
    ##
    plt.savefig(os.path.join(log_path,name),dpi=400,bbox_inches='tight')
    plt.savefig(os.path.join(save_path,name),dpi=400,bbox_inches='tight')
    print('Saving ML_result to '+os.path.join(log_path,name));  # plt.show()
    plt.close()
    ################################
    return None
def fwi_results_visualization(log_path,save_path,results_root,strategy_name):
    """  fwi_results_visualization for marmousi and overthrust   """
    #################   get general parameters
    print('Plotting the folder,',strategy_name)
    strategy_name=strategy_name.split('/')[-1]
    print(strategy_name)
    root1=os.path.join(results_root,strategy_name);
    root2=root1
    dirs=next(os.walk(os.path.join(root1)))[1]
    #################   Marmousi folders processing   ###################################################
    # dirs_marm=fnmatch.filter(dirs,'*Marmousi*')
    # dirs_2=fnmatch.filter(dirs_marm,'*f_z*')
    # init_model_variants=[]
    # for _dir_ in dirs_2:
    #     init_model_variants.append( _dir_.split('_f_z')[0] .split('Marmousi')[1] )
    #################
    dirs_marm=fnmatch.filter(dirs,'*Marmousi*')
    dirs_2=fnmatch.filter(dirs_marm,'*cnn*')
    init_model_variants=[]
    for _dir_ in dirs_2:
        init_model_variants.append( _dir_.split('_cnn')[0] .split('Marmousi')[1] )
    #################
    cnn_label_name=fnmatch.filter(dirs_marm,'*cnn*')
    #   construct cnn weights label
    cnn_label_name_='_cnn_w_'+cnn_label_name[0].split('_cnn_w_')[1]
    init_model_types=['_f_z',cnn_label_name_,'_true']
    for model_variant in init_model_variants:
        print(model_variant)
        dirs_=fnmatch.filter(dirs,'*Marmousi*')
        #################   recover file names
        paths=[ os.path.join(root1,'Marmousi'+model_variant+init_model_types[0]),
                os.path.join(root1,'Marmousi'+model_variant+init_model_types[1]),
                os.path.join(root1,'Marmousi'+model_variant+init_model_types[2]) ];    print(paths)
        # comparison_initial_models_with_fwi_misfits(paths,'marm_old.png',log_path=log_path,save_path=save_path)
        file_name='Marmousi'+model_variant+'_'+strategy_name+'.png'
        print('saving name to=',file_name)
        comparison_initial_models_with_fwi_misfits(paths,file_name,log_path=log_path,save_path=save_path)
        # comparison_initial_models_with_fwi_FFT_spectrums(paths,'marm_spectrums.png',log_path=log_path,save_path=save_path,pclip=0.0001,fmax=1,kmax=1)
    #####################################################################################
    #################   overthrust folders processing    ###################################################
    # dirs_over=fnmatch.filter(dirs,'*Overthrust*')
    # dirs_2=fnmatch.filter(dirs_over,'*f_z*')
    # init_model_variants=[]
    # for _dir_ in dirs_2:
    #     init_model_variants.append( _dir_.split('_f_z')[0] .split('Overthrust')[1] )
    #################
    dirs_over=fnmatch.filter(dirs,'*Overthrust*')
    dirs_2=fnmatch.filter(dirs_over,'*cnn*')
    init_model_variants=[]
    for _dir_ in dirs_2:
        init_model_variants.append( _dir_.split('_cnn')[0] .split('Overthrust')[1] )
    #################
    cnn_label_name=fnmatch.filter(dirs_over,'*cnn*')
    #   construct cnn weights label
    cnn_label_name_='_cnn_w_'+cnn_label_name[0].split('_cnn_w_')[1]
    init_model_types=['_f_z',cnn_label_name_,'_true']
    #################       Algorithm 1
    for model_variant in init_model_variants: 
        paths=[ os.path.join(root1,'Overthrust'+model_variant+init_model_types[0]),
                os.path.join(root1,'Overthrust'+model_variant+init_model_types[1]),
                os.path.join(root1,'Overthrust'+model_variant+init_model_types[2]) ];    print(paths)
        # comparison_initial_models_with_fwi_misfits(paths,'over_old.png',log_path=log_path,save_path=save_path)
        file_name='Overthrust'+model_variant+'_'+strategy_name+'.png'
        comparison_initial_models_with_fwi_misfits(paths,file_name,log_path=log_path,save_path=save_path)
        print('saving name to=',file_name)
    return None
def fwi_results_cgg_visualization(log_path,save_path,results_root,strategy_name):
    """ marm over old"""
    print('Plotting the folder,',strategy_name)
    strategy_name=strategy_name.split('/')[-1]
    print(strategy_name)
    root1=os.path.join(results_root,strategy_name);
    dirs=next(os.walk(os.path.join(root1)))[1]
    #################
    paths=[os.path.join(root1,dirs[0])]
    name=strategy_name+'_'+dirs[0]
    #################       Algorithm 1
    d=[]
    for path in paths:
        d.append(parse_denise_folder(path,denise_root='./') )
    ################################
    dx=25
    info_file=os.path.join('./','for_pasha','acq_data_parameters_cgg.pkl')
    with open(info_file,'rb') as input:
        acq_data=pickle.load(input)
    log_dict=acq_data['log_dict']
    log_loc=log_dict['loc']
    log=log_dict['data']
    log_idx = int(log_loc / 25)
    vh = log_loc * np.ones_like(log) / 1000 
    ################################
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 12.4/2*3      #width
    fig_size[1] = 5.0   #height
    plt.rcParams["figure.figsize"] = fig_size
    plt.rcParams["font.size"] = 14
    n_row=2;    n_col=3
    gs = gridspec.GridSpec(n_row,n_col)
    gs.update(left=0.0, right=0.92, wspace=0.0, hspace=0.0)
    # gs=GridSpec(n_row,n_col+1, width_ratios=[1,1,0.1], height_ratios=[1,1,1])
    # gs.update(wspace=0.0,hspace=0.0)
    ax=[None]*3
    labels=['a','b','c']
    a=1500;b=3000
    Fontsize=32
    labels_Fontsize=18
    text_Fontsize=30
    labelsize=14
    Fontsize=32    
    textFontsize=20

    if hasattr(d,'fwi_model_names')==False:
        return None
    fig=plt.figure()
    j=0;D=d[j]
    nx_orig=D.NX
    ind=D.fwi_model_names.index(os.path.join(D.save_folder,'model','modelTest_vp.bin'))
    ind=D.fwi_model_names.index( fnmatch.filter(D.fwi_model_names,'*vp_stage*')[-1] )
    print(D.fwi_model_names[ind])
    fwi_res=D.models[ind]
    x = np.arange(nx_orig) * D.DH / 1000
    y = np.arange(D.NY) * D.DH / 1000
    extent = np.array([x.min(), x.max(), y.min(), y.max()])
    i=0; row=0; col=0
    ax[i]=fig.add_subplot(gs[row,col]); 
    # ax[i].axes.xaxis.set_visible(False)
    ax[i].text(0.4,0.7, labels[i], fontsize=Fontsize,color = "white",weight="bold")
    ax[i].imshow(D.model_init.vp[:,0:nx_orig],extent=extent,vmin=a,vmax=b,aspect='auto');ax[i].invert_yaxis()
    ax[i].plot(vh, np.arange(len(log))*dx/1000, 'k--')
    ax[i].plot(vh + (log[::-1] - min(log)) / 1000, np.arange(len(log))*dx/1000, 'k')
    ax[i].set_ylabel('Depth, km',fontsize=labels_Fontsize)
    ax[i].set_xlabel('X, km',fontsize=labels_Fontsize)
    ax[i].tick_params(labelsize=labelsize)
    i=1; row=0; col=1
    ax[i]=fig.add_subplot(gs[row,col]); 
    ax[i].axes.yaxis.set_visible(False)
    ax[i].text(0.4,0.7, labels[i], fontsize=Fontsize,color = "white",weight="bold")
    last_image=ax[i].imshow(fwi_res[:,0:nx_orig],extent=extent,vmin=a,vmax=b,aspect='auto');ax[i].invert_yaxis()
    ax[i].plot(vh, np.arange(len(log))*dx/1000, 'k--')
    ax[i].set_xlabel('X, km',fontsize=labels_Fontsize)
    ax[i].tick_params(labelsize=labelsize)
    ax[i].plot(vh + (log[::-1] - min(log)) / 1000, np.arange(len(log))*dx/1000, 'k')
    #############################   misfits
    niter_list=[];misfit_list=[]
    j=0;D=d[j];path=paths[j]
    i=2; row=0; col=2
    m,m_data=plot_fwi_misfit2(os.path.join(path,'fld','seis_fwi_log.dat'),D)
    nstage=m_data['nstage']
    niter_stage=m_data['niter_stage']
    nstage_trans=m_data['nstage_trans']
    iteration=m_data['iteration']
    niter_list.append(len(iteration))
    misfit_list.append(m)
    ax[i]=fig.add_subplot(gs[row,col])
    ax[i].plot(iteration,m,'b-', linewidth=3, label='Evolution of the misfit function')
    ax[i].set_xlabel('Iteration number',fontsize=labels_Fontsize)
    ax[i].tick_params(labelsize=labelsize)
    for ii in range(1, nstage):
        ax[i].plot([nstage_trans[ii-1]+ii,nstage_trans[ii-1]+ii], [np.min(m),np.max(m)],'k--', linewidth=3)
    ax[i].axes.yaxis.set_visible(False)
    ######  x limits
    niter_list=np.asarray(niter_list)
    misfit_list=np.asarray(misfit_list)
    mf_min=0;mf_max=0;
    for misfit_curve in misfit_list:
        mf_min=np.min([np.min(misfit_curve),mf_min])
        mf_max=np.max([np.max(misfit_curve),mf_max])
    ax[2].set_xlim(left=1,right=np.max(niter_list))
    ax[2].set_ylim(bottom=mf_min,top=mf_max)
    y_pos=mf_min+(mf_max-mf_min)/5*4
    ax[2].text(10,y_pos, labels[2], fontsize=Fontsize,color = "black",weight="bold")
    # cbar.set_label(fontsize=22)
    #############################   append colorbar
    cbar_ax = fig.add_axes([0.93, 0.52, 0.02, 0.35])
    cbar=fig.colorbar(last_image,cax=cbar_ax)
    # cbar.set_label('V (km/sec)',labelpad=-40, y=1.05, rotation=0)   #horizontalalignment='center'
    cbar.ax.set_title('V (m/sec)',fontsize=18,pad=13.3)
    for t in cbar.ax.get_yticklabels():
        t.set_fontsize(16)
    #############################
    save_file_path1=os.path.join(log_path,name+'.png')
    save_file_path2=os.path.join(save_path,name+'.png')
    print('Saving ML_result to '+save_file_path1)
    print('Saving ML_result to '+save_file_path2)
    plt.savefig(save_file_path1,dpi=300,bbox_inches='tight')
    plt.savefig(save_file_path2,dpi=300,bbox_inches='tight')
    plt.close()
    return None
def fwi_results_cgg_visualization2(log_path,save_path,results_root,strategy_name):
    print('Plotting the folder,',strategy_name)
    strategy_name=strategy_name.split('/')[-1]
    print(strategy_name)
    root1=os.path.join(results_root,strategy_name);
    dirs=next(os.walk(os.path.join(root1)))[1]
    #################   process only 1 folder in the list
    # paths=[os.path.join(root1,dirs[0])]
    # name=strategy_name+'_'+dirs[0]
    # d=[]
    # for path in paths:
    #     d.append(parse_denise_folder(path,denise_root='./') )
    # D=d[0]
    #################   process full list
    for dir in dirs:
        path=os.path.join(root1,dir)
        D=parse_denise_folder(path,denise_root='./')
        d=[D]
        name=strategy_name+'_'+dir
        ################################
        if hasattr(D,'fwi_model_names')==False:
            print('Folder is empty!Folder is empty!Folder is empty!Folder is empty!Folder is empty!')
            return None
        ################################
        dx=25
        info_file=os.path.join('./','for_pasha','acq_data_parameters_cgg.pkl')
        with open(info_file,'rb') as input:
            acq_data=pickle.load(input)
        log_dict=acq_data['log_dict']
        log_loc=log_dict['loc']
        log=log_dict['data']
        log_idx = int(log_loc / 25)
        vh = log_loc * np.ones_like(log) / 1000 
        ################################
        lvp=log;   print(lvp.shape)
        # shear velocity, [m/s]
        lvs = lvp.copy() / (3 ** 0.5)
        lvs = np.where(lvp < 1.01 * np.min(lvp), 0, lvs)
        # density, [kg/m3] 
        lrho = 1e3*0.3 * lvp.copy()**0.25
        lrho = np.where(lvp < 1.01 * np.min(lvp), 1000, lrho)
        y = np.arange(D.NY) * D.DH / 1
        y2 = np.arange(len(lvp)) * D.DH / 1000
        ################################
        fig_size = plt.rcParams["figure.figsize"]
        fig_size[0] = 12.4/2*3      #width
        fig_size[1] = 6.6   #height
        plt.rcParams["figure.figsize"] = fig_size
        plt.rcParams["font.size"] = 14
        n_row=2;    n_col=3
        gs = gridspec.GridSpec(n_row,n_col)
        gs.update(left=0.0, right=0.92, wspace=0.0, hspace=0.5)
        # gs=GridSpec(n_row,n_col+1, width_ratios=[1,1,0.1], height_ratios=[1,1,1])
        # gs.update(wspace=0.0,hspace=0.0)
        ax=[None]*6
        labels=['a','b','c','d','e','f']
        a=1500;b=3000
        Fontsize=32
        labels_Fontsize=18
        text_Fontsize=30
        labelsize=14
        Fontsize=32    
        textFontsize=20

        fig=plt.figure()
        j=0;D=d[j]
        tmp=fnmatch.filter(D.fwi_model_names,'*vs_stage*')[-1] 
        tmp2=tmp.split('_stage_')[-1]
        print(D.fwi_model_names[D.fwi_model_names.index(os.path.join(D.save_folder,'model','modelTest_vp_stage_'+tmp2))])
        fwi_res_vp=D.models[D.fwi_model_names.index(os.path.join(D.save_folder,'model','modelTest_vp_stage_'+tmp2))]
        fwi_res_vs=D.models[D.fwi_model_names.index(os.path.join(D.save_folder,'model','modelTest_vs_stage_'+tmp2))]
        fwi_res_rho=D.models[D.fwi_model_names.index(os.path.join(D.save_folder,'model','modelTest_rho_stage_'+tmp2))]

        nx_orig=D.NX
        ind=D.fwi_model_names.index(os.path.join(D.save_folder,'model','modelTest_vp.bin'))
        ind=D.fwi_model_names.index( fnmatch.filter(D.fwi_model_names,'*vp_stage*')[-1] )
        print(D.fwi_model_names[ind])
        fwi_res=D.models[ind]
        x = np.arange(nx_orig) * D.DH / 1000
        y = np.arange(D.NY) * D.DH / 1000
        extent = np.array([x.min(), x.max(), y.min(), y.max()])
        #############################
        i=0; row=0; col=0
        ax[i]=fig.add_subplot(gs[row,col]); 
        # ax[i].axes.xaxis.set_visible(False)
        ax[i].text(0.4,0.7, labels[i], fontsize=Fontsize,color = "white",weight="bold")
        ax[i].imshow(D.model_init.vp[:,0:nx_orig],extent=extent,vmin=a,vmax=b,aspect='auto');ax[i].invert_yaxis()
        ax[i].plot(vh, np.arange(len(log))*dx/1000, 'k--')
        ax[i].plot(vh + (log[::-1] - min(log)) / 1000, np.arange(len(log))*dx/1000, 'k')
        ax[i].set_ylabel('Depth, km',fontsize=labels_Fontsize)
        ax[i].set_xlabel('X, km',fontsize=labels_Fontsize)
        ax[i].tick_params(labelsize=labelsize)
        i=1; row=0; col=1
        ax[i]=fig.add_subplot(gs[row,col]); 
        ax[i].axes.yaxis.set_visible(False)
        ax[i].text(0.4,0.7, labels[i], fontsize=Fontsize,color = "white",weight="bold")
        last_image=ax[i].imshow(fwi_res[:,0:nx_orig],extent=extent,vmin=a,vmax=b,aspect='auto');ax[i].invert_yaxis()
        ax[i].plot(vh, np.arange(len(log))*dx/1000, 'k--')
        ax[i].set_xlabel('X, km',fontsize=labels_Fontsize)
        ax[i].tick_params(labelsize=labelsize)
        ax[i].plot(vh + (log[::-1] - min(log)) / 1000, np.arange(len(log))*dx/1000, 'k')
        #############################
        vp_log=D.model_init.vp[::-1,log_idx];   vp_log1=vp_log[0:len(lvp)]
        score1=PCC(vp_log1,lvp[::-1])
        vp_log=fwi_res_vp[::-1,log_idx];        vp_log2=vp_log[0:len(lvp)]
        score2=PCC(vp_log2,lvp[::-1])
        PCC(vp_log1,lvp[::-1])
        PCC(vp_log2,lvp[::-1])
        PCC(vp_log2,vp_log2)
        PCC(vp_log2,-vp_log2)
        PCC(vp_log2,lvp[::-1])
        #############################
        i=3; row=1; col=0
        ax[i]=fig.add_subplot(gs[row,col]); 
        ax[i].text(0.4,0.7, labels[i], fontsize=Fontsize,color = "white",weight="bold")
        ax[i].plot(y2,lvp[::-1],'b--',label='well_log')
        ax[i].plot(y2,lvs[::-1], color='orange',linestyle='--'); 
        ax[i].plot(y2,lrho[::-1],'g--');
        ax[i].plot(y,D.model_init.vp[::-1,log_idx], 'b',label='vp'); 
        ax[i].plot(y,D.model_init.vs[::-1,log_idx], 'orange',label='vs')
        ax[i].plot(y,D.model_init.rho[::-1,log_idx],'g', label='rho');
        ax[i].set_xlim(left=0,right=y[-1])
        ax[i].set_ylim(bottom=0,top=3100)
        ax[i].grid()
        ax[i].tick_params(labelsize=labelsize)
        ax[i].set_xlabel('Depth, km',fontsize=labels_Fontsize)
        ax[i].set_ylabel('Velocity, m/sec',fontsize=labels_Fontsize)
        #############################
        vp_log=D.model_init.vp[::-1,log_idx];   vp_log=vp_log[0:len(lvp)]
        well_log=lvp[::-1]
        water_start=33
        score=      PCC(vp_log[water_start:],well_log[water_start:])
        score_mse=  MSE(vp_log[water_start:],well_log[water_start:])
        title_score='PCC(Vp predicted, vp well)='+numstr_2( score )#+'_'+numstr_2(score_mse)
        ax[i].set_title(title_score,fontsize=textFontsize)
        plt.legend(fontsize=labelsize,framealpha=0.1)
        #############################
        i=4; row=1; col=1
        ax[i]=fig.add_subplot(gs[row,col]); 
        ax[i].text(0.4,0.7, labels[i], fontsize=Fontsize,color = "white",weight="bold")
        ax[i].plot(y2,lvp[::-1],'b--',label='well_log')
        ax[i].plot(y2,lvs[::-1], color='orange',linestyle='--'); 
        ax[i].plot(y2,lrho[::-1],'g--');
        ax[i].plot(y,fwi_res_vp[::-1,log_idx], 'b',label='vp'); 
        ax[i].plot(y,fwi_res_vs[::-1,log_idx], 'orange',label='vs')
        ax[i].plot(y,fwi_res_rho[::-1,log_idx],'g', label='rho'); 
        ax[i].set_xlim(left=0,right=y[-1])
        ax[i].set_ylim(bottom=0,top=3100)
        ax[i].grid()
        ax[i].tick_params(labelsize=labelsize)
        ax[i].set_xlabel('Depth, km',fontsize=labels_Fontsize)
        vp_log=fwi_res_vp[::-1,log_idx];   vp_log=vp_log[0:len(lvp)]
        score=      PCC(vp_log[water_start:],well_log[water_start:])
        score_mse=  MSE(vp_log[water_start:],well_log[water_start:])
        title_score='PCC(Vp inverted, vp well)='+numstr_2( score )#+'_'+numstr_2(score_mse)
        ax[i].set_title(title_score,fontsize=textFontsize)
        ax[i].yaxis.set_major_formatter(plt.NullFormatter())
        plt.legend(fontsize=labelsize,framealpha=0.1)
        #############################   misfits
        niter_list=[];misfit_list=[]
        j=0;D=d[j];
        i=2; row=0; col=2
        m,m_data=plot_fwi_misfit2(os.path.join(path,'fld','seis_fwi_log.dat'),D)
        nstage=m_data['nstage']
        niter_stage=m_data['niter_stage']
        nstage_trans=m_data['nstage_trans']
        iteration=m_data['iteration']
        niter_list.append(len(iteration))
        misfit_list.append(m)
        ax[i]=fig.add_subplot(gs[row,col])
        ax[i].plot(iteration,m,'b-', linewidth=3, label='Evolution of the misfit function')
        ax[i].set_xlabel('Iteration number',fontsize=labels_Fontsize)
        ax[i].tick_params(labelsize=labelsize)
        for ii in range(1, nstage):
            ax[i].plot([nstage_trans[ii-1]+ii,nstage_trans[ii-1]+ii], [np.min(m),np.max(m)],'k--', linewidth=3)
        ax[i].axes.yaxis.set_visible(False)
        ######  x limits
        niter_list=np.asarray(niter_list)
        misfit_list=np.asarray(misfit_list)
        mf_min=0;mf_max=0;
        for misfit_curve in misfit_list:
            mf_min=np.min([np.min(misfit_curve),mf_min])
            mf_max=np.max([np.max(misfit_curve),mf_max])
        ax[2].set_xlim(left=1,right=np.max(niter_list))
        ax[2].set_ylim(bottom=mf_min,top=mf_max)
        y_pos=mf_min+4/5*(mf_max-mf_min)
        # to get rid of errors
        # ax[2].text(10,y_pos, labels[2], fontsize=Fontsize,color = "black",weight="bold")
        # cbar.set_label(fontsize=22)
        #############################   append colorbar
        cbar_ax = fig.add_axes([0.93, 0.52, 0.02, 0.35])
        cbar=fig.colorbar(last_image,cax=cbar_ax)
        # cbar.set_label('V (km/sec)',labelpad=-40, y=1.05, rotation=0)   #horizontalalignment='center'
        cbar.ax.set_title('V (m/sec)',fontsize=18,pad=13.3)
        for t in cbar.ax.get_yticklabels():
            t.set_fontsize(16)
        #############################
        save_file_path1=os.path.join(log_path,name+'.png')
        save_file_path2=os.path.join(save_path,name+'.png')
        print('Saving ML_result to '+save_file_path1)
        print('Saving ML_result to '+save_file_path2)
        plt.savefig(save_file_path1,dpi=300,bbox_inches='tight')
        plt.savefig(save_file_path2,dpi=300,bbox_inches='tight')
        plt.close()
    return None
def fwi_results_cgg_visualization2_2(log_path,save_path,results_root,strategy_name):
    """  plot the cgg fwi result with data misfits and comparison with well logs  """
    print('Plotting the folder,',strategy_name)
    strategy_name=strategy_name.split('/')[-1]
    print(strategy_name)
    root1=os.path.join(results_root,strategy_name);
    if os.path.exists(root1):
        dirs=next(os.walk(os.path.join(root1)))[1]
        plot_model_misfit=0
        #################
        if 'Marmousi' in dirs[0] or 'Overthrust' in dirs[0]:
            paths=[ os.path.join(root1,dirs[0]),
                    os.path.join(root1,dirs[0]),
                    os.path.join(root1,dirs[0]) ];    print(paths)
            file_name=strategy_name+'.png'
            print('saving name to=',file_name)
            comparison_initial_models_with_fwi_misfits(paths,file_name,log_path=log_path,save_path=save_path)
            return None
        #################   process only 1 folder in the list
        # paths=[os.path.join(root1,dirs[0])]
        # name=strategy_name+'_'+dirs[0]
        # d=[]
        # for path in paths:
        #     d.append(parse_denise_folder(path,denise_root='./') )
        # D=d[0]
        #################   process full list
        for dir in dirs:
            path=os.path.join(root1,dir)
            D=parse_denise_folder(path,denise_root='./')
            d=[D]
            name=strategy_name+'_'+dir
            ################################
            if hasattr(D,'fwi_model_names')==False:
                print('Folder is empty!Folder is empty!Folder is empty!Folder is empty!Folder is empty!')
                return None
            ################################
            dx=25
            info_file=os.path.join('./','for_pasha','acq_data_parameters_cgg.pkl')
            with open(info_file,'rb') as input:
                acq_data=pickle.load(input)
            log_dict=acq_data['log_dict']
            log_loc=log_dict['loc']
            log=log_dict['data']
            log_idx = int(log_loc / 25)
            vh = log_loc * np.ones_like(log) / 1000 
            lvp=log;   print(lvp.shape)
            # shear velocity, [m/s]
            lvs = lvp.copy() / (3 ** 0.5)
            lvs = np.where(lvp < 1.01 * np.min(lvp), 0, lvs)
            # density, [kg/m3] 
            lrho = 1e3*0.3 * lvp.copy()**0.25
            lrho = np.where(lvp < 1.01 * np.min(lvp), 1000, lrho)
            y = np.arange(D.NY) * D.DH / 1
            y2 = np.arange(len(lvp)) * D.DH / 1000
            ################################
            fig_size = plt.rcParams["figure.figsize"]
            fig_size[0] = 12.4/2*3      #width
            fig_size[1] = 6.6   #height
            plt.rcParams["figure.figsize"] = fig_size
            plt.rcParams["font.size"] = 14
            n_row=2;    n_col=3
            gs = gridspec.GridSpec(n_row,n_col)
            gs.update(left=0.0, right=0.92, wspace=0.0, hspace=0.5)
            # gs=GridSpec(n_row,n_col+1, width_ratios=[1,1,0.1], height_ratios=[1,1,1])
            # gs.update(wspace=0.0,hspace=0.0)
            ax=[None]*6
            labels=['a','b','c','d','e','f']
            a=1500;b=3000
            Fontsize=32
            labels_Fontsize=18
            text_Fontsize=30
            labelsize=14
            Fontsize=32    
            textFontsize=20

            fig=plt.figure()
            j=0;D=d[j]
            tmp=fnmatch.filter(D.fwi_model_names,'*vs_stage*')[-1] 
            tmp2=tmp.split('_stage_')[-1]
            print(D.fwi_model_names[D.fwi_model_names.index(os.path.join(D.save_folder,'model','modelTest_vp_stage_'+tmp2))])
            fwi_res_vp=D.models[D.fwi_model_names.index(os.path.join(D.save_folder,'model','modelTest_vp_stage_'+tmp2))]
            fwi_res_vs=D.models[D.fwi_model_names.index(os.path.join(D.save_folder,'model','modelTest_vs_stage_'+tmp2))]
            fwi_res_rho=D.models[D.fwi_model_names.index(os.path.join(D.save_folder,'model','modelTest_rho_stage_'+tmp2))]

            nx_orig=D.NX
            ind=D.fwi_model_names.index(os.path.join(D.save_folder,'model','modelTest_vp.bin'))
            ind=D.fwi_model_names.index( fnmatch.filter(D.fwi_model_names,'*vp_stage*')[-1] )
            print(D.fwi_model_names[ind])
            fwi_res=D.models[ind]
            x = np.arange(nx_orig) * D.DH / 1000
            y = np.arange(D.NY) * D.DH / 1000
            extent = np.array([x.min(), x.max(), y.min(), y.max()])
            #############################
            i=0; row=0; col=0
            ax[i]=fig.add_subplot(gs[row,col]); 
            # ax[i].axes.xaxis.set_visible(False)
            ax[i].text(0.4,0.7, labels[i], fontsize=Fontsize,color = "white",weight="bold")
            ax[i].imshow(D.model_init.vp[:,0:nx_orig],extent=extent,vmin=a,vmax=b,aspect='auto');ax[i].invert_yaxis()
            ax[i].plot(vh, np.arange(len(log))*dx/1000, 'k--')
            ax[i].plot(vh + (log[::-1] - min(log)) / 1000, np.arange(len(log))*dx/1000, 'k')
            ax[i].set_ylabel('Depth, km',fontsize=labels_Fontsize)
            ax[i].set_xlabel('X, km',fontsize=labels_Fontsize)
            ax[i].tick_params(labelsize=labelsize)
            i=1; row=0; col=1
            ax[i]=fig.add_subplot(gs[row,col]); 
            ax[i].axes.yaxis.set_visible(False)
            ax[i].text(0.4,0.7, labels[i], fontsize=Fontsize,color = "white",weight="bold")
            last_image=ax[i].imshow(fwi_res[:,0:nx_orig],extent=extent,vmin=a,vmax=b,aspect='auto');ax[i].invert_yaxis()
            ax[i].plot(vh, np.arange(len(log))*dx/1000, 'k--')
            ax[i].set_xlabel('X, km',fontsize=labels_Fontsize)
            ax[i].tick_params(labelsize=labelsize)
            ax[i].plot(vh + (log[::-1] - min(log)) / 1000, np.arange(len(log))*dx/1000, 'k')
            #############################
            vp_log=D.model_init.vp[::-1,log_idx];   vp_log1=vp_log[0:len(lvp)]
            score1=PCC(vp_log1,lvp[::-1])
            vp_log=fwi_res_vp[::-1,log_idx];        vp_log2=vp_log[0:len(lvp)]
            score2=PCC(vp_log2,lvp[::-1])
            PCC(vp_log1,lvp[::-1])
            PCC(vp_log2,lvp[::-1])
            PCC(vp_log2,vp_log2)
            PCC(vp_log2,-vp_log2)
            PCC(vp_log2,lvp[::-1])
            #############################
            i=3; row=1; col=0
            ax[i]=fig.add_subplot(gs[row,col]); 
            ax[i].text(0.4,2400, labels[i], fontsize=Fontsize,color = "black",weight="bold")
            ax[i].plot(y2,lvp[::-1],'b--',label='well_log')
            ax[i].plot(y2,lvs[::-1], color='orange',linestyle='--'); 
            ax[i].plot(y2,lrho[::-1],'g--');
            ax[i].plot(y,D.model_init.vp[::-1,log_idx], 'b',label='vp'); 
            ax[i].plot(y,D.model_init.vs[::-1,log_idx], 'orange',label='vs')
            ax[i].plot(y,D.model_init.rho[::-1,log_idx],'g', label='rho');
            ax[i].set_xlim(left=0,right=y[-1])
            ax[i].set_ylim(bottom=0,top=3100)
            ax[i].grid()
            ax[i].tick_params(labelsize=labelsize)
            ax[i].set_xlabel('Depth, km',fontsize=labels_Fontsize)
            ax[i].set_ylabel('Velocity, m/sec',fontsize=labels_Fontsize)
            vp_log=D.model_init.vp[::-1,log_idx];   vp_log=vp_log[0:len(lvp)]
            well_log=lvp[::-1]
            score=      PCC(vp_log[water_start:],well_log[water_start:])
            score_mse=  MSE(vp_log[water_start:],well_log[water_start:])
            title_score='PCC(Vp initial, Vp well)='+numstr_3( score )#+'_'+numstr_2(score_mse)
            ax[i].set_title(title_score,fontsize=textFontsize)
            plt.legend(fontsize=labelsize,framealpha=0.1)
            #############################
            i=4; row=1; col=1
            ax[i]=fig.add_subplot(gs[row,col]); 
            ax[i].text(0.4,2400, labels[i], fontsize=Fontsize,color = "black",weight="bold")
            ax[i].plot(y2,lvp[::-1],'b--',label='well_log')
            ax[i].plot(y2,lvs[::-1], color='orange',linestyle='--'); 
            ax[i].plot(y2,lrho[::-1],'g--');
            ax[i].plot(y,fwi_res_vp[::-1,log_idx], 'b',label='vp'); 
            ax[i].plot(y,fwi_res_vs[::-1,log_idx], 'orange',label='vs')
            ax[i].plot(y,fwi_res_rho[::-1,log_idx],'g', label='rho'); 
            ax[i].set_xlim(left=0,right=y[-1])
            ax[i].set_ylim(bottom=0,top=3100)
            ax[i].grid()
            ax[i].tick_params(labelsize=labelsize)
            ax[i].set_xlabel('Depth, km',fontsize=labels_Fontsize)
            vp_log=fwi_res_vp[::-1,log_idx];   vp_log=vp_log[0:len(lvp)]
            if score_to_use=='PCC':
                score=      PCC(vp_log[water_start:],well_log[water_start:])
            if score_to_use=='MSE':
                score=  MSE(vp_log[water_start:],well_log[water_start:])
            title_score=score_to_use+'(Vp inverted, Vp well)='+numstr_3( score )#+'_'+numstr_2(score_mse)
            ax[i].set_title(title_score,fontsize=textFontsize)
            ax[i].yaxis.set_major_formatter(plt.NullFormatter())
            plt.legend(fontsize=labelsize,framealpha=0.1)
            #############################   data misfits
            niter_list=[];misfit_list=[]
            i=2; row=0; col=2
            m,m_data=plot_fwi_misfit2(os.path.join(path,'fld','seis_fwi_log.dat'),D)
            nstage=m_data['nstage']
            niter_stage=m_data['niter_stage']
            nstage_trans=m_data['nstage_trans']
            iteration=m_data['iteration']
            niter_list.append(len(iteration))
            misfit_list.append(m)
            #############################   model misfits
            D.fwi_model_names
            fwi_vp_names=fnmatch.filter(D.fwi_model_names,'*vp_stage*')
            model_misfit=np.empty((len(fwi_vp_names)))
            iteration2=np.arange(len(model_misfit))
            for iter,fwi_vp_name in enumerate(fwi_vp_names):
                fwi_res=D.models[D.fwi_model_names.index(fwi_vp_name)]
                vp_log=fwi_res[::-1,log_idx];
                vp_log=vp_log[0:len(lvp)]
                score=      PCC(vp_log[water_start:],well_log[water_start:])
                model_misfit[iter]=score
            #############################
            ax[i]=fig.add_subplot(gs[row,col])
            ax_extra=ax[i].twinx()
            ax[i].plot(iteration,m,'b-', linewidth=3, label='Data misfit') #Evolution of the misfit function
            ax[i].set_xlabel('Iteration number',fontsize=labels_Fontsize)
            ax[i].text(iteration[np.min([7,len(iteration)])],(np.min(m)+6/10*(np.max(m)-np.min(m))), labels[i], fontsize=Fontsize,color = "black",weight="bold")
            ax[i].tick_params(labelsize=labelsize)
            ax[i].axes.yaxis.set_visible(False)
            for ii in range(1, nstage):
                ax[i].plot([nstage_trans[ii-1]+ii,nstage_trans[ii-1]+ii], [np.min(m),np.max(m)],'k--', linewidth=3)
            #############################
            if plot_model_misfit==1:
                ax_extra.plot(iteration2,model_misfit,'g-', linewidth=3, label='PCC well log comparison') #Evolution of the misfit function
                ax_extra.axes.yaxis.set_visible(False)
            #############################
            ######  x limits
            niter_list=np.asarray(niter_list)
            misfit_list=np.asarray(misfit_list)
            mf_min=0;mf_max=0;
            for misfit_curve in misfit_list:
                mf_min=np.min([np.min(misfit_curve),mf_min])
                mf_max=np.max([np.max(misfit_curve),mf_max])
            ax[2].set_xlim(left=1,right=np.max(niter_list))
            #############################   sety ylim
            ax[2].set_ylim(bottom=mf_min,top=mf_max)
            #############################
            y_pos=mf_min+4/5*(mf_max-mf_min)
            #############################   append colorbar
            cbar_ax = fig.add_axes([0.93, 0.52, 0.02, 0.35])
            cbar=fig.colorbar(last_image,cax=cbar_ax)
            # cbar.set_label('V (km/sec)',labelpad=-40, y=1.05, rotation=0)   #horizontalalignment='center'
            cbar.ax.set_title('V (m/sec)',fontsize=18,pad=13.3)
            for t in cbar.ax.get_yticklabels():
                t.set_fontsize(16)
            #############################
            save_file_path1=os.path.join(log_path,name+'.png')
            save_file_path2=os.path.join(save_path,name+'.png')
            print('Saving ML_result to '+save_file_path1)
            print('Saving ML_result to '+save_file_path2)
            plt.savefig(save_file_path1,dpi=300,bbox_inches='tight')
            plt.savefig(save_file_path2,dpi=300,bbox_inches='tight')
            plt.close()
            ss=1
    return None
def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / float(N)
def smooth(x,window_len=11,window='hanning'):
    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal
        
    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)
    ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']
    y=smooth(x,11,'flat')
    see also: 
    
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
 
    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")


    if window_len<3:
        return x


    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")


    s=numpy.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=numpy.ones(window_len,'d')
    else:
        w=eval('numpy.'+window+'(window_len)')

    y=numpy.convolve(w/w.sum(),s,mode='valid')
    return y
def fwi_results_cgg_visualization2_3(log_path,save_path,results_root,strategy_name):
    """  plot the cgg fwi result with data misfits and comparison with well logs  
         In this function we compare the inverted logs with the upscaled true well log
    """
    print('Plotting the folder,',strategy_name)
    strategy_name=strategy_name.split('/')[-1]
    root1=os.path.join(results_root,strategy_name);
    label_position_inside_axes=[0.04,0.97]
    if os.path.exists(root1):
        dirs=next(os.walk(os.path.join(root1)))[1]
        plot_model_misfit=0
        #################
        if 'Marmousi' in dirs[0] or 'Overthrust' in dirs[0]:
            paths=[ os.path.join(root1,dirs[0]),
                    os.path.join(root1,dirs[0]),
                    os.path.join(root1,dirs[0]) ];    print(paths)
            file_name=strategy_name+'.png'
            print('saving name to=',file_name)
            comparison_initial_models_with_fwi_misfits(paths,file_name,log_path=log_path,save_path=save_path)
            return None
        #################   process full list
        for dir in dirs:
            path=os.path.join(root1,dir)
            D=parse_denise_folder(path,denise_root='./')
            d=[D]
            name=strategy_name+'_'+dir
            ################################
            if hasattr(D,'fwi_model_names')==False:
                print('Folder is empty!Folder is empty!Folder is empty!Folder is empty!Folder is empty!')
                return None
            ################################    load well log data
            dx=25
            info_file=os.path.join('./','for_pasha','acq_data_parameters_cgg.pkl')
            with open(info_file,'rb') as input:
                acq_data=pickle.load(input)
            log_dict=acq_data['log_dict']
            log_loc=log_dict['loc']
            log=log_dict['data']
            log_idx = int(log_loc / 25)
            vh = log_loc * np.ones_like(log) / 1000 
            lvp=log;   print(lvp.shape)
            # shear velocity, [m/s]
            lvs = lvp.copy() / (3 ** 0.5)
            lvs = np.where(lvp < 1.01 * np.min(lvp), 0, lvs)
            # density, [kg/m3] 
            lrho = 1e3*0.3 * lvp.copy()**0.25
            lrho = np.where(lvp < 1.01 * np.min(lvp), 1000, lrho)
            y = np.arange(D.NY) * D.DH / 1
            y2 = np.arange(len(lvp)) * D.DH / 1000
            #############################upscale well log with moving average
            well_log=lvp[::-1]
            df = pd.DataFrame(data=well_log.ravel())
            lvp_upscaled6=df.rolling(window=int(2*(25/dx)+1),min_periods=1).mean()
            lvp_upscaled6[0:water_start]=1500
            lvp_upscaled7=df.rolling(window=int(2*(50/dx)+1),min_periods=1).mean()
            lvp_upscaled7[0:water_start]=1500
            lvp_upscaled8=df.rolling(window=int(2*(75/dx)+1),min_periods=1).mean()
            lvp_upscaled8[0:water_start]=1500

            plt.figure()
            plt.plot(well_log,label='0')
            plt.plot(lvp_upscaled6,label='50m window')
            plt.plot(lvp_upscaled7,label='100m window')
            plt.plot(lvp_upscaled8,label='150m window')
            plt.legend()
            plt.show()
            plt.savefig('test.png')
            plt.close()
            lvp_upscaled=np.copy(lvp_upscaled6)
            ################################    create figure
            fig_size = plt.rcParams["figure.figsize"]
            fig_size[0] = 12.4/2*3      #width
            fig_size[1] = 6.6   #height
            plt.rcParams["figure.figsize"] = fig_size
            plt.rcParams["font.size"] = 14
            n_row=2;    n_col=3
            gs = gridspec.GridSpec(n_row,n_col)
            gs.update(left=0.0, right=0.92, wspace=0.0, hspace=0.5)
            # gs=GridSpec(n_row,n_col+1, width_ratios=[1,1,0.1], height_ratios=[1,1,1])
            # gs.update(wspace=0.0,hspace=0.0)
            ax=[None]*6
            labels=['a','b','c','d','e','f']
            a=1500;b=3000
            Fontsize=32
            labels_Fontsize=18
            text_Fontsize=30
            labelsize=14
            Fontsize=32    
            textFontsize=20

            fig=plt.figure()
            j=0;D=d[j]
            tmp=fnmatch.filter(D.fwi_model_names,'*vs_stage*')[-1] 
            tmp2=tmp.split('_stage_')[-1]
            print(D.fwi_model_names[D.fwi_model_names.index(os.path.join(D.save_folder,'model','modelTest_vp_stage_'+tmp2))])
            fwi_res_vp=D.models[D.fwi_model_names.index(os.path.join(D.save_folder,'model','modelTest_vp_stage_'+tmp2))]
            fwi_res_vs=D.models[D.fwi_model_names.index(os.path.join(D.save_folder,'model','modelTest_vs_stage_'+tmp2))]
            fwi_res_rho=D.models[D.fwi_model_names.index(os.path.join(D.save_folder,'model','modelTest_rho_stage_'+tmp2))]

            nx_orig=D.NX
            ind=D.fwi_model_names.index(os.path.join(D.save_folder,'model','modelTest_vp.bin'))
            ind=D.fwi_model_names.index( fnmatch.filter(D.fwi_model_names,'*vp_stage*')[-1] )
            print(D.fwi_model_names[ind])
            fwi_res=D.models[ind]
            x = np.arange(nx_orig) * D.DH / 1000
            y = np.arange(D.NY) * D.DH / 1000
            extent = np.array([x.min(), x.max(), y.min(), y.max()])
            #############################   plot initial model for fwi and inverted result in the 1st raw
            i=0; row=0; col=0
            ax[i]=fig.add_subplot(gs[row,col]); 
            # ax[i].axes.xaxis.set_visible(False)
            # ax[i].text(0.4,0.7, labels[i], fontsize=Fontsize,color = "white",weight="bold")
            ax[i].text(label_position_inside_axes[0],label_position_inside_axes[1],labels[i], transform=ax[i].transAxes,fontsize=Fontsize,color = "white",weight='bold',va='top')
            ax[i].imshow(D.model_init.vp[:,0:nx_orig],extent=extent,vmin=a,vmax=b,aspect='auto');ax[i].invert_yaxis()
            ax[i].plot(vh, np.arange(len(log))*dx/1000, 'k--')
            ax[i].plot(vh + (log[::-1] - min(log)) / 1000, np.arange(len(log))*dx/1000, 'k')
            ax[i].set_ylabel('Depth, km',fontsize=labels_Fontsize)
            ax[i].set_xlabel('X, km',fontsize=labels_Fontsize)
            ax[i].tick_params(labelsize=labelsize)
            i=1; row=0; col=1
            ax[i]=fig.add_subplot(gs[row,col]); 
            ax[i].axes.yaxis.set_visible(False)
            ax[i].text(0.4,0.7, labels[i], fontsize=Fontsize,color = "white",weight="bold")
            last_image=ax[i].imshow(fwi_res[:,0:nx_orig],extent=extent,vmin=a,vmax=b,aspect='auto');ax[i].invert_yaxis()
            ax[i].plot(vh, np.arange(len(log))*dx/1000, 'k--')
            ax[i].set_xlabel('X, km',fontsize=labels_Fontsize)
            ax[i].tick_params(labelsize=labelsize)
            ax[i].plot(vh + (log[::-1] - min(log)) / 1000, np.arange(len(log))*dx/1000, 'k')
            #############################   calculate some logs
            vp_log=D.model_init.vp[::-1,log_idx];   vp_log1=vp_log[0:len(lvp)]
            score1=PCC(vp_log1,lvp[::-1])
            vp_log=fwi_res_vp[::-1,log_idx];        vp_log2=vp_log[0:len(lvp)]
            score2=PCC(vp_log2,lvp[::-1])
            #############################   plot initial model log and well log
            i=3; row=1; col=0
            ax[i]=fig.add_subplot(gs[row,col]); 
            # ax[i].text(0.4,2400, labels[i], fontsize=Fontsize,color = "black",weight="bold")
            ax[i].text(label_position_inside_axes[0],label_position_inside_axes[1],labels[i], transform=ax[i].transAxes,fontsize=Fontsize,color = "black",weight='bold',va='top')
            ax[i].plot(y2,lvp[::-1],'b--',label='vp_well_log')
            ax[i].plot(y2,lvp_upscaled,'b-.',label='upscaled_vp_well_log')
            ax[i].plot(y2,lvs[::-1], color='orange',linestyle='--'); 
            ax[i].plot(y2,lrho[::-1],'g--');
            ax[i].plot(y,D.model_init.vp[::-1,log_idx], 'b',label='vp'); 
            ax[i].plot(y,D.model_init.vs[::-1,log_idx], 'orange',label='vs')
            ax[i].plot(y,D.model_init.rho[::-1,log_idx],'g', label='rho');
            ax[i].set_xlim(left=0,right=y[-1])
            ax[i].set_ylim(bottom=0,top=3100)
            ax[i].grid()
            ax[i].tick_params(labelsize=labelsize)
            ax[i].set_xlabel('Depth, km',fontsize=labels_Fontsize)
            ax[i].set_ylabel('Velocity, m/sec',fontsize=labels_Fontsize)
            vp_log=D.model_init.vp[::-1,log_idx]
            vp_log=vp_log[0:len(lvp)]
            well_log=lvp[::-1]
            well_log=lvp_upscaled
            #############################
            score=      PCC(vp_log[water_start:],well_log[water_start:])
            score_mse=  MSE(vp_log[water_start:],well_log[water_start:])
            title_score='PCC(Vp initial, Vp well)='+numstr_3( score )#+'_'+numstr_2(score_mse)
            ax[i].set_title(title_score,fontsize=textFontsize)
            plt.legend(fontsize=labelsize,framealpha=0.1)
            #############################   plot inverted model log and well log
            i=4; row=1; col=1
            ax[i]=fig.add_subplot(gs[row,col]); 
            # ax[i].text(0.4,2400, labels[i], fontsize=Fontsize,color = "black",weight="bold")
            ax[i].text(label_position_inside_axes[0],label_position_inside_axes[1],labels[i], transform=ax[i].transAxes,fontsize=Fontsize,color = "black",weight='bold',va='top')
            ax[i].plot(y2,lvp[::-1],'b--',label='vp_well_log')
            ax[i].plot(y2,lvp_upscaled,'b-.',label='upscaled_vp_well_log')
            ax[i].plot(y2,lvs[::-1], color='orange',linestyle='--'); 
            ax[i].plot(y2,lrho[::-1],'g--');
            ax[i].plot(y,fwi_res_vp[::-1,log_idx], 'b',label='vp'); 
            ax[i].plot(y,fwi_res_vs[::-1,log_idx], 'orange',label='vs')
            ax[i].plot(y,fwi_res_rho[::-1,log_idx],'g', label='rho'); 
            ax[i].set_xlim(left=0,right=y[-1])
            ax[i].set_ylim(bottom=0,top=3100)
            ax[i].grid()
            ax[i].tick_params(labelsize=labelsize)
            ax[i].set_xlabel('Depth, km',fontsize=labels_Fontsize)
            vp_log=fwi_res_vp[::-1,log_idx];   vp_log=vp_log[0:len(lvp)]
            if score_to_use=='PCC':
                score=      PCC(vp_log[water_start:],well_log[water_start:])
            if score_to_use=='MSE':
                score=  MSE(vp_log[water_start:],well_log[water_start:])
            title_score=score_to_use+'(Vp inverted, Vp well)='+numstr_3( score )#+'_'+numstr_2(score_mse)
            ax[i].set_title(title_score,fontsize=textFontsize)
            ax[i].yaxis.set_major_formatter(plt.NullFormatter())
            plt.legend(fontsize=labelsize,framealpha=0.1)
            #############################   data misfits
            niter_list=[];misfit_list=[]
            i=2; row=0; col=2
            m,m_data=plot_fwi_misfit2(os.path.join(path,'fld','seis_fwi_log.dat'),D)
            nstage=m_data['nstage']
            niter_stage=m_data['niter_stage']
            nstage_trans=m_data['nstage_trans']
            iteration=m_data['iteration']
            niter_list.append(len(iteration))
            misfit_list.append(m)
            #############################   model misfits
            D.fwi_model_names
            fwi_vp_names=fnmatch.filter(D.fwi_model_names,'*vp_stage*')
            model_misfit=np.empty((len(fwi_vp_names)))
            iteration2=np.arange(len(model_misfit))
            for iter,fwi_vp_name in enumerate(fwi_vp_names):
                fwi_res=D.models[D.fwi_model_names.index(fwi_vp_name)]
                vp_log=fwi_res[::-1,log_idx];
                vp_log=vp_log[0:len(lvp)]
                score=      PCC(vp_log[water_start:],well_log[water_start:])
                model_misfit[iter]=score
            #############################
            ax[i]=fig.add_subplot(gs[row,col])
            ax_extra=ax[i].twinx()
            ax[i].plot(iteration,m,'b-', linewidth=3, label='Data misfit') #Evolution of the misfit function
            ax[i].set_xlabel('Iteration number',fontsize=labels_Fontsize)
            ax[i].text(label_position_inside_axes[0],label_position_inside_axes[1],labels[i], transform=ax[i].transAxes,fontsize=Fontsize,color = "black",weight='bold',va='top')
            ax[i].tick_params(labelsize=labelsize)
            ax[i].axes.yaxis.set_visible(False)
            for ii in range(1, nstage):
                ax[i].plot([nstage_trans[ii-1]+ii,nstage_trans[ii-1]+ii], [np.min(m),np.max(m)],'k--', linewidth=3)
            #############################
            if plot_model_misfit==1:
                ax_extra.plot(iteration2,model_misfit,'g-', linewidth=3, label='PCC well log comparison') #Evolution of the misfit function
                ax_extra.axes.yaxis.set_visible(False)
            #############################
            ######  x limits
            niter_list=np.asarray(niter_list)
            misfit_list=np.asarray(misfit_list)
            mf_min=0;mf_max=0;
            for misfit_curve in misfit_list:
                mf_min=np.min([np.min(misfit_curve),mf_min])
                mf_max=np.max([np.max(misfit_curve),mf_max])
            ax[2].set_xlim(left=1,right=np.max(niter_list))
            #############################   sety ylim
            ax[2].set_ylim(bottom=mf_min,top=mf_max)
            #############################
            y_pos=mf_min+4/5*(mf_max-mf_min)
            #############################   append colorbar
            cbar_ax = fig.add_axes([0.93, 0.52, 0.02, 0.35])
            cbar=fig.colorbar(last_image,cax=cbar_ax)
            # cbar.set_label('V (km/sec)',labelpad=-40, y=1.05, rotation=0)   #horizontalalignment='center'
            cbar.ax.set_title('V (m/sec)',fontsize=18,pad=13.3)
            for t in cbar.ax.get_yticklabels():
                t.set_fontsize(16)
            #############################
            save_file_path1=os.path.join(log_path,name+'.png')
            save_file_path2=os.path.join(save_path,name+'.png')
            print('Saving ML_result to '+save_file_path1)
            print('Saving ML_result to '+save_file_path2)
            plt.savefig(save_file_path1,dpi=300,bbox_inches='tight')
            plt.savefig(save_file_path2,dpi=300,bbox_inches='tight')
            plt.close()
            ss=1
    return None
def fwi_results_cgg_visualization2_4(log_path,save_path,results_root,strategy_name):
    """  plot the cgg fwi result with data misfits and comparison with well logs  
         In this function we compare the inverted logs with the upscaled true well log
    """
    print('Plotting the folder,',strategy_name)
    strategy_name=strategy_name.split('/')[-1]
    root1=os.path.join(results_root,strategy_name);
    label_position_inside_axes=[0.04,0.97]
    if os.path.exists(root1):
        dirs=next(os.walk(os.path.join(root1)))[1]
        plot_model_misfit=0
        #################   process full list
        for dir in dirs:
            path=os.path.join(root1,dir)
            D=parse_denise_folder(path,denise_root='./')
            d=[D]
            name=strategy_name+'_'+dir
            ################################
            if hasattr(D,'fwi_model_names')==False:
                print('Folder is empty!Folder is empty!Folder is empty!Folder is empty!Folder is empty!')
                exit()
            ################################    load well log data
            dx=25
            info_file=os.path.join('./','for_pasha','acq_data_parameters_cgg.pkl')
            with open(info_file,'rb') as input:
                acq_data=pickle.load(input)
            log_dict=acq_data['log_dict']
            log_loc=log_dict['loc']
            log=log_dict['data']
            log_idx = int(log_loc / 25)
            vh = log_loc * np.ones_like(log) / 1000 
            lvp=log;   print(lvp.shape)
            # shear velocity, [m/s]
            lvs = lvp.copy() / (3 ** 0.5)
            lvs = np.where(lvp < 1.01 * np.min(lvp), 0, lvs)
            # density, [kg/m3] 
            lrho = 1e3*0.3 * lvp.copy()**0.25
            lrho = np.where(lvp < 1.01 * np.min(lvp), 1000, lrho)
            y = np.arange(D.NY) * D.DH / 1
            y2 = np.arange(len(lvp)) * D.DH / 1000
            #############################upscale well log with moving average
            well_log=lvp[::-1]
            df = pd.DataFrame(data=well_log.ravel())
            lvp_upscaled6=df.rolling(window=int(2*(25/dx)+1),min_periods=1).mean()
            lvp_upscaled6[0:water_start]=1500
            lvp_upscaled7=df.rolling(window=int(2*(50/dx)+1),min_periods=1).mean()
            lvp_upscaled7[0:water_start]=1500
            lvp_upscaled8=df.rolling(window=int(2*(75/dx)+1),min_periods=1).mean()
            lvp_upscaled8[0:water_start]=1500

            plt.figure()
            plt.plot(well_log,label='0')
            plt.plot(lvp_upscaled6,label='50m window')
            plt.plot(lvp_upscaled7,label='100m window')
            plt.plot(lvp_upscaled8,label='150m window')
            plt.legend()
            plt.show()
            plt.savefig('test.png')
            plt.close()
            lvp_upscaled=np.copy(lvp_upscaled6)
            ################################    create figure
            fig_size = plt.rcParams["figure.figsize"]
            fig_size[0] = 12.4/2*3      #width
            fig_size[1] = 6.6   #height
            plt.rcParams["figure.figsize"] = fig_size
            plt.rcParams["font.size"] = 14
            n_row=2;    n_col=3
            gs = gridspec.GridSpec(n_row,n_col)
            gs.update(left=0.0, right=0.92, wspace=0.0, hspace=0.5)
            # gs=GridSpec(n_row,n_col+1, width_ratios=[1,1,0.1], height_ratios=[1,1,1])
            # gs.update(wspace=0.0,hspace=0.0)
            ax=[None]*6
            labels=['a','b','c','d','e','f']
            a=1500;b=3000
            Fontsize=32
            labels_Fontsize=18
            text_Fontsize=30
            labelsize=14
            Fontsize=32    
            textFontsize=20
            fig=plt.figure()
            j=0;D=d[j]
            tmp=fnmatch.filter(D.fwi_model_names,'*vs_stage*')[-1] 
            tmp2=tmp.split('_stage_')[-1]
            print(D.fwi_model_names[D.fwi_model_names.index(os.path.join(D.save_folder,'model','modelTest_vp_stage_'+tmp2))])
            fwi_res_vp=D.models[D.fwi_model_names.index(os.path.join(D.save_folder,'model','modelTest_vp_stage_'+tmp2))]
            fwi_res_vs=D.models[D.fwi_model_names.index(os.path.join(D.save_folder,'model','modelTest_vs_stage_'+tmp2))]
            fwi_res_rho=D.models[D.fwi_model_names.index(os.path.join(D.save_folder,'model','modelTest_rho_stage_'+tmp2))]

            nx_orig=D.NX
            ind=D.fwi_model_names.index(os.path.join(D.save_folder,'model','modelTest_vp.bin'))
            ind=D.fwi_model_names.index( fnmatch.filter(D.fwi_model_names,'*vp_stage*')[-1] )
            print(D.fwi_model_names[ind])
            fwi_res=D.models[ind]
            x = np.arange(nx_orig) * D.DH / 1000
            y = np.arange(D.NY) * D.DH / 1000
            extent = np.array([x.min(), x.max(), y.min(), y.max()])
            #############################   plot initial model for fwi and inverted result in the 1st raw
            i=0; row=0; col=0
            ax[i]=fig.add_subplot(gs[row,col]); 
            # ax[i].axes.xaxis.set_visible(False)
            # ax[i].text(0.4,0.7, labels[i], fontsize=Fontsize,color = "white",weight="bold")
            ax[i].text(label_position_inside_axes[0],label_position_inside_axes[1],labels[i], transform=ax[i].transAxes,fontsize=Fontsize,color = "white",weight='bold',va='top')
            ax[i].imshow(D.model_init.vp[:,0:nx_orig],extent=extent,vmin=a,vmax=b,aspect='auto');ax[i].invert_yaxis()
            ax[i].plot(vh, np.arange(len(log))*dx/1000, 'k--')
            ax[i].plot(vh + (log[::-1] - min(log)) / 1000, np.arange(len(log))*dx/1000, 'k')
            ax[i].set_ylabel('Depth, km',fontsize=labels_Fontsize)
            ax[i].set_xlabel('X, km',fontsize=labels_Fontsize)
            ax[i].tick_params(labelsize=labelsize)
            i=1; row=0; col=1
            ax[i]=fig.add_subplot(gs[row,col]); 
            ax[i].axes.yaxis.set_visible(False)
            ax[i].text(0.4,0.7, labels[i], fontsize=Fontsize,color = "white",weight="bold")
            last_image=ax[i].imshow(fwi_res[:,0:nx_orig],extent=extent,vmin=a,vmax=b,aspect='auto');ax[i].invert_yaxis()
            ax[i].plot(vh, np.arange(len(log))*dx/1000, 'k--')
            ax[i].set_xlabel('X, km',fontsize=labels_Fontsize)
            ax[i].tick_params(labelsize=labelsize)
            ax[i].plot(vh + (log[::-1] - min(log)) / 1000, np.arange(len(log))*dx/1000, 'k')
            #############################   calculate some logs
            vp_log=D.model_init.vp[::-1,log_idx];   vp_log1=vp_log[0:len(lvp)]
            score1=PCC(vp_log1,lvp[::-1])
            vp_log=fwi_res_vp[::-1,log_idx];        vp_log2=vp_log[0:len(lvp)]
            score2=PCC(vp_log2,lvp[::-1])
            #############################   plot initial model log and well log
            i=3; row=1; col=0
            ax[i]=fig.add_subplot(gs[row,col]); 
            # ax[i].text(0.4,2400, labels[i], fontsize=Fontsize,color = "black",weight="bold")
            ax[i].text(label_position_inside_axes[0],label_position_inside_axes[1],labels[i], transform=ax[i].transAxes,fontsize=Fontsize,color = "black",weight='bold',va='top')
            # ax[i].plot(y2,lvp_upscaled,'b-.',label='upscaled_vp_well_log')
            
            # ax[i].plot(y2,lvp[::-1],'b--',label=r"$\mathbf{V_p}$"+" well log")
            # ax[i].plot(y2,lvs[::-1], color='orange',linestyle='--',label=r"$\mathbf{V_s}$"+" well log") 
            # ax[i].plot(y2,lrho[::-1],'g--',label=r"$\mathbf{\rho}$" + " well log")
            # ax[i].plot(y,D.model_init.vp[::-1,log_idx], 'b',label=r"$\mathbf{V_p}$"); 
            # ax[i].plot(y,D.model_init.vs[::-1,log_idx], 'orange',label=r"$\mathbf{V_s}$")
            # ax[i].plot(y,D.model_init.rho[::-1,log_idx],'g', label=r"$\mathbf{\rho}$")

            ax[i].plot(y2,lvp[::-1],'b--',label=r"$V_p$"+" well log")
            ax[i].plot(y2,lvs[::-1], color='orange',linestyle='--',label=r"$V_s$"+" well log") 
            ax[i].plot(y2,lrho[::-1],'g--',label=r"$\rho$" + " well log")
            ax[i].plot(y,D.model_init.vp[::-1,log_idx], 'b',label=r"$V_p$"); 
            ax[i].plot(y,D.model_init.vs[::-1,log_idx], 'orange',label=r"$V_s$")
            ax[i].plot(y,D.model_init.rho[::-1,log_idx],'g', label=r"$\rho$")

            ax[i].set_xlim(left=0,right=y[-1])
            ax[i].set_ylim(bottom=0,top=3100)
            ax[i].grid()
            ax[i].tick_params(labelsize=labelsize)
            ax[i].set_xlabel('Depth, km',fontsize=labels_Fontsize)
            ax[i].set_ylabel('Velocity, m/sec',fontsize=labels_Fontsize)
            vp_log=D.model_init.vp[::-1,log_idx]
            vp_log=vp_log[0:len(lvp)]
            well_log=lvp[::-1]
            well_log=lvp_upscaled
            #############################
            score=      PCC(vp_log[water_start:],well_log[water_start:])
            score_mse=  MSE(vp_log[water_start:],well_log[water_start:])
            title_score='PCC($\mathit{V}$$_{p}$ initial, $\mathit{V}$$_{p}$ well)='+numstr_3( score )#+'_'+numstr_2(score_mse)
            # ax[i].set_title(title_score,fontsize=textFontsize)
            legend_properties = {'size':textFontsize}     #'weight':'bold'
            legend_loc=(1.69,-0.25)
            legend_loc=(2.5,-0.25)
            plt.legend(bbox_to_anchor=legend_loc,framealpha=1.0,loc='lower right',prop=legend_properties)  # bbox_to_anchor=(1.49,0.04), framealpha=0.1
            #############################   plot inverted model log and well log
            i=4; row=1; col=1
            ax[i]=fig.add_subplot(gs[row,col]); 
            # ax[i].text(0.4,2400, labels[i], fontsize=Fontsize,color = "black",weight="bold")
            ax[i].text(label_position_inside_axes[0],label_position_inside_axes[1],labels[i], transform=ax[i].transAxes,fontsize=Fontsize,color = "black",weight='bold',va='top')
            # ax[i].plot(y2,lvp_upscaled,'b-.',label='upscaled_vp_well_log')
            ax[i].plot(y2,lvp[::-1],'b--',label=r"$\mathbf{V_p}$"+" well log")
            ax[i].plot(y2,lvs[::-1], color='orange',linestyle='--'); 
            ax[i].plot(y2,lrho[::-1],'g--');
            ax[i].plot(y,fwi_res_vp[::-1,log_idx],'b',label=r"$\mathbf{V_p}$")
            ax[i].plot(y,fwi_res_vs[::-1,log_idx],'orange',label=r"$\mathbf{V_s}$")
            ax[i].plot(y,fwi_res_rho[::-1,log_idx],'g',label=r"$\mathbf{\rho}$")
            #################################################################################################################################################
            ax[i].set_xlim(left=0,right=y[-1])
            ax[i].set_ylim(bottom=0,top=3100)
            ax[i].grid()
            ax[i].tick_params(labelsize=labelsize)
            ax[i].set_xlabel('Depth, km',fontsize=labels_Fontsize)
            ax[i].axes.yaxis.set_visible(False)
            vp_log=fwi_res_vp[::-1,log_idx];   vp_log=vp_log[0:len(lvp)]
            if score_to_use=='PCC':
                score=      PCC(vp_log[water_start:],well_log[water_start:])
            if score_to_use=='MSE':
                score=  MSE(vp_log[water_start:],well_log[water_start:])
            title_score=score_to_use+'(Vp inverted, Vp well)='+numstr_3( score )#+'_'+numstr_2(score_mse)
            # ax[i].set_title(title_score,fontsize=textFontsize)
            # ax[i].yaxis.set_major_formatter(plt.NullFormatter())
            #################################################################################################################################################
            # plt.legend(framealpha=0.1,loc='lower right',prop=legend_properties)
            # plt.legend(bbox_to_anchor=(1.49,0.04),framealpha=0.1,loc='lower right',prop=legend_properties,borderaxespad=0)
            # plt.legend(bbox_to_anchor=(1.99,0.04),framealpha=1.0,loc='lower right',prop=legend_properties,borderaxespad=0)  # bbox_to_anchor=(1.49,0.04), framealpha=0.1
            #################################################################################################################################################
            #############################   data misfits
            niter_list=[];misfit_list=[]
            i=2; row=0; col=2
            m,m_data=plot_fwi_misfit2(os.path.join(path,'fld','seis_fwi_log.dat'),D)
            nstage=m_data['nstage']
            niter_stage=m_data['niter_stage']
            nstage_trans=m_data['nstage_trans']
            iteration=m_data['iteration']
            niter_list.append(len(iteration))
            misfit_list.append(m)
            #############################   model misfits
            fwi_vp_names=fnmatch.filter(D.fwi_model_names,'*vp_stage*')
            model_misfit=np.empty((len(fwi_vp_names)))
            iteration2=np.arange(len(model_misfit))
            for iter,fwi_vp_name in enumerate(fwi_vp_names):
                fwi_res=D.models[D.fwi_model_names.index(fwi_vp_name)]
                vp_log=fwi_res[::-1,log_idx];
                vp_log=vp_log[0:len(lvp)]
                score=      PCC(vp_log[water_start:],well_log[water_start:])
                model_misfit[iter]=score
            #############################
            ax[i]=fig.add_subplot(gs[row,col])
            ax[i].plot(iteration,m,'b-', linewidth=3, label='Data misfit') #Evolution of the misfit function
            ax[i].set_xlabel('Iteration number',fontsize=labels_Fontsize)
            ax[i].text(label_position_inside_axes[0],label_position_inside_axes[1],labels[i], transform=ax[i].transAxes,fontsize=Fontsize,color = "black",weight='bold',va='top')
            ax[i].tick_params(labelsize=labelsize)
            ax[i].axes.yaxis.set_visible(False)
            for ii in range(1, nstage):
                ax[i].plot([nstage_trans[ii-1]+ii,nstage_trans[ii-1]+ii], [np.min(m),np.max(m)],'k--', linewidth=3)
            #############################
            if plot_model_misfit==1:
                ax_extra=ax[i].twinx()
                ax_extra.plot(iteration2,model_misfit,'g-', linewidth=3, label='PCC well log comparison') #Evolution of the misfit function
                ax_extra.axes.yaxis.set_visible(False)
            #############################
            ######  x limits
            niter_list=np.asarray(niter_list)
            misfit_list=np.asarray(misfit_list)
            mf_min=0;mf_max=0;
            for misfit_curve in misfit_list:
                mf_min=np.min([np.min(misfit_curve),mf_min])
                mf_max=np.max([np.max(misfit_curve),mf_max])
            ax[2].set_xlim(left=1,right=np.max(niter_list))
            #############################   sety ylim
            ax[2].set_ylim(bottom=mf_min,top=mf_max)
            #############################
            y_pos=mf_min+4/5*(mf_max-mf_min)
            #############################   append colorbar
            cbar_ax = fig.add_axes([0.93, 0.52, 0.02, 0.35])
            cbar=fig.colorbar(last_image,cax=cbar_ax)
            cbar.ax.set_title('$\mathit{V}$$_{p}$ (m/sec)',fontsize=18,pad=13.3)
            for t in cbar.ax.get_yticklabels():
                t.set_fontsize(16)
            #############################
            save_file_path1=os.path.join(log_path,name+'.png')
            save_file_path2=os.path.join(save_path,name+'.png')
            if 'fwi_56_strategy_l5_del_low_gen__5hz_3' in name:
                pic_name_in_draft='cgg_1d_highfreq'
            elif 'fwi_56_strategy_l5_del_low_gen__5hz_3' in name:
                pic_name_in_draft='cgg_1d_highfreq'
            elif 'fwi_56_strategy_l5multi_cnn_13_special_weight_236_2_model__cgg_lin_vp_long_300_f_z_1_model__cgg_lin_vp_long' in name:
                pic_name_in_draft='cgg_cnnpred_highfreq'
            else:
                pic_name_in_draft=name
            save_file_path3=os.path.join(log_path,pic_name_in_draft+'.png')
            save_file_path4=os.path.join('./pictures_geophysics',pic_name_in_draft+'.png')
            print('Saving ML_result to '+save_file_path1)
            print('Saving ML_result to '+save_file_path2)
            plt.savefig(save_file_path1,dpi=300,bbox_inches='tight')
            plt.savefig(save_file_path2,dpi=300,bbox_inches='tight')
            plt.savefig(save_file_path3,dpi=300,bbox_inches='tight')
            plt.savefig(save_file_path4,dpi=300,bbox_inches='tight')
            plt.close()
    else:
        print('folder ',strategy_name,'is empty')
    return None
def plot_3_init_models_in_column(path,save_path,last_stage=100):
    ############    pickup field well log information
    info_file=os.path.join('./','for_pasha','acq_data_parameters_cgg.pkl')
    with open(info_file,'rb') as input:
        acq_data=pickle.load(input)
    log_dict=acq_data['log_dict']
    dx=25
    log_loc=log_dict['loc']
    log=log_dict['data']
    log_idx = int(log_loc / 25)
    vh = log_loc * np.ones_like(log) / 1000 
    lvp=log[::-1] 
    # shear velocity, [m/s]
    lvs = lvp.copy() / (3 ** 0.5)
    lvs = np.where(lvp < 1.01 * np.min(lvp), 0, lvs)
    # density, [kg/m3] 
    lrho = 1e3*0.3 * lvp.copy()**0.25
    lrho = np.where(lvp < 1.01 * np.min(lvp), 1000, lrho)

    well_log_original=lvp
    df = pd.DataFrame(data=well_log_original.ravel())
    # lvp_upscaled=df.rolling(window=int(2*(25/dx)+1),min_periods=1).mean()
    lvp_upscaled=df.rolling(window=int(2*(200/dx)+1),min_periods=1).mean()
    lvp_upscaled[0:water_start]=1500
    well_log=np.copy(lvp_upscaled)
    ################################ record sequential initial model prediction into the list
    name=path.split('/')[-1]
    folders=os.listdir(path)
    folders=fnmatch.filter(folders,'*stage*')
    folders=sorted(folders)
    m_list=[]
    for i in range(last_stage+1):
        name=path.split('/')[-1]
        fld=os.path.join(path,'stage'+str(i))
        if os.path.exists(fld):
            d=parse_denise_folder(fld)
            if hasattr(d,'model_init')==True:
                m_init=np.fliplr((d.model_init.vp).T)
                m_true=np.fliplr((d.model.vp).T)
                score=numstr_3(F_r2(m_init,m_true))
                # Plot_image(m_init.T,Show_flag=0,Save_flag=1,Title=name+'_m_init_stage_'+str(i)+'_R2(initial,true)='+score,Save_pictures_path=save_path)
                # smoothed_true_model=F_smooth(m_true,sigma_val=int(pars['target_smoothing_diameter']/pars['dx']))
                smoothed_true_model=F_smooth(m_true,sigma_val=int(300/25))
                smoothed_true_model[np.flipud(d.TAPER).T==0]=1500
                score=numstr_3(F_r2(smoothed_true_model,m_true))
                # Plot_image(smoothed_true_model.T,Show_flag=0,Save_flag=1,Title=name+'_m_target'+'_R2(initial,true)='+score,Save_pictures_path=save_path)
                # Plot_image(np.flipud(d.TAPER),Show_flag=0,Save_flag=1,Title=name+'taper',Save_pictures_path=save_path)
                m_list.append(m_init)
            else:   break
    ################################ append smoothed true model to the models list. Only for synthetic data
    if 'marm' in name or 'over' in name or 'Marm' in name or 'Over' in name:
        m_list.append(smoothed_true_model)
    n=len(m_list)
    print('Number of initial models=',n)
    ################################ calculate image extent
    nx_orig=d.NX        #-320
    x = np.arange(nx_orig) * d.DH / 1000
    y = np.arange(d.NY) * d.DH / 1000
    extent = np.array([x.min(), x.max(), y.min(), y.max()])
    ################################    set plotting parameters
    labels_Fontsize=18
    text_Fontsize=30
    labelsize=14
    Fontsize=32    
    textFontsize=20
    plt.rcParams["font.size"] = 14
    labels=['a) Apriori initial model','b) CNN-predicted initial model','c) CNN-target initial model','d','e','f','g','h','i','j','k','l','m','n','o']
    ################################
    if 'cgg' in name:
        vmin1=1500;vmax1=3000
        m_name='cgg'
    if 'marm' in name or 'over' in name or 'Marm' in name or 'Over' in name:
        vmin1=1500;vmax1=4500
        m_name='other'
    vmin1=vmin1/1000
    vmax1=vmax1/1000
    print('vmin1=',vmin1,',vmax1=',vmax1)

    if m_name=='cgg':
        n=2
        plotting_loop=[0,last_stage]
    elif m_name=='other':
        n=3     
        plotting_loop=[0,last_stage,last_stage+1]
    fig_size=[7,2.2*n]  #width, height
    fig,ax =  plt.subplots(nrows=n,ncols=1,figsize=fig_size)
    top_of_pictures=0.8/(n*2.2);  print(top_of_pictures)
    plt.subplots_adjust(left=0.0,bottom=0.0, 
                        right=1.0, top= 1-top_of_pictures, 
                        wspace=0.00, hspace=0.42)
    ################################
    for i,m_index in enumerate(plotting_loop):
        print(i,m_index)
        fld=os.path.join(path,'stage'+str(m_index))
        if i==0:
            im=ax[i].imshow(np.flipud(m_list[m_index].T)/1000,extent=extent,vmin=vmin1,vmax=vmax1,aspect='auto')
        else:
            ax[i].imshow(np.flipud(m_list[m_index].T)/1000,extent=extent,vmin=vmin1,vmax=vmax1,aspect='auto')
        #############################
        if m_name=='cgg':
            dx=d.DH
            ax[i].plot(vh, np.arange(len(log))*dx/1000, 'k--')
            ax[i].plot(vh + (log[::-1] - min(log)) / 1000, np.arange(len(log))*dx/1000, 'k')
            title=labels[i]
        else:
            score=(F_r2(m_list[m_index],m_true))
            # title=labels[i]+', $\mathit{R}$$^{2}$($\mathit{V}$$_{p}$ initial, $\mathit{V}$$_{p}$ true)='+str('{0:.3f}'.format(score))
            # title=labels[i]+', $\mathit{R}$$^{2}$='+str('{0:.3f}'.format(score))
            title=labels[i]+', $\mathit{R}$$^{2}$ (initial,true)='+str('{0:.3f}'.format(score))
        if i==n-1:
            ax[i].set_xlabel('X, km',fontsize=labels_Fontsize)
        ax[i].set_title(title,fontsize=labels_Fontsize)
        ax[i].set_ylabel('Depth, km',fontsize=labels_Fontsize)
        ax[i].tick_params(labelsize=labelsize)
        ax[i].invert_yaxis()
    ##########################################
    # cbar_ax = fig.add_axes([0.93, 0.52, 0.02, 0.35])
    # cbar=fig.colorbar(last_image,cax=cbar_ax)
    # cbar.ax.set_title('$\mathit{V}$$_{p}$ (m/sec)',fontsize=18,pad=13.3)
    # for t in cbar.ax.get_yticklabels():
    #     t.set_fontsize(16)

    # cax = fig.add_axes([0.75, 0.98, 7*0.2/fig.bbox_inches.xmax,11*0.02/fig.bbox_inches.ymax])
    # cbar=plt.colorbar(im,cax=cax,orientation='horizontal')
    # # cbar.ax.set_ylabel('$\mathit{V}$$_{p}$, km/sec',fontsize=labels_Fontsize,rotation=0,labelpad=100,loc='bottom')
    # cbar.ax.set_ylabel('$\mathit{V}$$_{p}$, km/sec',fontsize=labels_Fontsize)
    # cbar.ax.tick_params(labelsize=labelsize)

    cax = fig.add_axes([0.75, 0.98, 7*0.2/fig.bbox_inches.xmax,11*0.02/fig.bbox_inches.ymax])
    cbar=plt.colorbar(im,cax=cax,orientation='horizontal')
    # cbar.ax.set_ylabel('$\mathit{V}$$_{p}$, km/sec',fontsize=labels_Fontsize,rotation=0,labelpad=100,loc='bottom')
    cbar.ax.set_title('$\mathit{V}$$_{p}$, km/sec',fontsize=labels_Fontsize)
    cbar.ax.tick_params(labelsize=labelsize)
    ####################################3
    save_filename=os.path.join(save_path,name)
    print('save to ,',save_filename+'.png')
    fig.savefig(save_filename,dpi=300,bbox_inches='tight')
    plt.close()
    return None
def plot_init_models_from_seq_cnn_application(path,save_path,last_stage=100):
    ############    pickup well log information
    info_file=os.path.join('./','for_pasha','acq_data_parameters_cgg.pkl')
    with open(info_file,'rb') as input:
        acq_data=pickle.load(input)
    log_dict=acq_data['log_dict']
    log_loc=log_dict['loc']
    log=log_dict['data']
    log_idx = int(log_loc / 25)
    vh = log_loc * np.ones_like(log)/1000 
    lvp=log;   print(lvp.shape)
    ################################
    name=path.split('/')[-1]
    folders=os.listdir(path)
    folders=fnmatch.filter(folders,'*stage*')
    folders=sorted(folders)
    m_list=[]
    for i in range(last_stage+1):
        name=path.split('/')[-1]
        fld=os.path.join(path,'stage'+str(i))
        if os.path.exists(fld):
            d=parse_denise_folder(fld)
            if hasattr(d,'model_init')==True:
                m_init=np.fliplr((d.model_init.vp).T)
                m_true=np.fliplr((d.model.vp).T)
                score=numstr_3(F_r2(m_init,m_true))
                # Plot_image(m_init.T,Show_flag=0,Save_flag=1,Title=name+'_m_init_stage_'+str(i)+'_R2(initial,true)='+score,Save_pictures_path=save_path)
                # smoothed_true_model=F_smooth(m_true,sigma_val=int(pars['target_smoothing_diameter']/pars['dx']))
                smoothed_true_model=F_smooth(m_true,sigma_val=int(300/25))
                smoothed_true_model[np.flipud(d.TAPER).T==0]=1500
                score=numstr_3(F_r2(smoothed_true_model,m_true))
                # Plot_image(smoothed_true_model.T,Show_flag=0,Save_flag=1,Title=name+'_m_target'+'_R2(initial,true)='+score,Save_pictures_path=save_path)
                # Plot_image(np.flipud(d.TAPER),Show_flag=0,Save_flag=1,Title=name+'taper',Save_pictures_path=save_path)
                m_list.append(m_init)
            else:   break
    nx_orig=d.NX        #-320
    x = np.arange(nx_orig) * d.DH / 1000
    y = np.arange(d.NY) * d.DH / 1000
    extent = np.array([x.min(), x.max(), y.min(), y.max()])
    ################################
    labels_Fontsize=18
    text_Fontsize=30
    labelsize=14
    Fontsize=32    
    textFontsize=20
    plt.rcParams["font.size"] = 14
    labels=['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o']
    a=1500;b=4500
    ################################
    if 'cgg' in name:
        vmin1=1500;vmax1=3000
        m_name='cgg'
    if 'marm' in name or 'over' in name or 'Marm' in name or 'Over' in name:
        m_list.append(smoothed_true_model)
        vmin1=1500;vmax1=4500
        m_name='other'
    vmin1=vmin1/1000
    vmax1=vmax1/1000
    print('vmin1=',vmin1,',vmax1=',vmax1)
    
    n=len(m_list)
    print('Number of initial models=',n)
    # if m_name=='cgg':   n=n-1           
    fig_size=[7,2.2*n]  #width, height
    fig,ax =  plt.subplots(nrows=n,ncols=1,figsize=fig_size)
    top_of_pictures=0.8/(n*2.2);  print(top_of_pictures)
    plt.subplots_adjust(left=0.0,bottom=0.0, 
                        right=1.0, top= 1-top_of_pictures, 
                        wspace=0.00, hspace=0.42)
    for i in range(n):
        fld=os.path.join(path,'stage'+str(i))
        if i==0:
            im=ax[i].imshow(np.flipud(m_list[i].T)/1000,extent=extent,vmin=vmin1,vmax=vmax1,aspect='auto')
        else:
            ax[i].imshow(np.flipud(m_list[i].T)/1000,extent=extent,vmin=vmin1,vmax=vmax1,aspect='auto')
        #############################
        vp_log=m_list[i][log_idx,:]
        vp_log=vp_log[0:len(lvp)]
        well_log=lvp[::-1]
        if score_to_use=='PCC':
            score=      PCC(vp_log[water_start:],well_log[water_start:])
        if score_to_use=='MSE':
            score    =  MSE(vp_log[water_start:],well_log[water_start:])
        if score_to_use=='r2':
            score    =  F_r2(vp_log[water_start:],well_log[water_start:])
            # score    =  MSE(vp_log[water_start:],well_log[water_start:])
        #############################
        if m_name=='cgg':
            title=labels[i]+') Model # '+str(i)+', '+score_to_use+'($\mathit{V}$$_{p}$ initial log, $\mathit{V}$$_{p}$ well log)='+str('{0:.3f}'.format(score))
            dx=d.DH
            ax[i].plot(vh, np.arange(len(log))*dx/1000, 'k--')
            ax[i].plot(vh + (log[::-1] - min(log)) / 1000, np.arange(len(log))*dx/1000, 'k')
        else:
            score=(F_r2(m_list[i],m_true))
            title=labels[i]+') Model # '+str(i)+', $\mathit{R}$$^{2}$($\mathit{V}$$_{p}$ initial, $\mathit{V}$$_{p}$ true)='+str('{0:.3f}'.format(score))
        if i==n-1:
            ax[i].set_xlabel('X, km',fontsize=labels_Fontsize)
            if m_name=='other':
                title=labels[i]+') CNN-target model, $\mathit{R}$$^{2}$($\mathit{V}$$_{p}$ initial, $\mathit{V}$$_{p}$ true)='+str('{0:.3f}'.format(score))
        ax[i].set_title(title,fontsize=labels_Fontsize)
        ax[i].set_ylabel('Depth, km',fontsize=labels_Fontsize)
        ax[i].tick_params(labelsize=labelsize)
        ax[i].invert_yaxis()
    # cax = fig.add_axes([0.75, 0.98, 0.2, 0.02])
    cax = fig.add_axes([0.75, 0.98, 7*0.2/fig.bbox_inches.xmax,11*0.02/fig.bbox_inches.ymax])
    cbar=plt.colorbar(im,cax=cax,orientation='horizontal')
    cbar.ax.set_ylabel('$\mathit{V}$$_{p}$, km/sec',fontsize=labels_Fontsize,rotation=0,labelpad=100,loc='bottom')
    cbar.ax.tick_params(labelsize=labelsize)
    save_filename=os.path.join(save_path,name)
    print('save to ,',save_filename+'.png')
    fig.savefig(save_filename,dpi=300,bbox_inches='tight')  #,bbox_inches='tight'
    plt.close()
    return None
def plot_init_models_from_seq_cnn_application_cgg(path,save_path,last_stage=100):
    ############    pickup well log information
    info_file=os.path.join('./','for_pasha','acq_data_parameters_cgg.pkl')
    with open(info_file,'rb') as input:
        acq_data=pickle.load(input)
    log_dict=acq_data['log_dict']
    dx=25
    log_loc=log_dict['loc']
    log=log_dict['data']
    log_idx = int(log_loc / 25)
    vh = log_loc * np.ones_like(log) / 1000 
    lvp=log[::-1] 
    # shear velocity, [m/s]
    lvs = lvp.copy() / (3 ** 0.5)
    lvs = np.where(lvp < 1.01 * np.min(lvp), 0, lvs)
    # density, [kg/m3] 
    lrho = 1e3*0.3 * lvp.copy()**0.25
    lrho = np.where(lvp < 1.01 * np.min(lvp), 1000, lrho)

    well_log_original=lvp
    df = pd.DataFrame(data=well_log_original.ravel())
    # lvp_upscaled=df.rolling(window=int(2*(25/dx)+1),min_periods=1).mean()
    lvp_upscaled=df.rolling(window=int(2*(200/dx)+1),min_periods=1).mean()
    lvp_upscaled[0:water_start]=1500
    well_log=np.copy(lvp_upscaled)
    ################################
    name=path.split('/')[-1]
    folders=os.listdir(path)
    folders=fnmatch.filter(folders,'*stage*')
    folders=sorted(folders)
    m_list=[]
    for i in range(last_stage+1):
        name=path.split('/')[-1]
        fld=os.path.join(path,'stage'+str(i))
        if os.path.exists(fld):
            d=parse_denise_folder(fld)
            if hasattr(d,'model_init')==True:
                m_init=np.fliplr((d.model_init.vp).T)
                m_true=np.fliplr((d.model.vp).T)
                score=numstr_3(F_r2(m_init,m_true))
                # Plot_image(m_init.T,Show_flag=0,Save_flag=1,Title=name+'_m_init_stage_'+str(i)+'_R2(initial,true)='+score,Save_pictures_path=save_path)
                # smoothed_true_model=F_smooth(m_true,sigma_val=int(pars['target_smoothing_diameter']/pars['dx']))
                smoothed_true_model=F_smooth(m_true,sigma_val=int(300/25))
                smoothed_true_model[np.flipud(d.TAPER).T==0]=1500
                score=numstr_3(F_r2(smoothed_true_model,m_true))
                # Plot_image(smoothed_true_model.T,Show_flag=0,Save_flag=1,Title=name+'_m_target'+'_R2(initial,true)='+score,Save_pictures_path=save_path)
                # Plot_image(np.flipud(d.TAPER),Show_flag=0,Save_flag=1,Title=name+'taper',Save_pictures_path=save_path)
                m_list.append(m_init)
            else:   break
    nx_orig=d.NX        #-320
    x = np.arange(nx_orig) * d.DH / 1000
    y = np.arange(d.NY) * d.DH / 1000
    extent = np.array([x.min(), x.max(), y.min(), y.max()])
    ################################
    labels_Fontsize=18
    text_Fontsize=30
    labelsize=14
    Fontsize=32    
    textFontsize=20
    plt.rcParams["font.size"] = 14
    labels=['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o']
    a=1500;b=4500
    ################################
    if 'cgg' in name:
        vmin1=1500;vmax1=3000
        m_name='cgg'
    if 'marm' in name or 'over' in name or 'Marm' in name or 'Over' in name:
        m_list.append(smoothed_true_model)
        vmin1=1500;vmax1=4500
        m_name='other'
    vmin1=vmin1/1000
    vmax1=vmax1/1000
    print('vmin1=',vmin1,',vmax1=',vmax1)
    
    n=len(m_list)
    print('Number of initial models=',n)
    # if m_name=='cgg':   n=n-1           
    fig_size=[7,2.2*n]  #width, height
    fig,ax =  plt.subplots(nrows=n,ncols=1,figsize=fig_size)
    top_of_pictures=0.8/(n*2.2);  print(top_of_pictures)
    plt.subplots_adjust(left=0.0,bottom=0.0, 
                        right=1.0, top= 1-top_of_pictures, 
                        wspace=0.00, hspace=0.42)
    for i in range(n):
        fld=os.path.join(path,'stage'+str(i))
        if i==0:
            im=ax[i].imshow(np.flipud(m_list[i].T)/1000,extent=extent,vmin=vmin1,vmax=vmax1,aspect='auto')
        else:
            ax[i].imshow(np.flipud(m_list[i].T)/1000,extent=extent,vmin=vmin1,vmax=vmax1,aspect='auto')
        #############################
        min_periods=int(25/dx)
        lvp_upscaled_50=df.rolling(window=int(2*(50/dx)+1),min_periods=min_periods).mean().values
        lvp_upscaled_50[0:water_start]=1500
        lvp_upscaled_100=df.rolling(window=int(2*(100/dx)+1),min_periods=min_periods).mean().values
        lvp_upscaled_100[0:water_start]=1500
        lvp_upscaled_200=df.rolling(window=int(2*(200/dx)+1),min_periods=min_periods).mean().values
        lvp_upscaled_200[0:water_start]=1500
        #############################
        # Upscale Vp, Vs, and RHOB with Backus
        # Define Backus length and sampling interval
        # In real-world the former would be defined by modeling or estimated from seismic dominant frequency around the zone of interest at the well location (I recommend the latter for comparing logs to seismic inversion-derived properties).
        lvp_bks,lvs_bks,lrho_bks = br.rockphysics.backus(lvp.ravel(),lvs.ravel(),lrho.ravel(),100,dx)
        lvp_bks[0:water_start]=1500
        lvp_bks_200,lvs_bks_200,lrho_bks_200 = br.rockphysics.backus(lvp.ravel(),lvs.ravel(),lrho.ravel(),200,dx)
        lvp_bks_200[0:water_start]=1500
        lvp_bks_300,lvs_bks_300,lrho_bks_300 = br.rockphysics.backus(lvp.ravel(),lvs.ravel(),lrho.ravel(),300,dx)
        lvp_bks_300[0:water_start]=1500
        lvp_bks_500,lvs_bks_500,lrho_bks_500 = br.rockphysics.backus(lvp.ravel(),lvs.ravel(),lrho.ravel(),500,dx)
        lvp_bks_500[0:water_start]=1500
        lvp_bks_700,lvs_bks_700,lrho_bks_700 = br.rockphysics.backus(lvp.ravel(),lvs.ravel(),lrho.ravel(),700,dx)
        lvp_bks_700[0:water_start]=1500
        lvp_bks_900,lvs_bks_900,lrho_bks_900 = br.rockphysics.backus(lvp.ravel(),lvs.ravel(),lrho.ravel(),900,dx)
        lvp_bks_900[0:water_start]=1500
        #############################
        vp_log=m_list[i][log_idx,:]
        vp_log=vp_log[0:len(lvp)]
        compare_from=water_start
        # compare_from=0
        if score_to_use=='PCC':
            score=      PCC(vp_log[compare_from:],well_log[compare_from:])
        if score_to_use=='MSE':
            score    =  MSE(vp_log[compare_from:],well_log[compare_from:])
        if score_to_use=='r2':
            score    =  F_r2(vp_log[compare_from:],well_log[compare_from:])
            # score    =  MSE(vp_log[compare_from:],well_log[compare_from:])
        #############################
        plt.figure()
        plt.plot(vp_log,label='vp predicted '+str(i)+' times')
        plt.plot(well_log_original,label='well orig')
        # plt.plot(lvp_bks,label='lvp_bks')
        # plt.plot(lvp_bks_200,label='lvp_bks_200')
        # plt.plot(lvp_bks_300,label='lvp_bks_300')
        plt.plot(lvp_upscaled_50,label='well smoothed 50m half-window, '+str( PCC(vp_log[compare_from:],lvp_upscaled_50[compare_from:]) ) )
        plt.plot(lvp_upscaled_100,label='well smoothed 100m half-window, '+str(PCC(vp_log[compare_from:],lvp_upscaled_100[compare_from:])))
        plt.plot(lvp_upscaled_200,label='well smoothed 200m half-window, '+str(PCC(vp_log[compare_from:],lvp_upscaled_200[compare_from:])))
        plt.legend()
        plt.show()
        save_name=os.path.join(save_path,'log_from_stage'+str(i)+'.png');print(save_name)
        plt.savefig(save_name)
        plt.close()
        #############################
        if m_name=='cgg':
            title=labels[i]+') Model # '+str(i)+', '+score_to_use+'($\mathit{V}$$_{p}$ initial log, $\mathit{V}$$_{p}$ well log)='+str('{0:.3f}'.format(score))
            dx=d.DH
            ax[i].plot(vh, np.arange(len(log))*dx/1000, 'k--')
            ax[i].plot(vh + (log[::-1] - min(log)) / 1000, np.arange(len(log))*dx/1000, 'k')
        else:
            score=(F_r2(m_list[i],m_true))
            title=labels[i]+') Model # '+str(i)+', $\mathit{R}$$^{2}$($\mathit{V}$$_{p}$ initial, $\mathit{V}$$_{p}$ true)='+str('{0:.3f}'.format(score))
        if i==n-1:
            ax[i].set_xlabel('X, km',fontsize=labels_Fontsize)
            if m_name=='other':
                title=labels[i]+') CNN-target model, $\mathit{R}$$^{2}$($\mathit{V}$$_{p}$ initial, $\mathit{V}$$_{p}$ true)='+str('{0:.3f}'.format(score))
        ax[i].set_title(title,fontsize=labels_Fontsize)
        ax[i].set_ylabel('Depth, km',fontsize=labels_Fontsize)
        ax[i].tick_params(labelsize=labelsize)
        ax[i].invert_yaxis()
    # cax = fig.add_axes([0.75, 0.98, 0.2, 0.02])
    cax = fig.add_axes([0.75, 0.98, 7*0.2/fig.bbox_inches.xmax,11*0.02/fig.bbox_inches.ymax])
    cbar=plt.colorbar(im,cax=cax,orientation='horizontal')
    cbar.ax.set_ylabel('$\mathit{V}$$_{p}$, km/sec',fontsize=labels_Fontsize,rotation=0,labelpad=100,loc='bottom')
    cbar.ax.tick_params(labelsize=labelsize)
    save_filename=os.path.join(save_path,name)
    print('save to ,',save_filename+'.png')
    fig.savefig(save_filename,dpi=300,bbox_inches='tight')  #,bbox_inches='tight'
    plt.close()
    #############################
    compare_from=water_start
    # compare_from=0
    plt.figure()
    for i in range(len(m_list)):
        vp_log=m_list[i][log_idx,:]
        # plt.plot(lvp_bks[compare_from:len(lvp)],label='lvp_bks'+str( PCC(vp_log[compare_from:],lvp_upscaled_50[compare_from:]) ))
        plt.plot(vp_log[compare_from:len(lvp)],label='vp'+str(i)+','+str( PCC(vp_log[compare_from:len(lvp)],lvp_bks_300[compare_from:len(lvp)]) ),linestyle='dashed')
    plt.plot(well_log_original[compare_from:len(lvp)],label='well orig',linestyle='solid')
    # plt.plot(lvp_bks_200[compare_from:len(lvp)],label='lvp_bks_200')
    plt.plot(lvp_bks_300[compare_from:len(lvp)],label='lvp_bks_300')
    # plt.plot(lvp_bks_500[compare_from:len(lvp)],label='lvp_bks_500')
    plt.plot(lvp_bks_700[compare_from:len(lvp)],label='lvp_bks_700')
    plt.plot(lvp_bks_900[compare_from:len(lvp)],label='lvp_bks_900')
    # plt.plot(lvp_upscaled_50[compare_from:len(lvp)],label='W smoothed 50m, ',linestyle='dashed')
    # plt.plot(lvp_upscaled_100[compare_from:len(lvp)],label='W smoothed 100m, ',linestyle='dashed')
    # plt.plot(lvp_upscaled_200[compare_from:len(lvp)],label='W smoothed 200m, ',linestyle='dashed')
    plt.legend()
    plt.show()
    save_name=os.path.join(save_path,'all_predicted_logs.png');print(save_name)
    plt.savefig(save_name)
    plt.close()
    #############################
    return None
def comparison_initial_models_with_fwi_misfits_9_letters(paths,fname,log_path='./',save_path='./'):
    compare_with='true_model'
    # compare_with='cnn_target_model'
    plot_model_misfit=0
    d=[]
    for path in paths:
        d.append(parse_denise_folder(path,denise_root='./') )
    data_availability=[]
    for data_ in d:
        if hasattr(data_,'fwi_model_names')==False:
            data_availability.append(False)
        else:
            data_availability.append(True)
    ################################
    label_position_inside_axes=[0.04,0.97]
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
    # labels=['a','a','b','b','c','c','a','b','c','j','k','l','m','n','o']
    labels=['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o']
    labels=['a','b','d','e','g','h','c','f','i']
    a=1500;b=4500

    if data_availability[2]==True and compare_with!='true_model':
        D=d[2]
        nx_orig=D.NX
        cnn_target_model=D.model_init.vp

    fig=plt.figure()
    j=0;D=d[j]
    if data_availability[j]==True:
        nx_orig=D.NX        #-320
        x = np.arange(nx_orig) * D.DH / 1000
        y = np.arange(D.NY) * D.DH / 1000
        extent = np.array([x.min(), x.max(), y.min(), y.max()])
        ind=D.fwi_model_names.index( fnmatch.filter(D.fwi_model_names,'*vp_stage*')[-1] )
        fwi_res=D.models[ind]
        i=0; row=0; col=0
        ax[i]=fig.add_subplot(gs[row,col])
        ax[i].axes.xaxis.set_visible(False)
        ax[i].text(label_position_inside_axes[0],label_position_inside_axes[1],labels[i], transform=ax[i].transAxes,fontsize=Fontsize,color = "white",weight='bold',va='top')
        if compare_with=='true_model':
            score=numstr_3(F_r2(D.model_init.vp[:,0:nx_orig],D.model.vp[:,0:nx_orig]))
            ax[i].set_title('$\mathit{R}^{2}$(initial,true)='+score,fontsize=textFontsize)
        else:
            score2=numstr_3(F_r2(D.model_init.vp[:,0:nx_orig],cnn_target_model[:,0:nx_orig]))
            ax[i].set_title('$\mathit{R}^{2}$(initial,CNN-target initial)='+score2,fontsize=textFontsize)
        ax[i].imshow(D.model_init.vp[:,0:nx_orig],extent=extent,vmin=a,vmax=b,aspect='auto');ax[i].invert_yaxis()
        ax[i].set_ylabel('Depth, km',fontsize=labels_Fontsize)
        ax[i].tick_params(labelsize=labelsize)
        i=1; row=0; col=1
        ax[i]=fig.add_subplot(gs[row,col]); ax[i].axes.xaxis.set_visible(False);  ax[i].axes.yaxis.set_visible(False)
        ax[i].text(label_position_inside_axes[0],label_position_inside_axes[1],labels[i], transform=ax[i].transAxes,fontsize=Fontsize,color = "white",weight='bold',va='top')
        score=numstr_3(F_r2(fwi_res[:,0:nx_orig],D.model.vp[:,0:nx_orig]))
        ax[i].set_title('$\mathit{R}^{2}$(FWI result,true)='+score,fontsize=textFontsize)
        ax[i].tick_params(labelsize=labelsize)
        last_image=ax[i].imshow(fwi_res[:,0:nx_orig],extent=extent,vmin=a,vmax=b,aspect='auto');ax[i].invert_yaxis()
    ##
    j=1;D=d[j]
    if data_availability[j]==True:
        nx_orig=D.NX
        x = np.arange(nx_orig) * D.DH / 1000
        y = np.arange(D.NY) * D.DH / 1000
        extent = np.array([x.min(), x.max(), y.min(), y.max()])
        ind=D.fwi_model_names.index( fnmatch.filter(D.fwi_model_names,'*vp_stage*')[-1] )
        fwi_res=D.models[ind]
        i=2; row=1; col=0
        ax[i]=fig.add_subplot(gs[row,col]); 
        ax[i].axes.xaxis.set_visible(False);  
        # ax[i].axes.yaxis.set_visible(False)
        ax[i].text(label_position_inside_axes[0],label_position_inside_axes[1],labels[i],transform=ax[i].transAxes,fontsize=Fontsize,color = "white",weight='bold',va='top')
        if compare_with=='true_model':
            score=numstr_3(F_r2(D.model_init.vp[:,0:nx_orig],D.model.vp[:,0:nx_orig]))
            ax[i].set_title('$\mathit{R}^{2}$(initial,true)='+score,fontsize=textFontsize)
        else:
            score2=numstr_3(F_r2(D.model_init.vp[:,0:nx_orig],cnn_target_model[:,0:nx_orig]))
            ax[i].set_title('$\mathit{R}^{2}$(initial,CNN-target initial)='+score2,fontsize=textFontsize)
        ax[i].imshow(D.model_init.vp[:,0:nx_orig],extent=extent,vmin=a,vmax=b,aspect='auto');ax[i].invert_yaxis()
        ax[i].tick_params(labelsize=labelsize)
        ax[i].set_ylabel('Depth, km',fontsize=labels_Fontsize)
        i=3; row=1; col=1
        ax[i]=fig.add_subplot(gs[row,col]); ax[i].axes.xaxis.set_visible(False);  ax[i].axes.yaxis.set_visible(False)
        ax[i].text(label_position_inside_axes[0],label_position_inside_axes[1],labels[i], transform=ax[i].transAxes,fontsize=Fontsize,color = "white",weight='bold',va='top')
        score=numstr_3(F_r2(fwi_res[:,0:nx_orig],D.model.vp[:,0:nx_orig]))
        ax[i].set_title('$\mathit{R}^{2}$(FWI result,true)='+score,fontsize=textFontsize)
        ax[i].tick_params(labelsize=labelsize)
        last_image=ax[i].imshow(fwi_res[:,0:nx_orig],extent=extent,vmin=a,vmax=b,aspect='auto');ax[i].invert_yaxis()
    ##
    j=2;D=d[j]
    if data_availability[j]==True:
        nx_orig=D.NX
        x = np.arange(nx_orig) * D.DH / 1000
        y = np.arange(D.NY) * D.DH / 1000
        extent = np.array([x.min(), x.max(), y.min(), y.max()])

        ind=D.fwi_model_names.index( fnmatch.filter(D.fwi_model_names,'*vp_stage*')[-1] )
        fwi_res=D.models[ind]
        i=4; row=2; col=0
        ax[i]=fig.add_subplot(gs[row,col]); 
        ax[i].text(label_position_inside_axes[0],label_position_inside_axes[1],labels[i], transform=ax[i].transAxes,fontsize=Fontsize,color = "white",weight='bold',va='top')
        if compare_with=='true_model':
            score=numstr_3(F_r2(D.model_init.vp[:,0:nx_orig],D.model.vp[:,0:nx_orig]))
            ax[i].set_title('$\mathit{R}^{2}$(initial,true)='+score,fontsize=textFontsize)
        else:
            score2=numstr_3(F_r2(D.model_init.vp[:,0:nx_orig],cnn_target_model[:,0:nx_orig]))
            ax[i].set_title('$\mathit{R}^{2}$(initial,CNN-target initial)='+score2,fontsize=textFontsize)
        ax[i].imshow(D.model_init.vp[:,0:nx_orig],extent=extent,vmin=a,vmax=b,aspect='auto');ax[i].invert_yaxis()
        ax[i].set_xlabel('X, km',fontsize=labels_Fontsize)
        ax[i].set_ylabel('Depth, km',fontsize=labels_Fontsize)
        ax[i].tick_params(labelsize=labelsize)

        i=5; row=2; col=1
        ax[i]=fig.add_subplot(gs[row,col]); 
        ax[i].axes.yaxis.set_visible(False)
        ax[i].text(label_position_inside_axes[0],label_position_inside_axes[1],labels[i], transform=ax[i].transAxes,fontsize=Fontsize,color = "white",weight='bold',va='top')
        ax[i].set_xlabel('X, km',fontsize=labels_Fontsize)
        score=numstr_3(F_r2(fwi_res[:,0:nx_orig],D.model.vp[:,0:nx_orig]))
        ax[i].set_title('$\mathit{R}^{2}$(FWI result,true)='+score,fontsize=textFontsize)
        ax[i].tick_params(labelsize=labelsize)
        last_image=ax[i].imshow(fwi_res[:,0:nx_orig],extent=extent,vmin=a,vmax=b,aspect='auto');ax[i].invert_yaxis()
    ##
    cbar_ax = fig.add_axes([0.93, 0.12, 0.02, 0.75])
    cbar=fig.colorbar(last_image,cax=cbar_ax)
    cbar.ax.set_title('$\mathit{V}_{p}$ (m/sec)',fontsize=18,pad=13.3)
    for t in cbar.ax.get_yticklabels():
        t.set_fontsize(16)
    ##############################  misfits
    niter_list=[];
    misfit_list=[]
    model_misfit_list=[]
    j=0;D=d[j];path=paths[j]
    if data_availability[j]==True:
        i=6; row=0; col=2
        fwi_vp_names=fnmatch.filter(D.fwi_model_names,'*vp_stage*')
        model_misfit=np.empty((len(fwi_vp_names)))
        iteration2=np.arange(len(fwi_vp_names))
        for iter,fwi_vp_name in enumerate(fwi_vp_names):
            fwi_res=D.models[D.fwi_model_names.index(fwi_vp_name)]
            # score=numstr_3(F_r2(D.model_init.vp[:,0:nx_orig],D.model.vp[:,0:nx_orig]))
            score=      F_r2(fwi_res,D.model.vp)
            model_misfit[iter]=score
        m,m_data=plot_fwi_misfit2(os.path.join(path,'fld','seis_fwi_log.dat'),D)
        nstage=m_data['nstage']
        niter_stage=m_data['niter_stage'];
        nstage_trans=m_data['nstage_trans']
        iteration=m_data['iteration']
        niter_list.append(len(iteration))
        misfit_list.append(m)
        ax[i]=fig.add_subplot(gs[row,col])
        if plot_model_misfit==1:
            ax_extra=ax[i].twinx()
            ax_extra.plot(iteration2,model_misfit,'g-', linewidth=3, label='R2 model misfit') #Evolution of the misfit function
            ax_extra.axes.yaxis.set_visible(False)
        ax[i].plot(iteration,m,'b-', linewidth=3, label='Evolution of the misfit function')
        ax[i].text(label_position_inside_axes[0],label_position_inside_axes[1],labels[i], transform=ax[i].transAxes,fontsize=Fontsize,color = "black",weight='bold',va='top')
        # ax[i].text(0.4,0.7, labels[i], fontsize=Fontsize,color = "white",weight="bold")
        # ax[i].text(iteration[np.min([7,len(iteration)])],(np.min(m)+6/10*(np.max(m)-np.min(m))), labels[i], fontsize=Fontsize,color = "black",weight="bold")
        ax[i].tick_params(labelsize=labelsize)
        for ii in range(1, nstage):
            ax[i].plot([nstage_trans[ii-1]+ii,nstage_trans[ii-1]+ii], [np.min(m),np.max(m)],'k--', linewidth=3)
        ax[i].axes.yaxis.set_visible(False)
    j=1;D=d[j];path=paths[j]
    if data_availability[j]==True:
        i=7; row=1; col=2
        fwi_vp_names=fnmatch.filter(D.fwi_model_names,'*vp_stage*')
        model_misfit=np.empty((len(fwi_vp_names)))
        iteration2=np.arange(len(fwi_vp_names))
        for iter,fwi_vp_name in enumerate(fwi_vp_names):
            fwi_res=D.models[D.fwi_model_names.index(fwi_vp_name)]
            # score=numstr_3(F_r2(D.model_init.vp[:,0:nx_orig],D.model.vp[:,0:nx_orig]))
            score=      F_r2(fwi_res,D.model.vp)
            model_misfit[iter]=score
        m,m_data=plot_fwi_misfit2(os.path.join(path,'fld','seis_fwi_log.dat'),D)
        nstage=m_data['nstage']
        niter_stage=m_data['niter_stage'];
        nstage_trans=m_data['nstage_trans']
        iteration=m_data['iteration']
        niter_list.append(len(iteration))
        misfit_list.append(m)
        ax[i]=fig.add_subplot(gs[row,col])
        if plot_model_misfit==1:
            ax_extra=ax[i].twinx()
            ax_extra.plot(iteration2,model_misfit,'g-', linewidth=3, label='R2 model misfit') #Evolution of the misfit function
            ax_extra.axes.yaxis.set_visible(False)
        ax[i].plot(iteration,m,'b-', linewidth=3, label='Evolution of the misfit function')
        ax[i].text(label_position_inside_axes[0],label_position_inside_axes[1],labels[i], transform=ax[i].transAxes,fontsize=Fontsize,color = "black",weight='bold',va='top')
        ax[i].tick_params(labelsize=labelsize)
        for ii in range(1, nstage):
            ax[i].plot([nstage_trans[ii-1]+ii,nstage_trans[ii-1]+ii], [np.min(m),np.max(m)],'k--', linewidth=3)
        ax[i].axes.yaxis.set_visible(False)
    j=2;D=d[j];path=paths[j]
    if data_availability[j]==True:
        i=8; row=2; col=2
        fwi_vp_names=fnmatch.filter(D.fwi_model_names,'*vp_stage*')
        model_misfit=np.empty((len(fwi_vp_names)))
        iteration2=np.arange(len(fwi_vp_names))
        for iter,fwi_vp_name in enumerate(fwi_vp_names):
            fwi_res=D.models[D.fwi_model_names.index(fwi_vp_name)]
            # score=numstr_3(F_r2(D.model_init.vp[:,0:nx_orig],D.model.vp[:,0:nx_orig]))
            score=      F_r2(fwi_res,D.model.vp)
            model_misfit[iter]=score
        m,m_data=plot_fwi_misfit2(os.path.join(path,'fld','seis_fwi_log.dat'),D)
        nstage=m_data['nstage']
        niter_stage=m_data['niter_stage'];
        nstage_trans=m_data['nstage_trans']
        iteration=m_data['iteration']
        niter_list.append(len(iteration))
        misfit_list.append(m)
        ax[i]=fig.add_subplot(gs[row,col])
        if plot_model_misfit==1:
            ax_extra=ax[i].twinx()
            ax_extra.plot(iteration2,model_misfit,'g-', linewidth=3, label='R2 model misfit') #Evolution of the misfit function
            ax_extra.axes.yaxis.set_visible(False)
        ax[i].plot(iteration,m,'b-', linewidth=3, label='Evolution of the misfit function')
        ax[i].text(label_position_inside_axes[0],label_position_inside_axes[1],labels[i], transform=ax[i].transAxes,fontsize=Fontsize,color = "black",weight='bold',va='top')
        ax[i].tick_params(labelsize=labelsize)
        for ii in range(1, nstage):
            ax[i].plot([nstage_trans[ii-1]+ii,nstage_trans[ii-1]+ii], [np.min(m),np.max(m)],'k--', linewidth=3)
        ax[i].axes.yaxis.set_visible(False)
    ######  x limits
    niter_list=np.asarray(niter_list)
    misfit_list=np.asarray(misfit_list)
    mf_min=0;mf_max=0;
    for misfit_curve in misfit_list:
        mf_min=np.min([np.min(misfit_curve),mf_min])
        mf_max=np.max([np.max(misfit_curve),mf_max])
    y_pos=mf_min+(mf_max-mf_min)/5*4
    if data_availability[0]==True:
        ax[6].set_xlim(left=1,right=np.max(niter_list))
        ax[6].set_ylim(bottom=mf_min,top=mf_max)
        # ax[6].text(10,y_pos, labels[6], fontsize=Fontsize,color = "black",weight="bold")
    if data_availability[1]==True:
        ax[7].set_xlim(left=1,right=np.max(niter_list))
        ax[7].set_ylim(bottom=mf_min,top=mf_max)
        # ax[7].text(10,y_pos, labels[7], fontsize=Fontsize,color = "black",weight="bold")
    if data_availability[2]==True:
        ax[8].set_xlim(left=1,right=np.max(niter_list))
        ax[8].set_ylim(bottom=mf_min,top=mf_max)
        # ax[8].text(10,y_pos, labels[8], fontsize=Fontsize,color = "black",weight="bold")
    ######  saving
    save_file_path1=os.path.join(save_path,fname)
    save_file_path2=os.path.join(log_path,fname)
    print('Saving ML_result to '+save_file_path1+'  '+save_file_path2)
    plt.savefig(save_file_path1,dpi=300,bbox_inches='tight')
    plt.savefig(save_file_path2,dpi=300,bbox_inches='tight')
    # plt.show()
    plt.close()
    return None

# if name=='fwi_56_strategy_l5_del_low_gen__5hz_3_model__cgg_lin_vp_long':
#     pic_name_in_draft='cgg_1d_highfreq'
# elif name=='fwi_56_strategy_l5_del_low_gen__5hz_3':
#     pic_name_in_draft='cgg_1d_highfreq'
# elif name=='fwi_56_strategy_l5multi_cnn_13_special_weight_236_2_model__cgg_lin_vp_long_300_f_z_1_model__cgg_lin_vp_long':
#     pic_name_in_draft='cgg_cnnpred_highfreq'