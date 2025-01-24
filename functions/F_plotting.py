from functions.F_modules import *
def numstr(x):
    string = str('{0:.2f}'.format(x))
    return string
def F_nrms(mat,mat_true):
    nrms = np.linalg.norm((mat-mat_true),ord=2)/np.linalg.norm(mat_true,ord=2)
    return nrms
def F_r2(mat,mat_true):
    r2=1- (np.std(mat_true.flatten()-mat.flatten()) / np.std(mat_true.flatten())  )
    v1=mat.flatten()
    v2=mat_true.flatten()
    r2_2=r2_score(v1,v2)
    return r2_2
def tight_figure(fig,**kwargs):
    canvas = fig.canvas._get_output_canvas("png")
    print_method = getattr(canvas, 'print_png')
    print_method(io.BytesIO(), dpi=fig.dpi,
                 facecolor=fig.get_facecolor(), dryrun=True)
    renderer = fig._cachedRenderer
    bbox_inches = fig.get_tightbbox(renderer)
    bbox_artists = fig.get_default_bbox_extra_artists()
    bbox_filtered = []
    for a in bbox_artists:
        bbox = a.get_window_extent(renderer)
        if a.get_clip_on():
            clip_box = a.get_clip_box()
            if clip_box is not None:
                bbox = Bbox.intersection(bbox, clip_box)
            clip_path = a.get_clip_path()
            if clip_path is not None and bbox is not None:
                clip_path = \
                    clip_path.get_fully_transformed_path()
                bbox = Bbox.intersection(
                    bbox, clip_path.get_extents())
        if bbox is not None and (
                bbox.width != 0 or bbox.height != 0):
            bbox_filtered.append(bbox)

    if bbox_filtered:
        _bbox = Bbox.union(bbox_filtered)
        trans = Affine2D().scale(1.0 / fig.dpi)
        bbox_extra = TransformedBbox(_bbox, trans)
        bbox_inches = Bbox.union([bbox_inches, bbox_extra])

    pad = kwargs.pop("pad_inches", None)
    if pad is None:
        pad = plt.rcParams['savefig.pad_inches']

    bbox_inches = bbox_inches.padded(pad)

    tight_bbox.adjust_bbox(fig, bbox_inches, canvas.fixed_dpi)

    w = bbox_inches.x1 - bbox_inches.x0
    h = bbox_inches.y1 - bbox_inches.y0
    fig.set_size_inches(w,h)

def Plot_image(Data, Title='Title', c_lim='',x='',x_label='',y='',y_label='',
               dx='',dy='',Save_flag=0,Save_pictures_path='./Pictures',
               Reverse_axis=1,Curve='',Show_flag=1,Aspect='equal',write_fig_title=1,crd=''):
    os.makedirs(Save_pictures_path,exist_ok=True)
    if c_lim == '':  c_lim =[np.min(Data), np.max(Data)]
    if x == '':  x=(np.arange(np.shape(Data)[1]))
    if y == '':  y=(np.arange(np.shape(Data)[0]))
    if dx != '':  x=(np.arange(np.shape(Data)[1]))*dx
    if dy != '':  y=(np.arange(np.shape(Data)[0]))*dy
    extent = [x.min(), x.max(), y.min(), y.max()]
    #if Save_flag==1:#    plt.ion()
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
    if crd!='':
        plt.scatter(crd['x_rec'],crd['z_rec'],marker='.',c='black',s=5)
        plt.scatter(crd['x_src'],crd['z_src'],marker='^',c='black',s=49)
    ax = plt.gca()
    divider1 = make_axes_locatable((ax))
    cax1 = divider1.append_axes("right", size="2%", pad=0.05)
    cbar=plt.colorbar(cax=cax1)
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
        name=Save_pictures_path + '/' + Title + '.png'
        print('Saving to '+name)
        plt.savefig(name,bbox_inches='tight')
    if Show_flag==0:
        plt.show(block=False)
    else:
        if Show_flag == 2:
            a=1
        else:
            plt.show()
    plt.close()
    return None
def Plot_image_(Data, Title='Title', c_lim='',x='',x_label='',y='',y_label='',
               dx='',dy='',Save_flag=1,Save_pictures_path='./Pictures',
               Reverse_axis=1,Curve='',Show_flag=1,Aspect='equal'):
    # aspect - 'auto'
    if c_lim == '':  c_lim =[np.min(Data), np.max(Data)]
    if x == '':  x=(np.arange(np.shape(Data)[1]))
    if y == '':  y=(np.arange(np.shape(Data)[0]))
    if dx != '':  x=(np.arange(np.shape(Data)[1]))*dx
    if dy != '':  y=(np.arange(np.shape(Data)[0]))*dy
    extent = [x.min(), x.max(), y.min(), y.max()]
    fig=plt.figure()
    fig.dpi=330
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 18
    fig_size[1] = 3
    plt.rcParams["figure.figsize"] = fig_size
    plt.rcParams.update({'font.size': 20})
    plt.set_cmap('RdBu_r')
    # plt.axis(extent, Aspect)
    # plt.axis(extent, 'auto')
    plt.title('')
    if Reverse_axis == 1:
        plt.imshow(np.flipud(Data), extent=extent, interpolation='nearest',aspect=Aspect)
        plt.gca().invert_yaxis()
    else:
        plt.imshow((Data), extent=extent, interpolation='nearest',aspect=Aspect)
    if Curve != '':
        plt.plot(x, Curve, color='white', linewidth=1.2, linestyle='--')
    ax = plt.gca()
    plt.axis('equal')
    # plt.axis('tight')
    # tight_figure(fig)
    divider1 = make_axes_locatable((ax))
    cax1 = divider1.append_axes("right", size="2%", pad=0.05)
    cbar=plt.colorbar(cax=cax1)
    plt.clim(c_lim)
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    if Save_flag == 1:
        if not os.path.exists(Save_pictures_path):
            os.mkdir(Save_pictures_path)
        name=Save_pictures_path + '/' + Title + '.png'
        plt.savefig(name,bbox_inches='tight')
    if Show_flag==0:
        plt.show(block=False)
    else:
        if Show_flag == 2:
            a=1
        else:
            plt.show()
    plt.close()
    return None

def Plot_image2(Data,Curve='', Title='Title', c_lim='',x='',x_label='',y='',y_label='',
                Save_flag=0,Save_pictures_path='./Pictures',Reverse_axis=1):
    if c_lim == '':  c_lim =[np.min(Data), np.max(Data)]
    if x == '':  x=(np.arange(np.shape(Data)[1]))
    if y == '':  y=(np.arange(np.shape(Data)[0]))
    extent = [x.min(), x.max(), y.min(), y.max()]
    # x = x.reshape((1,) + x.shape)
    ##############################################
    plt.figure()
    plt.set_cmap('RdBu_r')
    plt.rcParams.update({'font.size': 14})
    # plt.subplot.left: 0.1
    # plt.gcf().subplots_adjust(left=0.1,right=0.1,top=0)
    # plt.gcf().subplots_adjust(left=0.1, right=0.1, top=0.1)
    plt.title(Title)
    if Reverse_axis == 1:
        plt.imshow(np.flipud(Data), extent=extent, interpolation='nearest', aspect='auto')
        plt.gca().invert_yaxis()
    else:
        plt.imshow(np.flipud(Data), extent=extent, interpolation='nearest', aspect='auto')
        # plt.imshow((Data), extent=extent, interpolation='none', aspect='auto')
    cbar=plt.colorbar()
    x0 = -33;   y0 = 1.054
    # cbar.set_label('Amplitude', labelpad=x0, y=y0,rotation=0)
    plt.clim(c_lim)
    if Curve != '':
        if len(np.shape(Curve)) == 2:
            Curve=Curve[0,:]
        plt.plot(x, Curve, color='white', linewidth=1.2, linestyle='--')
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    # plt.tight_layout()
    plt.autoscale()
    if Save_flag == 1:
        name=Save_pictures_path + '/' + Title + '.png'
        print(name)
        plt.savefig(name, dpi=800, bbox_inches="tight")
    plt.show(block=False)
    plt.close()
    ##############################################
    return cbar
def Plot_image3(Data, Title='Title', c_lim='',x='',x_label='',y='',y_label='',
               dx='',dy='',Save_flag=0,Save_pictures_path='./Pictures',
               Reverse_axis=1,Curve='',Show_flag=0,Aspect='equal'):
    if c_lim == '':  c_lim =[np.min(Data), np.max(Data)]
    if x == '':  x=(np.arange(np.shape(Data)[1]))
    if y == '':  y=(np.arange(np.shape(Data)[0]))
    if dx != '':  x=(np.arange(np.shape(Data)[1]))*dx
    if dy != '':  y=(np.arange(np.shape(Data)[0]))*dy
    extent = [x.min(), x.max(), y.min(), y.max()]
    plt.figure()
    plt.set_cmap('RdBu_r')
    # plt.axis(extent, Aspect)
    # plt.axis(extent, 'auto')
    plt.title(Title)
    if Reverse_axis == 1:
        plt.imshow(np.flipud(Data.T), extent=extent, interpolation='nearest',aspect=Aspect)
        # plt.imshow(np.flipud(Data), interpolation='nearest', aspect='auto')
        plt.gca().invert_yaxis()
    else:
        plt.imshow((Data.T), extent=extent, interpolation='nearest',aspect=Aspect)
        # plt.imshow((Data), interpolation='nearest', aspect='auto')
    ax = plt.gca()
    divider1 = make_axes_locatable((ax))
    cax1 = divider1.append_axes("right", size="2%", pad=0.05)
    cbar=plt.colorbar(cax=cax1)
    plt.clim(c_lim)
    if Curve != '':
        if len(np.shape(Curve)) == 2:
            Curve=Curve[0,:]
        plt.plot(x, Curve, color='white', linewidth=1.2, linestyle='--')
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    # plt.axis('equal')
    # plt.axis('tight')
    if Save_flag == 1:
        name=Save_pictures_path + '/' + Title + '.png'
        print(name)
        plt.savefig(name)
    if Show_flag==0:
        plt.show(block=False)
    else:
        plt.show()
    plt.close()
    return None
def Plot_image4(Data, Title='Title', c_lim='',x='',x_label='',y='',y_label='',
               dx='',dy='',Save_flag=0,Save_pictures_path='./Pictures',
               Reverse_axis=1,Curve='',Show_flag=1,Aspect='equal',fname=''):
    # aspect - 'auto'
    if fname=='':  fname=Title
    if c_lim == '':  c_lim =[np.min(Data), np.max(Data)]
    if x == '':  x=(np.arange(np.shape(Data)[1]))
    if y == '':  y=(np.arange(np.shape(Data)[0]))
    if dx != '':  x=(np.arange(np.shape(Data)[1]))*dx
    if dy != '':  y=(np.arange(np.shape(Data)[0]))*dy
    extent = [x.min(), x.max(), y.min(), y.max()]
    fig=plt.figure()
    fig.dpi=330
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 10.4
    fig_size[1] = 3.0
    plt.rcParams["figure.figsize"] = fig_size
    plt.set_cmap('RdBu_r')
    plt.rcParams.update({'font.size': 20})
    plt.title(Title)
    # plt.axis('tight')
    if Reverse_axis == 1:
        aa=plt.imshow(np.flipud(Data), extent=extent, interpolation='nearest',aspect=Aspect)
        # aa=plt.imshow(Data,extent=extent, interpolation='nearest',aspect=Aspect)
        plt.gca().invert_yaxis()
    else:
        aa=plt.imshow((Data), extent=extent, interpolation='nearest',aspect=Aspect)
        # plt.gca().invert_yaxis()
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    # ax = plt.gca()
    # divider1 = make_axes_locatable((ax))
    # cax1 = divider1.append_axes("right", size="2%", pad=0.05)       #0.05
    # cbar=plt.colorbar(cax=cax1)
    plt.colorbar()
    plt.clim(c_lim)
    # plt.axis(extent,Aspect)
    # plt.axis('equal')
    plt.axis('tight')
    # tight_figure(fig)
    if Save_flag == 1:
        if not os.path.exists(Save_pictures_path):
            os.mkdir(Save_pictures_path)
        name=Save_pictures_path + '/' + fname + '.png'
        print(name)
        plt.savefig(name)
    if Show_flag==0:
        # plt.show(block=False)
        do_nothing=1
    else:
        plt.show()
    plt.close()
    return None

def PLOT_data_patches(x,t,y,Name='',dx=1,dy=1,Title='',Save_flag=0):
    Title1='Input'
    Title2='Target'
    Title3='Output'
    ###############
    num=3
    ind=list(range(num, num+6))
    plt.figure(figsize=(8, 4))
    gs1 = gridspec.GridSpec(3, 6)
    gs1.update(wspace=0.33, hspace=0.25)  # set the spacing between axes.
    plt.subplots_adjust(left=0.07, bottom=0.1, right=0.9, top=0.9)
    cax = plt.axes([0.91, 0.1, 0.035, 0.8])
    for i in range(0,6):
        ax=plt.subplot(gs1[0,i])
        plt.imshow(x[ind[i],:,:,0].T, cmap=plt.get_cmap('RdBu_r'))
        if i==0:
            ax.set_ylabel(Title1)
        ax=plt.subplot(gs1[1, i])
        plt.imshow(t[ind[i], :, :, 0].T, cmap=plt.get_cmap('RdBu_r'))
        if i==0:
            ax.set_ylabel(Title2)
        ax=plt.subplot(gs1[2, i])
        plt.imshow(y[ind[i], :, :, 0].T, cmap=plt.get_cmap('RdBu_r'))
        if i == 0:
            ax.set_ylabel(Title3)
    plt.colorbar(cax=cax)
    plt.show()
    plt.close()
    return None
def PLOT_Spectrum_Analysis_Pad1(M1,M2,M3,M4,M5,M6,M7,M8,M9,Boundaries=0,Name='',dx=1,dy=1,df=1,
                                Plot_vertical_lines=1, Title='',Save_flag=0,fSpacex='',fSpacez=''):
    z_label_name = 'z, m'
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 20.4
    fig_size[1] = 9.0
    plt.rcParams["figure.figsize"] = fig_size
    plt.rcParams["font.size"] = 13
    fig, ax = plt.subplots(3,3)
    divider1 = make_axes_locatable((ax[0,0]))
    divider2 = make_axes_locatable((ax[0,1]))
    divider3 = make_axes_locatable((ax[0,2]))
    divider4 = make_axes_locatable((ax[1,0]))
    divider5 = make_axes_locatable((ax[1,1]))
    divider6 = make_axes_locatable((ax[1,2]))
    divider7 = make_axes_locatable((ax[2, 0]))
    divider8 = make_axes_locatable((ax[2, 1]))
    divider9 = make_axes_locatable((ax[2, 2]))
    cax1 = divider1.append_axes("right", size="2%", pad=0.05)
    cax2 = divider2.append_axes("right", size="2%", pad=0.05)
    cax3 = divider3.append_axes("right", size="2%", pad=0.05)
    cax4 = divider4.append_axes("right", size="2%", pad=0.05)
    cax5 = divider5.append_axes("right", size="2%", pad=0.05)
    cax6 = divider6.append_axes("right", size="2%", pad=0.05)
    cax7 = divider7.append_axes("right", size="2%", pad=0.05)
    cax8 = divider8.append_axes("right", size="2%", pad=0.05)
    cax9 = divider9.append_axes("right", size="2%", pad=0.05)
    x = np.arange(np.shape(M1)[1]) * dx
    y = np.arange(np.shape(M1)[0]) * dy
    f=np.arange(np.shape(M4)[0]) * df
    F_LIM = np.array([f.min(), f.max() / 2])
    extent1 = np.array([x.min(), x.max(), y.min(), y.max()])
    extent2 = np.array([y.min(), y.max(), F_LIM[0], F_LIM[1]])
    extent2 = np.array([y.min(), y.max(), f.min(), f.max()])
    extent2 = np.array([fSpacez.min(), fSpacez.max(), fSpacex.min(), fSpacex.max()])

    MIN = 0;    MAX = 0.02
    ########################    First row
    im1 = ax[0,0].imshow(np.flipud(M1), extent=extent1, aspect='auto')
    im2 = ax[0,1].imshow(np.flipud(M2), extent=extent1, aspect='auto')
    im3 = ax[0,2].imshow(np.flipud(M3), extent=extent1, aspect='auto')
    ax[0, 0].set_ylabel(z_label_name)
    ax[0, 0].set_xlabel('x (km)')
    ax[0, 1].set_ylabel(z_label_name)
    ax[0, 1].set_xlabel('x (km)')
    ax[0, 2].set_ylabel(z_label_name)
    ax[0, 2].set_xlabel('x (km)')
    ax[0, 0].invert_yaxis()
    ax[0, 1].invert_yaxis()
    ax[0, 2].invert_yaxis()
    ax[0, 0].set_title('dm true')
    ax[0, 1].set_title(Title + ',      dm 1 Hz')
    ax[0, 2].set_title('dm 3 Hz')
    ########################    Second row
    # im4=  ax[1,0].imshow(np.flipud(M4), extent=extent2, vmin=MIN, vmax=MAX, aspect='auto')
    # # im5 = ax[1,1].imshow(np.flipud(M5), extent=extent2, vmin=MIN, vmax=MAX, aspect='auto')
    # # im6 = ax[1,2].imshow(np.flipud(M6), extent=extent2, vmin=MIN, vmax=MAX, aspect='auto')
    # im5 = ax[1, 1].imshow(np.flipud(M5), extent=extent2, aspect='auto')
    # im6 = ax[1, 2].imshow(np.flipud(M6), extent=extent2, aspect='auto')
    # ax[1, 0].set_ylim(F_LIM)
    # ax[1, 1].set_ylim(F_LIM)
    # ax[1, 2].set_ylim(F_LIM)
    Mat = np.concatenate((M4, M5, M6), axis=0); MIN = np.min(Mat);MAX = 0.6* np.max(Mat)
    im4 = ax[1, 0].imshow(np.flipud(M4), extent=extent2, vmin=MIN, vmax=MAX, aspect='auto')
    im5 = ax[1, 1].imshow(np.flipud(M5), extent=extent2, vmin=MIN, vmax=MAX, aspect='auto')
    im6 = ax[1, 2].imshow(np.flipud(M6), extent=extent2, vmin=MIN, vmax=MAX, aspect='auto')
    ax[1, 0].set_ylabel('$K_{z}$ - vertical wavenumber')
    ax[1, 0].set_xlabel('$K_{x}$')
    ax[1, 1].set_xlabel('$K_{x}$')
    ax[1, 2].set_xlabel('$K_{x}$')
    ax[1, 0].set_title('|FFT2D (dm_true)|')
    ax[1, 1].set_title('|FFT2D (dm 1 Hz)|')
    ax[1, 2].set_title('|FFT2D (dm 3 Hz)|')
    ax[1, 0].invert_yaxis()
    ax[1, 1].invert_yaxis()
    ax[1, 2].invert_yaxis()
    ########################    Third row
    # im7 = ax[2, 0].imshow(np.flipud(M7), extent=extent2, aspect='auto')
    # im8 = ax[2, 1].imshow(np.flipud(M8), extent=extent2, aspect='auto')
    # im9 = ax[2, 2].imshow(np.flipud(M9), extent=extent2, aspect='auto')
    # ax[2, 0].set_ylim(F_LIM)
    # ax[2, 1].set_ylim(F_LIM)
    # ax[2, 2].set_ylim(F_LIM)
    im7 = ax[2, 0].imshow(np.flipud(M7), extent=extent2, vmin=MIN, vmax=MAX, aspect='auto')
    im8 = ax[2, 1].imshow(np.flipud(M8), extent=extent2, vmin=MIN, vmax=MAX, aspect='auto')
    im9 = ax[2, 2].imshow(np.flipud(M9), extent=extent2, vmin=MIN, vmax=MAX, aspect='auto')
    ax[2, 0].set_ylabel('$K_{z}$ - vertical wavenumber')
    ax[2, 0].set_xlabel('$K_{x}$')
    ax[2, 1].set_xlabel('$K_{x}$')
    ax[2, 2].set_xlabel('$K_{x}$')
    ax[2, 0].set_title('|FFT2D(dm 1 Hz)-FFT2D(dm 3 Hz)|')
    ax[2, 1].set_title('|FFT2D(dm 1 Hz)-FFT2D(dm_true)|')
    ax[2, 2].set_title('|FFT2D(dm 3 Hz)-FFT2D(dm_true)|')
    ax[2, 0].invert_yaxis()
    ax[2, 1].invert_yaxis()
    ax[2, 2].invert_yaxis()
    ########################
    cbar1 = plt.colorbar(im1, cax=cax1)
    cbar2 = plt.colorbar(im2, cax=cax2)
    cbar3 = plt.colorbar(im3, cax=cax3)
    cbar4 = plt.colorbar(im4, cax=cax4)
    cbar9 = plt.colorbar(im9, cax=cax9)
    cbar5 = plt.colorbar(im5, cax=cax5)
    cbar6 = plt.colorbar(im6, cax=cax6)
    cbar7 = plt.colorbar(im7, cax=cax7)
    cbar8 = plt.colorbar(im8, cax=cax8)
    plt.subplots_adjust(left=0.07, right=0.93, bottom=0.08,
        top=0.95, hspace=.49,wspace=.29)  # hspace,wspace- adjust space between subplots
    if Plot_vertical_lines==1:
        Boundaries2 = Boundaries * dx
        ax[0,0].axvline(Boundaries2,color='k', linestyle='-' ,linewidth=2.5)
        ax[0,1].axvline(Boundaries2, color='k', linestyle='-',linewidth=2.5)
        ax[0,2].axvline(Boundaries2, color='k', linestyle='-',linewidth=2.5)
    if Save_flag == 1:
        plt.savefig(Name, dpi=300)
        print('Saving result to ' + Name)
    plt.show(block=False)
    plt.close()
    # %
    return None
def PLOT_Spectrum_Analysis_Pad2(M1,M2,M3,M4,M5,M6,M7,M8,M9,M10,M11,M12,M13,M14,M15,Boundaries=0,Name='',dx=1,dy=1,df=1,
                                Plot_vertical_lines=1, Title='',Save_flag=0,fSpacex='',fSpacez=''):
    z_label_name = 'z, m'
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 25.4
    fig_size[1] = 15.0
    plt.rcParams["figure.figsize"] = fig_size
    plt.rcParams["font.size"] = 13
    fig, ax = plt.subplots(5,3)
    divider1 = make_axes_locatable((ax[0,0]))
    divider2 = make_axes_locatable((ax[0,1]))
    divider3 = make_axes_locatable((ax[0,2]))
    divider4 = make_axes_locatable((ax[1,0]))
    divider5 = make_axes_locatable((ax[1,1]))
    divider6 = make_axes_locatable((ax[1,2]))
    divider7 = make_axes_locatable((ax[2, 0]))
    divider8 = make_axes_locatable((ax[2, 1]))
    divider9 = make_axes_locatable((ax[2, 2]))
    divider10 = make_axes_locatable((ax[3, 0]))
    divider11 = make_axes_locatable((ax[3, 1]))
    divider12 = make_axes_locatable((ax[3, 2]))
    divider13= make_axes_locatable((ax[4, 0]))
    divider14 = make_axes_locatable((ax[4, 1]))
    divider15 = make_axes_locatable((ax[4, 2]))
    cax1 = divider1.append_axes("right", size="2%", pad=0.05)
    cax2 = divider2.append_axes("right", size="2%", pad=0.05)
    cax3 = divider3.append_axes("right", size="2%", pad=0.05)
    cax4 = divider4.append_axes("right", size="2%", pad=0.05)
    cax5 = divider5.append_axes("right", size="2%", pad=0.05)
    cax6 = divider6.append_axes("right", size="2%", pad=0.05)
    cax7 = divider7.append_axes("right", size="2%", pad=0.05)
    cax8 = divider8.append_axes("right", size="2%", pad=0.05)
    cax9 = divider9.append_axes("right", size="2%", pad=0.05)
    cax10 = divider10.append_axes("right", size="2%", pad=0.05)
    cax11 = divider11.append_axes("right", size="2%", pad=0.05)
    cax12 = divider12.append_axes("right", size="2%", pad=0.05)
    cax13= divider13.append_axes("right", size="2%", pad=0.05)
    cax14 = divider14.append_axes("right", size="2%", pad=0.05)
    cax15 = divider15.append_axes("right", size="2%", pad=0.05)
    x = np.arange(np.shape(M1)[1]) * dx
    y = np.arange(np.shape(M1)[0]) * dy
    f=np.arange(np.shape(M4)[0]) * df
    F_LIM = np.array([f.min(), f.max() / 2])
    extent1 = np.array([x.min(), x.max(), y.min(), y.max()])
    extent3 = np.array([y.min(), y.max(), F_LIM[0], F_LIM[1]])
    extent2 = np.array([y.min(), y.max(), f.min(), f.max()])
    extent2 = np.array([fSpacez.min(), fSpacez.max(), fSpacex.min(), fSpacex.max()])
    MIN = 0;    MAX = 0.02
    if Plot_vertical_lines==1:
        Boundaries2 = Boundaries * dx
        ax[0,0].axvline(Boundaries2,color='k', linestyle='-' ,linewidth=2.5)
        ax[0,1].axvline(Boundaries2, color='k', linestyle='-',linewidth=2.5)
        ax[0,2].axvline(Boundaries2, color='k', linestyle='-',linewidth=2.5)
    ########################    First row
    Mat = np.concatenate((M1, M2, M3), axis=0);MIN = np.min(Mat);MAX = 1.0 * np.max(Mat)
    im1 = ax[0, 0].imshow(np.flipud(M1), extent=extent1, vmin = MIN, vmax = MAX, aspect='auto')
    im2 = ax[0, 1].imshow(np.flipud(M2), extent=extent1, vmin = MIN, vmax = MAX, aspect='auto')
    im3 = ax[0, 2].imshow(np.flipud(M3), extent=extent1, vmin = MIN, vmax = MAX, aspect='auto')
    # im1 = ax[0,0].imshow(np.flipud(M1), extent=extent1, aspect='auto')
    # im2 = ax[0,1].imshow(np.flipud(M2), extent=extent1, aspect='auto')
    # im3 = ax[0,2].imshow(np.flipud(M3), extent=extent1, aspect='auto')
    ax[0, 0].set_ylabel(z_label_name)
    ax[0, 0].set_xlabel('x (km)')
    ax[0, 1].set_ylabel(z_label_name)
    ax[0, 1].set_xlabel('x (km)')
    ax[0, 2].set_ylabel(z_label_name)
    ax[0, 2].set_xlabel('x (km)')
    ax[0, 0].invert_yaxis()
    ax[0, 1].invert_yaxis()
    ax[0, 2].invert_yaxis()
    ax[0, 0].set_title('dm true')
    ax[0, 1].set_title(Title + ',      dm 1 Hz')
    ax[0, 2].set_title('dm 3 Hz')
    ########################    Second row
    # im4=  ax[1,0].imshow(np.flipud(M4), extent=extent2, vmin=MIN, vmax=MAX, aspect='auto')
    # # im5 = ax[1,1].imshow(np.flipud(M5), extent=extent2, vmin=MIN, vmax=MAX, aspect='auto')
    # # im6 = ax[1,2].imshow(np.flipud(M6), extent=extent2, vmin=MIN, vmax=MAX, aspect='auto')
    # im5 = ax[1, 1].imshow(np.flipud(M5), extent=extent2, aspect='auto')
    # im6 = ax[1, 2].imshow(np.flipud(M6), extent=extent2, aspect='auto')
    # ax[1, 0].set_ylim(F_LIM)
    # ax[1, 1].set_ylim(F_LIM)
    # ax[1, 2].set_ylim(F_LIM)
    Mat = np.concatenate((M4, M5, M6), axis=0); MIN = np.min(Mat);MAX = 0.6* np.max(Mat)
    im4 = ax[1, 0].imshow(np.flipud(M4), extent=extent2, vmin=MIN, vmax=MAX, aspect='auto')
    im5 = ax[1, 1].imshow(np.flipud(M5), extent=extent2, vmin=MIN, vmax=MAX, aspect='auto')
    im6 = ax[1, 2].imshow(np.flipud(M6), extent=extent2, vmin=MIN, vmax=MAX, aspect='auto')
    ax[1, 0].set_ylabel('$K_{z}$ - vertical wavenumber')
    ax[1, 0].set_xlabel('$K_{x}$')
    ax[1, 1].set_xlabel('$K_{x}$')
    ax[1, 2].set_xlabel('$K_{x}$')
    ax[1, 0].set_title('|FFT2D (dm_true)|')
    ax[1, 1].set_title('|FFT2D (dm 1 Hz)|')
    ax[1, 2].set_title('|FFT2D (dm 3 Hz)|')
    ax[1, 0].invert_yaxis()
    ax[1, 1].invert_yaxis()
    ax[1, 2].invert_yaxis()
    ########################    Third row
    # im7 = ax[2, 0].imshow(np.flipud(M7), extent=extent2, aspect='auto')
    # im8 = ax[2, 1].imshow(np.flipud(M8), extent=extent2, aspect='auto')
    # im9 = ax[2, 2].imshow(np.flipud(M9), extent=extent2, aspect='auto')
    # ax[2, 0].set_ylim(F_LIM)
    # ax[2, 1].set_ylim(F_LIM)
    # ax[2, 2].set_ylim(F_LIM)
    im7 = ax[2, 0].imshow(np.flipud(M7), extent=extent2, vmin=MIN, vmax=MAX, aspect='auto')
    im8 = ax[2, 1].imshow(np.flipud(M8), extent=extent2, vmin=MIN, vmax=MAX, aspect='auto')
    im9 = ax[2, 2].imshow(np.flipud(M9), extent=extent2, vmin=MIN, vmax=MAX, aspect='auto')
    ax[2, 0].set_ylabel('$K_{z}$ - vertical wavenumber')
    ax[2, 0].set_xlabel('$K_{x}$')
    ax[2, 1].set_xlabel('$K_{x}$')
    ax[2, 2].set_xlabel('$K_{x}$')
    ax[2, 0].set_title('|FFT2D(dm 1 Hz)-FFT2D(dm 3 Hz)|')
    ax[2, 1].set_title('|FFT2D(dm 1 Hz)-FFT2D(dm_true)|')
    ax[2, 2].set_title('|FFT2D(dm 3 Hz)-FFT2D(dm_true)|')
    ax[2, 0].invert_yaxis()
    ax[2, 1].invert_yaxis()
    ax[2, 2].invert_yaxis()
    ########################    Fourth row
    Mat = np.concatenate((M10, M11, M12), axis=0);MIN = np.min(Mat);MAX = 0.6 * np.max(Mat)
    im10 = ax[3, 0].imshow(np.flipud(M10), extent=extent3, vmin=MIN, vmax=MAX,aspect='auto')
    im11 = ax[3, 1].imshow(np.flipud(M11), extent=extent3, vmin=MIN, vmax=MAX,aspect='auto')
    im12 = ax[3, 2].imshow(np.flipud(M12), extent=extent3, vmin=MIN, vmax=MAX,aspect='auto')
    ax[3, 0].set_ylabel('$K_{z}$ - vertical wavenumber')
    ax[3, 0].set_xlabel('z, m')
    ax[3, 1].set_xlabel('z, m')
    ax[3, 2].set_xlabel('z, m')
    ax[3, 0].set_title('STFT' + '(dm_true('+str(Boundaries2) + ' m'+'))' )
    ax[3, 1].set_title('STFT' + '(dm_1Hz('+str(Boundaries2) + ' m'+'))' )
    ax[3, 2].set_title('STFT' + '(dm_3Hz('+str(Boundaries2) + ' m'+'))' )
    # ax[3, 0].invert_yaxis()
    # ax[3, 1].invert_yaxis()
    # ax[3, 2].invert_yaxis()
    ########################    Fifth row
    Mat = np.concatenate((M13, M14, M15), axis=0);MIN = np.min(Mat);MAX = 0.6 * np.max(Mat)
    im13 = ax[4, 0].imshow(np.flipud(M13), extent=extent3, vmin=MIN, vmax=MAX,aspect='auto')
    im14 = ax[4, 1].imshow(np.flipud(M14), extent=extent3, vmin=MIN, vmax=MAX,aspect='auto')
    im15 = ax[4, 2].imshow(np.flipud(M15), extent=extent3, vmin=MIN, vmax=MAX,aspect='auto')
    ax[4, 0].set_ylabel('$K_{z}$ - vertical wavenumber')
    ax[4, 0].set_xlabel('z, m')
    ax[4, 1].set_xlabel('z, m')
    ax[4, 2].set_xlabel('z, m')
    ax[4, 0].set_title('Crosscorrelation(dm 1 Hz,dm 3 Hz)')
    ax[4, 1].set_title('Crosscorrelation(dm_true,dm 1 Hz)')
    ax[4, 2].set_title('Crosscorrelation(dm_true,dm 3 Hz)')
    # ax[4, 0].invert_yaxis()
    # ax[4, 1].invert_yaxis()
    # ax[4, 2].invert_yaxis()
    ########################
    cbar1 = plt.colorbar(im1, cax=cax1)
    cbar2 = plt.colorbar(im2, cax=cax2)
    cbar3 = plt.colorbar(im3, cax=cax3)
    cbar4 = plt.colorbar(im4, cax=cax4)
    cbar5 = plt.colorbar(im5, cax=cax5)
    cbar6 = plt.colorbar(im6, cax=cax6)
    cbar7 = plt.colorbar(im7, cax=cax7)
    cbar8 = plt.colorbar(im8, cax=cax8)
    cbar9 = plt.colorbar(im9, cax=cax9)
    cbar10 = plt.colorbar(im10, cax=cax10)
    cbar11 = plt.colorbar(im11, cax=cax11)
    cbar12 = plt.colorbar(im12, cax=cax12)
    cbar13 = plt.colorbar(im13, cax=cax13)
    cbar14 = plt.colorbar(im14, cax=cax14)
    cbar15 = plt.colorbar(im15, cax=cax15)
    plt.subplots_adjust(left=0.07, right=0.93, bottom=0.08,
        top=0.95, hspace=.49,wspace=.29)  # hspace,wspace- adjust space between subplots
    if Save_flag == 1:
        plt.savefig(Name, dpi=300)
        print('Saving result to ' + Name)
    plt.show(block=False)
    plt.close()
    # %
    return None
def PLOT_Spectrum_Analysis_Pad3(M1,M2,M3,M4,M5,M6,M7,M8,M9,M10,M11,M12,M13,M14,M15,Curve1='',Curve2='',Boundaries=0,Name='',dx=1,dy=1,df=1,
                                Plot_vertical_lines=1, Title='',Save_flag=0,fSpacex='',fSpacez=''):
    if len(np.shape(Curve1)) == 2:
        Curve1=Curve1[0,:]
    if len(np.shape(Curve2)) == 2:
        Curve2=Curve2[0,:]
    z_label_name = 'z, m'
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 25.4
    fig_size[1] = 15.0
    Linewidth=3.6
    plt.rcParams["figure.figsize"] = fig_size
    plt.rcParams["font.size"] = 13
    fig, ax = plt.subplots(5,3)
    divider1 = make_axes_locatable((ax[0,0]))
    divider2 = make_axes_locatable((ax[0,1]))
    divider3 = make_axes_locatable((ax[0,2]))
    divider4 = make_axes_locatable((ax[1,0]))
    divider5 = make_axes_locatable((ax[1,1]))
    divider6 = make_axes_locatable((ax[1,2]))
    divider7 = make_axes_locatable((ax[2, 0]))
    divider8 = make_axes_locatable((ax[2, 1]))
    divider9 = make_axes_locatable((ax[2, 2]))
    divider10 = make_axes_locatable((ax[3, 0]))
    divider11 = make_axes_locatable((ax[3, 1]))
    divider12 = make_axes_locatable((ax[3, 2]))
    divider13= make_axes_locatable((ax[4, 0]))
    divider14 = make_axes_locatable((ax[4, 1]))
    divider15 = make_axes_locatable((ax[4, 2]))
    cax1 = divider1.append_axes("right", size="2%", pad=0.05)
    cax2 = divider2.append_axes("right", size="2%", pad=0.05)
    cax3 = divider3.append_axes("right", size="2%", pad=0.05)
    cax4 = divider4.append_axes("right", size="2%", pad=0.05)
    cax5 = divider5.append_axes("right", size="2%", pad=0.05)
    cax6 = divider6.append_axes("right", size="2%", pad=0.05)
    cax7 = divider7.append_axes("right", size="2%", pad=0.05)
    cax8 = divider8.append_axes("right", size="2%", pad=0.05)
    cax9 = divider9.append_axes("right", size="2%", pad=0.05)
    cax10 = divider10.append_axes("right", size="2%", pad=0.05)
    cax11 = divider11.append_axes("right", size="2%", pad=0.05)
    cax12 = divider12.append_axes("right", size="2%", pad=0.05)
    cax13= divider13.append_axes("right", size="2%", pad=0.05)
    cax14 = divider14.append_axes("right", size="2%", pad=0.05)
    cax15 = divider15.append_axes("right", size="2%", pad=0.05)
    x = np.arange(np.shape(M1)[1]) * dx
    y = np.arange(np.shape(M1)[0]) * dy
    f=np.arange(np.shape(M4)[0]) * df
    F_LIM = np.array([f.min(), f.max() / 4])
    extent1 = np.array([x.min(), x.max(), y.min(), y.max()])
    extent3 = np.array([y.min(), y.max(), F_LIM[0], F_LIM[1]])
    extent2 = np.array([y.min(), y.max(), f.min(), f.max()])
    extent2 = np.array([fSpacez.min(), fSpacez.max(), fSpacex.min(), fSpacex.max()])
    MIN = 0;    MAX = 0.02
    if Plot_vertical_lines==1:
        Boundaries2 = Boundaries * dx
        ax[0,0].axvline(Boundaries2,color='k', linestyle='-' ,linewidth=2.5)
        ax[0,1].axvline(Boundaries2, color='k', linestyle='-',linewidth=2.5)
        ax[0,2].axvline(Boundaries2, color='k', linestyle='-',linewidth=2.5)
    ########################    First row
    Mat = np.concatenate((M1, M2, M3), axis=0);MIN = np.min(Mat);MAX = 0.7 * np.max(Mat)
    im1 = ax[0, 0].imshow(np.flipud(M1), extent=extent1, vmin = MIN, vmax = MAX, aspect='auto')
    im2 = ax[0, 1].imshow(np.flipud(M2), extent=extent1, vmin = MIN, vmax = MAX, aspect='auto')
    im3 = ax[0, 2].imshow(np.flipud(M3), extent=extent1, vmin = MIN, vmax = MAX, aspect='auto')
    # im1 = ax[0,0].imshow(np.flipud(M1), extent=extent1, aspect='auto')
    # im2 = ax[0,1].imshow(np.flipud(M2), extent=extent1, aspect='auto')
    # im3 = ax[0,2].imshow(np.flipud(M3), extent=extent1, aspect='auto')
    ax[0, 0].set_ylabel(z_label_name)
    ax[0, 0].set_xlabel('x (km)')
    ax[0, 1].set_ylabel(z_label_name)
    ax[0, 1].set_xlabel('x (km)')
    ax[0, 2].set_ylabel(z_label_name)
    ax[0, 2].set_xlabel('x (km)')
    ax[0, 0].invert_yaxis()
    ax[0, 1].invert_yaxis()
    ax[0, 2].invert_yaxis()
    ax[0, 0].set_title('dm true')
    ax[0, 1].set_title(Title + ',      dm 1 Hz')
    ax[0, 2].set_title('dm 3 Hz')
    ########################    Second row
    # im4=  ax[1,0].imshow(np.flipud(M4), extent=extent2, vmin=MIN, vmax=MAX, aspect='auto')
    # # im5 = ax[1,1].imshow(np.flipud(M5), extent=extent2, vmin=MIN, vmax=MAX, aspect='auto')
    # # im6 = ax[1,2].imshow(np.flipud(M6), extent=extent2, vmin=MIN, vmax=MAX, aspect='auto')
    # im5 = ax[1, 1].imshow(np.flipud(M5), extent=extent2, aspect='auto')
    # im6 = ax[1, 2].imshow(np.flipud(M6), extent=extent2, aspect='auto')
    # ax[1, 0].set_ylim(F_LIM)
    # ax[1, 1].set_ylim(F_LIM)
    # ax[1, 2].set_ylim(F_LIM)
    Mat = np.concatenate((M4, M5, M6), axis=0); MIN = np.min(Mat);MAX = 0.6* np.max(Mat)
    im4 = ax[1, 0].imshow(np.flipud(M4), extent=extent2, vmin=MIN, vmax=MAX, aspect='auto')
    im5 = ax[1, 1].imshow(np.flipud(M5), extent=extent2, vmin=MIN, vmax=MAX, aspect='auto')
    im6 = ax[1, 2].imshow(np.flipud(M6), extent=extent2, vmin=MIN, vmax=MAX, aspect='auto')
    ax[1, 0].set_ylabel('$K_{z}$ - vertical wavenumber')
    ax[1, 0].set_xlabel('$K_{x}$')
    ax[1, 1].set_xlabel('$K_{x}$')
    ax[1, 2].set_xlabel('$K_{x}$')
    ax[1, 0].set_title('|FFT2D (dm_true)|')
    ax[1, 1].set_title('|FFT2D (dm 1 Hz)|')
    ax[1, 2].set_title('|FFT2D (dm 3 Hz)|')
    ax[1, 0].invert_yaxis()
    ax[1, 1].invert_yaxis()
    ax[1, 2].invert_yaxis()
    ########################    Third row
    # im7 = ax[2, 0].imshow(np.flipud(M7), extent=extent2, aspect='auto')
    # im8 = ax[2, 1].imshow(np.flipud(M8), extent=extent2, aspect='auto')
    # im9 = ax[2, 2].imshow(np.flipud(M9), extent=extent2, aspect='auto')
    # ax[2, 0].set_ylim(F_LIM)
    # ax[2, 1].set_ylim(F_LIM)
    # ax[2, 2].set_ylim(F_LIM)
    Mat = np.concatenate((M7, M8, M9), axis=0);MIN = np.min(Mat);       MAX = 0.6 * np.max(Mat)
    im7 = ax[2, 0].imshow(np.flipud(M7), extent=extent3, vmin=MIN, vmax=MAX, aspect='auto')
    im8 = ax[2, 1].imshow(np.flipud(M8), extent=extent3, vmin=MIN, vmax=MAX, aspect='auto')
    im9 = ax[2, 2].imshow(np.flipud(M9), extent=extent3, vmin=MIN, vmax=MAX, aspect='auto')

    ax[2, 0].set_ylabel('$K_{z}$ - vertical wavenumber')
    ax[2, 0].set_xlabel('$K_{x}$')
    ax[2, 1].set_xlabel('$K_{x}$')
    ax[2, 2].set_xlabel('$K_{x}$')
    # ax[2, 0].set_title('|FFT2D(dm 1 Hz)-FFT2D(dm 3 Hz)|')
    # ax[2, 1].set_title('|FFT2D(dm 1 Hz)-FFT2D(dm_true)|')
    # ax[2, 2].set_title('|FFT2D(dm 3 Hz)-FFT2D(dm_true)|')
    ax[2, 0].set_title('STFT_truespectrum')
    ax[2, 1].set_title('STFT_lowspectrum')
    ax[2, 2].set_title('STFT_highspectrum')
    # ax[2, 0].invert_yaxis()
    # ax[2, 1].invert_yaxis()
    # ax[2, 2].invert_yaxis()
    ########################    Fourth row
    Mat = np.concatenate((M10, M11, M12), axis=0);MIN = np.min(Mat);MAX = 0.6 * np.max(Mat)
    im10 = ax[3, 0].imshow(np.flipud(M10), extent=extent3, vmin=MIN, vmax=MAX,aspect='auto')
    im11 = ax[3, 1].imshow(np.flipud(M11), extent=extent3, vmin=MIN, vmax=MAX,aspect='auto')
    im12 = ax[3, 2].imshow(np.flipud(M12), extent=extent3, vmin=MIN, vmax=MAX,aspect='auto')

    ax[3, 0].set_ylabel('$K_{z}$ - vertical wavenumber')
    ax[3, 0].set_xlabel('z, m')
    ax[3, 1].set_xlabel('z, m')
    ax[3, 2].set_xlabel('z, m')
    # ax[3, 0].set_title('STFT' + '(dm_true('+str(Boundaries2) + ' m'+'))' )
    # ax[3, 1].set_title('STFT' + '(dm_1Hz('+str(Boundaries2) + ' m'+'))' )
    # ax[3, 2].set_title('STFT' + '(dm_3Hz('+str(Boundaries2) + ' m'+'))' )
    ax[3, 0].set_title('STFT_highspectrum-STFT_lowspectrum')
    ax[3, 1].set_title('STFT_truespectrum - STFT_lowspectrum')
    ax[3, 2].set_title('STFT_truespectrum - STFT_highspectrum')
    # ax[3, 0].invert_yaxis()
    # ax[3, 1].invert_yaxis()
    # ax[3, 2].invert_yaxis()
    ########################    Fifth row
    Mat = np.concatenate((M13, M14, M15), axis=0);MIN = np.min(Mat);MAX = 1.0 * np.max(Mat)
    im13 = ax[4, 0].imshow(np.flipud(M13), extent=extent3, vmin=MIN, vmax=MAX,aspect='auto')
    im14 = ax[4, 1].imshow(np.flipud(M14), extent=extent3, vmin=MIN, vmax=MAX,aspect='auto')
    im15 = ax[4, 2].imshow(np.flipud(M15), extent=extent3, vmin=MIN, vmax=MAX,aspect='auto')
    ax[4, 0].set_ylabel('$K_{z}$ - vertical wavenumber')
    ax[4, 0].set_xlabel('z, m')
    ax[4, 1].set_xlabel('z, m')
    ax[4, 2].set_xlabel('z, m')
    ax[4, 0].set_title('Crosscorrelation of STFT along x (dm 1 Hz,dm 3 Hz)')
    ax[4, 1].set_title('Crosscorrelation of STFT along x (dm_true,dm 1 Hz)')
    ax[4, 2].set_title('Crosscorrelation of STFT along x (dm_true,dm 3 Hz)')
    # ax[4, 0].invert_yaxis()
    # ax[4, 1].invert_yaxis()
    # ax[4, 2].invert_yaxis()
    ########################
    ax[2, 0].plot(y, Curve1, color='white', linewidth=Linewidth, linestyle='--')
    ax[2, 1].plot(y, Curve1, color='white', linewidth=Linewidth, linestyle='--')
    ax[2, 2].plot(y, Curve1, color='white', linewidth=Linewidth, linestyle='--')
    ax[3, 0].plot(y, Curve1, color='white', linewidth=Linewidth, linestyle='--')
    ax[3, 1].plot(y, Curve1, color='white', linewidth=Linewidth, linestyle='--')
    ax[3, 2].plot(y, Curve1, color='white', linewidth=Linewidth, linestyle='--')
    ax[4, 0].plot(y, Curve1, color='white', linewidth=Linewidth, linestyle='--')
    ax[4, 1].plot(y, Curve1, color='white', linewidth=Linewidth, linestyle='--')
    ax[4, 2].plot(y, Curve1, color='white', linewidth=Linewidth, linestyle='--')

    ax[2, 0].plot(y, Curve2, color='white', linewidth=Linewidth, linestyle='-.')
    ax[2, 1].plot(y, Curve2, color='white', linewidth=Linewidth, linestyle='-.')
    ax[2, 2].plot(y, Curve2, color='white', linewidth=Linewidth, linestyle='-.')
    ax[3, 0].plot(y, Curve2, color='white', linewidth=Linewidth, linestyle='-.')
    ax[3, 1].plot(y, Curve2, color='white', linewidth=Linewidth, linestyle='-.')
    ax[3, 2].plot(y, Curve2, color='white', linewidth=Linewidth, linestyle='-.')
    ax[4, 0].plot(y, Curve2, color='white', linewidth=Linewidth, linestyle='-.')
    ax[4, 1].plot(y, Curve2, color='white', linewidth=Linewidth, linestyle='-.')
    ax[4, 2].plot(y, Curve2, color='white', linewidth=Linewidth, linestyle='-.')
    ########################
    cbar1 = plt.colorbar(im1, cax=cax1)
    cbar2 = plt.colorbar(im2, cax=cax2)
    cbar3 = plt.colorbar(im3, cax=cax3)
    cbar4 = plt.colorbar(im4, cax=cax4)
    cbar5 = plt.colorbar(im5, cax=cax5)
    cbar6 = plt.colorbar(im6, cax=cax6)
    cbar7 = plt.colorbar(im7, cax=cax7)
    cbar8 = plt.colorbar(im8, cax=cax8)
    cbar9 = plt.colorbar(im9, cax=cax9)
    cbar10 = plt.colorbar(im10, cax=cax10)
    cbar11 = plt.colorbar(im11, cax=cax11)
    cbar12 = plt.colorbar(im12, cax=cax12)
    cbar13 = plt.colorbar(im13, cax=cax13)
    cbar14 = plt.colorbar(im14, cax=cax14)
    cbar15 = plt.colorbar(im15, cax=cax15)
    plt.subplots_adjust(left=0.07, right=0.93, bottom=0.08,
        top=0.95, hspace=.49,wspace=.29)  # hspace,wspace- adjust space between subplots
    if Save_flag == 1:
        plt.savefig(Name, dpi=300)
        print('Saving result to ' + Name)
    plt.show(block=False)
    plt.close()
    # %
    return None
def PLOT_Compare_spectrums(M1,M2,M3,Boundaries=0,Log_offset=-100500,
                           dx=1,dy=1,Name='',Plot_vertical_lines=0, Title='',Save_flag=0,
                           COEFF = 1,Extent_divider=1,Colorbar_max=-100500):
    Cbar_name = 'Normalized Amp'
    x_label_name = 'z, km'
    z_label_name = '$K_{z}$'
    x0 = 0.11;y0 = -0.21
    Title1='Input, Offset='+str(Log_offset)
    # Title1 = Name
    Title2='Target'
    Title3='Output'
    ######################
    Mat = np.concatenate((M1, M2, M3), axis=0)
    # COEFF = 1.0;
    MIN = np.min(Mat)
    MAX = COEFF * np.max(Mat)
    if Colorbar_max != -100500:
        MAX = Colorbar_max
    #%
    dx=dx;dy=dy
    #%
    x=np.arange(np.shape(M1)[1])*dx
    y = np.arange(np.shape(M1)[0])*dy
    extent = np.array([x.min(), x.max(), y.min(), y.max()])
    Y_LIM= np.array([y.min(), y.max()/Extent_divider])
    # matplotlib.rcParams.update({'font.size': 15})
    M1=np.flipud(M1)
    M2= np.flipud(M2)
    M3= np.flipud(M3)
    # %
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 6.4
    fig_size[1] = 8.0
    plt.rcParams["figure.figsize"] = fig_size
    plt.rcParams["font.size"] = 13
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1)
    #  allocate space for colorbar smartly
    divider1 = make_axes_locatable((ax1))
    divider2 = make_axes_locatable((ax2))
    divider3 = make_axes_locatable((ax3))
    cax1 = divider1.append_axes("right", size="2%", pad=0.05)
    cax2 = divider2.append_axes("right", size="2%", pad=0.05)
    cax3 = divider3.append_axes("right", size="2%", pad=0.05)
    im1=ax1.imshow(M1, extent=extent, vmin=MIN, vmax=MAX, aspect='auto')
    im2 = ax2.imshow(M2, extent=extent, vmin=MIN, vmax=MAX, aspect='auto')
    im3 = ax3.imshow(M3, extent=extent, vmin=MIN, vmax=MAX, aspect='auto')

    if Plot_vertical_lines==1:
        Boundaries2 = Boundaries * dx
        for i in range(np.shape(Boundaries)[0]):
            ax1.axvline(Boundaries2[i],color='k', linestyle='-' ,linewidth=2.5)
            ax2.axvline(Boundaries2[i], color='k', linestyle='-',linewidth=2.5)
            ax3.axvline(Boundaries2[i], color='k', linestyle='-',linewidth=2.5)
    ax1.set_ylabel(z_label_name)
    ax1.set_xlabel(x_label_name)
    ax2.set_ylabel(z_label_name)
    ax2.set_xlabel(x_label_name)
    ax3.set_ylabel(z_label_name)
    ax3.set_xlabel(x_label_name)
    ax1.invert_yaxis()
    ax2.invert_yaxis()
    ax3.invert_yaxis()
    ax1.xaxis.set_label_coords(x0, y0)
    ax2.xaxis.set_label_coords(x0, y0)
    ax3.xaxis.set_label_coords(x0, y0)
    ax1.set_title(Title1)
    ax2.set_title(Title2)
    ax3.set_title(Title3)

    ax1.set_ylim(Y_LIM)
    ax2.set_ylim(Y_LIM)
    ax3.set_ylim(Y_LIM)

    cbar1 = plt.colorbar(im1, cax=cax1)
    cbar2 = plt.colorbar(im2, cax=cax2)
    cbar3 = plt.colorbar(im3, cax=cax3)
    x0 = -50; y0 = 1.16
    cbar1.set_label(Cbar_name, labelpad=x0, y=y0, rotation=0)
    cbar2.set_label(Cbar_name, labelpad=x0, y=y0, rotation=0)
    cbar3.set_label(Cbar_name, labelpad=x0, y=y0, rotation=0)

    plt.subplots_adjust(bottom=0.08, top=0.95, hspace=.49, right=0.86)
    if Save_flag == 1:
        plt.savefig(Name, dpi=300)
        print('Saving ML_result to ' + Name)
    plt.show(block=False)
    plt.close()
    return None
def Plot_curves(Curve1,Curve2,Save_pictures_path,Name,Save_flag=0):
    plt.figure()
    # for i in range(np.shape(Curves,0)):
    plt.plot(Curve1)
    plt.plot(Curve2)
    plt.title('Curve')
    # plt.legend([''])
    if Save_flag==1:
        Name2 = Save_pictures_path + '/' + Name + '.png'
        plt.savefig(Name2, dpi=300)
    plt.show(block=False)
    plt.close()
    return None
def Plot_accuracy(history,Title='Title',Save_pictures_path='./Pictures',Save_flag=0):
    plt.figure()
    plt.plot(history['mean_absolute_error'])
    plt.plot(history['val_mean_absolute_error'])
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.axis('tight')
    plt.legend(['Training accuracy', 'Validation accuracy'], loc='lower right')
    if Save_flag == 1:
        name=Save_pictures_path + '/' + Title + '.png'
        print(name)
        plt.savefig(name)
    plt.show(block=False)
    plt.close()
    return None
def Plot_accuracy2(history,Title='Title',Save_pictures_path='./Pictures',Save_flag=0):
    plt.figure()
    plt.plot(history['coeff_determination'])
    plt.plot(history['val_coeff_determination'])
    plt.ylabel('R2')
    plt.xlabel('Epoch')
    plt.axis('tight')
    plt.ylim(-1,1)
    string=', R2 accuracy curve train/test='+numstr( history['coeff_determination'][len(history['coeff_determination'])-1] )+'/'+numstr(history['val_coeff_determination'][len(history['val_coeff_determination'])-1])
    plt.title(Title+string)
    plt.legend(['training R2','validation R2'], loc='lower right')
    if Save_flag == 1:
        name = Save_pictures_path + '/' + Title + '.png'
        print(name)
        plt.savefig(name)
    plt.show(block=False)
    plt.close()
    return None
def Plot_loss(history,Title='Title',Save_pictures_path='./Pictures',Save_flag=0):
    plt.figure()
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.yscale('log')
    plt.ylabel('Loss function')
    plt.xlabel('Epoch')
    plt.axis('tight')
    # len(history['coeff_determination'])
    # print(', R2 accuracy curve train/test='+numstr(history['coeff_determination'][-1])+'/'+numstr(history['val_coeff_determination'][-1]))
    string=', R2 accuracy curve train/test='+numstr( history['coeff_determination'][len(history['coeff_determination'])-1] )+'/'+numstr(history['val_coeff_determination'][len(history['val_coeff_determination'])-1])
    plt.title(Title)
    plt.legend(['Training', 'Validation'], loc='upper right')
    if Save_flag == 1:
        name=Save_pictures_path + '/' + Title + '.png'
        print(name)
        plt.savefig(name)
    plt.show(block=False)
    plt.close()
    return None
def Plot_loss_r2(history,Title='Title',Save_pictures_path='./Pictures',Save_flag=0):
    plt.figure()
    plt.plot(-np.array(history['loss']))
    plt.plot(-np.array(history['val_loss']))
    # plt.yscale('log')
    # ax.set_yscale('log')
    plt.ylabel('Loss function,R2')
    plt.xlabel('Epoch')
    plt.axis('tight')
    plt.legend(['Training', 'Validation'], loc='upper right')
    if Save_flag == 1:
        name=Save_pictures_path + '/' + Title + '.png'
        print(name)
        plt.savefig(name)
    plt.show(block=False)
    plt.close()
    return None
def plt_nb_T(vel, fname="Velocity", title="",
             ylabel="Depth (km)", xlabel="Distance (km)",
             cbar_label = "(km/s)",
             vmin=None, vmax=None,
             split_line=False,
             dx=25, dz=25, no_labels=False, origin_in_middle=False):
    plt.figure(figsize=(16,9))
    plt.set_cmap('RdBu_r')
    vel_image = vel[:,:].T
    extent=(0, dx * vel.shape[0] * 1e-3, dz * vel.shape[1] *1e-3, 0)
    if origin_in_middle:
        extent = (-dx * vel.shape[0] * .5e-3, dx * vel.shape[0] * .5e-3, dz * vel.shape[1] *1e-3, 0)
    plt.imshow(vel_image * 1e-3, origin='upper', extent=extent)
    plt.axis("tight")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.clim(vmin,vmax)
    cbar = plt.colorbar()
    cbar.ax.set_ylabel(cbar_label)
    if split_line:
        plt.axvline(x=extent[1]/2, color='black', linewidth=10, linestyle='-')
    
    if no_labels:
        plt.axis('off')
        plt.xlabel()
    plt.savefig(fname,bbox_inches='tight')
    plt.show()
def plt_nb_strip(vel, fname="Velocity", title="Velocity (km/s)",
                 ylabel="Depth (km)", xlabel="Distance (km)",
                 vmin=None, vmax=None,
                 dx=25, dz=25, no_labels=False, 
                 origin_in_middle=False, x_samples=200):
    plt_nb_T(vel[:x_samples,:], fname=fname, title=title,
             ylabel=ylabel, xlabel=xlabel,
             dx=dx, dz=dz, no_labels=no_labels,
             vmin=vmin, vmax=vmax,
             origin_in_middle=origin_in_middle)
def Plot_curves(Curve1,Save_pictures_path,Name,Save_flag=0):
    plt.figure()
    # for i in range(np.shape(Curves,0)):
    plt.plot(Curve1)
    plt.title(Name)
    # plt.legend([''])
    if Save_flag==1:
        Name2 = Save_pictures_path + '/' + Name + '.png'
        plt.savefig(Name2, dpi=300)
    plt.show(block=False)
    plt.close()
    return None
def Plot_spectrum(f,spectra,Save_pictures_path,Name,Save_flag=0):
    plt.figure()
    plt.plot(f, spectra)
    plt.xlabel('Hz')
    plt.grid()
    plt.xlim([0, 40])
    plt.title('Spectrum')
    if Save_flag==1:
        Name2 = Save_pictures_path + '/' + Name + '.png'
        plt.savefig(Name2, dpi=300)
    plt.show(block=False)
    plt.close()
    return None
def Plot_wavenumber_spectrum(Data, Title='Title', c_lim='',x='',x_label='',y='',y_label='',
               dx='',dy='',Save_flag=0,Save_pictures_path='./Pictures',
               Reverse_axis=1,Curve='',Show_flag=1,Aspect='auto',write_fig_title=1):
    if c_lim == '':  c_lim =[np.min(Data), np.max(Data)]
    if x == '':  x=(np.arange(np.shape(Data)[1]))
    if y == '':  y=(np.arange(np.shape(Data)[0]))
    if dx != '':  x=(np.arange(np.shape(Data)[1]))*dx
    if dy != '':  y=(np.arange(np.shape(Data)[0]))*dy
    extent = [x.min(), x.max(), y.min(), y.max()]
    # plt.ioff()
    fig=plt.figure()
    # plt.use('Agg')
    # plt.rcParams.update({'font.size':62})
    fig.dpi=330
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 10.4
    fig_size[1] = 4
    plt.rcParams["figure.figsize"] = fig_size
    # plt.rcParams["font.size"] = 132;# plt.axis(extent, Aspect)#; plt.axis(extent, 'auto')
    plt.set_cmap('RdBu_r')
    ax=plt.gca()
    # ax = fig.add_subplot(111)
    Fontsize=15
    plt.tick_params(labelsize=Fontsize)
    # plt.tick_params(labelsize=18)
    if write_fig_title==1:
        plt.title(Title)
    if Reverse_axis == 1:
        plt.imshow(np.flipud(Data),extent=extent,interpolation='nearest',aspect=Aspect)
        ax.invert_yaxis()
    else:
        plt.imshow((Data),extent=extent,interpolation='nearest',aspect=Aspect)
    plt.xlabel(x_label,fontsize=Fontsize+3)
    plt.ylabel(y_label,fontsize=Fontsize+3)
    ax.locator_params(axis='x', nbins=7)
    divider1 = make_axes_locatable((ax))
    cax1 = divider1.append_axes("right", size="2%", pad=0.05)
    cbar=plt.colorbar(cax=cax1)
    # cbar.set_label("(m/s)")
    cbar.ax.tick_params(labelsize=Fontsize)
    plt.clim(c_lim)
    # plt.rcParams('xtick',labelsize=Fontsize)
    plt.rc('xtick',labelsize=Fontsize)
    # plt.axis('equal')
    ########################################################
    if Save_flag==1:
        if not os.path.exists(Save_pictures_path):
            os.mkdir(Save_pictures_path)
        name=Save_pictures_path + '/' + Title + '.png'
        # plt.show(block=False)
        plt.show()
        plt.savefig(name,bbox_inches='tight')   # plt.savefig(name)
        # plt.close('all')
        plt.close(fig)
        print(name)
    else:
        plt.show()
        plt.close()
        # del fig
        # del plt
    return None
########################################################
# x_ticks=[]
# for i in range(len(x)):
#     x_ticks.append(str('{0:.4f}'.format(x[i])))
# x_ticks_to_plot=x_ticks[::100]
# # ax.set_xticklabels(x_ticks_to_plot,rotation=45)
# start, end = ax.get_xlim()
# ax.xaxis.set_ticks(np.arange(start, end, 2))

# plt.axis('equal')
# plt.axis('tight')
# tight_figure(fig)
def Plot_dataset_figures(dz,Save_pictures_path,NAME):
    pts=NAME.split('/')
    f_name=pts[-1]
    f_name=f_name[:-4]
    with open(NAME, 'rb') as f:
        data=np.load(f)
        M0=data['models'][0,:,:,0]
        # M1=data['input_data'][0,:,:,0]
        M2=data['output_data'][0,:,:,0]
        data.close()
    Plot_image(M0.T,Show_flag=0,Save_flag=1,dx=dz, dy=dz,Save_pictures_path=Save_pictures_path,Title=f_name+'_M0',Aspect='equal')
    Plot_image(M2.T,Show_flag=0,Save_flag=1,dx=dz, dy=dz,Save_pictures_path=Save_pictures_path,Title=f_name+'_M2',Aspect='equal')
    # print('model.shape=',M0.shape)
    print('target.shape=',M2.shape)
    return M0
def concat_vh(list_2d): 
    # define a function for vertically concatenating images of the same size  and horizontally 
    return cv2.vconcat([cv2.hconcat(list_h)  for list_h in list_2d]) 
def forceAspect(ax,aspect):
    im = ax.get_images()
    extent =  im[0].get_extent()
    ax.set_aspect(abs((extent[1]-extent[0])/(extent[3]-extent[2]))/aspect)
def colorbar(mappable):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import matplotlib.pyplot as plt
    last_axes = plt.gca()
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(mappable, cax=cax)
    plt.sca(last_axes)
    return cbar
def Plot_imgs_side_by_side(Data,Title='Title',c_lim='',
               Save_flag=0,Save_pictures_path='./Pictures',
               Show_flag=1,write_fig_title=1):
    Data=np.swapaxes(Data,1,2)
    img_tile = concat_vh([[Data[0,:,:],Data[1,:,:]],
                      [Data[2,:,:], Data[3,:,:]], 
                      [Data[4,:,:], Data[5,:,:]]])
    ####################
    if c_lim == '':  c_lim =[np.min(img_tile), np.max(img_tile)]
    # extent = [x.min(), x.max(), y.min(), y.max()]
    fig=plt.figure(dpi=330)
    # plt.figure(figsize=(9,3),dpi=330)
    plt.set_cmap('RdBu_r')
    if write_fig_title==1:
        plt.title(Title)
    val=17
    font_sz=23
    width=img_tile.shape[1]
    height=img_tile.shape[0]
    extent=(0,width,height, 0)

    ax = plt.gca()
    aspect=3
    im=ax.imshow(np.flipud(img_tile),interpolation='nearest',extent=extent,
    vmin=c_lim[0],vmax=c_lim[1])
    ax.set_aspect(aspect)
    ax.invert_yaxis()
    # im=plt.imshow(np.flipud(img_tile),interpolation='nearest')
    ax.text(val,height-val,           "a",fontsize=font_sz,fontweight='bold')
    ax.text(width/2,height-val,      "b",fontsize=font_sz,fontweight='bold')
    ax.text(val,     2/3*height-val, "c",fontsize=font_sz,fontweight='bold')
    ax.text(width/2,2/3*height-val, "d",fontsize=font_sz,fontweight='bold')
    ax.text(val,     1/3*height-val, "e",fontsize=font_sz,fontweight='bold')
    ax.text(width/2,1/3*height-val, "f",fontsize=font_sz,fontweight='bold')
    ax.tick_params(labelbottom=False,labelleft=False,bottom=False,left=False)
    cbar = plt.colorbar(im,ax=ax,fraction=0.03, pad=0.01)
    cbar_label = "(m/s)"
    cbar.set_label(cbar_label)
    # tight_figure(fig)
    if Save_flag == 1:
        if not os.path.exists(Save_pictures_path):
            os.mkdir(Save_pictures_path)
        name=Save_pictures_path + '/' + Title + '.png'
        print(name)
        # plt.savefig(name,bbox_inches='tight')
        fig.savefig(name,bbox_inches='tight')
    plt.show()
    plt.close()
    return None

# inp_orig_sizes = [M1, M2, M0, M0, M0, M0]
# PLOT_ML_Result4(inp_orig_sizes, history_flag=0, history=None, Boundaries=[],
# save_file_path=Save_pictures_path+'/'+f_name+'_'+str(counter)+'.png',dx=dx, dy=dz, 
# Title=NAME, Title2=NAME, Save_flag=1)
