#############################   other imports stack
import os,sys
sys.path.append('../')
import pickle

# info_file=os.path.join('./par','acq_data_parameters_cgg_correct.pkl')
# info_file=os.path.join('./data/acq_data_parameters_cgg.pkl')
# with open(info_file,'rb') as input:
#     acq_data=pickle.load(input)

# info_file=os.path.join('./par/acquisition_design.pkl')
# with open(info_file,'rb') as input:
#     acq_data=pickle.load(input)

from pathlib import Path
sys.path.append(str(Path('.').absolute().parent))
from functions.F_utils import *
from functions.F_modules import *
#############################   DENISE imports stack
import matplotlib as mpl
mpl.rcParams['figure.dpi']= 300
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
sys.path.append('./for_pasha')
from functions.utils import shared as sd
from functions.utils import shared as sd
from functions.utils import loaders as ld
from functions.utils import vis
import pyapi_denise as api
info_file=os.path.join('./par','acq_data_parameters_cgg_correct.pkl')
#####################   Fusion net  machine learning import #####################################
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
def conv_trans_block(in_dim,out_dim,act_fn):
    model = nn.Sequential(
        nn.ConvTranspose2d(in_dim, out_dim, kernel_size=3,
                           stride=2, padding=1, output_padding=1),
        nn.BatchNorm2d(out_dim),
        act_fn,)
    return model
def maxpool():
    pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
    return pool
def conv_block_3(in_dim,out_dim,act_fn,opt):
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
##########################################################################################
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
def MSE(x, y):
    """ Pearson's coefficient, rho = \frac{cov(x, y)}{std_x * std_y}    """
    x=x.flatten()
    y=y.flatten()
    difference_array = np.subtract(x,y)
    squared_array = np.square(difference_array)
    mse = squared_array.mean()
    return mse
##########################################################################################
#############################   utilities
def numstr_2(x):
    string = str('{0:.2f}'.format(x))
    return string
def numstr_3(x):
    string = str('{0:.3f}'.format(x))
    return string
def zero_below_freq(dat, fhi, dt, disable=False, reverse=False):
    """ Input zeros into frequency spectrum of data below or above specified frequency.
        by Oleg Ovcharenko, KAUST, 2021
       
    Args:
        dat(np.ndarray): 2D array [noffsets, ntimes]
        fhi(float): threshold frequency, Hz
        dt(float): temporal sampling, sec
        disable(bool): do nothing, return input data
        reverse(bool): when True, set zeros above fhi, otherwise below
    """
    if disable:
        return dat
    h, w = dat.shape[-2:]
    dat_fx = np.fft.rfft(dat, w)
    ff = np.fft.rfftfreq(dat.shape[-1], d=dt)
    if not reverse:
        where_to_zero = np.where(ff < fhi)[0]
    else:
        where_to_zero = np.where(ff >= fhi)[0]
    dat_fx[..., where_to_zero] = 0. + 0. * 1j
    out = np.fft.irfft(dat_fx, w)
    return out
def butter_bandpass(flo=None, fhi=None, fs=None, order=8, btype='band'):
    nyq = 0.5 * fs
    if btype == 'band':
        low = flo / nyq
        high = fhi / nyq
        lims = [low, high]
    elif btype == 'low':
        high = fhi / nyq
        lims = high
    elif btype == 'high':
        low = flo / nyq
        lims = low

    #b, a = scipy.signal.butter(order, lims, btype=btype)
    #return b, a
    sos = signal.butter(order, lims, btype=btype, output='sos')
    return sos
def bandpass(data, flo=None, fhi=None, dt=None, fs=None, order=4, btype='band', verbose=0, pad=(0, 8), upscale=0):
    """ Filter frequency content of 2D data in format [offset, time]

    Args:
        data (ndarray): [offset, time]
        flo (float): low coner frequency
        fhi (float): high corner frequency
        dt (float): sampling interval (introduced for back-compatibility), sec. You can enter either one dt or fs
        fs (float): 1/dt, sampling frequency, Hz
        order:
        btype (str): band, high or low
            * band: limit from both left and right
            * high: limit from right only
            * low: limit from left only
        verbose (bool): print details
       
    Example:
        filtered = bandpass(data, fhi=5, dt=0.002, order=8, btype='low')

    Returns:
        ndarray
    """
   
    if not fs:
        fs = 1/dt
   
    # if isinstance(data, torch.Tensor):
    # data = data.numpy()
   
    if upscale:
        no, nt = data.shape
        data = scipy.signal.resample(data, nt * upscale, axis=-1)
        fs = fs * upscale
       
    if pad:
        no, nt = data.shape
        tmp = np.zeros((no, nt + pad[0] + pad[1]))
        tmp[:, pad[0]:nt+pad[0]] = data
        data = tmp.copy()

    if verbose:
        print(f'Bandpass:\n\t{data.shape}\tflo={flo}\tfhi={fhi}\tfs={fs}')
    #b, a = butter_bandpass(flo, fhi, fs, order=order, btype=btype)
    #y = scipy.signal.filtfilt(b, a, data)

    sos = butter_bandpass(flo, fhi, fs, order=order, btype=btype)
    y = scipy.signal.sosfiltfilt(sos, data)
   
    if pad:
        y = y[:, pad[0]:-pad[1] if pad[1] else None]
   
    if upscale:
        y = y[:, ::upscale]
    return y
def F_calculate_log_number(path,Word,type='.png'):
    print(os.getcwd())
    Const = len(fnmatch.filter(os.listdir(path), Word + '*'))
    Name = path + '/' + Word + str(Const) + type
    while os.path.exists(Name):
        Const = Const + 1
        Name = path + '/' + Word + str(Const) + type
    return Const
def F_smooth(data, plot_flag=0, sigma_val=8):
    data2 = gaussian_filter(data, sigma=sigma_val, order=0, output=None,
                            mode="reflect", cval=0.0, truncate=4.0)
    if plot_flag == 1:
        a = np.min(data);
        b = np.max(data)
        Plot_image(data.T, Show_flag=1, c_lim=[a, b])
        Plot_image(data2.T, Show_flag=1, c_lim=[a, b])
    return data2
def load_bin(p, dims): 
    f = open(p); vp = np.fromfile (f, dtype=np.dtype('float32').newbyteorder ('<')); f.close();
    vp = vp.reshape(*dims); vp = np.transpose(vp); vp = np.flipud(vp); print(f"{vp.shape}"); return vp
def plot_model(v, title='', axis='on',folder_path='pictures',file_path='1.png',**kwargs): 
    plt.figure(); ax = plt.gca(); im = ax.imshow(np.flipud(v), cmap='RdBu_r', **kwargs); 
    plt.axis(axis); plt.title(title);   
    plt.axis('equal')
    divider = make_axes_locatable(ax); cax = divider.append_axes("right", size="5%", pad=0.05);plt.colorbar(im, cax=cax); #plt.pause(0.001)
    os.makedirs(folder_path,exist_ok=True);   
    strFile=os.path.join(folder_path,file_path)
    if os.path.isfile(strFile):
        os.remove(strFile)
    print('saving to ',strFile)
    plt.savefig(strFile,bbox_inches='tight');     plt.close()
def plot_acquisition(v,model,src=None, rec=None, title='',nx_orig=None,folder_path='pictures',file_path='1.png',**kwargs):
    mpl.rcParams['figure.dpi']= 300
    nx=model.nx; nz=model.nz; 
    nz,nx=v.shape
    dx=model.dx; 
    if nx_orig!=None:
        nx=nx_orig
    par = {'extent': [0,nx*dx,0,nz*dx],'cmap':'RdBu_r'}; par.update(kwargs);
    log=None
    cax_label=''
    plt.figure(); 
    ax = plt.gca(); 
    im = ax.imshow(v[:,0:nx],**par)
    plt.title(title); ax.invert_yaxis(); plt.xlabel('km'); plt.ylabel('km');
    if rec is not None: 
        map_rec = rec.x / dx < nx
        # plt.scatter(rec.x[map_rec], rec.y[map_rec], 1, color='m'); 
    if src is not None:
        map_src = src.x / dx < nx
        plt.scatter(src.x[map_src], src.y[map_src], 1, color='w'); 
    if log is not None:
        ax = plt.gca()
        _log = log['data']
        vh = log['loc'] * np.ones_like(_log)
        ax.plot(vh, np.arange(len(_log))*dx, 'k--')
        ax.plot(vh + (_log[::-1] - min(_log)) , np.arange(len(_log))*dx, 'k')
    divider = make_axes_locatable(ax); cax = divider.append_axes("right", size="5%", pad=0.05); 
    plt.colorbar(im, cax=cax, label=cax_label);
    plt.show(block=False)
    # plt.show()
    os.makedirs(folder_path,exist_ok=True)
    plt.savefig(os.path.join(folder_path,file_path),bbox_inches='tight');     
    print('plot_acquisition to ',os.path.join(folder_path,file_path))
    plt.close()
    return None
def plot_log_model__(mm, dx, nx0=None, nz0=None, _src=None, title='', log=None, log_location=None, cmap='RdBu_r', axis=True, cax_label='km/s', **kwargs):
    v = mm.copy() / 1000
    plt.figure(); ax = plt.gca();
    nz, nx = mm.shape[-2:]
    if _src is not None:
        map_src = _src.x / dx < nx0
        plt.scatter(_src.x[map_src]/1000, _src.y[map_src]/1000, 1, color='w'); 
    im = ax.imshow(v[:,:nx0], cmap=cmap, extent=[0, 
                                                 nx0 * dx / 1000 if nx0 else nx * dx / 1000, 
                                                 0, 
                                                 nz0 * dx/1000] if nz0 else nz * dx / 1000, 
                   origin='upper', **kwargs); 
    divider = make_axes_locatable(ax); 
    cax = divider.append_axes("right", size="5%", pad=0.05); cbar = plt.colorbar(im, cax=cax); cbar.set_label(cax_label);
    if axis:
        ax.set_xlabel('km'); ax.set_ylabel('km'); ax.set_title(title); 
    else:
        ax.axis('off')
    ax.invert_yaxis();
def plot_log_model(mm, dx, nx0=None, nz0=None, _src=None, title='', log=None, log_location=None, cmap='RdBu_r', axis=True, cax_label='km/s',folder_path='pictures',file_path='1.png',**kwargs):
    v = mm.copy() / 1000
    plt.figure(); ax = plt.gca();
    nz, nx = mm.shape[-2:]
    if _src is not None:
        map_src = _src.x / dx < nx0
        plt.scatter(_src.x[map_src]/1000, _src.y[map_src]/1000, 1, color='w'); 
    im = ax.imshow(v[:,:nx0], cmap=cmap, extent=[0, 
                                                 nx0 * dx / 1000 if nx0 else nx * dx / 1000, 
                                                 0, 
                                                 nz0 * dx/1000 if nz0 else nz * dx / 1000], 
                   origin='upper', **kwargs); 
    divider = make_axes_locatable(ax); 
    cax = divider.append_axes("right", size="5%", pad=0.05); cbar = plt.colorbar(im, cax=cax); cbar.set_label(cax_label);
    if axis:
        ax.set_xlabel('km'); ax.set_ylabel('km'); ax.set_title(title); 
    else:
        ax.axis('off')
    ax.invert_yaxis();

    if log is not None:
        vh = log_location * np.ones_like(log) / 1000 
        ax.plot(vh, np.arange(len(log))*dx/1000, 'k--')
        ax.plot(vh + (log[::-1] - min(log)) / 1000, np.arange(len(log))*dx/1000, 'k')
    plt.show(block=False)
    # plt.show()
    os.makedirs(folder_path,exist_ok=True)
    plt.savefig(os.path.join(folder_path,file_path),bbox_inches='tight');     
    print('plot_log_model to ',os.path.join(folder_path,file_path))
    plt.close()
def plot_shot(shot, title='', pclip=1.0,folder_path='pictures',file_path='1.png',show=0):
    vmax = pclip * np.max(np.abs(shot)); vmin = - vmax; 
    plt.figure(); 
    plt.imshow(shot.T, cmap='Greys', vmin=vmin, vmax=vmax); 
    plt.colorbar(); plt.axis('auto')
    plt.title(file_path); 
    os.makedirs(folder_path,exist_ok=True)
    plt.savefig(os.path.join(folder_path,file_path),dpi=300);     
    print('save to '+os.path.join(folder_path,file_path))
    if show==1: plt.show()
    plt.close()
def plot_spectrum(shot, dt,ps2=[],title='',fmax=None,folder_path='pictures',file_path='1.png',show=0):
    # fmax=7
    # filtered_shot2=bandpass(shot,fhi=3, dt=dt, order=8, btype='low')
    # plot_shot(shot,pclip=0.05,folder_path=folder_path,file_path='shot.png',show=0)
    # plot_shot(filtered_shot2, pclip=0.05,folder_path=folder_path,file_path='shot_filtered.png',show=0)
    ps = np.sum(np.abs(np.fft.fft(shot)) ** 2, axis=-2); 
    freqs = np.fft.fftfreq(len(ps), dt); 
    idx = np.argsort(freqs)
    causal = int(len(ps) // 2); 
    freqs, ps = freqs[idx], ps[idx];
    freqs = freqs[-causal:];
    ps = ps[-causal:]; 
    freqs = freqs[freqs < (fmax if fmax else np.max(freqs))]; 
    n = len(freqs); 
    ################
    plt.figure(); 
    FREQS=freqs[:n];PS=ps[:n]
    if ps2!=[]:
        [freqs_2,ps_2]=ps2       
        ps=ps/np.max(ps)
        ps_2=ps_2/np.max(ps_2)
    plt.plot(freqs[:n],ps[:n],label=title); 
    if ps2!=[]:
        plt.plot(freqs_2,ps_2,label='spectrum 2');
    plt.xlabel('Frequency (Hz)'); plt.ylabel('Gain'); plt.grid(True);
    os.makedirs(folder_path,exist_ok=True)
    strFile=os.path.join(folder_path,file_path)
    if os.path.isfile(strFile):
        os.remove(strFile)
    plt.savefig(strFile,dpi=300);   
    if show==1: plt.show()
    plt.close();    print('pic saved to ',strFile)
    ################
    # ps = np.sum(np.abs(np.fft.fft(filtered_shot2)) ** 2, axis=-2); 
    # freqs = np.fft.fftfreq(len(ps), dt); 
    # idx = np.argsort(freqs)
    # causal = int(len(ps) // 2); 
    # freqs, ps = freqs[idx], ps[idx];
    # freqs = freqs[-causal:];
    # ps = ps[-causal:]; 
    # freqs = freqs[freqs < (fmax if fmax else np.max(freqs))]; 
    # n = len(freqs); 
    # file_path='spectrum_filtered.png'
    # plt.figure(); 
    # FREQS=freqs[:n];PS=ps[:n]
    # if ps2!=[]:
    #     [freqs_2,ps_2]=ps2       
    #     ps=ps/np.max(ps)
    #     ps_2=ps_2/np.max(ps_2)
    # plt.plot(freqs[:n],ps[:n],label=title); 
    # if ps2!=[]:
    #     plt.plot(freqs_2,ps_2,label='spectrum 2');
    # plt.xlabel('Frequency (Hz)'); plt.ylabel('Gain'); plt.grid(True);
    # os.makedirs(folder_path,exist_ok=True)
    # strFile=os.path.join(folder_path,file_path)
    # if os.path.isfile(strFile):
    #     os.remove(strFile)
    # plt.savefig(strFile,dpi=300)
    # if show==1: plt.show()
    # plt.close();    print('pic saved to ',strFile)
    return FREQS,PS
def plot_logs(m1, m2, idx=2,folder_path='pictures',file_path='1.png'):
    file_name=file_path.split('.')[0]
    plt.figure()
    plt.plot(m1.vp[::-1,idx], 'b--',label='well_log')
    plt.plot(m1.vs[::-1,idx], color='orange',linestyle='--')
    plt.plot(m1.rho[::-1,idx],'g--')
    plt.plot(m2.vp[::-1,idx], 'b',label='vp')
    plt.plot(m2.vs[::-1,idx], 'orange',label='vs'); 
    plt.plot(m2.rho[::-1,idx],'g', label='rho'); plt.legend() 
    os.makedirs(folder_path,exist_ok=True)
    plt.axis('tight')
    log1=m2.vp[::-1,idx]
    log2=m1.vp[::-1,idx]
    crop_nz=min(len(log1),len(log2))
    log1=log1[0:crop_nz]
    log2=log2[0:crop_nz]
    score_str=PCC(log1,log2)
    score_str='PCC='+str(score_str)
    plt.title(file_name+' '+score_str)
    print('saving to ',os.path.join(folder_path,file_name+'_'+(score_str)+'.png'))
    plt.savefig(os.path.join(folder_path,file_name+'_'+(score_str)+'.png'))
    plt.close()
def is_empty(p): return False if (os.path.exists(p) and [f for f in os.listdir(p) if f != '.gitignore']) else True
def plot_fwi_misfit(filename,d):#######   plot fwi objective curve
    # estimate number of iterations 
    if os.path.exists(filename):
        print('Parse misfit file ,'+filename)
        data = np.genfromtxt(filename)
    else:
        return None
    if data.size == 0:
        print('empty misfit log')
        return None
    else:
        print('misfit data exists')   
    if data.ndim==1:
        print('only one iteration finished')
        return None
    else:
        nit, npara = np.shape(data)
    iteration = np.linspace(1,nit,nit,dtype=int)
    # estimate number of stages
    tmp = np.max(data[:,-1])
    nstage = tmp.astype(int)
    # estimate number of iterations / FWI stage
    niter_stage = np.zeros(nstage, dtype=int)
    h=1
    stage = 1
    for i in range(1, nit):       
        if data[i,-1] == stage:
            h = h + 1
        if data[i,-1] != stage:
            print('Parse misfit file ,'+filename)
            niter_stage[stage-1] = h - 1
            h = 1
            stage = stage + 1
        if i == (nit-1):
            niter_stage[stage-1] = h - 1
    # define iteration number for transition from one stage to the next
    nstage_trans = np.cumsum(niter_stage)
    # normalize misfit function to maximum value
    data[:,4] = data[:,4] / np.max(data[:,4])
    # plot stage boundaries
    ##########################  plot normalized misfit function
    plt.figure()
    for i in range(1, nstage):
        plt.semilogy([nstage_trans[i-1]+i,nstage_trans[i-1]+i], [np.min(data[:,4]),np.max(data[:,4])],'k--', linewidth=2)
    # plot misfit function
    plt.semilogy(iteration,data[:,4],'b-', linewidth=1, label='Evolution of the misfit function')
    # scale and annotate axis
    a = plt.gca()
    a.set_xticklabels(a.get_xticks())
    a.set_yticklabels(a.get_yticks())
    # plt.ylim(0,1)
    plt.ylabel('Normalized misfit function')
    plt.xlabel('Iteration no.')
    plt.autoscale(enable=True, axis='y')    #, tight=True
    # add FWI stages and iteration numbers per stage 
    for i in range(1, nstage+1):
        stage_title = "stage" + "%0.*f" %(0,np.fix(i)) + "\n (" + "%0.*f" %(0,np.fix(niter_stage[i-1])) + " iterations)"
        plt.text(nstage_trans[i-1] - 0.85 * niter_stage[i-1], 0.6, stage_title, color='black')
    # plt.tight_layout()
    # figure output
    #plt.savefig('test.png', format='png', dpi=100)
    print('saving to '+os.path.join(d.pictures_folder,'normalized_misfit.png'))
    plt.savefig(os.path.join(d.pictures_folder,'normalized_misfit.png'),dpi=300 )
    plt.show();plt.close()
    ##########################  plot L2 misfit function 4th from last column in data file seis_fwilog.dat
    plt.figure()
    chosen_column=-4
    for i in range(1, nstage):
        plt.semilogy([nstage_trans[i-1]+i,nstage_trans[i-1]+i], [np.min(data[:,chosen_column]),np.max(data[:,chosen_column])],'k--', linewidth=2)
    # plot misfit function
    plt.semilogy(iteration,data[:,chosen_column],'b-', linewidth=1, label='Evolution of the misfit function')
    # scale and annotate axis
    a = plt.gca()
    a.set_xticklabels(a.get_xticks())
    a.set_yticklabels(a.get_yticks())
    # plt.ylim(0,1)
    plt.ylabel('misfit function')
    plt.xlabel('Iteration no.')
    plt.autoscale(enable=True, axis='y')    #, tight=True
    # add FWI stages and iteration numbers per stage 
    for i in range(1, nstage+1):
        stage_title = "stage" + "%0.*f" %(0,np.fix(i)) + "\n (" + "%0.*f" %(0,np.fix(niter_stage[i-1])) + " iterations)"
        plt.text(nstage_trans[i-1] - 0.85 * niter_stage[i-1], 0.012, stage_title, color='black')
    # plt.tight_layout()
    # figure output
    #plt.savefig('test.png', format='png', dpi=100)
    print('saving to '+os.path.join(d.pictures_folder,'misfit.png'))
    plt.savefig(os.path.join(d.pictures_folder,'misfit.png'),dpi=300)
    plt.show();plt.close()
    return data[:,4]
def plot_fwi_misfit2(filename,d):#######   plot fwi objective curve
    d.pictures_folder=os.path.join(d.save_folder)
    # estimate number of iterations 
    if os.path.exists(filename):
        print('Parse misfit file ,'+filename)
        data = np.genfromtxt(filename)
    else:
        if hasattr(d,'misfit_data')==True:
            data=d.misfit_data
        else:
            return None
    if data.size == 0:
        print('empty misfit log')
        return None
    else:
        print('misfit data exists')   
    if data.ndim==1:
        print('only one iteration finished')
        data=np.expand_dims(data,axis=0)
        data2=np.repeat(data,2,axis=0)
        # return None
    else:
        nit, npara = np.shape(data)
    nit, npara = np.shape(data)
    iteration = np.linspace(1,nit,nit,dtype=int)
    # estimate number of stages
    tmp = np.max(data[:,-1]);   nstage = tmp.astype(int)
    # estimate number of iterations / FWI stage
    niter_stage = np.zeros(nstage, dtype=int)
    h=1
    stage = 1
    for i in range(1, nit):       
        if data[i,-1] == stage:
            h = h + 1
        if data[i,-1] != stage:
            print('Parse misfit file ,'+filename)
            niter_stage[stage-1] = h - 1
            h = 1
            stage = stage + 1
        if i == (nit-1):
            niter_stage[stage-1] = h - 1
    data_orig=data
    # define iteration number for transition from one stage to the next
    nstage_trans = np.cumsum(niter_stage)
    # normalize misfit function to maximum value
    plt.figure()
    plt.plot(iteration,data[:,4],'b-', linewidth=1, label='Evolution of the misfit function')
    # plt.savefig(os.path.join(d.pictures_folder,'original_misfit.png'),dpi=300 )
    # plt.show();
    plt.close()
    # data[:,4] = data[:,4] / np.max(np.abs(data[:,4]))
    # data[:,4] = data[:,4] / (data[:,-2])**2
    # plot stage boundaries
    ##########################  plot normalized misfit function
    plt.figure()
    for i in range(1, nstage):
        plt.semilogy([nstage_trans[i-1]+i,nstage_trans[i-1]+i], [np.min(data[:,4]),np.max(data[:,4])],'k--', linewidth=2)
    # plot misfit function
    plt.semilogy(iteration,data[:,4],'b-', linewidth=1, label='Evolution of the misfit function')
    # scale and annotate axis
    a = plt.gca()
    a.set_xticklabels(a.get_xticks())
    a.set_yticklabels(a.get_yticks())
    # plt.ylim(0,1)
    plt.ylabel('Normalized misfit function')
    plt.xlabel('Iteration no.')
    plt.autoscale(enable=True, axis='y')    #, tight=True
    # add FWI stages and iteration numbers per stage 
    for i in range(1, nstage+1):
        stage_title = "stage" + "%0.*f" %(0,np.fix(i)) + "\n (" + "%0.*f" %(0,np.fix(niter_stage[i-1])) + " iterations)"
        plt.text(nstage_trans[i-1] - 0.85 * niter_stage[i-1], 0.6, stage_title, color='black')
    # plt.tight_layout()
    # figure output
    #plt.savefig('test.png', format='png', dpi=100)
    print('saving to '+os.path.join(d.pictures_folder,'normalized_misfit.png'))
    # plt.savefig(os.path.join(d.pictures_folder,'normalized_misfit.png'),dpi=300 )
    # plt.show();
    plt.close()
    ##########################  plot L2 misfit function 4th from last column in data file seis_fwilog.dat
    data=data_orig
    plt.figure()
    chosen_column=-4
    for i in range(1, nstage):
        plt.semilogy([nstage_trans[i-1]+i,nstage_trans[i-1]+i], [np.min(data[:,chosen_column]),np.max(data[:,chosen_column])],'k--', linewidth=2)
    # plot misfit function
    plt.semilogy(iteration,data[:,chosen_column],'b-', linewidth=1, label='Evolution of the misfit function')
    # scale and annotate axis
    a = plt.gca()
    a.set_xticklabels(a.get_xticks())
    a.set_yticklabels(a.get_yticks())
    # plt.ylim(0,1)
    plt.ylabel('misfit function')
    plt.xlabel('Iteration no.')
    plt.autoscale(enable=True, axis='y')    #, tight=True
    # add FWI stages and iteration numbers per stage 
    for i in range(1, nstage+1):
        stage_title = "stage" + "%0.*f" %(0,np.fix(i)) + "\n (" + "%0.*f" %(0,np.fix(niter_stage[i-1])) + " iterations)"
        plt.text(nstage_trans[i-1] - 0.85 * niter_stage[i-1], 0.012, stage_title, color='black')
    # plt.tight_layout()
    # figure output
    #plt.savefig('test.png', format='png', dpi=100)
    print('saving to '+os.path.join(d.pictures_folder,'misfit.png'))
    # plt.savefig(os.path.join(d.pictures_folder,'misfit.png'),dpi=300)
    # plt.show();
    plt.close()
    mdata={ 'nstage_trans':nstage_trans,
            'niter_stage':niter_stage,
            'nstage':nstage,
            'iteration':iteration}
    return data[:,4],mdata
def record_misfit(filename,d):#######   plot fwi objective curve
    # estimate number of iterations 
    if os.path.exists(filename):
        data = np.genfromtxt(filename)
        if data.size == 0:
            return 'empty'
        else:
            return data
    else:
        return 'empty'
def plot_fwi_step_length(filename,d):######  Plot step length evolution
    if os.path.exists(filename):
        data = np.genfromtxt(filename)
    else:
        return None
    # estimate number of iterations
    if data.size == 0:
        print('empty misfit log')
        return None
    else:
        print('data not empty')  
    if data.ndim==1:
        print('only one iteration finished')
        return None
    else:
        nit,npara=np.shape(data) 
    iteration = np.linspace(1,nit,nit,dtype=int)
    # estimate number of stages
    tmp = np.max(data[:,-1])
    nstage = tmp.astype(int)
    # estimate number of iterations / FWI stage
    niter_stage = np.zeros(nstage, dtype=int)
    h=1
    stage = 1
    for i in range(1, nit):       
        if data[i,-1] == stage:
            h = h + 1
        if data[i,-1] != stage:
            niter_stage[stage-1] = h - 1
            h = 1
            stage = stage + 1
        if i == (nit-1):
            niter_stage[stage-1] = h - 1
    # define iteration number for transition from one stage to the next
    nstage_trans = np.cumsum(niter_stage)
    # normalize misfit function to maximum value
    data[:,4] = data[:,4] / np.max(data[:,4])
    plt.figure()
    for i in range(1, nstage):
        plt.semilogy([nstage_trans[i-1]+i,nstage_trans[i-1]+i], [np.min(data[:,0]),np.max(data[:,0])],'k--', linewidth=2)
    # plot misfit function
    plt.semilogy(iteration,data[:,0],'b-', linewidth=3)
    # scale and annotate axis
    a = plt.gca()
    a.set_xticklabels(a.get_xticks())
    a.set_yticklabels(a.get_yticks())
    plt.ylabel('Optimum step length from parabolic line search')
    plt.xlabel('Iteration no.')
    plt.autoscale(enable=True, axis='y', tight=True)
    # add FWI stages and iteration numbers per stage 
    for i in range(1, nstage+1):
        stage_title = "stage" + "%0.*f" %(0,np.fix(i)) + "\n (" + "%0.*f" %(0,np.fix(niter_stage[i-1])) + " iterations)"
        plt.text(nstage_trans[i-1] - 0.85 * niter_stage[i-1], np.max(data[:,0]) - 0.4 * np.max(data[:,0]), stage_title,color='black')
    plt.tight_layout()
    # figure output
    plt.savefig(os.path.join(d.pictures_folder,'misfit_2_opt_step_length.png'),bbox_inches='tight')
    plt.show();plt.close;
    return None
######     same plots with misfits
def test_function():
    a=1
    return a
#############################   important denise scripts
def create_velocity_model_file(results_folder,stage=1,pars=None):
    #############################
    smoothing_radius=100
    vp1=1500;vp2=4500;
    os.listdir(results_folder)
    parent_path=os.path.join(str(Path(results_folder).absolute().parent))
    base_velocity_models_file=os.path.join(parent_path,'velocity_models_file.hdf5')
    if os.path.exists(base_velocity_models_file):
        print('uploading velocity models from ',base_velocity_models_file)
        print('smoothing_radius=',smoothing_radius)
        MODELS=load_file(base_velocity_models_file,'models')
        MODELS_INIT_FINAL=load_file(base_velocity_models_file,'models_init')
        water_taper=load_file(base_velocity_models_file,'water_taper')
    # else:
    if stage>0:
        ####    data_source
        path=os.path.join(parent_path,'stage'+str(stage-1))
        directory=os.path.join(path,'fld')
        #     # d.save_folder=
        # else:
        d=api.Denise(verbose=0)
        FILE=os.path.join(directory,'seis_inversion.inp')
        if not os.path.exists(FILE):
            FILE=os.path.join(directory,'seis_forward.inp')
        d._parse_inp_file(fname=FILE)
        if os.path.exists((os.path.join(directory,'model'))):
            if len(os.listdir(os.path.join(directory,'model')))>0:
                d.save_folder=directory
                models,fnames=d.get_fwi_models(return_filenames=True,keys=['vp'])
                index=-1
                if np.isnan(models[index].sum()):
                    while np.isnan(models[index].sum()):
                        index=index-1
                model_picked_up_from_previous_stage=models[index]
                print('model_picked_up_from_previous_stage=',fnames[-1])
                model_picked_up_from_previous_stage2=np.copy(model_picked_up_from_previous_stage)
                model_picked_up_from_previous_stage2=np.rot90(model_picked_up_from_previous_stage2,3)
                nx,nz=model_picked_up_from_previous_stage2.shape
        else:
            ss=1
            if os.path.exists(os.path.join(path,'denise_data.pkl')):
                with open(os.path.join(path,'denise_data.pkl'),'rb') as input:
                    d=pickle.load(input)
                ss=1
                model_picked_up_from_previous_stage=d
                d.fwi_model_names
        #########   fix sizes
        nx,nz=MODELS.shape
        MODELS=MODELS[0:nx,:]
        MODELS_INIT_FINAL=MODELS_INIT_FINAL[0:nx,:]
        #########   Smooth the model picked up from previous stage
        # new_initial_model=F_smooth(new_initial_model,sigma_val=int(smoothing_radius/pars['dx']))
        # new_initial_model[:,0:20]=1500
        #########   pickup CNN data
        d=parse_denise_folder(path)
        if not os.path.exists(os.path.join(path,'denise_data.pkl')):
            with open(os.path.join(path,'denise_data.pkl'), 'wb') as output:
                pickle.dump(d,output,protocol=4)
        check_passed=False
        if hasattr(d,'fwi_model_names')==True:
            model_files_list=[]
            for ii in d.fwi_model_names:
                model_files_list.append(ii.split('/')[-1])
            model_files_list=fnmatch.filter(model_files_list,'*vp_stage*')
            print('Last result file=',model_files_list[-1])
            for name in d.fwi_model_names:
                if 'vp_stage_2_it_5' in name:
                    check_passed=True
            if check_passed==True:
                name_to_load=os.path.join(d.save_folder,'model','modelTest_vp.bin')
                if name_to_load in d.fwi_model_names:    check2_passed=True
                else:   check2_passed=False
                name_to_load=os.path.join(d.save_folder,'model','modelTest_vp_stage_2_it_5.bin')
                if name_to_load in d.fwi_model_names:    check2_passed=True
                else:   check2_passed=False
                if check2_passed==True:
                    ind=d.fwi_model_names.index(name_to_load)
                    if np.isnan( d.models[ind].sum() )==False:
                        vp_res=d.models[ind]
                        ########
                        df = pd.DataFrame(np.arange(len(d.models)),columns=['index_in_array'])
                        df.insert(0,'filename',d.fwi_model_names)
                        df2=df[df['filename'].str.contains("modelTest_vp_stage")]
                        m_i_ra=np.array(d.models)[np.array(df2['index_in_array'],dtype=int)]
                        m_i_names=df2['filename'].to_list()
                        ########
                        # Plot_image(np.fliplr(d.model.vp.T).T,Show_flag=0,Save_flag=1,Title='_m_i_ra_0',Save_pictures_path='./pictures_for_check')
                        m_i_ra2=np.swapaxes(m_i_ra,1,2)
                        m_i_ra2=np.flip(m_i_ra2,2)
                        # Plot_image(m_i_ra2[0,::].T,Show_flag=0,Save_flag=1,Title='_m_i_ra_0',Save_pictures_path='./pictures_for_check')
                        # Plot_image(d.TAPER, Show_flag=0,Save_flag=1,Title='d.TAPER',Save_pictures_path=results_folder)
                        parameters={'file_path':os.path.join(path,path.split('/')[-1]+'.npz'),
                                'Models':       np.fliplr((d.model.vp).T),
                                'Models_init':  np.fliplr((d.model_init.vp).T),
                                'input_data':   np.fliplr(vp_res.T),
                                'm_i_ra':   m_i_ra2,
                                'm_i_names':   m_i_names,
                                'taper':   np.fliplr(d.TAPER.T),    # water_taper
                                'dx':d.DH,
                                'dz':d.DH}
        #########   Load CNN
        with open(os.path.join('./fwi','dataset_to_create_09_09.pkl'),'rb') as input:
            data_dict=pickle.load(input)
        scaling_constants_dict=data_dict['scaling_constants_dict']
        parser = ArgumentParser();  opt = parser.parse_args()
        with open(os.path.join(pars['prediction_path'],'opt.txt'),'r') as f:
            opt.__dict__=json.load(f)  
        ######################################################    
        scaling_range='standardization' #'-11'
        type_of_input=opt.inp_channels_type
        m_i_ra=parameters['m_i_ra'];
        dm_i_ra=m_i_ra-parameters['Models_init']
        taper=water_taper
        model_initial=np.squeeze(parameters['Models_init'])
        smoothed_true_model=F_smooth(MODELS,sigma_val=int(100/pars['dx']))    #100 or 300
        smoothed_true_model[taper==0]=1500
        dm_i_ra_scaled=np.empty_like(dm_i_ra)
        dm_i_ra_scaled2=np.empty((dm_i_ra.shape[0],opt.img_height,opt.img_width))
        Plot_image(taper.T, Show_flag=0,Save_flag=1,Title='water_taper',Save_pictures_path=results_folder)
        ###
        for i in range(dm_i_ra.shape[0]):
            dm_i_ra_scaled[i,::]=scaling_data(dm_i_ra[i,::],scaling_constants_dict,'x',scaling_range=scaling_range).squeeze()
            dm_i_ra_scaled2[i,:,:]=imresize(dm_i_ra_scaled[i,::].squeeze(),[opt.img_height,opt.img_width])
        taper=imresize(taper,[opt.img_height,opt.img_width]);    taper[taper<1]=0;   # imresize introduces real number
        model_initial=imresize(model_initial,[opt.img_height,opt.img_width])
        model_initial[taper==0]=1500
        model_initial_scaled=scaling_data(model_initial,scaling_constants_dict,'init_model',scaling_range=scaling_range).squeeze()
        ##############  fill the input data matrix
        x_=np.copy(dm_i_ra_scaled2)
        img_A=np.empty((opt.channels,opt.img_height,opt.img_width))
        #######################
        if type_of_input == '1dv':
            img_A[0, :, :] = x_[-1, ::]
        elif type_of_input == '1_fwi_res':
            img_A[0,:,:]=imresize(fwi_result_scaled.squeeze(),[opt.img_height,opt.img_width])
        elif type_of_input == '1m_1taper':
            img_A[0, :, :] = x_[-1, ::]+model_initial_scaled
            img_A[1, :, :] = taper
        elif type_of_input == '1dv_1taper':
            img_A[0, :, :] = x_[-1, ::]
            img_A[1, :, :] = taper
        elif type_of_input == '1dv_1init':
            img_A[0, :, :] = x_[-1, ::]
            img_A[1, :, :] = model_initial_scaled
        elif type_of_input == '1dv_1init_1sign':
            img_A[0, :, :] = x_[-1, ::]
            img_A[1, :, :] =model_initial_scaled
            img_A[2, :, :]=np.sign(x_[-1, ::])
        elif type_of_input == 'dm_i_1init':
            for i in range(10):
                img_A[i,::]=imresize(dm_i_ra_scaled[i,::].squeeze(),[opt.img_height,opt.img_width])
            img_A[10, :, :]=model_initial_scaled
        elif type_of_input == 'dm_i_1taper':
            for i in range(10):
                img_A[i,::]=imresize(dm_i_ra_scaled[i,::].squeeze(),[opt.img_height,opt.img_width])
            img_A[10, :, :]=taper
        elif type_of_input == 'dm_i_1init_1taper':
            for i in range(10):
                img_A[i,::]=imresize(dm_i_ra_scaled[i,::].squeeze(),[opt.img_height,opt.img_width])
            img_A[10, :, :]=model_initial_scaled
            img_A[11, :, :]=taper
        elif type_of_input == 'dm_i_1init_1sign_1taper':
            for i in range(10):
                img_A[i,::]=imresize(dm_i_ra_scaled[i,::].squeeze(),[opt.img_height,opt.img_width])
            img_A[10, :, :]=model_initial_scaled
            img_A[11, :, :]=np.sign(x_[-1, ::])
            img_A[12, :, :]=taper
        elif type_of_input == '1grad_1dv':
            img_A[0, :, :] = x_[-2, ::]
            img_A[1, :, :] = x_[-1, ::]
        elif type_of_input == 'allgrad_1dv':
            img_A = x_[-opt.channels:, ::]
        elif type_of_input == 'only_grads':
            x_ = x_[0:-1, ::]  # select only gradients for input
            selected_channels = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 49]
            img_A = x_[selected_channels, ::]  # variant 1
        elif type_of_input == 'dm_i':
            for i in range(img_A.shape[0]):
                img_A[i,::]=imresize(dm_i_ra_scaled[i,::].squeeze(),[opt.img_height,opt.img_width])
        elif type_of_input == 'm_i':
            for i in range(img_A.shape[0]):
                img_A[i,::]=imresize(m_i_ra_scaled[i,::].squeeze(),[opt.img_height,opt.img_width])
        #######################
        Plot_image(smoothed_true_model.T, Show_flag=0,Save_flag=1,Title='smoothed_true_model'+numstr(F_r2(smoothed_true_model,MODELS)),Save_pictures_path=results_folder)
        Plot_image(taper.T, Show_flag=0,Save_flag=1,Title='taper',Save_pictures_path=results_folder)
        Plot_image(model_initial.T, Show_flag=0,Save_flag=1,Title='model_initial',Save_pictures_path=results_folder)
        Plot_image(model_initial_scaled.T, Show_flag=0,Save_flag=1,Title='model_initial_scaled',Save_pictures_path=results_folder)
        # Plot_image(img_A[0,10, :, :].T, Show_flag=0,Save_flag=1,Title='img_A[0,10, :, :]',Save_pictures_path=results_folder)
        # Plot_image(img_A[0,11, :, :].T, Show_flag=0,Save_flag=1,Title='img_A[0,11, :, :]',Save_pictures_path=results_folder)
        # Plot_image(img_A[0,12, :, :].T, Show_flag=0,Save_flag=1,Title='img_A[0,12, :, :]',Save_pictures_path=results_folder)
        Plot_image(dm_i_ra_scaled2[9,:,:].T, Show_flag=0,Save_flag=1,Title='data_for_sign',Save_pictures_path=results_folder)
        ###
        pars['prediction_path'].split('/')[-1]
        num=(pars['prediction_path'].split('/')[-1].split('predictions_')[-1])
        weights=fnmatch.filter(glob(pars['prediction_path']+'/*'),'*.pth')[-1]
        # weights2=os.path.join(pars['prediction_path'],'generator_weights.pth')
        # weights=opt.generator_model_name
        #########   Apply CNN to the model picked up from previous stage
        fusion_net_parameters=(opt.channels,1,opt.number_of_filters)
        print('fusion net parameters=(input_nc,output_nc,ngf)',fusion_net_parameters)
        ######################################################
        torch.backends.cudnn.benchmark = True
        torch.manual_seed(123)
        torch.cuda.manual_seed(123)
        np.random.seed(123)
        random.seed(123)

        cuda = True if torch.cuda.is_available() else False
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # device = torch.device("cpu")
        # cuda = False

        print('cuda=',cuda)
        print('device=',device)
        # generator=(FusionGenerator(fusion_net_parameters[0],fusion_net_parameters[1],fusion_net_parameters[2],opt));
        generator=nn.DataParallel(FusionGenerator(fusion_net_parameters[0],fusion_net_parameters[1],fusion_net_parameters[2],opt))
        # generator=nn.DataParallel(FusionGenerator(fusion_net_parameters[0],fusion_net_parameters[1],fusion_net_parameters[2],opt) )
        # generator=nn.DataParallel(FusionGenerator(fusion_net_parameters[0],fusion_net_parameters[1],fusion_net_parameters[2],opt),output_device=device) #,device_ids=[gpu_number]

        generator.to(device)
        # generator = generator.cuda()

        generator.load_state_dict(torch.load(weights,map_location=device ))
        generator.eval()
        Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        for ii in range(img_A.shape[0]):
            Plot_image(img_A[ii,::].T, Show_flag=0,Save_flag=1,Title='ch_'+str(ii)+'_',Save_pictures_path=results_folder)
        img_A=np.expand_dims(img_A,axis=0)
        real_A=Variable(torch.from_numpy(img_A).to(device).type(Tensor))

        fake_B = generator(real_A)
        fake_B2=fake_B.cpu().detach().numpy().squeeze()
        fake_B3=imresize(fake_B2,[nx,nz])
        if type_of_input=='m_i_ra':
            fake_B4=scaling_data_back(fake_B3,scaling_constants_dict,'fwi_res',scaling_range=scaling_range).squeeze()
            new_initial_model=fake_B4
        else:
            fake_B4=scaling_data_back(fake_B3,scaling_constants_dict,'t',scaling_range=scaling_range).squeeze()
            new_initial_model=fake_B4+parameters['Models_init']
        predicted_initial=np.copy(new_initial_model)
        predicted_initial.shape
        print(' '+results_folder)
        ######################################################
        Plot_image(fake_B4.T, Show_flag=0,Save_flag=1,Title='predicted_update_',Save_pictures_path=results_folder)
        Plot_image(predicted_initial.T, Show_flag=0,Save_flag=1,c_lim=[1500,3000],Title='predicted_initial_model_R2_3000_m_true_'+numstr(F_r2(predicted_initial,MODELS)),Save_pictures_path=results_folder)
        Plot_image(predicted_initial.T, Show_flag=0,Save_flag=1,c_lim=[1500,4500],Title='predicted_initial_model_R2_4500_m_true_'+numstr(F_r2(predicted_initial,MODELS)),Save_pictures_path=results_folder)
        #############################   cgg log data
        with open(info_file,'rb') as input:
            acq_data=pickle.load(input)
        log_dict=acq_data['log_dict']
        log_loc=log_dict['loc']
        log=log_dict['data']
        log_idx = int(log_loc / 25)
        log_idx2=620
        vh = log_loc * np.ones_like(log) / 1000 
        lvp=log;   print(lvp.shape)
        # shear velocity, [m/s]
        lvs = lvp.copy() / (3 ** 0.5)
        lvs = np.where(lvp < 1.01 * np.min(lvp), 0, lvs)
        # density, [kg/m3] 
        lrho = 1e3*0.3 * lvp.copy()**0.25
        lrho = np.where(lvp < 1.01 * np.min(lvp), 1000, lrho)
        ######################################################
        predicted_initial2=np.copy(predicted_initial)   #    lower box condition applied
        predicted_initial3=np.copy(predicted_initial)   #    substitute bottom with
        parent_initial_model=load_file(base_velocity_models_file,'models_init')
        ############   mask the bottom part of the model with 1d initial model
        ############    lower box condition
        upper_bound=100
        box_max=parent_initial_model[:,upper_bound:]
        predicted_initial2[:,upper_bound:]=np.where(predicted_initial[:,upper_bound:]<box_max,box_max,predicted_initial[:,upper_bound:])
        predicted_initial3[:,upper_bound:]=box_max
        ######################################################
        Plot_image(predicted_initial2, Show_flag=0,Save_flag=1,c_lim=[1500,4500],Title='predicted_initial_model_R2_4500(m,true)='+numstr(F_r2(predicted_initial,MODELS)),Save_pictures_path=results_folder)
        labelsize=14
        Fontsize=32    
        textFontsize=20
        fig=plt.figure()
        plt.plot(lvp[::-1],'b--',label='well_log')
        plt.plot(parent_initial_model[log_idx,:],'b',label='init_stage0'); 
        plt.plot(predicted_initial[log_idx,:],label='init_stage'+str(stage)); 
        plt.plot(predicted_initial2[log_idx,:],label='init_stage'+str(stage)+'_bounded#1'); 
        plt.plot(predicted_initial3[log_idx,:],label='init_stage'+str(stage)+'_bounded#2'); 
        # plt.plot(parent_initial_model   [log_idx2,:],'b',label='init_stage2_0'); 
        # plt.plot(predicted_initial      [log_idx2,:],label='init_stage2_'+str(stage)); 
        # plt.plot(predicted_initial2     [log_idx2,:],label='init_stage2_'+str(stage)+'_bounded'); 
        plt.xlim(left=0)
        plt.ylim(bottom=0,top=3100)
        plt.grid()
        # plt.tick_params(labelsize=labelsize)
        plt.xlabel('Depth, km')
        plt.ylabel('Velocity, m/sec')
        vp_log=parent_initial_model[log_idx,:];   vp_log=vp_log[0:len(lvp)]
        well_log=lvp[::-1]
        water_start=33
        water_start=0
        score=      PCC(vp_log[water_start:],well_log[water_start:])
        score_mse=  MSE(vp_log[water_start:],well_log[water_start:])
        title_score='PCC(Vp initial, Vp well)='+numstr_3( score )#+'_'+numstr_2(score_mse)
        plt.title(title_score,fontsize=textFontsize)
        plt.legend(fontsize=labelsize,framealpha=0.1)
        nname=os.path.join(results_folder,'logs.png');print(nname)
        plt.savefig(nname)
        ######################################################
        new_initial_model2=F_smooth(new_initial_model,sigma_val=int(100/pars['dx']))
        new_initial_model3=F_smooth(new_initial_model,sigma_val=int(50/pars['dx']))
        new_initial_model4=F_smooth(new_initial_model,sigma_val=int(200/pars['dx']))
        new_initial_model5=F_smooth(new_initial_model,sigma_val=int(300/pars['dx']))
        new_initial_model6=F_smooth(new_initial_model,sigma_val=int(400/pars['dx']))
        new_initial_model7=F_smooth(new_initial_model,sigma_val=int(500/pars['dx']))
        # Plot_image(new_initial_model2.T, Show_flag=0,Save_flag=1,c_lim=[1500,4500],Title='new_initial_model2, R2(m,true)='+numstr(F_r2(new_initial_model2,MODELS)),Save_pictures_path=results_folder)
        # Plot_image(new_initial_model3.T, Show_flag=0,Save_flag=1,c_lim=[1500,4500],Title='new_initial_model3, R2(m,true)='+numstr(F_r2(new_initial_model3,MODELS)),Save_pictures_path=results_folder)
        # Plot_image(new_initial_model4.T, Show_flag=0,Save_flag=1,c_lim=[1500,4500],Title='new_initial_model4, R2(m,true)='+numstr(F_r2(new_initial_model4,MODELS)),Save_pictures_path=results_folder)
        # Plot_image(new_initial_model5.T, Show_flag=0,Save_flag=1,c_lim=[1500,4500],Title='new_initial_model5, R2(m,true)='+numstr(F_r2(new_initial_model5,MODELS)),Save_pictures_path=results_folder)
        # Plot_image(new_initial_model6.T, Show_flag=0,Save_flag=1,c_lim=[1500,4500],Title='new_initial_model6, R2(m,true)='+numstr(F_r2(new_initial_model6,MODELS)),Save_pictures_path=results_folder)
        # Plot_image(new_initial_model7.T, Show_flag=0,Save_flag=1,c_lim=[1500,4500],Title='new_initial_model7, R2(m,true)='+numstr(F_r2(new_initial_model7,MODELS)),Save_pictures_path=results_folder)
        ######################################################  chosen initial model
        if 'cgg' in base_velocity_models_file:
            # MODELS_INIT_FINAL=predicted_initial3    
            MODELS_INIT_FINAL=predicted_initial2   #    lower box condition applied
        else:
            new_initial_model8=F_smooth(predicted_initial2,sigma_val=int(300/pars['dx']))
            new_initial_model8[taper==0]=1500
            # MODELS_INIT_FINAL=predicted_initial   #    use prediction with no further processing
            # MODELS_INIT_FINAL=predicted_initial2   #    lower box condition applied
            MODELS_INIT_FINAL=new_initial_model8   #    lower box condition applied
        # MODELS_INIT_FINAL=predicted_initial
        tmp=np.copy(predicted_initial)
        Plot_image(tmp.T, Show_flag=0,Save_flag=1,c_lim=[1500,3000],Title='tmp_'+numstr(F_r2(predicted_initial2,MODELS)),Save_pictures_path=results_folder)
        tmp2=scipy.ndimage.filters.gaussian_filter1d(tmp,20,axis=0)
        tmp3=np.copy(tmp)
        tmp3[:,90:]=scipy.ndimage.filters.gaussian_filter1d(tmp[:,90:],130,axis=0)
        Plot_image(tmp2.T, Show_flag=0,Save_flag=1,c_lim=[1500,3000],
            Title='tmp2_'+numstr(F_r2(predicted_initial2,MODELS)),Save_pictures_path=results_folder)
        Plot_image(tmp3.T, Show_flag=0,Save_flag=1,c_lim=[1500,3000],
            Title='tmp3_'+numstr(F_r2(predicted_initial2,MODELS)),Save_pictures_path=results_folder)
        tmp4=F_smooth(tmp3,sigma_val=int(300/pars['dx']))
        Plot_image(tmp4.T, Show_flag=0,Save_flag=1,c_lim=[1500,3000],
            Title='tmp4_'+numstr(F_r2(predicted_initial2,MODELS)),Save_pictures_path=results_folder)
        MODELS_INIT_FINAL=tmp4
        MODELS_INIT_FINAL[taper==0]=1500
        ######################################################
        Plot_image(predicted_initial2.T, Show_flag=0,Save_flag=1,c_lim=[1500,3000],Title='predicted_initial2_R2_m_true_3000_'+numstr(F_r2(predicted_initial2,MODELS)),Save_pictures_path=results_folder)
        Plot_image(predicted_initial2.T, Show_flag=0,Save_flag=1,c_lim=[1500,4500],Title='predicted_initial2_R2_m_true_4500_'+numstr(F_r2(predicted_initial2,MODELS)),Save_pictures_path=results_folder)
        Plot_image(predicted_initial3.T, Show_flag=0,Save_flag=1,c_lim=[1500,3000],Title='predicted_initial3_R2_m_true_3000_'+numstr(F_r2(predicted_initial3,MODELS)),Save_pictures_path=results_folder)
        Plot_image(predicted_initial3.T, Show_flag=0,Save_flag=1,c_lim=[1500,4500],Title='predicted_initial3_R2_m_true_4500_'+numstr(F_r2(predicted_initial3,MODELS)),Save_pictures_path=results_folder)
        Plot_image(MODELS_INIT_FINAL.T, Show_flag=0,Save_flag=1,c_lim=[1500,3000],Title='chosen init. model, R2(m,true)_3000='+numstr(F_r2(MODELS_INIT_FINAL,MODELS)),Save_pictures_path=results_folder)
        Plot_image(MODELS_INIT_FINAL.T, Show_flag=0,Save_flag=1,c_lim=[1500,4500],Title='chosen init. model, R2(m,true)_4500='+numstr(F_r2(MODELS_INIT_FINAL,MODELS)),Save_pictures_path=results_folder)
        ######################################################
        # Plot_image(model_picked_up_from_previous_stage2.T, Show_flag=0,Save_flag=1,Title='m2_color_R2(m,true)='+numstr(F_r2(model_picked_up_from_previous_stage2,MODELS)),Save_pictures_path=results_folder)
        # Plot_image(model_picked_up_from_previous_stage2.T, Show_flag=0,Save_flag=1,c_lim=[vp1,vp2],Title='m2, R2(m,true)='+numstr(F_r2(model_picked_up_from_previous_stage2,MODELS)),Save_pictures_path=results_folder)
        Plot_image(new_initial_model.T, Show_flag=0,Save_flag=1,c_lim=[1500,3000],Title='new_initial_model3000, R2(m,true)='+numstr(F_r2(new_initial_model,MODELS)),Save_pictures_path=results_folder)
        Plot_image(parameters['Models_init'].T, Show_flag=0,Save_flag=1,c_lim=[1500,3000],Title='old_initial_model3000, R2(m,true)='+numstr(F_r2(parameters['Models_init'],MODELS)),Save_pictures_path=results_folder)
        Plot_image(new_initial_model.T, Show_flag=0,Save_flag=1,c_lim=[1500,4500],Title='new_initial_model4500, R2(m,true)='+numstr(F_r2(new_initial_model,MODELS)),Save_pictures_path=results_folder)
        Plot_image(parameters['Models_init'].T, Show_flag=0,Save_flag=1,c_lim=[1500,4500],Title='old_initial_model4500, R2(m,true)='+numstr(F_r2(parameters['Models_init'],MODELS)),Save_pictures_path=results_folder)

        Plot_image(new_initial_model.T-parameters['Models_init'].T, Show_flag=0,Save_flag=1,Title='initial_model_diff',Save_pictures_path=results_folder)
        Plot_image(img_A[0,9,:,:].T, Show_flag=0,Save_flag=1,Title='m_i_ra10',Save_pictures_path=results_folder)
        # Plot_image(parameters['Models_init'].T, Show_flag=0,Save_flag=1,c_lim=[1500,3000],Title='initial_model_old',Save_pictures_path=results_folder)
        # Plot_image(new_initial_model.T, Show_flag=0,Save_flag=1,c_lim=[1500,3000],Title='initial_model_new',Save_pictures_path=results_folder)
        # Plot_image(new_initial_model.T, Show_flag=0,Save_flag=1,c_lim=[1500,4500],Title='initial_model_new1500_4500',Save_pictures_path=results_folder)
        Plot_image(fake_B2.T, Show_flag=0,Save_flag=1,Title='fake_B2',Save_pictures_path=results_folder)
        Plot_image(fake_B3.T, Show_flag=0,Save_flag=1,Title='fake_B3',Save_pictures_path=results_folder)
        Plot_image(fake_B4.T, Show_flag=0,Save_flag=1,Title='fake_B4',Save_pictures_path=results_folder)
        Plot_image(MODELS.T, Show_flag=0,Save_flag=1,c_lim=[vp1,vp2],Title='true',Save_pictures_path=results_folder)
        ######################################################
    #   create initial velocity model file for the fwi stage 
    # exit()
    MODELS_INIT_FINAL[taper==0]=1500
    velocity_models_file=os.path.join(results_folder,'velocity_models_file.hdf5')
    f = h5py.File(velocity_models_file,'w')
    f.create_dataset('models',data=MODELS)
    f.create_dataset('models_init',data=MODELS_INIT_FINAL)
    f.create_dataset('water_taper',data=water_taper)
    f.create_dataset('dx', data=pars['dx'])
    f.create_dataset('dz', data=pars['dz']);f.close()
    print('finished.halas!!')
    res={'models': MODELS, 'models_init': MODELS_INIT_FINAL,
        'water_taper':water_taper,'dx':pars['dx']}
    return res
def copy_su_from_to(path_from, path_to):
    """ Copy files matching *.su.* from one folder to another folder"""
    os.makedirs(path_to, exist_ok=True)
    su_files = glob(path_from + '/*.su.*')
    su_files = [f for f in su_files if '.it' not in f]
    print(f'Found {len(su_files)} *.su.* files in {path_from}')
    commands = []
    print(f'Copy files from {path_from} to {path_to}')
    for f in su_files:
        os.system(f'cp {f} {path_to}')
def cleaning_script(results_folder):
    """clear the denise folder in order not to overload shaheen file limit  """
    cleaning_commands=[]
    # cleaning_commands.append(f"rm -f {os.path.join(results_folder,'fld','su')+'/*'}\n")
    cleaning_commands.append(f"rm -f {os.path.join(results_folder,'fld','start')+'/*su.shot*'}\n")
    cleaning_commands.append(f"rm -f {os.path.join(results_folder,'fld','start')+'/*.denise.*'}\n")
    cleaning_commands.append(f"rm -f {os.path.join(results_folder,'fld','taper/*.bin.*')}\n")
    cleaning_commands.append(f"rm -f {os.path.join(results_folder,'fld','model/*.bin.*')}\n")
    cleaning_commands.append(f"rm -f {os.path.join(results_folder,'fld','jacobian/*.bin.*')}\n")
    cleaning_commands.append(f"rm -f {os.path.join(results_folder,'fld','jacobian/*.old.*')}\n")
    cleaning_commands.append(f"rm -r {os.path.join(results_folder,'fld','log')}\n")
    cleaning_commands.append(f"rm -r {os.path.join(results_folder,'fld','gravity')}\n")
    cleaning_commands.append(f"rm -r {os.path.join(results_folder,'fld','picked_times')}\n")
    cleaning_commands.append(f"rm -r {os.path.join(results_folder,'fld','snap')}\n")
    cleaning_commands.append(f"rm -r {os.path.join(results_folder,'fld','receiver')}\n")
    cleaning_commands.append(f"rm -r {os.path.join(results_folder,'fld','source')}\n")
    cleaning_commands.append(f"find {os.path.join(results_folder,'fld')}  -empty -type d -delete\n")
    # cleaning_commands.append(f"rm -r {os.path.join(results_folder,'fld','start')}\n")
    # cleaning_commands.append(f"rm -r {os.path.join(results_folder,'fld','taper')}\n")
    return cleaning_commands
def extend(x, ez, ex):
    """ x shape=[Nz,Nx] """
    if ex > 0:
        x = np.concatenate((x, np.flip(x[:, -ex:], -1)), 1)              # OX
    if ez > 0:
        x = np.concatenate((x, x.min() * np.ones((ez, x.shape[1]))), 0)  # OZ
    return x
def denise_fwi(models_path,results_path,home_directory,calculation_spacing=50,pars=dict(),mode='plotting',plot_flag=0):
    """   create standard Denise folder for starting FWI.    """
    #########################   start logging
    f=open(os.path.join(results_path,'log_mode_'+pars['data_mode']+'.txt'), 'w')
    sys.stdout=Tee(sys.stdout,f)
    T1=datetime.datetime.now();
    print('Start of program time=',T1)
    #########################   setting paths
    print('simulation parameters=',pars)
    d=api.Denise(pars['root_denise'],verbose=1)
    d.INV_MOD_OUT=1
    d.save_folder=os.path.join(results_path,'fld','')   #    api._cmd('rm -r '+d.save_folder)
    d.set_paths(makedirs=True)
    #########################   Loading velocity from file
    print('loading vel models from',models_path)
    tmp=load_file(models_path,'models')
    if np.max(tmp)<10:
        units='km/sec'
        unit_coefficient=1
    else:
        units='m/sec'
        unit_coefficient=10e-4
    print(units)
    Models_orig=load_file(models_path,'models')* 1000*unit_coefficient
    Models_init_orig=load_file(models_path,'models_init')* 1000*unit_coefficient
    water_taper=load_file(models_path,'water_taper')
    #########################   identify the name of velocity model
    tmp1=models_path.split('/')[-1]
    tmp2=tmp1.split('.hdf5')[0]
    tmp3=tmp2.split('model__')[-1]
    velocity_model_name=tmp3
    if 'cgg' in velocity_model_name or pars['gen_mode']=='test_real_data':
        real_data_flag=1
    else:   
        real_data_flag=0
    print('processing real data=',real_data_flag)
    #########################
    nx_orig,nz_orig=Models_orig.shape
    if plot_flag==1:
        Plot_image(water_taper.T, Show_flag=0,Save_flag=1,Title='water_taper_',Save_pictures_path=os.path.join(d.save_folder,'pictures'))
        Plot_image(Models_orig.T, Show_flag=0,Save_flag=1,Title='Models_orig_',Save_pictures_path=os.path.join(d.save_folder,'pictures'))
        Plot_image(Models_init_orig.T, Show_flag=0,Save_flag=1,Title='Models_init_orig_',Save_pictures_path=os.path.join(d.save_folder,'pictures'))
    #########################   calculate coordinate of last source
    if pars['last_source_position']=='nx':
        nx_last_source_position=nx_orig
    else:
        nx_last_source_position=pars['last_source_position']
    #########################   extend model in x direction
    if pars['extend_model_x']==True:
        Models_orig = ( extend(Models_orig.T, 0, 320) ).T
        Models_init_orig = ( extend(Models_init_orig.T, 0, 320) ).T
        water_taper = ( extend(water_taper.T, 0, 320) ).T
    dx = load_file(models_path, 'dx'); dx = float(dx)
    dz = load_file(models_path, 'dz'); dz = float(dz)
    ########    resizing if needed
    dx_new=calculation_spacing;  dz_new=calculation_spacing
    Models=Models_orig; Models_init=Models_init_orig
    #########################   adaptation of old variables to new denise paradigm
    dx=dx_new;  dz=dz_new;  
    n1,n2 = Models.shape;
    if n1<n2:   nx=n2;  nz=n1
    else:       nx=n1;  nz=n2
    #########################   Divide computation area between processors, if nx,nz is not dividable by number of processors, crop the area
    print(f'nx:\t{nx}');   print(f'nz:\t{nz}')
    #########################
    # d.NPROCX=2;   d.NPROCY=1
    d.NPROCX=pars['NPROCX'];   d.NPROCY = pars['NPROCY']
    ######################### domain division
    import multiprocessing
    n_proc=(multiprocessing.cpu_count()/(d.NPROCX*d.NPROCY)-1)*d.NPROCX*d.NPROCY
    n_proc=32
    if not (nx/d.NPROCX).is_integer():
        nx_new=int(np.floor(nx/d.NPROCX)*d.NPROCX    )
        print('nx/d.NPROCX is not whole number')
    else:   nx_new=int(nx)
    if not (nz/d.NPROCY).is_integer():
        nz_new=int(np.floor(nz/d.NPROCY)*d.NPROCY)
        print('ny/d.NPROCY is not whole number')
    else:        nz_new=int(nz)
    if not (n_proc/d.NPROCY/d.NPROCX).is_integer():
        print('n_proc/d.NPROCY/d.NPROCX is not whole number')
    #########################
    Models=Models[0:nx_new,0:nz_new]
    Models_init=Models_init[0:nx_new,0:nz_new]
    water_taper=water_taper[0:nx_new,0:nz_new]
    #   move below make_vp_vs_rho!!!!!!!!!!!!!!!!!
    #########################   fix taper units notation: water=0,rocks=1
    if water_taper[0,-1]==0:    
        water_taper=(-water_taper+1)
    else:   
        print(water_taper.shape)
    #########################   shift taper for better gradient calculation
    taper_shift=int(pars['taper_shift']/pars['dz'])
    if taper_shift!=0:
        water_taper_shifted=np.roll(water_taper,-taper_shift,axis=1)
        water_taper_shifted[:,-taper_shift:]=1
    else:   water_taper_shifted=np.copy(water_taper)
    test=Models*water_taper
    ########################    calculate minimum depth in shifted water taper
    tmp=np.empty(( water_taper_shifted.shape[0] ))
    for i in range(water_taper_shifted.shape[0]):
        aa=np.where(water_taper_shifted[i,:]==0)
        tmp[i]=( aa[0].max() )
    water_sz=tmp.min(); 
    #########################   Flip into denise format. for better plotting
    vp=np.flipud(Models.T)
    nz, nx = vp.shape
    vp_init=np.flipud(Models_init.T)
    water_taper=np.flipud(water_taper.T)
    water_taper_shifted=np.flipud(water_taper_shifted.T)
    #########################
    vp,vs,rho=make_vp_vs_rho(vp,water_taper)
    # plot_model(np.concatenate((vp,vs,rho), 1),folder_path=os.path.join(d.save_folder,'pictures'),file_path='true_models.png')
    model = api.Model(vp, vs, rho, dx)
    d.set_model(model)  #   set before forward modelling
    vplim = {'vmax': model.vp.max(), 'vmin': model.vp.min()}
    vslim = {'vmax': model.vs.max(), 'vmin': model.vs.min()}
    rholim ={'vmax':model.rho.max(), 'vmin': model.rho.min()}
    vlims = {'vp': vplim, 'vs': vslim, 'rho': rholim}
    vp_init,vs_init,rho_init=make_vp_vs_rho(vp_init,water_taper)
    model_init = api.Model(vp_init,vs_init,rho_init,dx)
    # plot_model(np.concatenate((vp_init,vs_init,rho_init), 1),**vlims['vp'],folder_path=os.path.join(d.save_folder,'pictures'),file_path='.png')
    ########################    set physics and velocity component to record
    d.PHYSICS=1
    d.SEISMO=2
    d.QUELLTYPB=4
    ########################    set box constraints
    if real_data_flag==1:
        d.VPUPPERLIM=model.vp.max()+1000
        d.VSUPPERLIM=model.vs.max()+500
        d.RHOUPPERLIM=model.rho.max()+500
        d.VPLOWERLIM=model.vp.min()
        d.VSLOWERLIM=model.vs.min()
        d.RHOLOWERLIM=model.rho.min()
    else:
        d.VPUPPERLIM=model.vp.max()
        d.VSUPPERLIM=model.vs.max()
        d.RHOUPPERLIM=model.rho.max()
        d.VPLOWERLIM=model.vp.min()
        d.VSLOWERLIM=model.vs.min()
        d.RHOLOWERLIM=model.rho.min()
    ########################    create taper file for fwi (SWS_TAPER_FILE).custom tapering. Option 1. Apply shifted taper to gradients. 
    d.SWS_TAPER_FILE = 1
    d.set_taper(water_taper_shifted)
    ########################    create taper file for fwi (SWS_TAPER_FILE).horizontal taper
    d.SWS_TAPER_GRAD_HOR=1
    # # d.EXP_TAPER_GRAD_HOR=3  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! Important parameter
    # water_sz=int(400/d.DH)    before. old method. in training dataset
    d.GRADT1=int(water_sz-6)
    d.GRADT2=int(d.GRADT1+5)
    d.GRADT3=490;    d.GRADT4=500
    ########################    set dt
    d._calc_nt_dt()
    calculated_dt=d.DT
    print('Calculated dt=',d.DT)
    d.TIME=6.0
    ########################   record snapshotsssh -L 12490:gpu213-10:12490 plotnips@glogin.ibex.kaust.edu.sa
    if real_data_flag==1:
        with open(info_file,'rb') as input:
            acq_data=pickle.load(input)
        #### acq_data = {'src':src,'rec':rec,'wavelet':bpw,'src_x0':src_x0,'rec_x0':rec_x0,'dsrc':dsrc,'drec':drec,'dDT':dDT}
        src=acq_data['src']
        rec=acq_data['rec']
        bpw=acq_data['wavelet']
        src_x0=acq_data['src_x0']
        rec_x0=acq_data['rec_x0']
        dsrc=acq_data['dsrc']
        drec=acq_data['drec']
        d.DT=acq_data['dDT'] 
        log_dict=acq_data['log_dict']
        d.QUELLART = 3  
        d.WRITE_STF = 0
        d.REC_INCR_X = dsrc
        d.N_STREAMER=len(rec)
        ###### copy_su_from_to('./for_pasha/out_for_pasha/su/field',os.path.join(d.save_folder,'su'))
    else:
        info_file=os.path.join('./data/acq_data_parameters_cgg_correct.pkl') 
        if os.path.exists(info_file):
            with open(info_file,'rb') as input:
                acq_data=pickle.load(input)
        Acquisition_type='OBN'
        Acquisition_type='Marine'
        if Acquisition_type=='OBN':
            ###################     OBN acquisition.set sources, receivers
            drec = 20.
            dsrc = 1*160.  #55 or 10 or 2
            if pars['data_mode']=='fwi_intermediate_smoothing_test':
                dsrc = 30*160.  #55 or 10 or 2
            depth_src = 40.  # source depth [m]
            depth_rec=(water_sz+1)*dz
            xsrc1 = 800.  # 1st source position [m]
            xsrc2 = nx*dx-xsrc1
            xrec1 = 200.
            xrec2 = nx*dx-xrec1
        elif Acquisition_type=='Marine':
            ###################     Marine acquisition.set sources, receivers
            drec = 20.
            dsrc = 200
            # dsrc = 320
            depth_src=40
            depth_rec =60.
            distance_between_streamer_and_receivers=140
            # xsrc1 = 800.  # xsrc1 - 1st source position [m]
            xsrc1=int(np.ceil(0.05*nx))*dx      #5% offset 
            xsrc2 = nx*dx-xsrc1
            xrec1 = xsrc1+distance_between_streamer_and_receivers
            streamer_length=8000
            if streamer_length>nx*dx:
                streamer_length=nx*dx-1000
            xrec2 = xrec1+streamer_length
            if pars['data_gen_mode']=='pseudo_field':       #   CGG_NW_Australia
                d.QUELLART = 3  
                d.WRITE_STF = 0
                # hasattr(acq_data,'drec')
                if 'dDT' in acq_data.keys():
                    d.DT = acq_data['dDT']
                else:
                    d.DT = acq_data['DT']
                d.DT=calculated_dt
                d.DT=0.002
                d.NT=6/d.DT
                d.NT = 3000
                drec=acq_data['drec']
                xrec1=acq_data['rec_x0']
                rec_=acq_data['rec']
                depth_rec=rec_.y[0]
                # dsrc=acq_data['dsrc']  
                dsrc=pars['dsrc']
                if 'short_version' in pars['data_mode']:
                    dsrc=6000
                xsrc1=acq_data['src_x0']
                xsrc2 = nx_last_source_position*dx-xsrc1
                src_=acq_data['src']
                depth_src=src_.y[0]
        ###################  Wrap into sources,receivers arrays
        xrec = np.arange(xrec1, xrec2+dx,drec)
        yrec = depth_rec * (xrec / xrec)
        xsrc = np.arange(xsrc1,xsrc2+dx,dsrc)
        ysrc = depth_src* xsrc/xsrc
        ################### Wrap into api
        src = api.Sources(xsrc,ysrc)
        if pars['data_gen_mode']=='pseudo_field':
            ###################
            rec=acq_data['rec']
            wavelet=acq_data['wavelet']
            src.wavelets=np.repeat(wavelet,len(src),0)
        else:
            rec = api.Receivers(xrec,yrec)
        if Acquisition_type=='Marine':
            d.REC_INCR_X = dsrc
            d.N_STREAMER=len(rec)
    ###################  fwi stages
    replace_strategy_for_true_smoothed_model=0
    d.fwi_stages = []
    if pars['data_mode']=='cnn_13':
        d.STEPMAX=200
        ################### gradient step length estimation
        ##########  testshots for gradient step length estimation,    (TESTSHOT_START,TESTSHOT_END,TESTSHOT_INCR) = 1,17,2
        d.TESTSHOT_START=1; d.TESTSHOT_END=len(src._ones);  d.TESTSHOT_INCR=5
        d.EPS_SCALE=0.0001
        d.SCALEFAC=1.2
        ################### seismic data bandwidth
        TIME_FILT=1
        d.ITERMAX=5
        ################### define fwi stages
        ########### make strong smoothing on low frequency data. [0,6] hz data
        freq=8; wd_damp=0
        d.add_fwi_stage(fc_low=0.0,fc_high=freq,time_filt=TIME_FILT,normalize=2,
            e_precond=3,spatfilter=0,pro=1e-2,
            wd_damp=wd_damp,wd_damp1=wd_damp)
        freq=10; wd_damp=0
        d.add_fwi_stage(fc_low=0.0,fc_high=freq,time_filt=TIME_FILT,normalize=2,
            e_precond=3,spatfilter=0,pro=1e-5,
            wd_damp=wd_damp,wd_damp1=wd_damp)
        for i,stage in enumerate(d.fwi_stages):
            print(f'Stage {i+1}:\n\t{stage}\n')
    elif pars['data_mode']=='cnn_13_for_real_data':
        d.EXP_TAPER_GRAD_HOR=3
        ################### gradient step length estimation
        ##########  testshots for gradient step length estimation,    (TESTSHOT_START,TESTSHOT_END,TESTSHOT_INCR) = 1,17,2
        d.TESTSHOT_START=1; d.TESTSHOT_END=len(src._ones);  d.TESTSHOT_INCR=5
        ################### seismic data bandwidth
        d.FC_SPIKE_1=7
        d.FC_SPIKE_2=15
        TIME_FILT=1
        d.ITERMAX=5
        ################### define fwi stages
        ########### make strong smoothing on low frequency data. [0,6] hz data
        freq=8; wd_damp=0
        d.add_fwi_stage(fc_low=0.0,fc_high=freq,time_filt=TIME_FILT,normalize=2,
            e_precond=3,spatfilter=0,pro=1e-2,
            wd_damp=wd_damp,wd_damp1=wd_damp)
        freq=10; wd_damp=0
        d.add_fwi_stage(fc_low=0.0,fc_high=freq,time_filt=TIME_FILT,normalize=2,
            e_precond=3,spatfilter=0,pro=1e-2,
            wd_damp=wd_damp,wd_damp1=wd_damp)
        for i,stage in enumerate(d.fwi_stages):
            print(f'Stage {i+1}:\n\t{stage}\n')
    elif pars['data_mode']=='cnn_13_special':
        d.EXP_TAPER_GRAD_HOR=3
        d.STEPMAX=200
        ################### gradient step length estimation
        ##########  testshots for gradient step length estimation,    (TESTSHOT_START,TESTSHOT_END,TESTSHOT_INCR) = 1,17,2
        d.TESTSHOT_START=1; d.TESTSHOT_END=len(src._ones);  d.TESTSHOT_INCR=5
        d.EPS_SCALE=0.0001
        d.SCALEFAC=1.2
        ################### seismic data bandwidth
        d.FC_SPIKE_1=7
        d.FC_SPIKE_2=15
        TIME_FILT=1
        d.ITERMAX=5
        ################### define fwi stages
        ########### make strong smoothing on low frequency data. [0,6] hz data
        freq=8; wd_damp=0
        d.add_fwi_stage(fc_low=0.0,fc_high=freq,time_filt=TIME_FILT,normalize=2,
            e_precond=3,spatfilter=0,pro=1e-10,
            wd_damp=wd_damp,wd_damp1=wd_damp)
        freq=10; wd_damp=0
        d.add_fwi_stage(fc_low=0.0,fc_high=freq,time_filt=TIME_FILT,normalize=2,
            e_precond=3,spatfilter=0,pro=1e-10,
            wd_damp=wd_damp,wd_damp1=wd_damp)
        for i,stage in enumerate(d.fwi_stages):
            print(f'Stage {i+1}:\n\t{stage}\n')
    elif pars['data_mode']=='cnn_13_short_version':
        d.EXP_TAPER_GRAD_HOR=3
        d.STEPMAX=200
        ################### gradient step length estimation
        ##########  testshots for gradient step length estimation,    (TESTSHOT_START,TESTSHOT_END,TESTSHOT_INCR) = 1,17,2
        d.TESTSHOT_START=1; d.TESTSHOT_END=len(src._ones);  d.TESTSHOT_INCR=5
        d.EPS_SCALE=0.0001
        d.SCALEFAC=1.2
        ################### seismic data bandwidth
        d.FC_SPIKE_1=7
        d.FC_SPIKE_2=15
        TIME_FILT=1
        d.ITERMAX=1
        ################### define fwi stages
        ########### make strong smoothing on low frequency data. [0,6] hz data
        freq=8; wd_damp=0
        d.add_fwi_stage(fc_low=0.0,fc_high=freq,time_filt=TIME_FILT,normalize=2,
            e_precond=3,spatfilter=0,pro=1e-2,
            wd_damp=wd_damp,wd_damp1=wd_damp)
        # freq=10; wd_damp=0
        # d.add_fwi_stage(fc_low=0.0,fc_high=freq,time_filt=TIME_FILT,normalize=2,
        #     e_precond=3,spatfilter=0,pro=1e-5,
        #     wd_damp=wd_damp,wd_damp1=wd_damp)
        for i,stage in enumerate(d.fwi_stages):
            print(f'Stage {i+1}:\n\t{stage}\n')
    elif pars['data_mode']=='cnn_16':   #   new strategy. 23.11.21
        # d.EXP_TAPER_GRAD_HOR=3
        d.STEPMAX=200
        ################### gradient step length estimation
        ##########  testshots for gradient step length estimation,    (TESTSHOT_START,TESTSHOT_END,TESTSHOT_INCR) = 1,17,2
        d.TESTSHOT_START=1; d.TESTSHOT_END=len(src._ones);  d.TESTSHOT_INCR=5
        ################### seismic data bandwidth
        TIME_FILT=1
        d.ITERMAX=5
        ################### define fwi stages
        ########### make strong smoothing on low frequency data. [0,6] hz data
        freq=8; wd_damp=0
        d.add_fwi_stage(fc_low=0.0,fc_high=freq,time_filt=TIME_FILT,normalize=2,
            e_precond=3,spatfilter=0,pro=1e-10,
            wd_damp=wd_damp,wd_damp1=wd_damp)
        freq=10; wd_damp=0
        d.add_fwi_stage(fc_low=0.0,fc_high=freq,time_filt=TIME_FILT,normalize=2,
            e_precond=3,spatfilter=0,pro=1e-10,
            wd_damp=wd_damp,wd_damp1=wd_damp)
        for i,stage in enumerate(d.fwi_stages):
            print(f'Stage {i+1}:\n\t{stage}\n')
    elif pars['data_mode']=='fwi_56_strategy_l2':   
        d.ITERMAX=150
        par_stages_fwi={'inv_vs_iter': 3,
                        'inv_rho_iter':5,
                        'normalize': 2,
                        'order': 6,
                        'time_filt':1}
        freqs_low = [0, 0, 0, 0,0]
        freqs_high = [6, 6, 7,8,8]
        grad_smoothing = [1.5, 1, 0.5,0.25, 0.125]
        spatfilters = [4, 4, 4,4,4]
        pro=[1e-2,1e-2,1e-2,1e-2,1e-2]
        LNORM=2
        # LNORM = 2 # LNORM = 5 - global correlation norm (Choi & Alkhalifah 2012)  # LNORM = 6 - envelope objective functions after Chi, Dong and Liu (2014) - EXPERIMENTAL; LNORM = 7 - NIM objective function after Chauris et al. (2012) and Tejero et al. (2015) - EXPERIMENTAL
        for i,freq in enumerate(freqs_high):
            d.add_fwi_stage(fc_low=freqs_low[i],
                            fc_high=freq, 
                            wd_damp=grad_smoothing[i],
                            wd_damp1=grad_smoothing[i],
                            spatfilter=spatfilters[i],
                            lnorm=LNORM,pro=pro[i],**par_stages_fwi)
        for i,fwi_stage in enumerate(d.fwi_stages):
            print(f'Stage {i+1}:\n\t{fwi_stage}\n')
    elif pars['data_mode']=='fwi_56_strategy_l5':   
        d.ITERMAX=150
        par_stages_fwi={'inv_vs_iter': 3,
                        'inv_rho_iter':5,
                        'normalize': 2,
                        'order': 6,
                        'time_filt':1}
        freqs_low = [0, 0, 0, 0,0]
        freqs_high = [6, 6, 7,8,8]
        grad_smoothing = [1.5, 1, 0.5,0.25, 0.125]
        spatfilters = [4, 4, 4,4,4]
        pro=[1e-2,1e-2,1e-2,1e-2,1e-2]
        LNORM=5
        # LNORM = 2 # LNORM = 5 - global correlation norm (Choi & Alkhalifah 2012)  # LNORM = 6 - envelope objective functions after Chi, Dong and Liu (2014) - EXPERIMENTAL; LNORM = 7 - NIM objective function after Chauris et al. (2012) and Tejero et al. (2015) - EXPERIMENTAL
        for i,freq in enumerate(freqs_high):
            d.add_fwi_stage(fc_low=freqs_low[i],
                            fc_high=freq, 
                            wd_damp=grad_smoothing[i],
                            wd_damp1=grad_smoothing[i],
                            spatfilter=spatfilters[i],
                            lnorm=LNORM,pro=pro[i],**par_stages_fwi)
        for i,fwi_stage in enumerate(d.fwi_stages):
            print(f'Stage {i+1}:\n\t{fwi_stage}\n')
    elif pars['data_mode']=='fwi_56_strategy_special_l2':   
        d.ITERMAX=150
        d.EXP_TAPER_GRAD_HOR=3
        par_stages_fwi={'inv_vs_iter': 3,
                        'inv_rho_iter':5,
                        'normalize': 2,
                        'order': 6,
                        'time_filt':1}
        freqs_low = [0, 0, 0, 0,0]
        freqs_high = [6, 6, 7,8,8]
        grad_smoothing = [1.5, 1, 0.5,0.25, 0.125]
        spatfilters = [4, 4, 4,4,4]
        pro=[1e-2,1e-2,1e-2,1e-2,1e-2]
        LNORM=2
        # LNORM = 2 # LNORM = 5 - global correlation norm (Choi & Alkhalifah 2012)  # LNORM = 6 - envelope objective functions after Chi, Dong and Liu (2014) - EXPERIMENTAL; LNORM = 7 - NIM objective function after Chauris et al. (2012) and Tejero et al. (2015) - EXPERIMENTAL
        for i,freq in enumerate(freqs_high):
            d.add_fwi_stage(fc_low=freqs_low[i],
                            fc_high=freq, 
                            wd_damp=grad_smoothing[i],
                            wd_damp1=grad_smoothing[i],
                            spatfilter=spatfilters[i],
                            lnorm=LNORM,pro=pro[i],**par_stages_fwi)
        for i,fwi_stage in enumerate(d.fwi_stages):
            print(f'Stage {i+1}:\n\t{fwi_stage}\n')
    elif pars['data_mode']=='fwi_56_strategy_special_l5':   
        d.ITERMAX=150
        d.EXP_TAPER_GRAD_HOR=3
        par_stages_fwi={'inv_vs_iter': 3,
                        'inv_rho_iter':5,
                        'normalize': 2,
                        'order': 6,
                        'time_filt':1}
        freqs_low = [0, 0, 0, 0,0]
        freqs_high = [6, 6, 7,8,8]
        grad_smoothing = [1.5, 1, 0.5,0.25, 0.125]
        spatfilters = [4, 4, 4,4,4]
        pro=[1e-2,1e-2,1e-2,1e-2,1e-2]
        LNORM=5
        # LNORM = 2 # LNORM = 5 - global correlation norm (Choi & Alkhalifah 2012)  # LNORM = 6 - envelope objective functions after Chi, Dong and Liu (2014) - EXPERIMENTAL; LNORM = 7 - NIM objective function after Chauris et al. (2012) and Tejero et al. (2015) - EXPERIMENTAL
        for i,freq in enumerate(freqs_high):
            d.add_fwi_stage(fc_low=freqs_low[i],
                            fc_high=freq, 
                            wd_damp=grad_smoothing[i],
                            wd_damp1=grad_smoothing[i],
                            spatfilter=spatfilters[i],
                            lnorm=LNORM,pro=pro[i],**par_stages_fwi)
        for i,fwi_stage in enumerate(d.fwi_stages):
            print(f'Stage {i+1}:\n\t{fwi_stage}\n')
    elif pars['data_mode']=='fwi_60_strategy_fullband_l2':   
        """   cgg full-band FWI workflow  """
        d.ITERMAX=150
        par_stages_fwi={'inv_vs_iter': 0,
                        'inv_rho_iter':0,
                        'normalize': 2,
                        'order': 6,
                        'time_filt':1}
        freqs_low = [0, 0, 0, 0,0,0]
        freqs_high = [3, 4, 5, 6, 7, 8]
        grad_smoothing = [2, 1.5, 1.0, 0.5, 0.25, 0.125]
        spatfilters = [4, 4, 0, 0, 0, 0]
        pro=[1e-2,1e-2,1e-2,1e-2,1e-2,1e-2]
        LNORM=2
        # LNORM = 2 # LNORM = 5 - global correlation norm (Choi & Alkhalifah 2012)  # LNORM = 6 - envelope objective functions after Chi, Dong and Liu (2014) - EXPERIMENTAL; LNORM = 7 - NIM objective function after Chauris et al. (2012) and Tejero et al. (2015) - EXPERIMENTAL
        for i,freq in enumerate(freqs_high):
            d.add_fwi_stage(fc_low=freqs_low[i],
                            fc_high=freq, 
                            wd_damp=grad_smoothing[i],
                            wd_damp1=grad_smoothing[i],
                            spatfilter=spatfilters[i],
                            lnorm=LNORM,pro=pro[i],**par_stages_fwi)
        for i,fwi_stage in enumerate(d.fwi_stages):
            print(f'Stage {i+1}:\n\t{fwi_stage}\n')
    elif pars['data_mode']=='fwi_60_strategy_fullband_l5':   
        """   cgg full-band FWI workflow  """
        d.ITERMAX=150
        par_stages_fwi={'inv_vs_iter': 0,
                        'inv_rho_iter':0,
                        'normalize': 2,
                        'order': 6,
                        'time_filt':1}
        freqs_low = [0, 0, 0, 0,0,0]
        freqs_high = [3, 4, 5, 6, 7, 8]
        grad_smoothing = [2, 1.5, 1.0, 0.5, 0.25, 0.125]
        spatfilters = [4, 4, 0, 0, 0, 0]
        pro=[1e-2,1e-2,1e-2,1e-2,1e-2,1e-2]
        LNORM=5
        # LNORM = 2 # LNORM = 5 - global correlation norm (Choi & Alkhalifah 2012)  # LNORM = 6 - envelope objective functions after Chi, Dong and Liu (2014) - EXPERIMENTAL; LNORM = 7 - NIM objective function after Chauris et al. (2012) and Tejero et al. (2015) - EXPERIMENTAL
        for i,freq in enumerate(freqs_high):
            d.add_fwi_stage(fc_low=freqs_low[i],
                            fc_high=freq, 
                            wd_damp=grad_smoothing[i],
                            wd_damp1=grad_smoothing[i],
                            spatfilter=spatfilters[i],
                            lnorm=LNORM,pro=pro[i],**par_stages_fwi)
        for i,fwi_stage in enumerate(d.fwi_stages):
            print(f'Stage {i+1}:\n\t{fwi_stage}\n')
    elif pars['data_mode']=='fwi_60_strategy_special_fullband_l2':   
        """   cgg full-band FWI workflow  """
        d.EXP_TAPER_GRAD_HOR=3
        d.ITERMAX=150
        par_stages_fwi={'inv_vs_iter': 0,
                        'inv_rho_iter':0,
                        'normalize': 2,
                        'order': 6,
                        'time_filt':1}
        freqs_low = [0, 0, 0, 0,0,0]
        freqs_high = [3, 4, 5, 6, 7, 8]
        grad_smoothing = [2, 1.5, 1.0, 0.5, 0.25, 0.125]
        spatfilters = [4, 4, 0, 0, 0, 0]
        pro=[1e-2,1e-2,1e-2,1e-2,1e-2,1e-2]
        LNORM=2
        # LNORM = 2 # LNORM = 5 - global correlation norm (Choi & Alkhalifah 2012)  # LNORM = 6 - envelope objective functions after Chi, Dong and Liu (2014) - EXPERIMENTAL; LNORM = 7 - NIM objective function after Chauris et al. (2012) and Tejero et al. (2015) - EXPERIMENTAL
        for i,freq in enumerate(freqs_high):
            d.add_fwi_stage(fc_low=freqs_low[i],
                            fc_high=freq, 
                            wd_damp=grad_smoothing[i],
                            wd_damp1=grad_smoothing[i],
                            spatfilter=spatfilters[i],
                            lnorm=LNORM,pro=pro[i],**par_stages_fwi)
        for i,fwi_stage in enumerate(d.fwi_stages):
            print(f'Stage {i+1}:\n\t{fwi_stage}\n')
    elif pars['data_mode']=='fwi_60_strategy_special_fullband_l5':   
        """   cgg full-band FWI workflow  """
        d.EXP_TAPER_GRAD_HOR=3
        d.ITERMAX=150
        par_stages_fwi={'inv_vs_iter': 0,
                        'inv_rho_iter':0,
                        'normalize': 2,
                        'order': 6,
                        'time_filt':1}
        freqs_low = [0, 0, 0, 0,0,0]
        freqs_high = [3, 4, 5, 6, 7, 8]
        grad_smoothing = [2, 1.5, 1.0, 0.5, 0.25, 0.125]
        spatfilters = [4, 4, 0, 0, 0, 0]
        pro=[1e-2,1e-2,1e-2,1e-2,1e-2,1e-2]
        LNORM=5
        # LNORM = 2 # LNORM = 5 - global correlation norm (Choi & Alkhalifah 2012)  # LNORM = 6 - envelope objective functions after Chi, Dong and Liu (2014) - EXPERIMENTAL; LNORM = 7 - NIM objective function after Chauris et al. (2012) and Tejero et al. (2015) - EXPERIMENTAL
        for i,freq in enumerate(freqs_high):
            d.add_fwi_stage(fc_low=freqs_low[i],
                            fc_high=freq, 
                            wd_damp=grad_smoothing[i],
                            wd_damp1=grad_smoothing[i],
                            spatfilter=spatfilters[i],
                            lnorm=LNORM,pro=pro[i],**par_stages_fwi)
        for i,fwi_stage in enumerate(d.fwi_stages):
            print(f'Stage {i+1}:\n\t{fwi_stage}\n')
    if pars['current_data_type']=='true' and replace_strategy_for_true_smoothed_model==1:
        d.fwi_stages=[]
        TIME_FILT=1
        d.ITERMAX=150
        par_stages_fwi = {'normalize': 2,
                 'order': 4,
                 'time_filt':TIME_FILT}
        freqs_high = [6,9,10]
        grad_smoothing = [0,0,0]
        spatfilters = [0,0,0]
        pro=[1e-2,1e-2,1e-2]
        inv_vp_iter= [0,0,0]
        inv_vs_iter= [0,0,0]
        inv_rho_iter=[0,0,0]
        LNORM=5
        repeat_stages_n_times=2
        freqs_high=freqs_high*repeat_stages_n_times
        grad_smoothing=grad_smoothing*repeat_stages_n_times
        spatfilters=spatfilters*repeat_stages_n_times
        pro=pro*repeat_stages_n_times
        inv_vp_iter=inv_vp_iter*repeat_stages_n_times
        inv_vs_iter=inv_vs_iter*repeat_stages_n_times
        inv_rho_iter=inv_rho_iter*repeat_stages_n_times
        # LNORM = 2
        # LNORM = 5 - global correlation norm (Choi & Alkhalifah 2012)
        # LNORM = 6 - envelope objective functions after Chi, Dong and Liu (2014) - EXPERIMENTAL
        # LNORM = 7 - NIM objective function after Chauris et al. (2012) and Tejero et al. (2015) - EXPERIMENTAL
        for i,freq in enumerate(freqs_high):
            d.add_fwi_stage(fc_high=freq, 
                            wd_damp=grad_smoothing[i],
                            wd_damp1=grad_smoothing[i],
                            spatfilter=spatfilters[i],
                            inv_vp_iter= inv_vp_iter[i], 
                            inv_vs_iter= inv_vs_iter[i], 
                            inv_rho_iter=inv_rho_iter[i],
                            lnorm=LNORM,
                            pro=pro[i],
                            **par_stages_fwi)
            print(f'Stage {i+1}:\n\t{d.fwi_stages[i]}\n')
    ###################
    if plot_flag==1:
        plot_acquisition(model.vp,model,src, rec, 'Vp',**vlims['vp'],folder_path=os.path.join(results_path,'pictures'),file_path='model_vp.png')
        plot_acquisition(model.vs,model,src, rec, 'Vs',**vlims['vs'],folder_path=os.path.join(results_path,'pictures'),file_path='model_vs.png')
        plot_acquisition(model.rho,model,src, rec, 'rho',**vlims['rho'],folder_path=os.path.join(results_path,'pictures'),file_path='model_rho.png')
        plot_acquisition(model_init.vp,model,src, rec, 'Vp initial',**vlims['vp'],folder_path=os.path.join(results_path,'pictures'),file_path='model_vp_init.png')
        plot_acquisition(model_init.vs,model,src, rec, 'Vs initial',**vlims['vs'],folder_path=os.path.join(results_path,'pictures'),file_path='model_vs_init.png')
        plot_acquisition(model_init.rho,model,src,rec,'Rho initial',**vlims['rho'],folder_path=os.path.join(results_path,'pictures'),file_path='model_rho_init.png')
        plot_acquisition(water_taper,model,src, rec, 'taper',folder_path=os.path.join(results_path,'pictures'),file_path='model_taper.png')
    ########################
    print(f'NSRC:\t{len(src)}');   print(f'NREC:\t{len(rec)}')
    parallelization_command='mpirun -np '+str(n_proc);  
    if mode=='generate_task_files':
        d.filename=os.path.join(d.save_folder,'seis_forward.inp')
        d.MFILE=os.path.join(d.save_folder,'start/model')
        d.forward(model, src, rec, run_command=parallelization_command,disable=True)
    #   Inspect shots
    flag_plot_shots=0
    if flag_plot_shots==1:
        shots = d.get_shots(keys=['_y'])
        if shots!=[]:
            print(f'Read {len(shots)} shots {shots[0].shape} into list')
            # Plot 2 shots and their respective power spectra
            it_list=[int(np.floor(x)) for x in np.linspace(0, len(shots)-1, 2)]
            it_list=[0]
            for i in it_list:
                plot_shot(shots[i], pclip=0.05, title=str(i),folder_path=os.path.join(d.save_folder,'pictures'),file_path='shot_orig'+str(i)+'.png',show=0)
                freqs,ps=plot_spectrum(shots[i], d.DT, fmax=30,folder_path=os.path.join(d.save_folder,'pictures'),file_path='spectrum_shots_'+str(i)+'.png')
                ss=1
    log_idx = int(model_init.nx/2)
    # plot_logs(model,model_init,log_idx,folder_path=os.path.join(d.save_folder,'pictures'),file_path='logs.png')
    ##################     Create FWI input file. Use default parameters, except high- and low-pass corner frequencies of Butterworth filter.
    if mode=='generate_task_files':
        d.filename=os.path.join(d.save_folder,'seis_inversion.inp')
        d.MFILE=os.path.join(d.save_folder,'start/model_init')
        d.fwi(model_init, src,rec, parallelization_command, disable=True)
    ##################     Create forward modelling, started from inverted initial model, input file. Create forward modelling, started from initial model, input file.
    if real_data_flag==1:
        if mode=='generate_task_files':
            d.filename=os.path.join(d.save_folder,'seis_forward_next_fdmd.inp')
            d.MFILE=os.path.join(d.save_folder,'start/inverted_model')
            # d.DATA_DIR=os.path.join(d.save_folder,'su_modelled/seis')
            os.makedirs(os.path.join(d.save_folder,'su_modelled'),exist_ok=True)
            d.SEIS_FILE_P=os.path.join(d.save_folder,'su_modelled','seis_p.su')
            d.forward(model,src,rec,disable=True,_write_acquisition_flag=0)
            ####
            d.filename=os.path.join(d.save_folder,'seis_forward_next_fdmd_from_init_model.inp')
            d.MFILE=os.path.join(d.save_folder,'start/model_init')
            os.makedirs(os.path.join(d.save_folder,'su_modelled_init_model'),exist_ok=True)
            d.SEIS_FILE_P=os.path.join(d.save_folder,'su_modelled_init_model','seis_p.su')
            d.forward(model_init,src,rec,disable=True,_write_acquisition_flag=0)
    ##################  wrap all data in denise structure and pickle to file
    T2=datetime.datetime.now()
    print('Program finished after',T2-T1);  print('time now=',T2)
    print('processed data in ',results_path)
    d.help()
    return None
def denise_plotting(results_path,pars=None):
    """ results_path- path containing direcory fld (denise 'outputs' folder)"""
    #####################################
    directory=os.path.join(results_path,'fld');
    d = api.Denise(verbose=0)
    d._parse_inp_file(fname=os.path.join(directory,'seis_inversion.inp'))    # 'seis_inversion.inp', 'seis_forward.inp'
    d.filename=os.path.join(directory,'seis_inversion.inp')
    d.save_folder=directory;
    print('Plotting pictures in directory ',os.path.join(d.save_folder,'pictures'))
    os.makedirs(os.path.join(d.save_folder,'pictures'),exist_ok=True)
    api._cmd(f"rm -r {os.path.join(d.save_folder,'pictures/*')}")
    # from pathlib import Path;   [f.unlink() for f in Path(os.path.join(d.save_folder,'pictures')).glob("*") if f.is_file()] 
    #####################################   load src,rec
    os.listdir(results_path)
    if not os.path.exists(os.path.join(results_path,'denise_data.pkl')):
        xrec=np.load(os.path.join(d.save_folder,'receiver','rec_x.npy'))
        yrec=np.load(os.path.join(d.save_folder,'receiver','rec_y.npy'))
        xsrc=np.load(os.path.join(d.save_folder,'source','src_x.npy'))
        ysrc=np.load(os.path.join(d.save_folder,'source','src_y.npy'))
        rec = api.Receivers(xrec, yrec)
        src = api.Sources(xsrc, ysrc)    
    else:
        with open(os.path.join(results_path,'denise_data.pkl'),'rb') as input:
            d = pickle.load(input)
        rec=d.rec
        src=d.src
    #####################################   load velocity models
    dx=d.DH
    vp,fnames = d.get_fwi_start(return_filenames=True,keys=['.vp'])
    vs,fnames = d.get_fwi_start(return_filenames=True,keys=['.vs'])
    rho,fnames = d.get_fwi_start(return_filenames=True,keys=['.rho'])
    model = api.Model(vp[0],vs[0],rho[0], dx)
    vplim = {'vmax': model.vp.max(), 'vmin': model.vp.min()}
    vslim = {'vmax': model.vs.max(), 'vmin': model.vs.min()}
    rholim={'vmax': model.rho.max(), 'vmin': model.rho.min()}
    dv_lim={'vmax': 500, 'vmin': -500}
    vlims = {'vp': vplim, 'vs': vslim,'rho': rholim,'dv': dv_lim}
    vp_init,fnames = d.get_fwi_start(return_filenames=True,keys=['init.vp'])
    vs_init,fnames = d.get_fwi_start(return_filenames=True,keys=['init.vs'])
    rho_init,fnames = d.get_fwi_start(return_filenames=True,keys=['init.rho'])
    model_init = api.Model(vp_init[0],vs_init[0],rho_init[0],dx)
    for component in ['vp','vs','rho']:   
        f_name='init_'+component;   r2='_r2(initial,true)_'+numstr(F_r2(getattr(model_init,component),getattr(model,component)));   
        plot_acquisition(getattr(model_init,component),model,src,rec,f_name+r2,**vlims[component],folder_path=os.path.join(d.save_folder,'pictures'),file_path=f_name+r2+'.png')
        f_name='true'+component+'_'
        plot_acquisition(getattr(model,component),model,src,rec,f_name,**vlims[component],folder_path=os.path.join(d.save_folder,'pictures'),file_path=f_name+'.png')
    ############################7#########   plot wavefields
    plot_wavefields=0
    if plot_wavefields==1 and len(os.listdir(os.path.join(directory,'snap')) )!=0:
        wavefields,fnames=d.get_snapshots(return_filenames=True)
        for m, f in zip(wavefields,fnames):
            for i in np.arange(m.shape[0]):
                f_name=(f.split('/')[-1])+'_snap_'+str(i)
                plot_acquisition(m[i,::],model,src,rec,f_name,folder_path=os.path.join(d.save_folder,'pictures'),file_path=f_name+'.png')
    #####################################   shots
    plot_shots=1
    if plot_shots==1:
        if os.path.exists(os.path.join(directory,'su')):
            shots = d.get_shots(keys=['_y'])
            if shots!=[]:
                print(f'Read {len(shots)} shots {shots[0].shape} into list')
                it_list=[int(np.floor(x)) for x in np.linspace(0, len(shots)-1, 2)]
                it_list=[0]
                for i in it_list:
                    plot_shot(shots[i], pclip=0.05, title=str(i),folder_path=os.path.join(d.save_folder,'pictures'),
                        file_path='shot'+str(i)+'.png',show=0)
                    freqs,ps=plot_spectrum(shots[i], d.DT, fmax=30,folder_path=os.path.join(d.save_folder,'pictures'),file_path='spectrum_shots_'+str(i)+'.png')
            del shots
    plot_models=1
    if plot_models==1:
        ##################  plot gradients
        grads,fnames = d.get_fwi_gradients(return_filenames=True)
        for m,f in zip(grads, fnames):
            f_name=(f.split('/')[-1]).split('.')[0]
            plot_acquisition(m,model,src,rec,f_name,folder_path=os.path.join(d.save_folder,'pictures'),file_path=f_name+'.png')
        ##################  plot inverted models
        # models, fnames = d.get_fwi_models([component, 'stage'], return_filenames=True)
        # plot_model(m ,**vlims[component],folder_path=os.path.join(d.save_folder,'pictures'),file_path=f_name+'.png')
        comps=['vp']
        models,fnames = d.get_fwi_models(return_filenames=True,keys=comps)
        for m, f in zip(models, fnames):
            f_name=(f.split('/')[-1]).split('.')[0]
            # for component in ['vp']:    #,'vs','rho'
            for component in comps:    #,'vs','rho'
                if component in f_name: 
                    r2='_r2(m_i,true)_'+numstr(F_r2(m,getattr(model,component)))
                    plot_acquisition(m,model,src,rec,f_name+r2,**vlims[component],folder_path=os.path.join(d.save_folder,'pictures'),file_path=f_name+r2+'.png')
                    plot_acquisition(m-getattr(model_init,component),model,src,rec,f_name+r2,**vlims['dv'],folder_path=os.path.join(d.save_folder,'pictures'),file_path=f_name+'_dv(m_i,m_init)'+r2+'.png')
                    # plot_acquisition(m-getattr(model,component),model,src,rec,f_name,**vlims['dv'],folder_path=os.path.join(d.save_folder,'pictures'),file_path=f_name+'_difference_with_true.png')
        ##################  plot tapers
        models, fnames = d.get_fwi_tapers(return_filenames=True)
        for m, f in zip(models,fnames):
            f_name=(f.split('/')[-1]).split('.')[0]
            # plot_model(m,folder_path=os.path.join(d.save_folder,'pictures'),file_path=f_name+'.png')
            plot_acquisition(m,model,src,rec,f_name,folder_path=os.path.join(d.save_folder,'pictures'),file_path=f_name+'.png')
    #####################################   plot fwi objective curve
    filename =os.path.join(directory,'seis_fwi_log.dat')
    plot_fwi_misfit(filename,d)
    plot_logs(model,model_init,33,folder_path=os.path.join(d.save_folder,'pictures'),file_path='logs.png')
    #####################################  Plot step length evolution
    plot_fwi_step_length(filename,d)
    #####################################  Record data to hdf5 file
    if pars!=None:
        if pars['record']==1:
            models_,fnames_ = d.get_fwi_models(return_filenames=True,keys='vp.bin') #   vp model on last iteration
            vp_res=models_[0]
            dv_res=vp_res  # model on last iteration - initial fwi model
            parameters={'file_path':os.path.join(pars['dataset_path'],'model_'+results_path.split('/')[-1]+'.npz'),
                'Models':       np.fliplr(getattr(model,'vp').T),
                'Models_init':  np.fliplr(getattr(model_init,'vp').T),
                'input_data':   np.fliplr(dv_res.T),
                'dx':dx,
                'dz':dx}
    return None
def denise_plotting2(d,results_path,pars=None):
    if hasattr(d,'model')==True:
        d=parse_denise_folder(results_path)
    """ results_path- path containing direcory fld (denise 'outputs' folder)"""
    directory=os.path.join(results_path,'fld')
    if 'outputs' in results_path:   directory=results_path
    d.pictures_folder=os.path.join(results_path,'pictures')
    os.makedirs(d.pictures_folder,exist_ok=True)
    print('Plotting pictures in directory ',d.pictures_folder)
    ##################################### lets assume I extended models in fwi for better gradients on 320 nx samples
    nx_orig=d.NX-320
    nx_orig=d.NX
    #####################################
    # api._cmd(f"rm -r {os.path.join(d.save_folder,'pictures')}")
    # from pathlib import Path;   [f.unlink() for f in Path(os.path.join(d.save_folder,'pictures')).glob("*") if f.is_file()] 
    #####################################  Plot fwi objective curve,step length evolution
    plot_misfits=1
    if plot_misfits==1:
        filename =os.path.join(directory,'seis_fwi_log.dat')
        if os.path.exists(filename):
            plot_fwi_misfit(filename,d)
            plot_fwi_step_length(filename,d)
        else:
            if hasattr(d,'misfit_data')==True:
                plot_fwi_misfit(filename,d)
                plot_fwi_step_length(filename,d)
    #####################################   load src,rec
    if not os.path.exists(os.path.join(results_path,'denise_data.pkl')):
        xrec=np.load(os.path.join(d.save_folder,'receiver','rec_x.npy'))
        yrec=np.load(os.path.join(d.save_folder,'receiver','rec_y.npy'))
        xsrc=np.load(os.path.join(d.save_folder,'source','src_x.npy'))
        ysrc=np.load(os.path.join(d.save_folder,'source','src_y.npy'))
        rec = api.Receivers(xrec, yrec)
        src = api.Sources(xsrc, ysrc)    
    else:
        rec=d.rec
        src=d.src
    # models,fnames = d.get_fwi_models(return_filenames=True,keys=['vp'])
    #####################################   load velocity models
    plot_models=1
    if plot_models==1:
        load_from_folder=1
        if hasattr(d,'model')==True:
            if not (getattr(d.model,'vp') is None):
                if getattr(d.model,'vp').size!=0:
                    dx=d.DH
                    vplim = {'vmax':1500,'vmin':4500}
                    vslim = {'vmax': d.model.vs.max(), 'vmin': 0 }
                    rholim={'vmax': d.model.rho.max(), 'vmin': 1000 }
                    dv_lim={'vmax': 500, 'vmin': -500}
                    vlims = {'vp': vplim, 'vs': vslim,'rho': rholim,'dv': dv_lim}
                    for component in ['vp','vs','rho']:   
                        f_name='init_'+component;       r2='_r2(initial,true)_'+numstr3(F_r2(getattr(d.model_init,component),getattr(d.model,component))); 
                        f_name2='init_2_'+component;    r2_2='_r2(initial,true)_'+numstr3(F_r2(getattr(d.model_init,component)[:,0:nx_orig],getattr(d.model,component)[:,0:nx_orig]));  
                        plot_acquisition(getattr(d.model_init,component),d.model,src,rec,f_name2+r2_2,**vlims[component],nx_orig=nx_orig,folder_path=d.pictures_folder,file_path=f_name2+r2_2+'.png')
                        f_name='true'+component+'_'
                        plot_acquisition(getattr(d.model,component),d.model,src,rec,f_name,**vlims[component],nx_orig=nx_orig,folder_path=d.pictures_folder,file_path=f_name+'.png')
                    model=d.model
                    model_init=d.model_init
                    load_from_folder=0
        if load_from_folder==1:
            dx=d.DH
            vp,fnames = d.get_fwi_start(return_filenames=True,keys=['.vp'])
            vs,fnames = d.get_fwi_start(return_filenames=True,keys=['.vs'])
            rho,fnames = d.get_fwi_start(return_filenames=True,keys=['.rho'])
            model = api.Model(vp[0],vs[0],rho[0], dx)
            vp_init,fnames = d.get_fwi_start(return_filenames=True,keys=['init.vp']);
            vs_init,fnames = d.get_fwi_start(return_filenames=True,keys=['init.vs']);
            rho_init,fnames = d.get_fwi_start(return_filenames=True,keys=['init.rho']);
            if vp_init==[]: vp_init=vp;
            if vs_init==[]: vs_init=vs;
            if rho_init==[]: rho_init=rho;

            model_init = api.Model(vp_init[0],vs_init[0],rho_init[0],dx)
            d.model_init=model_init
            d.model=model
            vplim = {'vmax':1500,'vmin':4500}
            vslim = {'vmax':d.model.vs.max(), 'vmin': 0 }
            rholim={'vmax': d.model.rho.max(), 'vmin': 1000 }
            dv_lim={'vmax': 500, 'vmin': -500}
            # vplim = {'vmax': model_init.vp.max(), 'vmin': model_init.vp.min()}
            # vslim = {'vmax': model_init.vs.max(), 'vmin': model_init.vs.min()}
            # rholim={'vmax': model_init.rho.max(), 'vmin': model_init.rho.min()}
            # dv_lim={'vmax': 500, 'vmin': -500}
            vlims = {'vp': vplim, 'vs': vslim,'rho': rholim,'dv': dv_lim}
            for component in ['vp','vs','rho']:
                f_name='init_'+component;   r2='_r2(initial,true)_'+numstr3(F_r2(getattr(model_init,component)[:,0:nx_orig],getattr(model,component)[:,0:nx_orig] ) );   
                plot_acquisition(getattr(model_init,component),model,src,rec,f_name+r2,**vlims[component],nx_orig=nx_orig,folder_path=d.pictures_folder,file_path=f_name+r2+'.png')
                f_name='true'+component+'_'
                plot_acquisition(getattr(model,component),model,src,rec,f_name,**vlims[component],nx_orig=nx_orig,folder_path=d.pictures_folder,file_path=f_name+'.png')
    #####################################   shots
    plot_shots=1
    if '/var/' in os.getcwd():
        plot_shots=0
    if plot_shots==1:
        if os.path.exists(os.path.join(directory,'su')):
            shots = d.get_shots(keys=['_y'])
            shots,filenames= d.get_shots(return_filenames=True)
            if shots!=[]:
                print(f'Read {len(shots)} shots {shots[0].shape} into list')
                it_list=[int(np.floor(x)) for x in np.linspace(0,len(shots)-1)]
                it_list=np.arange(0,len(shots)-1,int(len(shots)/3)).tolist()
                # it_list=[0]
                for i in it_list:
                    # filtered_shot=bandpass(shot[i],flo=corner_frequency,dt=dt,order=8,btype='high')
                    filtered_shot=bandpass(shots[i],fhi=6,dt=d.DT,order=8,btype='low')
                    freqs,ps=plot_spectrum(shots[i],d.DT,fmax=15,folder_path=d.pictures_folder,file_path='spectrum_shots_'+str(i)+'.png')
                    plot_shot(shots[i], pclip=0.05, title=str(i),folder_path=d.pictures_folder,file_path='shot'+str(i)+'.png',show=0)
                    freqs,ps=plot_spectrum(filtered_shot,d.DT, fmax=15,folder_path=d.pictures_folder,file_path='spectrum_low_freq_precisely_shots_'+str(i)+'.png')
                    plot_shot(filtered_shot,pclip=0.05, title=str(i),folder_path=d.pictures_folder,file_path='shot_low_freq_precisely_'+str(i)+'.png',show=0)
            del shots
    ############################7#########   plot wavefields
    plot_wavefields=0
    if plot_wavefields==1 and len(os.listdir(os.path.join(directory,'snap')) )!=0:
        wavefields,fnames=d.get_snapshots(return_filenames=True)
        for m, f in zip(wavefields,fnames):
            for i in np.arange(m.shape[0]):
                f_name=(f.split('/')[-1])+'_snap_'+str(i)
                plot_acquisition(m[i,::],model,src,rec,f_name,nx_orig=nx_orig,folder_path=os.path.join(d.save_folder,'pictures'),file_path=f_name+'.png')
    #####################################   plot_gradients
    plot_gradients=1
    if plot_gradients==1:
        if hasattr(d,'grads')==True:
            # if not (getattr(d.grads) is None):
            #     if getattr(d.grads).size!=0:
            for m,f in zip(d.grads,d.grad_names):
                f_name=(f.split('/')[-1]).split('.')[0]
                plot_acquisition(m,d.model,src,rec,f_name,nx_orig=nx_orig,folder_path=d.pictures_folder,file_path=f_name+'.png')
        else:
            if os.path.exists((os.path.join(directory,'jacobian'))):
                if len(os.listdir(os.path.join(directory,'jacobian')))>0:
                    grads,fnames = d.get_fwi_gradients(return_filenames=True)
                    for m,f in zip(grads, fnames):
                        f_name=(f.split('/')[-1]).split('.')[0]
                        plot_acquisition(m,d.model,src,rec,f_name,nx_orig=nx_orig,folder_path=d.pictures_folder,file_path=f_name+'.png')
    ##################  plot inverted models
    plot_inverted_models=1
    if plot_inverted_models==1:
        if hasattr(d,'fwi_model_names')==True:
            print(d.fwi_model_names)
            comps=['vp']
            # comps=['vp','vs','rho']
            for m, f in zip(d.models,d.fwi_model_names):
                f_name=(f.split('/')[-1]).split('.')[0]
                # for component in ['vp']:    #,'vs','rho'
                for component in comps:    #,'vs','rho'
                    if component in f_name: 
                        r2='_r2(m_i,true)_'+numstr3(F_r2(m[:,0:nx_orig],getattr(d.model,component)[:,0:nx_orig] ))
                        plot_acquisition(m,d.model,src,rec,f_name+r2,**vlims[component],nx_orig=nx_orig,folder_path=d.pictures_folder,file_path=f_name+r2+'.png')
                        plot_acquisition(m-getattr(d.model_init,component),d.model,src,rec,f_name+r2,nx_orig=nx_orig,folder_path=d.pictures_folder,file_path=f_name+'_dv(m_i,m_init)'+r2+'.png')
                        # plot_acquisition(m-getattr(model,component),model,src,rec,f_name,**vlims['dv'],folder_path=d.pictures_folder,file_path=f_name+'_difference_with_true.png')
                    
            ##################  plot tapers
            # plot_acquisition(d.taper.vp,d.model,src,rec,f_name,folder_path=d.pictures_folder,file_path=f_name+'_taper.png')
            plot_logs(d.model,d.model_init,33,folder_path=d.pictures_folder,file_path='logs_true_init.png')
        else:
            comps=['vp']
            comps=['vp','vs','rho']
            if os.path.exists((os.path.join(directory,'model'))):
                if len(os.listdir(os.path.join(directory,'model')))>0:
                    for comp in comps:
                        models,fnames = d.get_fwi_models(return_filenames=True,keys=comp)
                        for m, f in zip(models, fnames):
                            f_name=(f.split('/')[-1]).split('.')[0]
                            # for component in ['vp']:    #,'vs','rho'
                            for component in comps:    #,'vs','rho'
                                if component in f_name: 
                                    r2='_r2(m_i,true)_'+numstr_3(F_r2(m[:,0:nx_orig],getattr(model,component)[:,0:nx_orig]))
                                    plot_acquisition(m,model,src,rec,f_name+r2,**vlims[component],nx_orig=nx_orig,folder_path=d.pictures_folder,file_path=f_name+r2+'.png')
                                    plot_acquisition(m-getattr(model_init,component),model,src,rec,f_name+r2,nx_orig=nx_orig,folder_path=d.pictures_folder,file_path=f_name+'_dv(m_i,true)'+r2+'.png')
                                    models, fnames = d.get_fwi_tapers(return_filenames=True)
                    for m, f in zip(models,fnames):
                        f_name=(f.split('/')[-1]).split('.')[0]
                        plot_acquisition(m,model,src,rec,f_name,nx_orig=nx_orig,folder_path=d.pictures_folder,file_path=f_name+'.png')
    return None
def denise_plotting3(d,results_path,pars=None):
    """ results_path- path containing direcory fld (denise 'outputs' folder)"""
    directory=os.path.join(results_path,'fld')
    d.pictures_folder=os.path.join(results_path,'pictures')
    # api._cmd('rm -r '+os.path.join(results_path,'pictures/*'))
    os.makedirs(d.pictures_folder,exist_ok=True)
    path=results_path
    #####################################
    all_flags_value=1
    flag_plot_logs=all_flags_value
    flag_plot_shots=all_flags_value
    flag_plot_shots2=all_flags_value
    flag_plot_inverted_models=all_flags_value
    # flag_plot_logs=0
    # flag_plot_shots=0
    # flag_plot_shots2=0
    # flag_plot_inverted_models=0
    ##################################### lets assume I extended models in fwi for better gradients on 320 nx samples
    nx_orig=d.NX-320
    ##################  
    with open(info_file,'rb') as input:
        acq_data=pickle.load(input)
    #### acq_data = {'src':src,'rec':rec,'wavelet':bpw,'src_x0':src_x0,'rec_x0':rec_x0,'dsrc':dsrc,'drec':drec,'dDT':dDT}
    dx=d.DH
    src=acq_data['src']
    rec=acq_data['rec']
    bpw=acq_data['wavelet']
    src_x0=acq_data['src_x0']
    rec_x0=acq_data['rec_x0']
    dsrc=acq_data['dsrc']
    drec=acq_data['drec']
    d.DT=acq_data['dDT'] 
    log_dict=acq_data['log_dict']
    log_loc=log_dict['loc']
    wlog=log_dict['data']
    log_idx = int(log_loc / dx)
    d.REC_INCR_X = dsrc
    d.N_STREAMER=len(rec)
    #####################################   plot inverted models via logs
    lvp=wlog;   print(lvp.shape)
    lvp=np.repeat(wlog,1000,axis=1)
    # shear velocity, [m/s]
    lvs = lvp.copy() / (3 ** 0.5)
    lvs = np.where(lvp < 1.01 * np.min(lvp), 0, lvs)
    # density, [kg/m3] 
    lrho = 1e3*0.3 * lvp.copy()**0.25
    lrho = np.where(lvp < 1.01 * np.min(lvp), 1000, lrho)
    model_log = api.Model(lvp, lvs, lrho, dx)
    vvr = {'vp': None, 'vs': None, 'rho': None}
    d.fwi_model_names
    for k in vvr.keys():
        if os.path.exists(os.path.join(directory,'model')):
            vvr[k], fnames = d.get_fwi_models([k + '_stage'], return_filenames=True)
        else:
            fnames=fnmatch.filter(d.fwi_model_names,'*'+k+'_stage*')
            models_list=[]
            for fname in fnames:
                ind=d.fwi_model_names.index(fname)
                models_list.append(d.models[ind])
            vvr[k]=models_list
    #####################################flag_plot_logs
    if flag_plot_logs==1:
        # lvp, _ = ld.load_hh(os.path.join('./for_pasha/data_cgg', 'cgg_log.hh'), 1)
        # lvp = np.flipud(lvp.T)
        # lvp = lvp[::factor, :]
        # lvp_ = lvp[:-5, :nx0]
        # plot_logs(model_log,api.Model(vvr['vp'][0], vvr['vs'][0], vvr['rho'][0],dx) ,int(log_loc/dx))
        # api.Model(vvr['vp'][0], vvr['vs'][0], vvr['rho'][0],dx)
        # api.Model(vvr['vp'][0], vvr['vs'][0], vvr['rho'][0],dx)
        if len(vvr['vp'])>0:
            for i in range(len(vvr['vp'])):
                print('directory=',directory)
                if len(fnames)>0:
                    stage_name=fnames[i].split('_stage_')[-1].split('.bin')[0]
                    plot_logs(model_log,api.Model(vvr['vp'][i], vvr['vs'][i], vvr['rho'][i],dx),int(log_loc/dx),folder_path=d.pictures_folder,file_path='logs_stage_'+stage_name+'.png' )
                    if len(vvr['vp'])>1:
                        plot_logs(model_log,api.Model(vvr['vp'][-1], vvr['vs'][-1], vvr['rho'][-1],dx),int(log_loc/dx),folder_path=d.pictures_folder,file_path='logs_model_last_init.png')
    plot_logs(model_log,d.model_init,int(log_loc/dx),folder_path=d.pictures_folder,file_path='logs_true_init.png')
    taper=d.TAPER
    plot_model(d.TAPER,folder_path=d.pictures_folder,file_path='_taper.png')
    #####################################
    nx_orig=600;    nz_orig=d.model.vp.shape[0]               
    vplim = {'vmax':1500,'vmin':3000}
    vslim = {'vmax': d.model.vs.max(), 'vmin': 0 }
    rholim={'vmax': d.model.rho.max(), 'vmin': 1000 }
    dv_lim={'vmax': 500, 'vmin': -500}
    vlims = {'vp': vplim, 'vs': vslim,'rho': rholim,'dv': dv_lim}
    vplim2 = {'vmax':1500/1000,'vmin':3000/1000}
    vslim2 = {'vmax': d.model.vs.max()/1000, 'vmin': 0 }
    rholim2={'vmax': d.model.rho.max()/1000, 'vmin': 1000/1000 }
    dv_lim2={'vmax': 500/1000, 'vmin': -500/1000}
    vlims_kms = {'vp': vplim2, 'vs': vslim2,'rho': rholim2,'dv': dv_lim2}
    # #####################################   plot initial true velocity models
    for component in ['vp','vs','rho']:   
        f_name='init_'+component;   
        r2='_r2(initial,true)_'+numstr3(F_r2(   getattr(d.model_init,component)[0:nz_orig,0:nx_orig],getattr(d.model,component)[0:nz_orig,0:nx_orig] )  );
        plot_acquisition(getattr(d.model_init,component),d.model,src,rec,f_name+r2,**vlims[component],folder_path=d.pictures_folder,file_path=f_name+r2+'.png')
        plot_log_model(getattr(d.model_init,component),dx,nx_orig,nz_orig,_src=src,title=f_name+r2,log=wlog,log_location=log_loc,**vlims_kms[component],folder_path=d.pictures_folder,file_path=f_name+r2+'_2.png')
        f_name='true'+component+'_'
        plot_acquisition(getattr(d.model,component),d.model,src,rec,f_name,**vlims[component],folder_path=d.pictures_folder,file_path=f_name+'.png')
        plot_log_model(getattr(d.model,component),dx,nx_orig,nz_orig,_src=src,title=f_name+r2,log=wlog,log_location=log_loc,**vlims_kms[component],folder_path=d.pictures_folder,file_path=f_name+'_2.png')
    # #####################################   plot inverted models
    getattr(d.model_init,component).shape
    comps=['vp','vs','rho']
    if flag_plot_inverted_models==1:
        if hasattr(d,'fwi_model_names')==True:
            for m, f in zip(d.models,d.fwi_model_names):
                f_name=(f.split('/')[-1]).split('.')[0]
                for component in comps:    #,'vs','rho'
                    if component in f_name: 
                        r2='_r2(m_i,true)_'+numstr3(F_r2(m,getattr(d.model,component)))
                        # plot_acquisition(m,d.model,src,rec,f_name+r2,**vlims[component],folder_path=d.pictures_folder,file_path=f_name+r2+'.png')
                        # plot_acquisition(m-getattr(d.model_init,component),d.model,src,rec,f_name+r2,folder_path=d.pictures_folder,file_path=f_name+'_dv(m_i,m_init)'+r2+'.png')
                        plot_log_model(m,dx,nx_orig,nz_orig,_src=src,title=f_name+r2,log=wlog,log_location=log_loc,**vlims_kms[component],folder_path=d.pictures_folder,file_path=f_name+'__'+r2+'.png')
                        plot_log_model(m-getattr(d.model_init,component),dx,nx_orig,nz_orig,_src=src,title=f_name+r2,log=wlog,log_location=log_loc,folder_path=d.pictures_folder,file_path=f_name+'__dv(m_i,m_init)'+r2+'.png')
    #####################################
    if os.path.exists(os.path.join(results_path,'fld','su_modelled')):
        # #####################################   compare field shots and modelled shots on last iteration
        if flag_plot_shots==1:
            shots_field,filenames_field=d.get_shots_from_directory(os.path.join(path,'fld','su'),keys=['_p'],return_filenames=True)
            shots_modelled,filenames_modelled=d.get_shots_from_directory(os.path.join(path,'fld','su_modelled'),keys=['_p'],return_filenames=True)
            if len(shots_modelled)>1:
                for ishot in [13, 65]:
                    print(ishot)
                    shot_s = divmax(shots_modelled[ishot])
                    shot_f = divmax(shots_field[ishot])
                    freqs,ps=plot_spectrum(shot_s,d.DT,fmax=15,folder_path=d.pictures_folder,file_path='spectrum_shot_s_'+str(ishot)+'.png')
                    freqs,ps=plot_spectrum(shot_f,d.DT,fmax=15,folder_path=d.pictures_folder,file_path='spectrum_shot_f_'+str(ishot)+'.png')
                    vis.plot_shot(shot_f, pclip=0.0125,folder_path=os.path.join(path,'pictures'),file_path='shot_field_'+str(ishot)+'.png')
                    vis.plot_shot(np.concatenate([np.flip(shot_s, 0), shot_f],axis=0), pclip=0.0125,folder_path=os.path.join(path,'pictures'),file_path='shot_side_by_side_comparison_'+str(ishot)+'.png')
                    vis.plot_compare_stripes(shot_s,shot_f,pclip=0.0125,folder_path=os.path.join(path,'pictures'),file_path='shot_comparison_'+str(ishot)+'.png',colorbar=False, dt=0.002, dx=25)
            ss=1
        # #####################################   compare field shots and modelled shots on initial model
        if flag_plot_shots2==1:
            shots_field,filenames_field=d.get_shots_from_directory(os.path.join(path,'fld','su'),keys=['_p'],return_filenames=True)
            shots_field[0].shape
            for ishot in [0,1,2,3,11,31,51,71]:
                print(ishot)
                shot_f = divmax(shots_field[ishot])
                freqs,ps=plot_spectrum(shot_f,d.DT,fmax=15,folder_path=d.pictures_folder,title='shot_'+str(ishot),file_path='spectrum_shot_f_'+str(ishot)+'.png')
                vis.plot_shot(shot_f, pclip=0.0125,folder_path=os.path.join(path,'pictures'),title='shot_'+str(ishot),file_path='shot_field_'+str(ishot)+'.png')
            shots_modelled,filenames_modelled=d.get_shots_from_directory(os.path.join(path,'fld','su_modelled_init_model'),keys=['_p'],return_filenames=True)
            if len(shots_modelled)>1:
                for ishot in [6,11,31,51,71]:
                # for ishot in range(len(shots_modelled)):
                    print(ishot)
                    shot_s = divmax(shots_modelled[ishot])
                    shot_f = divmax(shots_field[ishot])
                    freqs,ps=plot_spectrum(shot_s,d.DT,fmax=15,folder_path=d.pictures_folder,title='shot_'+str(ishot),file_path='spectrum_init_model_shot_s_'+str(ishot)+'.png')
                    freqs,ps=plot_spectrum(shot_f,d.DT,fmax=15,folder_path=d.pictures_folder,title='shot_'+str(ishot),file_path='spectrum_shot_f_'+str(ishot)+'.png')
                    vis.plot_shot(shot_f, pclip=0.0125,folder_path=os.path.join(path,'pictures'),title='shot_'+str(ishot),file_path='shot_field_'+str(ishot)+'.png')
                    vis.plot_shot(np.concatenate([np.flip(shot_s, 0), shot_f],axis=0), pclip=0.0125,title='shot_'+str(ishot),folder_path=os.path.join(path,'pictures'),file_path='shot_side_by_side_comparison_withinitmodel_'+str(ishot)+'.png')
                    vis.plot_compare_stripes(shot_s,shot_f,pclip=0.0125,folder_path=os.path.join(path,'pictures'),title='shot_'+str(ishot),file_path='shot_comparison_withinitmodel_'+str(ishot)+'.png',colorbar=False, dt=0.002, dx=25)
        #####################################                            
    return None
def denise_plotting_from_denise_folder(results_path,pars=None):
    """ results_path- path containing direcory fld (denise 'outputs' folder)"""
    directory=os.path.join(results_path)
    #####################################   read denise structure
    d=api.Denise(verbose=0)
    denise_input_files=[os.path.join(directory,'seis_inversion.inp'),os.path.join(directory,'seis.inp')]
    for input_file in denise_input_files:
        if os.path.exists(input_file):
            FILE=input_file
    if  'FILE' in globals() or 'FILE' in locals():
        d._parse_inp_file(fname=FILE)
        d.save_folder=directory
    else:
        # api._cmd(f"rm -r {os.path.join(path)}")
        return None
    d.pictures_folder=os.path.join(directory,'pictures')
    os.makedirs(d.pictures_folder,exist_ok=True)
    print('Plotting pictures in directory ',d.pictures_folder)
    load_from_folder=1
    #####################################  Plot fwi objective curve,step length evolution
    plot_misfits=1
    if plot_misfits==1:
        filename =os.path.join(directory,'seis_fwi_log.dat')
        if os.path.exists(filename):
            plot_fwi_misfit(filename,d)
            plot_fwi_step_length(filename,d)
    #####################################   load src,rec
    xrec=np.load(os.path.join(d.save_folder,'receiver','rec_x.npy'))
    yrec=np.load(os.path.join(d.save_folder,'receiver','rec_y.npy'))
    xsrc=np.load(os.path.join(d.save_folder,'source','src_x.npy'))
    ysrc=np.load(os.path.join(d.save_folder,'source','src_y.npy'))
    rec = api.Receivers(xrec,yrec)
    src = api.Sources(xsrc,ysrc)
    #####################################   load velocity models
    #models,fnames = d.get_fwi_models(return_filenames=True,keys=['vp'])
    plot_models=1
    if plot_models==1:
        dx=d.DH
        vp,fnames = d.get_fwi_start(return_filenames=True)
        vp,fnames = d.get_fwi_start(return_filenames=True,keys=['.vp'])
        vs,fnames = d.get_fwi_start(return_filenames=True,keys=['.vs'])
        rho,fnames = d.get_fwi_start(return_filenames=True,keys=['.rho'])
        model = api.Model(vp[0],vs[0],rho[0], dx)
        vp_init,fnames = d.get_fwi_start(return_filenames=True,keys=['init.vp'])
        vs_init,fnames = d.get_fwi_start(return_filenames=True,keys=['init.vs'])
        rho_init,fnames = d.get_fwi_start(return_filenames=True,keys=['init.rho'])
        # vp_init,fnames = d.get_fwi_start(return_filenames=True,keys= ['vp'])
        # vs_init,fnames = d.get_fwi_start(return_filenames=True,keys= ['vs'])
        # rho_init,fnames = d.get_fwi_start(return_filenames=True,keys=['rho'])
        if vp_init==[]: vp_init=vp;
        if vs_init==[]: vs_init=vs;
        if rho_init==[]: rho_init=rho;
        model_init = api.Model(vp_init[0],vs_init[0],rho_init[0],dx)
        vplim = {'vmax': model_init.vp.max(), 'vmin': model_init.vp.min()}
        vslim = {'vmax': model_init.vs.max(), 'vmin': model_init.vs.min()}
        rholim={'vmax': model_init.rho.max(), 'vmin': model_init.rho.min()}
        dv_lim={'vmax': 500, 'vmin': -500}
        vlims = {'vp': vplim, 'vs': vslim,'rho': rholim,'dv': dv_lim}
        d.model_init=model_init
        d.model=model
        for component in ['vp','vs','rho']:
            f_name='init_'+component;   r2='_r2(initial,true)_'+numstr3(F_r2(getattr(model_init,component),getattr(model,component)));   
            plot_acquisition(getattr(model_init,component),model,src,rec,f_name+r2,**vlims[component],folder_path=d.pictures_folder,file_path=f_name+r2+'.png')
            f_name='true'+component+'_'
            plot_acquisition(getattr(model,component),model,src,rec,f_name,**vlims[component],folder_path=d.pictures_folder,file_path=f_name+'.png')
    #####################################   shots
    plot_shots=1
    field_data_dir=os.path.join(results_path,'su','field')
    if plot_shots==1:
        if os.path.exists(field_data_dir):
            filenames=d._get_filenames(field_data_dir)
            shots=d._from_su(filenames)
            # shots,filenames= d.get_shots(return_filenames=True)     #'_y'
            if shots!=[]:
                print(f'Read {len(shots)} shots {shots[0].shape} into list')
                it_list=[int(np.floor(x)) for x in np.linspace(0,len(shots)-1)]
                it_list=np.arange(0,len(shots)-1,int(len(shots)/3)).tolist()
                # it_list=[11]
                for i in it_list:
                    # filtered_shot=bandpass(shot[i],flo=corner_frequency,dt=dt,order=8,btype='high')
                    filtered_shot=bandpass(shots[i],fhi=6,dt=d.DT,order=8,btype='low')
                    freqs,ps=plot_spectrum(shots[i],d.DT,fmax=10,folder_path=d.pictures_folder,file_path='spectrum_shots_'+str(i)+'.png')
                    plot_shot(shots[i], pclip=0.05, title=str(i),folder_path=d.pictures_folder,file_path='shot'+str(i)+'.png',show=0)
                    freqs,ps=plot_spectrum(filtered_shot,d.DT, fmax=10,folder_path=d.pictures_folder,file_path='spectrum_low_freq_precisely_shots_'+str(i)+'.png')
                    plot_shot(filtered_shot,pclip=0.05, title=str(i),folder_path=d.pictures_folder,file_path='shot_low_freq_precisely_'+str(i)+'.png',show=0)
            del shots
    ##################  plot inverted models
    plot_inverted_models=1
    if plot_inverted_models==1:
        comps=['vp']
        if os.path.exists((os.path.join(directory,'model'))):
            if len(os.listdir(os.path.join(directory,'model')))>0:
                models,fnames = d.get_fwi_models(return_filenames=True,keys=comps)
                for m, f in zip(models, fnames):
                    f_name=(f.split('/')[-1]).split('.')[0]
                    # for component in ['vp']:    #,'vs','rho'
                    for component in comps:    #,'vs','rho'
                        if component in f_name: 
                            r2='_r2(m_i,true)_'+numstr3(F_r2(m,getattr(model,component)))
                            plot_acquisition(m,model,src,rec,f_name+r2,**vlims[component],folder_path=d.pictures_folder,file_path=f_name+r2+'.png')
                            # plot_acquisition(m-getattr(model_init,component),model,src,rec,f_name+r2,folder_path=d.pictures_folder,file_path=f_name+'_dv(m_i,true)'+r2+'.png')
                models, fnames = d.get_fwi_tapers(return_filenames=True)
                for m, f in zip(models,fnames):
                    f_name=(f.split('/')[-1]).split('.')[0]
                    plot_acquisition(m,model,src,rec,f_name,folder_path=d.pictures_folder,file_path=f_name+'.png')   
    #####################################   plot_gradients
    plot_gradients=1
    if plot_gradients==1:
        if os.path.exists((os.path.join(directory,'jacobian'))):
            if len(os.listdir(os.path.join(directory,'jacobian')))>0:
                grads,fnames = d.get_fwi_gradients(return_filenames=True)
                for m,f in zip(grads, fnames):
                    f_name=(f.split('/')[-1]).split('.')[0]
                    plot_acquisition(m,d.model,src,rec,f_name,folder_path=d.pictures_folder,file_path=f_name+'.png')
    return None
def plot_multi_cnn_results(path):
    paths_=[]
    paths_2=next(os.walk(os.path.join('./logs')))[1]
    for p_ in paths_2:
        paths_.append(os.path.join('./logs',p_))
    paths_=sorted(paths_,key=os.path.getmtime)
    saving_path=paths_[-1]
    if os.path.exists(os.path.join(path,'denise_data.pkl')):
        with open(os.path.join(path,'denise_data.pkl'),'rb') as input:
            d=pickle.load(input)
    f_name=path.split('/')[-3]+'_'+path.split('/')[-2]+'_'+path.split('/')[-1]
    component='vp'
    if 'cgg' in path:
        vplim = {'vmax':1500,'vmin':3000}
    else:
        vplim = {'vmax':1500,'vmin':4500}
    r2='_r2_initial_true_'+numstr3(F_r2(   getattr(d.model_init,component),getattr(d.model,component) )  );
    plot_acquisition(getattr(d.model_init,component),d.model,d.src,d.rec,f_name+r2,**vplim,folder_path=saving_path,file_path=f_name+r2+'.png')    
    return None
#############################
def save_denise_file(pars):
    """save denise file to hdf5 file to be picked by pytorch-gan/pix2pix
    DEPRECATED in 07.07.21"""
    ##########################  create directory for dataset
    path=pars['file_path']
    os.makedirs(str(Path(path).parent),exist_ok=True)
    ##########################  rewrite file
    rewrite_file=1
    if os.path.exists(path) and rewrite_file==1:
        print('file '+path+' exists, rewriting')
        os.remove(path)
    if os.path.exists(path) and rewrite_file==0:
        print('file '+path+' exists, no rewriting, skipped processing')
        return None
    ##########################
    Models=pars['Models']
    Models_init=pars['Models_init']
    input_data=pars['input_data']-Models_init
    output_data= F_smooth(Models,sigma_val=int(100/pars['dx']))-Models_init     #100 or 300
    taper=pars['taper']
    ##########################  cropping according to taper
    # Plot_image(input_data.squeeze().T, Show_flag=0,Save_flag=1,Title='input_data_start'+path.split('/')[-1].split('.')[0],Save_pictures_path='./Pictures')
    # print((np.where(input_data[33,:]==0))[0].size)
    ind=np.where(taper==1)
    ix1=ind[0].min();   ix2=ind[0].max()
    iz1=ind[1].min();
    iz1=21
    iz1=31  #   to avoid zero pixels due to error tapers
    iz2=ind[1].max()
    Models=Models[ix1:ix2,:];
    Models_init=Models_init[ix1:ix2,:]
    input_data=input_data[ix1:ix2,iz1:iz2];
    # Plot_image(input_data.squeeze().T, Show_flag=0,Save_flag=1,Title='input_data_step1'+path.split('/')[-1].split('.')[0],Save_pictures_path='./Pictures')
    output_data=output_data[ix1:ix2,iz1:iz2];
    ##########################  custom cropping
    ix1=40;   ix2=-60
    iz1=0;   iz2=-1
    # iz1=(np.where(input_data[33,:]==0))[0].size     # crop zero update    !!!!!!!!!!!!!!!!!!!!!!!!!!!DEPRECATED!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # print(iz1)
    Models=Models[ix1:ix2,:]
    Models_init=Models_init[ix1:ix2,:]
    input_data=input_data[ix1:ix2,iz1:iz2];
    # Plot_image(input_data.squeeze().T, Show_flag=0,Save_flag=1,Title='input_data_step2'+path.split('/')[-1].split('.')[0],Save_pictures_path='./Pictures')
    output_data=output_data[ix1:ix2,iz1:iz2];
    ##########################   
    # print((np.where(input_data[33,:]==0))[0].size)
    # print((np.where(input_data2[33,:]==0))[0].size)
    # print((np.where(Models[33,:]==1500))[0].size)
    # print((np.where(Models2[33,:]==1500))[0].size)
    # print((np.where(Models3[33,:]==1500))[0].size)
    # Plot_image(input_data.squeeze().T, Show_flag=0,Save_flag=1,Title='input_data_'+path.split('/')[-1].split('.')[0],Save_pictures_path='./Pictures')
    # Plot_image(input_data2.squeeze().T, Show_flag=0,Save_flag=1,Title='input_data2_'+path.split('/')[-1].split('.')[0],
    #     Save_pictures_path='./Pictures')
    # Plot_image(input_data3.squeeze().T, Show_flag=0,Save_flag=1,Title='input_data3_'+path.split('/')[-1].split('.')[0],
    #     Save_pictures_path='./Pictures')
    # Plot_image(Models.squeeze().T, Show_flag=0,Save_flag=1,Title='Models_'+path.split('/')[-1].split('.')[0],
    #     Save_pictures_path='./Pictures')
    # Plot_image(Models2.squeeze().T, Show_flag=0,Save_flag=1,Title='Models2_'+path.split('/')[-1].split('.')[0],
    #     Save_pictures_path='./Pictures')
    # Plot_image(Models3.squeeze().T, Show_flag=0,Save_flag=1,Title='Models3_'+path.split('/')[-1].split('.')[0],
    #     Save_pictures_path='./Pictures')
    # Plot_image(taper.squeeze().T, Show_flag=0,Save_flag=1,Title='taper_'+path.split('/')[-1].split('.')[0],
    #     Save_pictures_path='./Pictures')
    # Plot_image(input_data.squeeze().T, Show_flag=0,Save_flag=1,Title='input_data_'+path.split('/')[-1].split('.')[0],
    #     Save_pictures_path='./Pictures')
    # Plot_image(output_data.squeeze().T,Show_flag=0,Save_flag=1,Title='output_data_'+path.split('/')[-1].split('.')[0],
    #     dx=20,dy=20,Save_pictures_path='./Pictures')
    # Plot_image(Models.squeeze().T, Show_flag=0,Save_flag=1,Title='Models_'+path.split('/')[-1].split('.')[0],
    #     dx=20,dy=20,Save_pictures_path='./Pictures')
    # Plot_image(Models_init.squeeze().T,Show_flag=0,Save_flag=1,Title='Models_init_'+path.split('/')[-1].split('.')[0],
        # dx=20,dy=20,Save_pictures_path='./Pictures')
    ##########################
    input_data_real_amplitudes=np.expand_dims(input_data,axis=(0,-1))
    output_data_real_amplitudes=np.expand_dims(output_data,axis=(0,-1))
    Models=np.expand_dims(Models,axis=(0,-1))
    Models_init=np.expand_dims(Models_init,axis=(0,-1))
    ##########################
    input_data,scaler_x=scaling_data_01(input_data,preconditioning=False,
        visualize_scaling_results=0,save_pictures_path='./pictures/'+'1.png')
    output_data,scaler_t=scaling_data_01(output_data,preconditioning=False)
    # Plot_image(input_data.squeeze().T,Show_flag=1,Save_flag=1,Title='input_data_'+path.split('/')[-1].split('.')[0],dx=20,dy=20)
    # Plot_image(output_data.squeeze().T,Show_flag=1,Save_flag=1,Title='output_data_'+path.split('/')[-1].split('.')[0],dx=20,dy=20)
    # Plot_image(input_data_real_amplitudes.squeeze().T,Show_flag=1,Save_flag=1,Title='input_data_real_amp'+path.split('/')[-1].split('.')[0],dx=20,dy=20)
    # Plot_image(output_data_real_amplitudes.squeeze().T,Show_flag=1,Save_flag=1,Title='output_data_real_amp'+path.split('/')[-1].split('.')[0],dx=20,dy=20)
    print('saving to '+path)
    # print('cropped=',Models.shape)
    # print('taper.shape=',taper.shape)
    # print('pars[Models]=',pars['Models'].shape)
    np.savez(path,input_data=input_data,output_data=output_data,
        models_init=Models_init,models=Models,dx=pars['dx'],dz=pars['dz'],
        input_data_real_amplitudes=input_data_real_amplitudes,
        output_data_real_amplitudes=output_data_real_amplitudes,
        scaler_x=scaler_x,scaler_t=scaler_t,taper=taper)
    return None
def save_denise_file2(pars):
    """save denise file to hdf5 file to be picked by pytorch-gan/pix2pix"""
    ##########################  create directory for dataset
    path=pars['file_path']
    os.makedirs(str(Path(path).parent),exist_ok=True)
    ##########################  rewrite file
    rewrite_file=1
    if os.path.exists(path) and rewrite_file==1:
        print('file '+path+' exists, rewriting')
        os.remove(path)
    if os.path.exists(path) and rewrite_file==0:
        print('file '+path+' exists, no rewriting, skipped processing')
        return None
    ##########################
    Models=pars['Models']
    Models_init=pars['Models_init']
    input_data=pars['input_data']-Models_init
    ##########################  smooth targets with specific radius
    output_data= F_smooth(Models,sigma_val=int(100/pars['dx']))-Models_init     #100 or 300
    ##########################  read taper shape
    taper=pars['taper']
    ind=np.where(taper==1)
    ix1=ind[0].min();   ix2=ind[0].max()
    iz1=ind[1].min();   iz2=ind[1].max()
    ##########################  cropping according to taper from bottom, and to leave water from above
    Models=Models           [ix1:ix2,0:iz2]
    Models_init=Models_init [ix1:ix2,0:iz2]
    ##########################  cropping according to taper
    input_data=input_data   [ix1:ix2,iz1:iz2]
    output_data=output_data [ix1:ix2,iz1:iz2]
    ##########################  custom Ox cropping to get rid of acquisition edge artifacts
    ix1=40;   ix2=-60
    Models=Models[ix1:ix2,:]
    Models_init=Models_init[ix1:ix2,:]
    input_data=input_data[ix1:ix2,:]
    output_data=output_data[ix1:ix2,:]
    ##########################  scaling
    input_data_real_amplitudes=np.expand_dims(input_data,axis=(0,-1))
    output_data_real_amplitudes=np.expand_dims(output_data,axis=(0,-1))
    Models=np.expand_dims(Models,axis=(0,-1))
    Models_init=np.expand_dims(Models_init,axis=(0,-1))
    input_data,scaler_x=scaling_data_01(input_data,preconditioning=False,
        visualize_scaling_results=0,save_pictures_path='./pictures/'+'1.png')
    output_data,scaler_t=scaling_data_01(output_data,preconditioning=False)
    ##########################  recording
    print('saving to '+path)
    print('input_data',input_data.shape)
    print('Models',Models.shape)
    print('upper taper width',Models.shape[2]-input_data.shape[2])
    np.savez(path,input_data=input_data,output_data=output_data,
        models_init=Models_init,models=Models,dx=pars['dx'],dz=pars['dz'],
        input_data_real_amplitudes=input_data_real_amplitudes,
        output_data_real_amplitudes=output_data_real_amplitudes,
        scaler_x=scaler_x,scaler_t=scaler_t)
    return None
def save_denise_file3(pars):
    """save denise file to hdf5 file to be picked by pytorch-gan/pix2pix.
    Input data is not cropped """
    ##########################  create directory for dataset
    path=pars['file_path']
    os.makedirs(str(Path(path).parent),exist_ok=True)
    file_name=path.split('/')[-1]
    file_name=file_name.split('.npz')[0]
    ##########################  rewrite file
    rewrite_file=1
    if os.path.exists(path) and rewrite_file==1:
        print('file '+path+' exists, rewriting')
        os.remove(path)
    if os.path.exists(path) and rewrite_file==0:
        print('file '+path+' exists, no rewriting, skipped processing')
        return None
    ##########################
    taper=pars['taper']
    Models=pars['Models']
    Models_init=pars['Models_init']
    input_data=pars['input_data']-Models_init
    ##########################  smooth targets with specific radius
    input_data[taper==0]=0
    smoothed_model=F_smooth(Models,sigma_val=int(100/pars['dx']))    #100 or 300
    smoothed_model[taper==0]=1500
    output_data=smoothed_model-Models_init 
    ##########################  plotting    ????    ,c_lim=[1500,4500]
    # Plot_image(taper.T,Show_flag=0,Save_flag=1,Title=file_name+'_taper',Save_pictures_path='./pictures_for_check')
    # Plot_image(input_data.T,Show_flag=0,Save_flag=1,Title=file_name+'_input_data',Save_pictures_path='./pictures_for_check')
    # Plot_image(output_data.T,Show_flag=0,Save_flag=1,Title=file_name+'_output_data',Save_pictures_path='./pictures_for_check')
    # Plot_image(Models_init.T,Show_flag=0,Save_flag=1,Title=file_name+'_Models_init',Save_pictures_path='./pictures_for_check')
    ##########################  scaling
    input_data_real_amplitudes=np.expand_dims(input_data,axis=(0,-1))
    output_data_real_amplitudes=np.expand_dims(output_data,axis=(0,-1))
    Models=np.expand_dims(Models,axis=(0,-1))
    Models_init=np.expand_dims(Models_init,axis=(0,-1))
    input_data,scaler_x=scaling_data_01(input_data,)
    output_data,scaler_t=scaling_data_01(output_data)
    ##########################  recording
    print('saving to '+path)
    print('input_data',input_data.shape)
    np.savez(path,input_data=input_data,output_data=output_data,
        models_init=Models_init,models=Models,dx=pars['dx'],dz=pars['dz'],
        input_data_real_amplitudes=input_data_real_amplitudes,
        output_data_real_amplitudes=output_data_real_amplitudes,
        scaler_x=scaler_x,scaler_t=scaler_t,taper=taper)
    return None
def save_denise_file4(pars):
    """Actual version of the code.
    save denise file to hdf5 file to be picked by pytorch-gan/pix2pix.
    Input data is not cropped.  2 input channels:initial model and model update. 
    And scale each other according to the dataset maximums."""
    ##########################  create directory for dataset
    path=pars['file_path']
    dataset_path=str(Path(path).parent)
    os.makedirs(dataset_path,exist_ok=True)
    file_name=path.split('/')[-1]
    file_name=file_name.split('.npz')[0]
    ##########################  rewrite file
    rewrite_file=1
    if os.path.exists(path) and rewrite_file==1:
        print('file '+path+' exists, rewriting')
        os.remove(path)
    if os.path.exists(path) and rewrite_file==0:
        print('file '+path+' exists, no rewriting, skipped processing')
        return None
    ##########################
    scaling_range=pars['scaling_range']
    taper=pars['taper']
    Models=pars['Models']
    Models_init=pars['Models_init']
    fwi_result=pars['input_data']
    input_data=pars['input_data']-Models_init
    dm_i_ra=pars['m_i_ra']-Models_init
    ##########################  Create target by smoothing targets with specific radius
    input_data[taper==0]=0
    smoothed_true_model=F_smooth(Models,sigma_val=int(pars['target_smoothing_diameter']/pars['dx']))    #100 or 300
    smoothed_true_model[taper==0]=1500
    output_data=smoothed_true_model-Models_init 
    ##########################  plotting    ????    ,c_lim=[1500,4500]
    # Plot_image(taper.T,Show_flag=0,Save_flag=1,Title=file_name+'_taper',Save_pictures_path='./pictures_for_check')
    # Plot_image(input_data.T,Show_flag=0,Save_flag=1,Title=file_name+'_input_data',Save_pictures_path='./pictures_for_check')
    # Plot_image(output_data.T,Show_flag=0,Save_flag=1,Title=file_name+'_output_data',Save_pictures_path='./pictures_for_check')
    # Plot_image(Models_init.T,Show_flag=0,Save_flag=1,Title=file_name+'_Models_init',Save_pictures_path='./pictures_for_check')
    # Plot_image(smoothed_true_model.T,Show_flag=0,Save_flag=1,Title=file_name+'_smoothed_true_model',Save_pictures_path='./pictures_for_check')
    ##########################  in my notation data should be 4-channel
    fwi_result=np.expand_dims(fwi_result,axis=(0,-1))
    input_data_real_amplitudes=np.expand_dims(input_data,axis=(0,-1))
    output_data_real_amplitudes=np.expand_dims(output_data,axis=(0,-1))
    Models=np.expand_dims(Models,axis=(0,-1))
    Models_init=np.expand_dims(Models_init,axis=(0,-1))
    ##########################  create scaling dictionary for different variables
    scaling_constants_dict={'x':[2010],'t':[2700],'init_model':[5500,1500],'model':[7510,1500],'fwi_res':[5900,600]}    #   dataset with outliers
    #   I chose 1450 because marmousi sample has some values of 1460
    scaling_constants_dict={'x':[1600],'t':[2700],'init_model':[4200,1500],'model':[4700,1450],'fwi_res':[4700,1450]}    #   dataset constrained between 1500 4700, with testing samples with outliers. generator_2
    # with open(os.path.join('./fwi','dataset_to_create.pkl'),'rb') as input:
    # with open(os.path.join('./fwi','dataset_to_create_new.pkl'),'rb') as input:
    with open(os.path.join('./fwi','dataset_to_create_09_09.pkl'),'rb') as input:
        data_dict=pickle.load(input)
    scaling_constants_dict=data_dict['scaling_constants_dict']
    with open(os.path.join(dataset_path,'scaling_constants_dict.pkl'), 'wb') as output:
        pickle.dump(scaling_constants_dict,output,protocol=4)
    # with open(os.path.join(dataset_path,'scaling_constants_dict.pkl'),'rb') as input:
    #     scaling_constants_dict=pickle.load(input)
    ##########################  scaling
    input_data_scaled=scaling_data(input_data,scaling_constants_dict,'x',scaling_range=scaling_range)
    print(' !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!input_data.min(),input_data.max()=',input_data.min(),input_data.max(),'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    output_data_scaled=scaling_data(output_data,scaling_constants_dict,'t',scaling_range=scaling_range)
    Models_init_scaled=scaling_data(Models_init,scaling_constants_dict,'init_model',scaling_range=scaling_range)
    fwi_result_scaled=scaling_data(fwi_result,scaling_constants_dict,'fwi_res',scaling_range=scaling_range)
    smoothed_true_model_scaled=scaling_data(smoothed_true_model,scaling_constants_dict,'model',scaling_range=scaling_range)
    # 'm_i_ra':   m_i_ra,'m_i_names':   m_i_names, ?????????????
    m_i_ra=pars['m_i_ra']
    m_i_ra=np.expand_dims(m_i_ra,axis=-1)
    m_i_ra_scaled=np.empty_like(m_i_ra)
    dm_i_ra_scaled=np.empty_like(m_i_ra)
    for i in range(m_i_ra.shape[0]):
        m_i_ra_scaled[i,::]=scaling_data(m_i_ra[i,::],scaling_constants_dict,'fwi_res',scaling_range=scaling_range)
        dm_i_ra_scaled[i,::]=scaling_data(dm_i_ra[i,::],scaling_constants_dict,'x',scaling_range=scaling_range)
    ##########################  check scale back
    # fwi_result_scaled_back=scaling_data_back(fwi_result_scaled,scaling_constants_dict,'fwi_res')
    # input_data_rescaled_back=scaling_data_01_back(input_data_scaled,scaling_constants_dict,'x')
    # Plot_image(Models_init_scaled.squeeze().T,Show_flag=0,Save_flag=1,Title=file_name+'_Models_init_scaled',Save_pictures_path='./pictures_for_check')
    # Plot_image(input_data_scaled.squeeze().T,Show_flag=0,Save_flag=1,Title=file_name+'_input_data_scaled',Save_pictures_path='./pictures_for_check')
    # Plot_image(output_data_scaled.squeeze().T,Show_flag=0,Save_flag=1,Title=file_name+'_output_data_scaled',Save_pictures_path='./pictures_for_check')
    # input_data_diff=fwi_result-fwi_result_scaled_back
    # Plot_image(input_data_diff.squeeze().T,Show_flag=0,Save_flag=1,Title=file_name+'_input_data_diff',Save_pictures_path='./pictures_for_check')
    ##########################  recording
    print('saving to '+path)
    print('input_data',input_data.shape)
    print('fwi_result.min',fwi_result.min())
    print('fwi_result.max',fwi_result.max())
    ##########################  todo fix the double save of m_i_ra
    np.savez(path,
        input_data_real_amplitudes=input_data_real_amplitudes,
        output_data_real_amplitudes=output_data_real_amplitudes,
        input_data=input_data_scaled,output_data=output_data_scaled,
        models_init=Models_init,models_init_scaled=Models_init_scaled,
        models=Models,dx=pars['dx'],dz=pars['dz'],
        fwi_result=fwi_result,fwi_result_scaled=fwi_result_scaled,
        m_i_ra=m_i_ra,m_i_ra_scaled=m_i_ra_scaled,
        dm_i_ra=dm_i_ra,dm_i_ra_scaled=dm_i_ra_scaled,
        smoothed_true_model=smoothed_true_model,
        smoothed_true_model_scaled=smoothed_true_model_scaled,
        taper=taper,scaling_range=scaling_range)
    return None
def remake_taper_save_file(new_dataset_folder,old_dataset_folder,source_file):
    """Read source file, rescale it, save to new destination."""   
    with open(os.path.join(old_dataset_folder,'scaling_constants_dict.pkl'),'rb') as input:
        scaling_constants_dict=pickle.load(input)
    with open(os.path.join(new_dataset_folder,'scaling_constants_dict.pkl'), 'wb') as output:
        pickle.dump(scaling_constants_dict,output,protocol=4)
    print('process ',source_file)
    with open(os.path.join(old_dataset_folder,source_file), 'rb') as f:
        data = np.load(f, allow_pickle=True)
        input_data_scaled = data['input_data']
        output_data_scaled = data['output_data']
        input_data_real_amplitudes= data['input_data_real_amplitudes']
        output_data_real_amplitudes= data['output_data_real_amplitudes']
        models_init =data['models_init']
        models_init_scaled =data['models_init_scaled']
        models = data['models']
        fwi_result=data['fwi_result']
        fwi_result_scaled=data['fwi_result_scaled']
        smoothed_true_model=data['smoothed_true_model']
        smoothed_true_model_scaled=data['smoothed_true_model_scaled']
        taper=data['taper']
        m_i_ra_scaled=data['m_i_ra_scaled']
        dm_i_ra_scaled=data['dm_i_ra_scaled']
        dm_i_ra=data['dm_i_ra']
        m_i_ra=data['m_i_ra']
        dz = data['dz']
        dx = data['dx']
        data.close()
    ############     
    pic_folder='./Pictures'
    Plot_image(models.squeeze().T,Show_flag=0, Save_flag=1, Title='model', Save_pictures_path=pic_folder)
    Plot_image((taper).T,Show_flag=0, Save_flag=1, Title='water_taper', Save_pictures_path=pic_folder)
    tmp=taper*models.squeeze()
    Plot_image(tmp.T,Show_flag=0, Save_flag=1, Title='model_tapered', Save_pictures_path=pic_folder)
    print(tmp[100,:])
    tmp=taper*input_data_real_amplitudes.squeeze()
    Plot_image(input_data_real_amplitudes.squeeze().T,Show_flag=0, Save_flag=1, Title='input_data_real_amplitudes', Save_pictures_path=pic_folder)
    Plot_image(tmp.T,Show_flag=0, Save_flag=1, Title='input_data_real_amplitudes_tapered', Save_pictures_path=pic_folder)
    Plot_image((input_data_real_amplitudes.squeeze()-tmp).T,Show_flag=0, Save_flag=1, Title='diff_input_data_real_amplitudes_', Save_pictures_path=pic_folder)
        
    tmp=taper*output_data_real_amplitudes.squeeze()
    Plot_image((output_data_real_amplitudes.squeeze()-tmp).T,Show_flag=0, Save_flag=1, Title='diff_output_data_real_amplitudes_', Save_pictures_path=pic_folder)
    Plot_image(output_data_real_amplitudes.squeeze().T,Show_flag=0, Save_flag=1, Title='output_data_real_amplitudes', Save_pictures_path=pic_folder)
    ############
    path=os.path.join(new_dataset_folder,source_file.split('/')[-1])
    input_data_scaled=scaling_data(input_data_real_amplitudes,scaling_constants_dict,'x')
    output_data_scaled=scaling_data(output_data_real_amplitudes,scaling_constants_dict,'t')
    models_init_scaled=scaling_data(models_init,scaling_constants_dict,'init_model')
    fwi_result_scaled=scaling_data(fwi_result,scaling_constants_dict,'fwi_res')
    smoothed_true_model_scaled=scaling_data(smoothed_true_model,scaling_constants_dict,'model')   
    ############
    # np.savez(path,
    #     input_data_real_amplitudes=input_data_real_amplitudes,
    #     output_data_real_amplitudes=output_data_real_amplitudes,
    #     input_data=input_data_scaled,output_data=output_data_scaled,
    #     models_init=models_init,models_init_scaled=models_init_scaled,
    #     models=models,dx=dx,dz=dz,
    #     fwi_result=fwi_result,
    #     fwi_result_scaled=fwi_result_scaled,
    #     m_i_ra=m_i_ra,m_i_ra_scaled=m_i_ra_scaled,
    #     dm_i_ra=dm_i_ra,dm_i_ra_scaled=dm_i_ra_scaled,
    #     smoothed_true_model=smoothed_true_model,
    #     smoothed_true_model_scaled=smoothed_true_model_scaled,
    #     taper=taper)
    print('saving to '+path)
    print('fwi_result.min,max',fwi_result.min(),fwi_result.max())
    return None
def parse_denise_folder(path,denise_root='./'):
    print(os.getcwd())
    print('Reading data from folder:',path)
    directory=os.path.join(path,'fld');
    os.listdir(path)
    if os.path.exists(os.path.join(path,'denise_data.pkl')):
        with open(os.path.join(path,'denise_data.pkl'),'rb') as input:
            d=pickle.load(input)
    else:
        d=api.Denise(denise_root,verbose=0)
        # d=api.Denise(verbose=0)
        denise_input_files=[os.path.join(directory,'seis.inp'),os.path.join(directory,'seis_inversion.inp')]
        for input_file in denise_input_files:
            if os.path.exists(input_file):
                FILE=input_file
        if  'FILE' in globals() or 'FILE' in locals():
            d._parse_inp_file(fname=FILE)
            d.save_folder=directory
        ############################################################    
        ##########  process seis_fwi_log.dat
        filename =os.path.join(directory,'seis_fwi_log.dat')
        if os.path.exists(filename):
            d.misfit_data=np.genfromtxt(filename)
        ##########  process 'model','jacobian' folder
        """move contents of model and jacobian folders to denise.data.pkl file."""
        if os.path.exists(os.path.join(path,'fld','model')):
            if len(os.listdir(os.path.join(path,'fld','model')))>0:
                models,fwi_model_names = d.get_fwi_models(return_filenames=True) #keys=['vp']
                d.models=models
                d.fwi_model_names=fwi_model_names
        if os.path.exists(os.path.join(path,'fld','jacobian')):
            if len(os.listdir(os.path.join(path,'fld','jacobian'))):
                grads,grad_names=d.get_fwi_gradients(return_filenames=True)
                d.grads=grads
                d.grad_names=grad_names
        ##########  process 'taper' folder
        if os.path.exists(os.path.join(path,'fld','taper')):
            if len(os.listdir(os.path.join(path,'fld','taper')))>0:
                taper,fnames=d.get_fwi_tapers(return_filenames=True)
                print('len(taper)=',len(taper))
                if 'vp' in d.taper._defined==True:
                    d.taper.vp=taper[-1]
                if len(taper)>1:
                    d.TAPER=taper[-1]
        ##########  process 'start' folder
        if 'vp' not in d.model._defined:
            if os.path.exists(os.path.join(path,'fld','start')):
                if len(os.listdir(os.path.join(path,'fld','start')))==0:
                    return None
                else:
                    print('path',path)
                    print('d._root_start=',d._root_start)
                    true_vp,init_fnames=d.get_fwi_start(return_filenames=True,keys='/model.vp')
                    true_vs,init_fnames=d.get_fwi_start(return_filenames=True,keys='/model.vs')
                    true_rho,init_fnames=d.get_fwi_start(return_filenames=True,keys='/model.rho')
                    model=api.Model(true_vp[0],true_vs[0],true_rho[0],d.DH)
                    d.model=model
        if hasattr(d,'model_init')==False:
            if os.path.exists(os.path.join(path,'fld','start')):
                init_vp,init_fnames=d.get_fwi_start(return_filenames=True,keys='model_init.vp')
                init_vs,init_fnames=d.get_fwi_start(return_filenames=True,keys='model_init.vs')
                init_rho,init_fnames=d.get_fwi_start(return_filenames=True,keys='model_init.rho')
                # print('path=',path)
                model_init=api.Model(init_vp[0],init_vs[0],init_rho[0],d.DH)
                d.model_init=model_init
        ##########  process 'source','receiver' folder
        if hasattr(d,'src')==False:
            if os.path.exists(os.path.join(path,'fld','source')):
                xsrc=np.load(os.path.join(d.save_folder,'source','src_x.npy'))
                ysrc=np.load(os.path.join(d.save_folder,'source','src_y.npy'))
                d.src = api.Sources(xsrc, ysrc)
            if os.path.exists(os.path.join(path,'fld','receiver')):
                xrec=np.load(os.path.join(d.save_folder,'receiver','rec_x.npy'))
                yrec=np.load(os.path.join(d.save_folder,'receiver','rec_y.npy'))
                d.rec = api.Receivers(xrec, yrec)
        if hasattr(d,'src')==False:
            nx=d.NX
            water_sz=20
            xsrc1 = 600.  # 1st source position [m]
            xsrc2 = nx*d.DH-xsrc1
            drec = 20.
            dsrc = 2*160.  #55 or 10
            xrec1 = 200.
            depth_src = 40.  # source depth [m]
            depth_rec=(water_sz+1)*d.DH
            xrec2 = nx*d.DH-xrec1
            xrec2 = nx*d.DH-xrec1
            xsrc = np.arange(xsrc1,xsrc2+d.DH,dsrc)
            ysrc = depth_src* xsrc/xsrc
            xrec = np.arange(xrec1, xrec2+d.DH,drec)
            yrec = depth_rec * (xrec / xrec)
            d.rec = api.Receivers(xrec, yrec)
            d.src = api.Sources(xsrc, ysrc)
        if hasattr(d,'grads')==False or hasattr(d,'fwi_model_names')==False:
            denise_data_file=os.path.join(path,'denise_data.pkl')
            # './fwi/cgg_real_data/fwi_56_strategy_l2/multi_cnn_13_special_weight_675_4_model___cgg_lin_vp_long_300_f_z_stage3/denise_data.pkl'
            # '/var/remote/lustre/project/k1404/pavel/DENISE-Black-Edition/fwi/cgg_real_data/fwi_56_strategy_l2_del_low_gen__5hz_1/model__cgg_lin_vp_long'
            if os.path.exists(denise_data_file):
                with open(denise_data_file,'rb') as input:
                    d=pickle.load(input)
            else:
                denise_data_file=os.path.join(path,'fld','denise_data.pkl')
                if os.path.exists(denise_data_file):
                    with open(denise_data_file,'rb') as input:
                        d=pickle.load(input)
    return d
def divmax(x): return x / np.max(np.abs(x))
def filter_seismic_data(path,dt=0.002,corner_frequency=5,fmax=10):
    # d=api.Denise('./',verbose=1)
    d=api.Denise('/home/plotnips/Dropbox/Log_extrapolation/scripts/DENISE-Black-Edition-master',verbose=0)
    shots,filenames=d.get_shots_from_directory(path,return_filenames=True)
    print('low frequency data corner frequency, ',corner_frequency)
    print(f'Read {len(shots)} shots {shots[0].shape} into list')
    counter=0
    for shot,filename in zip(shots,filenames):
        ################    processing stage 1
        filtered_shot=bandpass(shot,fhi=6, dt=dt,order=8, btype='low')
        ################    processing stage 2
        filtered_shot2=bandpass(shot,flo=corner_frequency+0.8,dt=dt,order=8,btype='high')
        filtered_shot2_=bandpass(shot,flo=corner_frequency+0.8,fhi=6,dt=dt,order=8,btype='band') #for plotting purposes
        ################    processing stage 3
        filtered_shot3=zero_below_freq(filtered_shot2,corner_frequency+0.1,dt, disable=False,reverse=False)
        filtered_shot3_=zero_below_freq(filtered_shot2_,corner_frequency+0.1,dt, disable=False,reverse=False)
        ################
        plotting_flag=1
        if plotting_flag==1:
            shots_list=[0,22,44,66]
            if counter in shots_list:
                plot_shot(shot,pclip=0.05,folder_path='./pictures_for_check',file_path='shot_'+str(counter)+'.png',show=0)
                plot_shot(filtered_shot2,pclip=0.05,folder_path='./pictures_for_check',file_path='shot'+str(counter)+'_stage2.png',show=0)
                freqs,ps=plot_spectrum(shot,dt,fmax=fmax,folder_path='./pictures_for_check',file_path='spectrum_'+str(counter)+'_original.png')
                freqs,ps=plot_spectrum(filtered_shot2,dt,fmax=fmax,folder_path='./pictures_for_check',file_path='spectrum_'+str(counter)+'stage2.png')
        ################    chosen processing    
        with segyio.su.open(filename,"r+",endian='little',ignore_geometry=True) as dst:
            dst.trace=filtered_shot2
        # shot_read,read_filename=d.get_shots_from_datadir(keys=[filename],return_filenames=True); shot_read=shot_read[0]
        counter=counter+1
    return None
#############################      main code
def denise_folder_process(mode,path,save_path='./datasets/test',denise_root='./',pars=None):
    if os.path.exists(os.path.join(path,'fld')):
        directory=os.path.join(path,'fld')
    else:   directory=os.path.join(path)
    # api._cmd('rm -r '+os.path.join(path,'pictures/*'))
    if 'delete_pictures_' in mode:
        print('delete pictures from ',path)
        api._cmd(f"rm -f {os.path.join(path,'*.png')}\n")
        api._cmd(f"rm -r {os.path.join(path,'pictures')}\n")
        api._cmd(f"rm -r {os.path.join(path,'fld','pictures')}\n")
        return None
    if 'delete_certain_folder'==mode:
        print(f"rm -r {os.path.join(path)}\n")
        api._cmd(f"rm -r {os.path.join(path)}\n")
    if 'delete'==mode:    # delete fld folder inpath
        print(f"rm -r {os.path.join(path,'fld')}\n")
        # api._cmd(f"rm -r {os.path.join(path,'fld')}\n")
        return None
    if 'outputs' in path:   directory=path
    ##########  precheck for empty folders
    if 'clean_empty_folders' in mode:
        if not os.path.exists(os.path.join(path,'denise_data.pkl')):
            FILE=os.path.join(directory,'seis_inversion.inp')
            if not os.path.exists(FILE):
                FILE=os.path.join(directory,'seis_forward.inp')
                if not os.path.exists(FILE):
                    api._cmd(f"rm -r {os.path.join(path)}")
                    return None
        else:
            api._cmd(f"rm -r {directory}")
            return None
    if 'delete_empty_folders'==mode:
        if not os.path.exists(os.path.join(path,'fld')):
            api._cmd(f"rm -r {os.path.join(path)}")
        else:
            return None
    if 'delete_empty_folders2'==mode:
        if not os.path.exists(os.path.join(path,'fld')) and not os.path.exists(os.path.join(path,'denise_data.pkl')):
            api._cmd(f"rm -r {os.path.join(path)}")
        else:
            return None
    if 'clean_everything_except_denise_data_pkl'==mode:
        if os.path.exists(os.path.join(path,'fld')):
            api._cmd(f"rm -r {os.path.join(path,'fld')}")
            # api._cmd(f"rm {os.path.join(path,'*.out')}\n")
            # api._cmd(f"rm {os.path.join(path,'*.err')}\n")
            api._cmd(f"rm {os.path.join(path,'*.hdf5')}\n")
            api._cmd(f"rm {os.path.join(path,'*.png')}\n")
            api._cmd(f"rm {os.path.join(path,'*.txt')}\n")
        else:
            return None
    ##########  
    # api._cmd(f"rm -r {os.path.join(path,'fld')}\n")
    # api._cmd(f"rm -f {os.path.join(path,'fwi.sh')}\n")
    # api._cmd(f"rm -f {os.path.join(path,'*.png')}\n")
    # return None
    ########### api._cmd('rm -r '+path);    return None
    ##########  precheck for empty folders
    # if not os.path.exists(directory):
    #     api._cmd(f"rm -r {os.path.join(path)}")
    #     return None
    # else:
    #     if len(os.listdir(directory))==0:
    #         api._cmd(f"rm -r {os.path.join(path)}")
    #         return None
    #     else:
    #         api._cmd(f"rm -f {os.path.join(path,'seis_forward.inp')}")
    #         api._cmd(f"rm -f {os.path.join(path,'seis_fwi_log.dat')}")
    #         return None
    ###########     check for denise_data.pkl file or alternative data sources. if not, delete folder
    print('processing '+path)
    if os.path.exists(os.path.join(path,'denise_data.pkl')):
        try:
            with open(os.path.join(path,'denise_data.pkl'),'rb') as input:
                d=pickle.load(input)
        except:
            print('problem pickling file',path)
            # api._cmd(f"rm -r {path}")
            return None
    else:
        d=api.Denise(denise_root,verbose=0)
        denise_input_files=[os.path.join(directory,'seis.inp'),os.path.join(directory,'seis_inversion.inp')]
        for input_file in denise_input_files:
            if os.path.exists(input_file):
                FILE=input_file
        if  'FILE' in globals() or 'FILE' in locals():
            d._parse_inp_file(fname=FILE)
            d.save_folder=directory
        else:
            # api._cmd(f"rm -r {os.path.join(path)}")
            return None
    ########### check for damaged fwi parameters (taper)
    # if os.path.exists(os.path.join(directory,'taper')):
    #     if len(os.listdir(os.path.join(directory,'taper')))!=0:
    #         taper2,fnames2=d.get_fwi_tapers(return_filenames=True,keys=['taper.bin'])
    #         taper,fnames=d.get_fwi_tapers(return_filenames=True)
    #         print('len(taper)=',len(taper))
    #         taper=taper[-1]
    # else:
    #     if os.path.exists(os.path.join(path,'denise_data.pkl')):
    #         taper=d.taper
    #         taper=taper.vp
    # if 'taper' in globals() or 'taper' in locals():
    # # if True:
    #     val=taper[int(taper.shape[0]/2),int(taper.shape[1]/2)]
    #     if val==0:
    #         api._cmd(f"rm -r {os.path.join(path)}")
    #         return None
    # # plot_model(taper,folder_path=os.path.join('pictures'),file_path='1.png')
    # # models_,fnames_=d.get_fwi_models(return_filenames=True,keys='vp.bin') #   vp model on last iteration
    # # vp_res=models_[0]
    # # plot_model(vp_res,folder_path=os.path.join('pictures'),file_path='2.png')
    ##########  
    # api._cmd(f"rm -r {os.path.join(directory,'su')}")
    if 'd' in globals() or 'd' in locals():
        # d._parse_inp_file(fname=os.path.join(directory,'seis_inversion.inp'))    # 'seis_inversion.inp', 'seis_forward.inp'
        # d.filename=os.path.join(directory,'seis_inversion.inp')
        # d.save_folder=directory
        if 'multi_cnn_plotting'==mode:
            plot_multi_cnn_results(path)
            ss=1
        if 'crop_zero_freqs' in mode:
            """data processing investigation.high-pass filter the seismogram, then crop zero frequencies"""
            print('data processing investigation.high-pass filter the seismogram, then crop zero frequencies')
            ###############################     apply low pass filter by cropping fourier coefficients
            d.pictures_folder=os.path.join(path,'pictures');    os.makedirs(d.pictures_folder,exist_ok=True)
            dt=d.DT
            # corner_frequency=4  #+0.8
            # corner_frequency=5  #+0.8
            corner_frequency=pars['corner_frequency']
            if 'delete_low_freqs' not in pars.keys():
                delete_low_freqs=False
            else:       delete_low_freqs=pars['delete_low_freqs']
            print('low frequency data corner frequency, ',corner_frequency)
            fmax=10     # xlim in spectrum plotting
            if os.path.exists(os.path.join(directory,'su')):
                # shots,filenames=d.get_shots_from_datadir(keys=['_p'],return_filenames=True)
                shots,filenames=d.get_shots_from_directory(os.path.join(directory,'su'),keys=['_p'],return_filenames=True)
                # shots,filenames=d.get_shots(keys=['_p'],return_filenames=True)
                print(f'Read {len(shots)} shots {shots[0].shape} into list')
                counter=0
                for shot,filename in zip(shots,filenames):
                    ################    processing stage 1
                    filtered_shot=bandpass(shot,fhi=6, dt=dt,order=8, btype='low')
                    # filtered_shot=shot
                    ################    processing stage 2
                    filtered_shot2=bandpass(shot,flo=corner_frequency,dt=dt,order=8,btype='high')
                    filtered_shot2_=bandpass(shot,flo=corner_frequency,fhi=6,dt=dt,order=8,btype='band') #for plotting purposes
                    ################    processing stage 3
                    # filtered_shot3=np.copy(filtered_shot2)
                    filtered_shot3=zero_below_freq(filtered_shot2,corner_frequency,dt, disable=False,reverse=False)
                    # filtered_shot3_=np.copy(filtered_shot2)
                    filtered_shot3_=zero_below_freq(filtered_shot2_,corner_frequency,dt, disable=False,reverse=False)
                    ################
                    plotting_flag=1
                    if plotting_flag==1:
                        shots_list=[0,22,44,66]
                        if counter in shots_list:
                            plot_shot(shot,pclip=0.05,folder_path=d.pictures_folder,file_path='shot_'+str(counter)+'.png',show=0)
                            # plot_shot(filtered_shot, pclip=0.05,folder_path=d.pictures_folder,file_path='shot_stage1.png',show=0)
                            plot_shot(filtered_shot2,pclip=0.05,folder_path=d.pictures_folder,file_path='shot'+str(counter)+'_stage2.png',show=0)
                            # plot_shot(filtered_shot2_,pclip=0.05,folder_path=d.pictures_folder,file_path='shot_stage2_precise.png',show=0)
                            # plot_shot(filtered_shot3,pclip=0.05,folder_path=d.pictures_folder,file_path='shot_stage3.png',show=0)
                            # plot_shot(filtered_shot3_,pclip=0.05,folder_path=d.pictures_folder,file_path='shot_stage3_precise.png',show=0)
                            ###############
                            freqs,ps=plot_spectrum(shot,d.DT,fmax=fmax,folder_path=d.pictures_folder,file_path='spectrum_'+str(counter)+'_original.png')
                            # freqs,ps=plot_spectrum(filtered_shot,d.DT,fmax=fmax,folder_path=d.pictures_folder,file_path='spectrum_stage1.png')
                            freqs,ps=plot_spectrum(filtered_shot2,d.DT,fmax=fmax,folder_path=d.pictures_folder,file_path='spectrum_'+str(counter)+'stage2.png')
                            # freqs,ps=plot_spectrum(filtered_shot2_,d.DT,fmax=fmax,folder_path=d.pictures_folder,file_path='spectrum_stage2_.png')
                            # freqs,ps=plot_spectrum(filtered_shot3,d.DT,fmax=fmax,folder_path=d.pictures_folder,file_path='spectrum_stage3.png')
                            # freqs,ps=plot_spectrum(filtered_shot3_,d.DT,fmax=fmax,folder_path=d.pictures_folder,file_path='spectrum_stage3_.png')
                    ################    chosen processing    
                    # filtered_shot_=filtered_shot3
                    with segyio.su.open(filename,"r+",endian='little',ignore_geometry=True) as dst:
                        if delete_low_freqs==False: dst.trace=filtered_shot2    #only high-pass filtered data
                        else:   dst.trace=filtered_shot3    #high-pass filtered data, frequencies belof Fc are zeroed out
                    ################
                    # shot_read,read_filename=d.get_shots_from_datadir(keys=[filename],return_filenames=True)
                    shot_read,read_filename=d.get_shots_from_directory(os.path.join(directory,'su'),keys=[filename],return_filenames=True)
                    shot_read=shot_read[0]

                    # # plot_shot(filtered_shot_,pclip=0.05,folder_path=d.pictures_folder,file_path='filtered_shot_.png',show=0)
                    # # # plot_shot(filtered_shot_-shot_read,pclip=0.05,folder_path=d.pictures_folder,file_path='shot_diff.png',show=0)
                    # # # shot_read_to_plot=bandpass(shot_read,flo=corner_frequency,fhi=6,dt=dt,order=8,btype='band')
                    # shot_read_to_plot=shot_read
                    # # freqs,ps=plot_spectrum(filtered_shot_,d.DT,fmax=10,folder_path=d.pictures_folder,file_path='spectrum_shot_read_to_plot.png')
                    # freqs,ps=plot_spectrum(shot_read,d.DT,fmax=10,folder_path=d.pictures_folder,file_path='spectrum_'+str(counter)+'_shot_read_to_plot.png')
                    counter=counter+1
        if 'check' in mode:
            """deprecated function. 18.07.21"""
            # perform action, if folder does not contain results
            # action: return None, or delete folder
            perform_action=False
            action='return_none'
            # action='delete_folder'
            output_file=fnmatch.filter(os.listdir(path),'*.out')    ##  investigate output file for signs of finishing the program
            output_file=sorted(output_file)
            # if not output_file:     #   deprecated
            if os.path.exists(os.path.join(directory,'model')):
                models,fnames=d.get_fwi_models(return_filenames=True)
                if fnames!=[]:
                    dd,check_3_stage=d.get_fwi_models(return_filenames=True,keys='stage_2')
                    if check_3_stage==[]:
                        perform_action=True
                        # perform_action=False
                else:
                    perform_action=True
            else:   perform_action=True
            #####   deprecated
            # else:
            #     with open(os.path.join(path,output_file[-1])) as f:
            #         lines = f.readlines()
            #     occurence=0
            #     for line in lines:
            #         if 'Total real time of program' in line:
            #             occurence=occurence+1
            #     if occurence!=2:
            #         # return None
            #         print('occurrence',occurence)
            ##  investigate misfit file
            misfit=record_misfit(os.path.join(directory,'seis_fwi_log.dat'),d)
            if misfit=='empty' or misfit.ndim==1:
                perform_action=True
            else:
                if np.isnan(misfit).any()==False:
                    models_,fnames_=d.get_fwi_models(return_filenames=True,keys='vp.bin') #   vp model on last iteration
                else:
                    perform_action=True
            if perform_action==True:
                if action=='return_none':
                    return None
                elif action=='delete_folder':
                    # api._cmd(f"echo hi")
                    api._cmd(f"rm -r {os.path.join(path)}")
                else:   
                    return None
        if 'save' in mode:
            # if os.path.exists(os.path.join(path,'fld','model')):
            #     if len(os.listdir(os.path.join(path,'fld','model'))):
            #         models_,fwi_model_names_ = d.get_fwi_models(return_filenames=True,keys=['vp.bin']) #keys=['vp']
            #         if len(models_)>0:
            #             vp_res=models_[0]
            #             vp,fnames_=d.get_fwi_start(return_filenames=True,keys=['vp'])
            #             vp_init,fnames_ = d.get_fwi_start(return_filenames=True,keys=['init.vp'])
            #             if os.path.exists(os.path.join(directory,'taper')):
            #                 if len(os.listdir(os.path.join(directory,'taper')))==0:
            #                     raise Exception("Taper file does not exist")
            #                 else:
            #                     tapers,taper_names=d.get_fwi_tapers(return_filenames=True,keys=['taper.bin'])
            #                     taper=tapers[0]     #   np.fliplr(tapers[0].T)
            #             # print(taper[-25:,22])
            #             # plot_model(taper,folder_path=os.path.join('pictures'),file_path='taper_'+path.split('/')[-1]+'.png')
            #             parameters={'file_path':os.path.join(save_path,path.split('/')[-1]+'.npz'),
            #                 'Models':       np.fliplr(vp[0].T),
            #                 'Models_init':  np.fliplr(vp_init[0].T),
            #                 'input_data':   np.fliplr(vp_res.T),
            #                 'taper':   np.fliplr(taper.T),
            #                 'dx':d.DH,
            #                 'dz':d.DH}
            # else:
            d=parse_denise_folder(path)
            if hasattr(d,'fwi_model_names')==True:
                check_passed=False
                if hasattr(d,'fwi_model_names')==True:
                    model_files_list=[]
                    for ii in d.fwi_model_names:
                        model_files_list.append(ii.split('/')[-1])
                    model_files_list=fnmatch.filter(model_files_list,'*vp_stage*')
                    if len(model_files_list)==0:
                        return None
                    print('Last result file=',model_files_list[-1])
                    for name in d.fwi_model_names:
                        # if 'vp_stage_3' in name:
                        # if 'vp_stage_2_it_4' in name:
                        #     check_passed=True;
                        if 'vp_stage_2_it_5' in name:
                            check_passed=True
                    if check_passed==True:
                        name_to_load=os.path.join(d.save_folder,'model','modelTest_vp.bin')
                        if name_to_load in d.fwi_model_names:    check2_passed=True
                        else:   check2_passed=False
                        name_to_load=os.path.join(d.save_folder,'model','modelTest_vp_stage_2_it_5.bin')
                        if name_to_load in d.fwi_model_names:    check2_passed=True
                        else:   check2_passed=False
                        # name_to_load=os.path.join(d.save_folder,'model','modelTest_vp_stage_2_it_4.bin')
                        # if name_to_load in d.fwi_model_names:    check2_passed=True
                        # else:   check2_passed=False
                        if check2_passed==True:
                            ind=d.fwi_model_names.index(name_to_load)
                            if np.isnan( d.models[ind].sum() )==False:
                                vp_res=d.models[ind]
                                ########
                                df = pd.DataFrame(np.arange(len(d.models)),columns=['index_in_array'])
                                df.insert(0,'filename',d.fwi_model_names)
                                df2=df[df['filename'].str.contains("modelTest_vp_stage")]
                                m_i_ra=np.array(d.models)[np.array(df2['index_in_array'],dtype=int)]
                                # m_i_ra=[]
                                # for i in range(tmp.shape[0]):
                                #     m_i_ra.append(tmp[i,::])
                                m_i_names=df2['filename'].to_list()
                                ########
                                if hasattr(d,'TAPER')==True:
                                    water_taper=d.TAPER
                                else:
                                    # calculated_water_taper=calculate_water_taper(np.fliplr((d.model.vp).T))
                                    taper=calculate_water_taper(np.flipud(d.model.vp))
                                    taper=np.flipud(taper)
                                    taper=taper+1
                                    taper[taper==2]=0
                                    d.set_taper(taper)
                                    water_taper=d.taper.vp
                                # Plot_image(np.fliplr(d.model.vp.T).T,Show_flag=0,Save_flag=1,Title='_m_i_ra_0',Save_pictures_path='./pictures_for_check')
                                m_i_ra2=np.swapaxes(m_i_ra,1,2)
                                m_i_ra2=np.flip(m_i_ra2,2)
                                # Plot_image(m_i_ra2[0,::].T,Show_flag=0,Save_flag=1,Title='_m_i_ra_0',Save_pictures_path='./pictures_for_check')
                                if m_i_ra2.shape[0]==10:
                                    parameters={'file_path':os.path.join(save_path,path.split('/')[-1]+'.npz'),
                                            'Models':       np.fliplr((d.model.vp).T),
                                            'Models_init':  np.fliplr((d.model_init.vp).T),
                                            'input_data':   np.fliplr(vp_res.T),
                                            'm_i_ra':   m_i_ra2,
                                            'm_i_names':   m_i_names,
                                            'taper':   np.fliplr(water_taper.T),
                                            'dx':d.DH,
                                            'dz':d.DH}
                    # else:
                    #     # denise_folder_process('plot',dir_, save_path=dataset_path)
                    #     denise_plotting2(d,path,pars=None)
            if 'parameters' in globals() or 'parameters' in locals():
                if np.isnan( parameters['input_data'].sum() )==False:
                    parameters.update({'scaling_range':pars['scaling_range'] } )
                    parameters.update({'target_smoothing_diameter':pars['target_smoothing_diameter'] } )
                    save_denise_file4(parameters)
        if 'clean_' in mode:
            # object_directory=os.path.join(directory,'su')
            # if os.path.exists(object_directory):
            #     if len(os.listdir(object_directory) ) != 0:
            #         files=os.listdir(object_directory); files=sorted(files)
            #         files1=fnmatch.filter(files,'*x.su*')
            #         files2=fnmatch.filter(files,'*y.su*')
            #         files_to_delete=files1[1:]+files2[1:]
            #         for file_name in files_to_delete:
            #             os.remove(os.path.join(object_directory,file_name))
                    # if os.path.exists(os.path.join(directory,'pictures')):
            #################
            cleaning_commands=cleaning_script(path)
            for cmd in cleaning_commands:
                api._cmd(cmd)
        if 'optimizing_space' in mode:
            # api._cmd(f"rm -r {os.path.join(path,'fld','pictures')}\n")
            if not os.path.exists( os.path.join(path,'denise_data.pkl') ):
                d=parse_denise_folder(path)
                # ##########  dump to pickle file
                with open(os.path.join(path,'denise_data.pkl'), 'wb') as output:
                    pickle.dump(d,output,protocol=4)
                # with open(os.path.join(path,'denise_data.pkl'),'rb') as input:
                #     d2=pickle.load(input)
            ##########  delete files
            api._cmd(f"rm -r {os.path.join(path,'fld','jacobian')}\n")
            api._cmd(f"rm -r {os.path.join(path,'fld','model')}\n")
            api._cmd(f"rm -r {os.path.join(path,'fld','taper')}\n")
            api._cmd(f"rm -r {os.path.join(path,'fld','start')}\n")
            api._cmd(f"rm -r {os.path.join(path,'fld','source')}\n")
            api._cmd(f"rm -r {os.path.join(path,'fld','receiver')}\n")
            api._cmd(f"rm -r {os.path.join(path,'fld','wavelet')}\n")
            api._cmd(f"rm -r {os.path.join(path,'fld','su')}\n")
            api._cmd(f"rm -r {os.path.join(path,'fld','su_modelled')}\n")
            api._cmd(f"rm -r {os.path.join(path,'fld','su_modelled_init_model')}\n")
            api._cmd(f"rm -r {os.path.join(path,'fld','log')}\n")
            api._cmd(f"rm -r {os.path.join(path,'fld','jacobian')}\n")
            api._cmd(f"rm -f {os.path.join(path,'fld','seis_forward.inp')}\n")

            # api._cmd(f"rm -f {os.path.join(path,'fld','seis_fwi_log.dat')}\n")
            # api._cmd(f"rm {os.path.join(path,'*.out')}\n")
            # api._cmd(f"rm {os.path.join(path,'*.err')}\n")
            # api._cmd(f"rm {os.path.join(path,'*.hdf5')}\n")
            api._cmd(f"find {os.path.join(path,'fld')}  -empty -type d -delete\n")

            # api._cmd(f"rm -r {os.path.join(path,'pictures')}\n")
            # api._cmd(f"rm -r {os.path.join(path,'fld')}\n")

            api._cmd(f"rm -r {os.path.join(path,'fld','pictures')}\n")
            # api._cmd(f"rm -f {os.path.join(path,'*.png')}\n")
            api._cmd(f"rm -f {os.path.join(path,'*.py')}\n")
            # api._cmd(f"rm {os.path.join(path,'*.sh')}\n")

            # api._cmd(f"rm -r {os.path.join(path,'fld')}\n")
            # api._cmd(f"rm -f {os.path.join(path,'*.txt')}\n")
            aa=1
        if 'optimize_results' in mode:
            """move contents of model and jacobian folders to denise.data.pkl file."""
            if os.path.exists(os.path.join(path,'fld','model')):
                if len(os.listdir(os.path.join(path,'fld','model'))):
                    models,fwi_model_names = d.get_fwi_models(return_filenames=True) #keys=['vp']
                    d.models=models
                    d.fwi_model_names=fwi_model_names
            if os.path.exists(os.path.join(path,'fld','jacobian')):
                if len(os.listdir(os.path.join(path,'fld','jacobian'))):
                    grads,grad_names=d.get_fwi_gradients(return_filenames=True)
                    d.grads=grads
                    d.grad_names=grad_names
            with open(os.path.join(path,'denise_data.pkl'), 'wb') as output:
                pickle.dump(d,output,pickle.HIGHEST_PROTOCOL)
            api._cmd(f"rm -r {os.path.join(path,'fld','model')}\n")
            api._cmd(f"rm -r {os.path.join(path,'fld','jacobian')}\n")
            a=1
        if 'copy_pictures' in mode:
            """ copy pictures to separate folder for better interpretation"""
            results_folder_name=path.split('/')[-2]+'_'+path.split('/')[-1]
            api._cmd(f"scp -r {os.path.join(path,'pictures')} {os.path.join('./fwi_result_pictures',results_folder_name)}")
            file_=fnmatch.filter(os.listdir(path), '*.err')[0]
            api._cmd(f"scp -r {os.path.join(path,file_)} {os.path.join('./fwi_result_pictures',results_folder_name,file_)}")
            file_=fnmatch.filter(os.listdir(path), '*.out')[0]
            api._cmd(f"scp -r {os.path.join(path,file_)} {os.path.join('./fwi_result_pictures',results_folder_name,file_)}")
        # if 'plot' in mode:        #   if mode=='plot':
        if 'plot'==mode:
            """ results_path- path containing direcory fld (denise 'outputs' folder)"""
            # denise_plotting(path,pars=None)
            denise_plotting2(d,path,pars=None)
        if mode=='plot_with_logs':
            """ results_path- path containing direcory fld (denise 'outputs' folder)"""
            ss=1
            # denise_plotting(path,pars=None)
            denise_plotting3(d,path,pars=None)    
        if 'next_fdmd' in mode:
            # if not os.path.exists( os.path.join(path,'denise_data.pkl') ):
            d=parse_denise_folder(path)
            if hasattr(d,'fwi_model_names')==True:
                model_files_list=d.fwi_model_names
                ind_vp=d.fwi_model_names.index(fnmatch.filter(model_files_list,'*vp_stage*')[-1])
                ind_vs=d.fwi_model_names.index(fnmatch.filter(model_files_list,'*vs_stage*')[-1])
                ind_rho=d.fwi_model_names.index(fnmatch.filter(model_files_list,'*rho_stage*')[-1])
                model = api.Model(d.models[ind_vp], d.models[ind_vs], d.models[ind_rho],d.DH)
                d.MFILE=os.path.join(d.save_folder,'start/inverted_model')
                d.set_model(model)
                d._write_model()
                # d.filename=os.path.join(d.save_folder,'seis_forward_next_fdmd.inp')
                # d.filename=os.path.join(d.save_folder,'trash.inp')
                # d.SEIS_FILE_P=os.path.join(d.save_folder,'su_comparison','seis_p.su')
                # d.forward(model,d.src,d.rec,disable=True,_write_acquisition_flag=0)
        if 'real_data_plotting_session' in mode:
            d=parse_denise_folder(path)
            denise_plotting3(d,path,pars=None)
            # denise_plotting2(d,path,pars=None)
    ss=1
    return None
def create_sbatch_file_for_fwi_folder(batch_file_name,results_folder):
    #SBATCH --job-name=_cgg_cnn
    model_name=results_folder.split('/')[-1]
    str1='#!/bin/bash\n'
    str1 = str1 + '#SBATCH -N 40\n'
    str1 = str1 + '#SBATCH --partition=workq\n'
    str1 = str1 + '#SBATCH -t 24:00:00\n'
    str1 = str1 + '#SBATCH --account=k1404\n'
    str1 = str1 + '#SBATCH --job-name='+'_'+model_name+'\n'
    str1 = str1 + '#SBATCH -o ' + os.path.join(results_folder,'%J_'+model_name) + '.out\n'
    str1 = str1 + '#SBATCH -e ' + os.path.join(results_folder,'%J_'+model_name) + '.err\n'
    str1=str1+'export DENISE=/project/k1404/pavel/DENISE-Black-Edition\n'
    str1=str1+f"module swap PrgEnv-gnu PrgEnv-intel\n"   
    str1=str1+f"module swap PrgEnv-cray PrgEnv-intel\n"   
    str1=str1+f"module load madagascar\n"
    str1=str1+f"module list\n"
    str1=str1+f"source /project/k1404/pavel/DENISE-Black-Edition/denise_env/bin/activate\n"
    str1=str1+f"which python\n"
    str1=str1+f"pwd\n"
    simulation_pars_file=os.path.join(results_folder,'fld','seis_inversion.inp')
    fwi_workflow_file=os.path.join(results_folder,'fld','seis_fwi.inp')
    str1=str1+f"srun --hint=nomultithread --ntasks-per-node=32 --ntasks-per-socket=16 --cpus-per-task=1 $DENISE/bin/denise {simulation_pars_file} {fwi_workflow_file}\n"
    str1=str1+f"srun -n 1 python {os.path.join(results_folder,'post_processing_script.py')}\n"

    f = open(batch_file_name,'w');  f.write(str1);f.close()
    print(batch_file_name)
    denise_root='/project/k1404/pavel/DENISE-Black-Edition'

    imports = 'import sys,os\n'
    imports = imports+f"sys.path.append(os.getcwd())\n"
    imports = imports+f"sys.path.append('/lustre/project/k1404/pavel/DENISE-Black-Edition')\n"
    imports = imports+'from F_utils import *\n'
    imports = imports+'from F_plotting import *\n'
    imports = imports+'from F_fwi import *\n'
    imports = imports+'import fnmatch\n'
    imports = imports+'from glob import glob\n'
    imports = imports+'import numpy as np\n'
    imports = imports+'import pyapi_denise_pavel as api\n'
    post_processing = imports
    post_processing = post_processing+f"results_folder='{results_folder}'\n"
    post_processing = post_processing+f"denise_root='{denise_root}'\n"
    post_processing = post_processing+f"denise_folder_process('plot',results_folder,denise_root=denise_root)\n"
    post_processing = post_processing+f"denise_folder_process('optimizing_space_',results_folder,denise_root=denise_root)\n"
    post_processing = post_processing+f"denise_folder_process('plot',results_folder,denise_root=denise_root)\n"
    post_processing_script_name = os.path.join(results_folder,'post_processing_script.py')
    print(post_processing_script_name)
    f = open(post_processing_script_name,'w');  f.write(post_processing);   f.close()
    return None
#############################      shaheen launch
def launch_batch_of_jobs(job,batch,res_folder,run_fwi_flag=1):
    number=batch[0]
    velocity_model_name=number
    if 'cgg' in str(velocity_model_name):    real_data_flag=1
    else:   real_data_flag=0
    ########################################
    if real_data_flag==0:
        model_name='model_'+str(number)
        results_folder=os.path.join(res_folder,model_name);    os.makedirs(results_folder,exist_ok=True)
        # if len(os.listdir(os.path.join(results_folder,'fld','model')) )==0:
        str1 = job + '#SBATCH --job-name='+'_'+model_name+'\n'
        str1 = str1 + '#SBATCH -o ' + os.path.join(results_folder,'%J_'+res_folder.split('/')[-1]+'_'+model_name) + '.out\n'
        str1 = str1 + '#SBATCH -e ' + os.path.join(results_folder,'%J_'+res_folder.split('/')[-1]+'_'+model_name) + '.err\n'
        str1=str1+'export DENISE=/project/k1404/pavel/DENISE-Black-Edition\n'
        # str1=str1+f"source ~/.bashrc\n"
        str1=str1+f"module swap PrgEnv-gnu PrgEnv-intel\n"   
        str1=str1+f"module swap PrgEnv-cray PrgEnv-intel\n"   
        str1=str1+f"module load madagascar\n"
        str1=str1+f"module list\n"
        str1=str1+f"source /project/k1404/pavel/DENISE-Black-Edition/denise_env/bin/activate\n"
        str1=str1+f"which python\n"
        for number in batch:
            model_name='model_'+str(number)
            results_folder=os.path.join(res_folder,model_name);
            str1=str1+f"srun -n 1 python {os.path.join(results_folder,'data_generation_script.py')}\n"
            str1=str1+'export GEN='+os.path.join(results_folder,'fld')+'\n'
            str1=str1+'srun --hint=nomultithread --ntasks-per-node=32 --ntasks-per-socket=16 --cpus-per-task=1 $DENISE/bin/denise $GEN/seis_forward.inp $GEN/seis_fwi.inp\n'
            str1=str1+f"srun -n 1 python {os.path.join(results_folder,'field_data_processing_script.py')}\n"
            str1=str1+'srun --hint=nomultithread --ntasks-per-node=32 --ntasks-per-socket=16 --cpus-per-task=1 $DENISE/bin/denise $GEN/seis_inversion.inp $GEN/seis_fwi.inp\n'
            ####### str1=str1+f"srun -n 1 python {os.path.join(results_folder,'plotting_script.py')}\n"
            str1=str1+f"srun -n 1 python {os.path.join(results_folder,'post_processing_script.py')}\n"
            cleaning_commands=cleaning_script(results_folder)
            clean_script=''
            for cmd in cleaning_commands:
                clean_script=clean_script+cmd
            # str1=str1+clean_script
        batch_file_name=os.path.join(results_folder,'fwi.sh')
        f = open(batch_file_name,'w');  f.write(str1);f.close()
    return batch_file_name
def generate_data_for_batch_of_jobs(generator_to_use,denise_fwi,gen_mode,res_folder,calculation_spacing,pars,number):
    rewrite_task=1
    model_name='model_'+str(number)
    results_folder=os.path.join(res_folder,model_name);    
    os.makedirs(results_folder,exist_ok=True)
    filename_model=os.path.join(results_folder,model_name+'.hdf5')
    if 'initial_velocity_models_source' in pars.keys():
        initial_velocity_models_source=pars['initial_velocity_models_source']
    else:
        initial_velocity_models_source='generator'
    if initial_velocity_models_source=='generator':
        generated_model,initial_model,water_taper=generator_to_use(gen_mode,model_name,dh=calculation_spacing,out_shape=pars['out_shape'])
    elif initial_velocity_models_source=='cnn_prediction':
        """     prediction for cgg model    """
        prediction_path=pars['prediction_path']
        files_path=os.listdir(prediction_path)
        files_path=fnmatch.filter(files_path,'*.npz')
        for file_ in files_path:
            print(file_)
            if model_name in file_ and '100' in file_:
                NAME=file_
                print('true ',NAME)
        # unpack data from file with predictions
        with open(os.path.join(prediction_path,NAME), 'rb') as f:
            data = np.load(f)
            generated_model = data['models']
            initial_model=data['predicted_initial_model']
            water_taper=data['water_taper']
            calculation_spacing = data['dx']
            data.close()
    elif initial_velocity_models_source=='multi_cnn':
        prediction_path=pars['prediction_path']
        D=parse_denise_folder(prediction_path)
        calculation_spacing = D.DH
        generated_model = np.rot90(D.model.vp,k=3)
        water_taper=np.rot90(D.TAPER,k=3)
        initial_model=np.rot90(D.model_init.vp,k=3)
        ############    some horizontal smoothing, deprecated
        # res=create_velocity_model_file('./fwi/multi/multi_cnn_13_special_weight_236_2/model__cgg_lin_vp_long_300_f_z/stage1',stage=1,pars={'data_mode': 'cnn_13_special', 'dx': 25, 'dz': 25, 'out_shape': [496, 150], 'dsrc': 200, 'data_gen_mode': 'pseudo_field', 'taper_shift': 0, 'last_source_position': 'nx', 'corner_frequency': 5, 'full_band': False, 'delete_low_freqs': True, 'current_data_type': 'record_cnn_data', 'prediction_path': './predictions/predictions_236', 'NNODES': 10, 'NPROCX': 4, 'NPROCY': 1, 'gen_mode': 'synthetic_data', 'extend_model_x': False})
        # generated_model=res['models']
        # water_taper=res['water_taper']
        # initial_model=res['models_init']
    if not os.path.exists(filename_model) or rewrite_task==1:
        f = h5py.File(filename_model,'w')
        f.create_dataset('models',data=generated_model)
        f.create_dataset('models_init',data=initial_model)
        f.create_dataset('water_taper',data=water_taper)
        f.create_dataset('dx', data=calculation_spacing)
        f.create_dataset('dz', data=calculation_spacing);f.close()
    return None
#############################   sfgpu codes
def fwi_full(models_path,results_path,home_directory,i_model,flag_plotting=0,iter=10):
    import m8r as sf
    # %%    Imports
    results_file_name = str((i_model))
    print('Model # ',i_model)
    # %%    Loading velocity from file
    flag_modeling = 1
    T1=datetime.datetime.now();print(T1)

    Models=load_file(models_path,'models')
    if np.max(Models)<10:
        units='km/sec'
        unit_coefficient=1
    else:
        units='m/sec'
        unit_coefficient=10e-3
    print(units)

    Models=load_file(models_path,'models')* 1000*unit_coefficient
    Models_init=load_file(models_path,'models_init')* 1000*unit_coefficient

    dx = load_file(models_path, 'dx'); dx = float(dx)
    dz = load_file(models_path, 'dz'); dz = float(dz)
    
    dx_new=50
    dz_new=50

    if dx_new!=dx or dz_new!=dz:
        Models=F_resize(Models,dx=dx,dz=dz,dx_new=dx_new,dz_new=dz_new)
        Models_init=F_resize(Models_init,dx=dx,dz=dz,dx_new=dx_new,dz_new=dz_new)
        Nx=Models.shape[0]
        tmp=models_path.split(sep="_")
        if tmp[1]=='Marmousi':
            val=12
            append=1
        elif tmp[1]=='Overthrust':
            val=12
            append=1
        else:
            append=0
        if append==1:
            water=np.ones((Nx,val))*1500
            Models=np.concatenate(      [water,Models],axis=1)*unit_coefficient
            Models_init=np.concatenate( [water,Models_init],axis=1)*unit_coefficient
            Models[:,0:18]=1500
            Models_init[:,0:18]=1500
    # %%
    if os.path.exists(results_path)==0:
        os.mkdir(results_path)
    if os.path.exists(results_path) and flag_modeling==1:
        if len(os.listdir(results_path)):
            cmd((f"rm -r {results_path+'/*'}"))
    os.chdir(results_path)
    print(os.getcwd())
    os.mkdir('./Pictures')
    f = open('./log.txt', 'w')
    sys.stdout=Tee(sys.stdout,f)
    # %%
    Models=Models[0:,0:]  #700
    Models_init=Models_init[0:,0:]
    print('Models',Models.shape)
    print('Models_init',Models_init.shape)
    # %%    first dimension of mat should be Nz, second Nz
    path2 = os.getcwd() + '/model_.rsf'
    np_to_rsf(Models,path2,d1=dx_new,d2=dz_new)
    cmd((f"sfcp < {'model_.rsf'} --out=stdout > model.rsf"));     cmd((f"sfrm {'model_.rsf'}"))

    # init1=Models_init
    # init2=init1
    # aa=F_smooth(init1[:,18:],sigma_val=int(1400/dx))

    # aa2=gaussian_filter(init1[:,18:],sigma=int(1400/dx),order=0,output=None,mode="reflect",cval=0.0,truncate=4.0)
    # init2[:,18:]=aa2
    # Models_init=init2

    path2 = os.getcwd() + '/init_model_.rsf'
    np_to_rsf(Models_init,path2,d1=dx_new,d2=dz_new)
    cmd((f"sfcp < {'init_model_.rsf'} --out=stdout > init_model.rsf"));     cmd((f"sfrm {'init_model_.rsf'}"))

    #   smooth
    # cmd((f"sfsmooth < {'init_model.rsf'} repeat=5 rect1=20 rect2=10 --out=stdout> init_model2_variant.rsf"))  # original
    # Models_init=rsf_to_np('init_model2_variant.rsf')
    # cmd((f"sfsmooth < {'vel.rsf'} repeat=20 rect1=60 rect2=50 --out=stdout> smvel.rsf"))  # original
    # cmd((f"sfsmooth < {'vel.rsf'} repeat=5 rect1=25 rect2=25 > smvel.rsf"))
    
    # %%    modeling, FWI
    cmd("sfdd < model.rsf form=native | sfput label1=Depth  unit1=m label2=Lateral unit2=m --out=stdout > vel.rsf")
    model_name='vel.rsf'
    model_orig = sf.Input(model_name)
    Nx = model_orig.int("n2");  # Nx =1531
    gxbeg=0;    #x-begining index of receivers, starting from 0
    gzbeg=2;    #z-begining index of receivers, starting from 0
    sxbeg=5;    #x-begining index of sources, starting from 0
    szbeg=2     #z-begining index of sources, starting from 0
    dt=0.004
    T_max=2.5
    nt = int(T_max / dt + 1)
    print('max time=',T_max)
    print('nt=',nt)
    Source_Spacing=800
    ns=(Nx-2*sxbeg)*dx_new//Source_Spacing+1
    # ns=(Nx-2*sxbeg)*dx_new//dx_new +1
    ng=Nx
    Precondition='n'
    filtering=0
    # ns=10
    # ng=10
    ######## Old pars
    iter1=iter
    iter2=2
    iter3=2
    iter4=2
    # iter1=2
    # iter2=2
    # iter3=2
    # iter4=2
    Rbell1=3;Rbell2=1;Rbell3=1;Rbell4=1
    Fm=3
    T_max_stage1 = 5        #   2 sec
    T_max_stage2 = 1
    T_max_stage3 = 1
    T_max_stage4 = 1        #   12 iterations (max possible length)
    F_cutoff=6
    Nplo=20
    ######## New pars
    Fm=5
    T_max_stage1 = 5
    ########
    ng_lim=Nx-2*gxbeg
    if ng>ng_lim:
        ng=ng_lim
    ns_lim=Nx-2*sxbeg
    if ns>ns_lim:
        ns=ns_lim
    jsx= Source_Spacing//dx_new  #source x-axis jump interval
    jgx= (Nx-2*gxbeg)//ng     #receiver x-axis jump interval

    print(f"Nx = {Nx}")
    print(f"jgx = {jgx}")
    print(f"jsx = {jsx}")
    print(f"ng*jgx+gxbeg = {ng*jgx+gxbeg}")
    print(f"ns*jsx+sxbeg=  {ns*jsx+sxbeg}")
    print(f"ns = {ns}")
    print(f"ng = {ng}")
    print(f"Fm = {Fm}")
    print(f"Rbell1 = {Rbell1}")
    print(f"Rbell2 = {Rbell2}")
    print(f"Rbell3 = {Rbell3}")
    print(f"Rbell4 = {Rbell4}")
    print(f"T_max_stage1 = {T_max_stage1}")
    print(f"T_max_stage2 = {T_max_stage2}")
    print(f"T_max_stage3 = {T_max_stage3}")
    print(f"T_max_stage4 = {T_max_stage4}")
    print(f"iter1 = {iter1}")
    print(f"iter2 = {iter2}")
    print(f"iter3 = {iter3}")
    print(f"iter4 = {iter4}")
    if flag_modeling==1:
        #   Stage 1########################################################################################
        cmd((f"sfgenshots< {'vel.rsf'} csdgather=n fm={Fm} amp=1 dt={dt} ns={ns} ng={ng} nt={int(T_max_stage1/dt + 1)} "
                f"sxbeg={sxbeg} szbeg={szbeg} jsx={jsx} jsz=0 gxbeg={gxbeg} gzbeg={gzbeg} jgx={jgx} jgz=0> shots.rsf"))
        # cmd((f"sfbandpass < {'shots.rsf'} nplo={Nplo} flo={F_cutoff} verb=y --out=stdout > shots_filtered.rsf"))
        # cmd((f"sfspectra < {'shots_filtered.rsf'} all=y --out=stdout > spectra_filtered.rsf"))
        # spectra_filtered=rsf_to_np('spectra_filtered.rsf')
        # spectra_filtered=spectra_filtered.astype('float64')
        cmd((f"sfspectra < {'shots.rsf'} all=y --out=stdout > spectra.rsf"))
        cmd((f"sfgpufwi  < {'init_model.rsf'} shots={'shots.rsf'} > "
        # cmd((f"sfgpufwi  < {'init_model.rsf'} shots={'shots_filtered.rsf'} > "
             f"{'vsnaps_.rsf'} grads={'grads_.rsf'} objs={'objs_.rsf'} illums={'illums_.rsf'} "
             f" niter={str(iter1)} precon={Precondition} verb=y rbell={str(Rbell1)} "))
        cmd((f"sfcp < {'grads_.rsf'} --out=stdout > grads.rsf"));   cmd((f"sfrm {'grads_.rsf'}"))
        cmd((f"sfcp < {'objs_.rsf'} --out=stdout > objs.rsf"));     cmd((f"sfrm {'objs_.rsf'}"))
        cmd((f"sfcp < {'illums_.rsf'} --out=stdout > illums.rsf"));     cmd((f"sfrm {'illums_.rsf'}"))
        cmd((f"sfcp < {'vsnaps_.rsf'} --out=stdout > vsnaps.rsf"));     cmd((f"sfrm {'vsnaps_.rsf'}"))
        cmd((f"sfcp < {'shots.rsf'} --out=stdout > shots_stdout.rsf")); 
        cmd((f"sfrm {'shots.rsf'}"))
        shots=rsf_to_np('shots_stdout.rsf')
        cmd((f"sfrm {'shots_stdout.rsf'}"))
        # %%    Record results
        true=rsf_to_np('vel.rsf')
        grads=rsf_to_np('grads.rsf')
        illums=rsf_to_np('illums.rsf')
        vsnaps=rsf_to_np('vsnaps.rsf')
        #   Stage 2########################################################################################
        init_model2=vsnaps[-1, :, :]
        path2 = os.getcwd() + '/init_model2.rsf'
        np_to_rsf(init_model2,path2,d1=dx_new,d2=dz_new)

        cmd((f"sfgenshots< {'vel.rsf'} csdgather=n fm={Fm} amp=1 dt={dt} ns={ns} ng={ng} nt={int(T_max_stage2/ dt + 1)} "
            f"sxbeg={sxbeg} szbeg={szbeg} jsx={jsx} jsz=0 gxbeg={gxbeg} gzbeg={gzbeg} jgx={jgx} jgz=0> shots2.rsf"))
        cmd((f"sfbandpass < {'shots2.rsf'} nplo={Nplo} flo={F_cutoff} verb=y --out=stdout > shots_filtered2.rsf"))
        cmd((f"sfspectra < {'shots2.rsf'} all=y --out=stdout > spectra2.rsf"))

        cmd((f"sfgpufwi  < {'init_model2.rsf'} shots={'shots2.rsf'} > "
        # cmd((f"sfgpufwi  < {'init_model2.rsf'} shots={'shots_filtered2.rsf'} > "
             f"{'vsnaps_2.rsf'} grads={'grads_2.rsf'} objs={'objs_2.rsf'} illums={'illums_2.rsf'} "
             f" niter={str(iter2)} precon={'n'} verb=y rbell={str(Rbell2)} "))
        cmd((f"sfcp < {'grads_2.rsf'} --out=stdout > grads2.rsf"));       cmd((f"sfrm {'grads_2.rsf'}"))
        cmd((f"sfcp < {'objs_2.rsf'} --out=stdout > objs2.rsf"));         cmd((f"sfrm {'objs_2.rsf'}"))
        cmd((f"sfcp < {'illums_2.rsf'} --out=stdout > illums2.rsf"));     cmd((f"sfrm {'illums_2.rsf'}"))
        cmd((f"sfcp < {'vsnaps_2.rsf'} --out=stdout > vsnaps2.rsf"));     cmd((f"sfrm {'vsnaps_2.rsf'}"))
        cmd((f"sfcp < {'shots2.rsf'} --out=stdout > shots_stdout2.rsf")) 
        cmd((f"sfrm {'shots2.rsf'}"))
        shots2=rsf_to_np('shots_stdout2.rsf')
        cmd((f"sfrm {'shots_stdout2.rsf'}"))
        grads2=rsf_to_np('grads2.rsf')
        illums2=rsf_to_np('illums2.rsf')
        vsnaps2=rsf_to_np('vsnaps2.rsf')
        # #   Stage 3########################################################################################
        # init_model3=vsnaps2[-1, :, :]
        # path2 = os.getcwd() + '/init_model3.rsf'
        # np_to_rsf(init_model3,path2,d1=dx_new,d2=dz_new)
        # cmd((f"sfgenshots< {'vel.rsf'} csdgather=n fm={Fm} amp=1 dt={dt} ns={ns} ng={ng} nt={int(T_max_stage3/ dt + 1)} "
        #     f"sxbeg={sxbeg} szbeg={szbeg} jsx={jsx} jsz=0 gxbeg={gxbeg} gzbeg={gzbeg} jgx={jgx} jgz=0> shots3.rsf"))
        # cmd((f"sfbandpass < {'shots3.rsf'} nplo={Nplo} flo={F_cutoff} verb=y --out=stdout > shots_filtered3.rsf"))

        # cmd((f"sfspectra < {'shots3.rsf'} all=y --out=stdout > spectra3.rsf"))
        # # cmd((f"sfgpufwi  < {'init_model3.rsf'} shots={'shots_filtered3.rsf'} > "
        # cmd((f"sfgpufwi  < {'init_model3.rsf'} shots={'shots3.rsf'} > "
        #      f"{'vsnaps_3.rsf'} grads={'grads_3.rsf'} objs={'objs_3.rsf'} illums={'illums_3.rsf'} "
        #      f" niter={str(iter3)} precon={'n'} verb=y rbell={str(Rbell3)} "))
        # cmd((f"sfcp < {'grads_3.rsf'} --out=stdout > grads3.rsf"));       cmd((f"sfrm {'grads_3.rsf'}"))
        # cmd((f"sfcp < {'objs_3.rsf'} --out=stdout > objs3.rsf"));         cmd((f"sfrm {'objs_3.rsf'}"))
        # cmd((f"sfcp < {'illums_3.rsf'} --out=stdout > illums3.rsf"));     cmd((f"sfrm {'illums_3.rsf'}"))
        # cmd((f"sfcp < {'vsnaps_3.rsf'} --out=stdout > vsnaps3.rsf"));     cmd((f"sfrm {'vsnaps_3.rsf'}"))
        # cmd((f"sfcp < {'shots3.rsf'} --out=stdout > shots_stdout3.rsf")) 
        # cmd((f"sfrm {'shots3.rsf'}"))
        # shots3=rsf_to_np('shots_stdout3.rsf')
        # cmd((f"sfrm {'shots_stdout3.rsf'}"))
        # grads3=rsf_to_np('grads3.rsf')
        # illums3=rsf_to_np('illums3.rsf')
        # vsnaps3=rsf_to_np('vsnaps3.rsf')
        # #   Stage 4########################################################################################
        # init_model4=vsnaps3[-1, :, :]
        # path2 = os.getcwd() + '/init_model4.rsf'
        # np_to_rsf(init_model4,path2,d1=dx_new,d2=dz_new)
        # cmd((f"sfgenshots< {'vel.rsf'} csdgather=n fm={Fm} amp=1 dt={dt} ns={ns} ng={ng} nt={int(T_max_stage4/dt + 1)} "
        #     f"sxbeg={sxbeg} szbeg={szbeg} jsx={jsx} jsz=0 gxbeg={gxbeg} gzbeg={gzbeg} jgx={jgx} jgz=0> shots4.rsf"))
        # cmd((f"sfbandpass < {'shots4.rsf'} nplo={Nplo} flo={F_cutoff} verb=y --out=stdout > shots_filtered4.rsf"))
        
        # cmd((f"sfspectra < {'shots4.rsf'} all=y --out=stdout > spectra4.rsf"))
        # # cmd((f"sfgpufwi  < {'init_model4.rsf'} shots={'shots_filtered4.rsf'} > "
        # cmd((f"sfgpufwi  < {'init_model4.rsf'} shots={'shots4.rsf'} > "
        #      f"{'vsnaps_4.rsf'} grads={'grads_4.rsf'} objs={'objs_4.rsf'} illums={'illums_4.rsf'} "
        #      f" niter={str(iter4)} precon={'n'} verb=y rbell={str(Rbell4)} "))
        # cmd((f"sfcp < {'grads_4.rsf'} --out=stdout >    grads4.rsf"));       cmd((f"sfrm {'grads_4.rsf'}"))
        # cmd((f"sfcp < {'objs_4.rsf'} --out=stdout >     objs4.rsf"));         cmd((f"sfrm {'objs_4.rsf'}"))
        # cmd((f"sfcp < {'illums_4.rsf'} --out=stdout >   illums4.rsf"));     cmd((f"sfrm {'illums_4.rsf'}"))
        # cmd((f"sfcp < {'vsnaps_4.rsf'} --out=stdout >   vsnaps4.rsf"));     cmd((f"sfrm {'vsnaps_4.rsf'}"))
        # cmd((f"sfcp < {'shots4.rsf'} --out=stdout > shots_stdout4.rsf")) 
        # cmd((f"sfrm {'shots4.rsf'}"))
        # shots4=rsf_to_np('shots_stdout4.rsf')
        # cmd((f"sfrm {'shots_stdout4.rsf'}"))
        # grads4=rsf_to_np(   'grads4.rsf')
        # illums4=rsf_to_np(  'illums4.rsf')
        # vsnaps4=rsf_to_np(  'vsnaps4.rsf')
        # ########################################################################################
        # cmd((f"sfrm {'shots_filtered1.rsf'}"))
        # cmd((f"sfrm {'shots_filtered2.rsf'}"))
        # cmd((f"sfrm {'shots_filtered3.rsf'}"))
        # cmd((f"sfrm {'shots_filtered4.rsf'}"))
    ########################################################################################
    # vsnaps=np.concatenate((vsnaps,vsnaps2,vsnaps3,vsnaps4),axis=0)
    # grads=np.concatenate((grads,grads2,grads3,grads4),axis=0)
    # illums=np.concatenate((illums,illums2,illums3,illums4),axis=0)
    ########################################################################################
    objs=rsf_to_np('objs.rsf')
    # objs2=rsf_to_np('objs2.rsf')
    # objs3=rsf_to_np('objs3.rsf')
    # objs4=rsf_to_np('objs4.rsf')
    spectra=rsf_to_np('spectra.rsf')
    spectra=spectra.astype('float64')
    # spectra2=rsf_to_np('spectra2.rsf')
    # spectra2=spectra2.astype('float64')
    shots_filtered=rsf_to_np('shots_filtered.rsf')
    shots_filtered=shots_filtered.astype('float64')
    
    if len(vsnaps.shape)==2:
        vsnaps=np.expand_dims(vsnaps,axis=0)
        grads= np.expand_dims(grads, axis=0)
        illums=np.expand_dims(illums, axis=0)
    file = sf.Input('spectra.rsf')
    nf=file.int("n1")
    df = file.float("d1")

    print(f"model.shape = {grads.shape}")
    print(f"shots.shape = {shots.shape}")
    print(f"vsnaps.shape ={vsnaps.shape}")
    print(f"objs.shape={objs.shape}")
    path= './Pictures'
    f = h5py.File('model_res' + results_file_name + '.hdf5', 'w')
    f.create_dataset('result', data=vsnaps[-1, :, :])
    f.create_dataset('dx', data=dx_new)
    f.create_dataset('dz', data=dz_new)
    f.close()

    # Plot_curves(spectra,Save_pictures_path=path,Name='spectra',Save_flag=1)
    T2 = datetime.datetime.now();
    print('Program execution time', T2 - T1)
    print(os.getcwd())

    # if flag_plotting==1:
    print(os.getcwd())
    a=np.min(true)
    b=np.max(true)
    Nit=grads.shape[0]
    R2 = np.zeros((Nit))
    for i in range(0,Nit,1):
        R2[i] = F_r2(vsnaps[i,:,:].T,true.T)
    
    Plot_image(true.T,Show_flag=0,Save_flag=1,Title="true",Save_pictures_path=path)
    # Plot_image(init1.T,Show_flag=0,Save_flag=1,Title="init1_"+numstr3(F_r2(init1.T,true.T)),Save_pictures_path=path,c_lim=[a,b])
    # Plot_image(init2.T,Show_flag=0,Save_flag=1,Title="init2_"+numstr3(F_r2(init2.T,true.T)),Save_pictures_path=path,c_lim=[a,b])

    Plot_image(Models_init.T,Show_flag=0,Save_flag=1,Title="init_stage1_"+numstr3(F_r2(Models_init.T,true.T)),Save_pictures_path=path,c_lim=[a,b])
    # Plot_image(init_model2.T,Show_flag=0,Save_flag=1,Title="init_stage2_"+numstr3(F_r2(init_model2.T,true.T)),Save_pictures_path=path,c_lim=[a,b])
    Plot_image((true-Models_init).T, Show_flag=0, Save_flag=1, Title="all_wavenumbers", Save_pictures_path=path)
    f = np.linspace(0, nf * df, num=nf)
    Plot_spectrum(f,spectra,Save_pictures_path=path,Name='spectra',Save_flag=1)
    # Plot_spectrum(f,spectra_filtered,Save_pictures_path=path,Name='spectra_filtered',Save_flag=1)
    # Plot_spectrum(f,spectra2,Save_pictures_path=path,Name='spectra2',Save_flag=1)
    # Plot_image(smvel.T,Show_flag=0,Save_flag=1,Title="smvel"+numstr3(F_r2(smvel.T,true.T)),Save_pictures_path=path,c_lim=[a,b])
    Plot_image(vsnaps[-1,:,:].T-true.T,Show_flag=0,Save_flag=1,Title="final_model_true_difference_"+numstr3(F_r2(vsnaps[-1,:,:].T,true.T)),Save_pictures_path=path)
    Plot_image(Models_init.T-true.T,Show_flag=0,Save_flag=1,Title="true_init_difference_"+numstr3(F_r2(Models_init.T,true.T)),Save_pictures_path=path)
    Plot_image(vsnaps[-1,:,:].T-Models_init.T,Show_flag=0,Save_flag=1,Title="dv_high_",Save_pictures_path=path,c_lim=[-10,10])
    Plot_image(vsnaps[-1, :, :].T-Models_init.T, Show_flag=0, Save_flag=1, Title="dv_high2_", Save_pictures_path=path,c_lim=[-500,500])
    Plot_image(vsnaps[-1, :, :].T - Models_init.T, Show_flag=0, Save_flag=1, Title="dv_high_orig_colors_", Save_pictures_path=path)
    Plot_image(vsnaps[-1, :, :].T,Show_flag=0,Save_flag=1,Title="final_"+numstr3(R2[-1]),Save_pictures_path=path,c_lim=[a,b])
    Plot_curves(objs,Save_pictures_path=path,Name= 'obj',Save_flag=1)
    # Plot_curves(objs2,Save_pictures_path=path,Name='obj2',Save_flag=1)
    # Plot_curves(objs3,Save_pictures_path=path,Name='obj3',Save_flag=1)
    # Plot_curves(objs4,Save_pictures_path=path,Name='obj4',Save_flag=1)
    Plot_curves(R2, Save_pictures_path=path, Name='model_misfit', Save_flag=1)

    end_value=Nit
    start_value=end_value-12
    start_value=0
    vector=18*np.ones((vsnaps.shape[1],1))
    if start_value<=0:
        start_value=0
    for i in range(start_value,end_value-1,100):
        Plot_image(grads[i,:,:].T,Show_flag=0,Save_flag=1,Title="grad_"+str(i),Save_pictures_path=path,Curve=vector)
        Plot_image(illums[i, :, :].T, Show_flag=0, Save_flag=1, Title="illums_" + str(i), Save_pictures_path=path,Curve=vector)
        Plot_image(vsnaps[i,:,:].T,Show_flag=0,Save_flag=1,Title="model_"+str(i)+'_'+numstr3(R2[i]),Save_pictures_path=path,c_lim=[a,b],Curve=vector)
        if i!=vsnaps.shape[0]:
            Plot_image(vsnaps[i+1,:,:].T-vsnaps[i,:,:].T,Show_flag=0,Save_flag=1,Title="dv_"+str(i),Save_pictures_path=path,Curve=vector)

    # for i in range(0,np.min([6,Nit]),1):
    #     Plot_image(grads[i,:,:].T,Show_flag=0,Save_flag=1,Title="grad_"+str(i),Save_pictures_path=path)
    #     Plot_image(vsnaps[i,:,:].T,Show_flag=0,Save_flag=1,Title="model_"+str(i)+'_'+numstr(R2[i]),Save_pictures_path=path,c_lim=[a,b])
    #     Plot_image(illums[i, :, :].T, Show_flag=0, Save_flag=1, Title="illums_" + str(i), Save_pictures_path=path)
    for i in range(0,np.min([7,shots.shape[0] ]),3):
            Plot_image(shots[i,:,:].T,Show_flag=0,Save_flag=1,Title="shot_1_"+str(i),Save_pictures_path=path,Aspect='auto')
            # Plot_image(shots2[i, :, :].T,Show_flag=0,Save_flag=1,Title="shot_2_" + str(i), Save_pictures_path=path,Aspect='auto')
            # Plot_image(shots3[i, :, :].T,Show_flag=0,Save_flag=1,Title="shot_3_" + str(i), Save_pictures_path=path,Aspect='auto')
            # Plot_image(shots_filtered[i, :, :].T,Show_flag=0,Save_flag=1,Title="shots_filtered" + str(i), Save_pictures_path=path,Aspect='auto')
    print('flag_plotting end',flag_plotting)
    # os.chdir('../../')
    os.chdir(home_directory)
    T2 = datetime.datetime.now()
    print('Program finish',T2-T1)
def fwi(models_path,results_path,home_directory,
    i_model='',flag_plotting=0,iter=10,Fm=5,delete_model_folder=1,
    t_max_stage=5,Source_Spacing=800,flag_modeling=1,Rbell1=10,
    calculation_spacing=50,crop_models_flag=1):
    import m8r as sf;   T1=datetime.datetime.now();
    print('Start of program time=',T1)
    # %%
    initial_model_txt_name='init_model'
    temporary_initial_model_txt_name='init_model_temporary'
    temporary_vel_model_txt_name='vel_temporary'
    result_model_txt_name='model_res_new_params'
    result_model_txt_name='model_res_iter_'+str(iter)
    #########
    print(results_path)
    tmp=results_path.split('/')[-1]
    MODEL_FILENAME=tmp.split('_')[1]
    # %%    Loading velocity from file
    print('models_path',models_path)
    if models_path[-4:]=='.rsf':
        print('exist=',os.path.exists(models_path))
        if not os.path.exists(models_path):
            return None
        Models_orig=rsf_to_np(models_path)
        tmp='';
        for ii in models_path.split('/')[0:-1]: tmp=tmp+ii+'/'
        tmp=tmp+initial_model_txt_name+'.rsf'
        Models_init_orig=rsf_to_np(tmp)
        file=sf.Input(tmp)
        dx=file.int("d1")
        dz = file.float("d2")
    else:
        ################   file recording
        savepath_for_dataset='../datasets/dataset_vl_gen'
        aa=results_path.split('/')
        res_dir=''
        for ii in aa[0:-1]:
            res_dir=res_dir+ii+'/'
        test_status=0
        print('hi start of recording')
        if models_path=='./models/model_Seam2_full.hdf5':
            crop_pars=return_CNN_target_size(res_dir,test_status,'model'+i_model)
            save_data_file_in_full_size(crop_pars,res_dir,test_status,savepath_for_dataset,'model'+i_model,append_new_files_flag=0)
        #     save_data_file2(crop_pars,res_dir,test_status,savepath_for_dataset,1,'model'+i_model)
        Models=load_file(models_path,'models')
        if np.max(Models)<10:
            units='km/sec'
            unit_coefficient=1
        else:
            units='m/sec'
            unit_coefficient=10e-4
        print(units)
        Models_orig=load_file(models_path,'models')* 1000*unit_coefficient
        Models_init_orig=load_file(models_path,'models_init')* 1000*unit_coefficient
        dx = load_file(models_path, 'dx'); dx = float(dx)
        dz = load_file(models_path, 'dz'); dz = float(dz)
    #################   crop model from sides along ox
    if crop_models_flag==1:
        if   MODEL_FILENAME=='Marmousi':  i1=105;i2=-109
        elif MODEL_FILENAME=='Overthrust':  i1=74;i2=-71
        elif MODEL_FILENAME=='Seam':  i1=76;i2=475
        elif MODEL_FILENAME=='Seam2':  i1=180;i2=400
        Models_orig=Models_orig[i1:i2,:]
        Models_init_orig=Models_init_orig[i1:i2,:]
    ########    resizing if needed
    dx_new=calculation_spacing;  dz_new=calculation_spacing
    # if dx_new==50:  filtering=0
    # elif dx_new==25:  filtering=0
    filtering=0
    if dx_new!=dx or dz_new!=dz:
        print('resizing')
        Models=F_resize(Models_orig,dx=dx,dz=dz,dx_new=dx_new,dz_new=dz_new)
        Models_init=F_resize(Models_init_orig,dx=dx,dz=dz,dx_new=dx_new,dz_new=dz_new)
        ##########
        # val=12;Nx=Models.shape[0];water=np.ones((Nx,val))*1500
        # Models=np.concatenate(      [water,Models],axis=1     )
        # Models_init=np.concatenate( [water,Models_init],axis=1)
        # Models[:,0:18]=1500
        # Models_init[:,0:18]=1500
        ##########
        water_sz=(np.where(Models[0,:]==1500))[0].size
        Models=Models[:,(water_sz-18):]
        Models_init=Models_init[:,(water_sz-18):]
    else:   
        Models=Models_orig; Models_init=Models_init_orig
        water_sz=(np.where(Models[0,:]==1500))[0].size
    #########
    print(Models[0,:40])
    print(Models_orig[0,:40])
    print(Models_init_orig[0,:40])
    print(Models_init[0,:40])
    # %%    Rewrite results folder or not
    rewrite_flag=1
    if os.path.exists(results_path)==0:
        os.mkdir(results_path)
    if os.path.exists(results_path+'/'+result_model_txt_name+'.hdf5') and flag_modeling==1:
        if rewrite_flag==0:
            print('Data in following path is already calculated '+results_path+'/'+result_model_txt_name+'.hdf5')
            return None
            # res_name=fnmatch.filter(os.listdir(results_path),'model_res'+'*')
            # if os.path.exists(results_path+'/vsnaps.rsf') and os.stat(results_path+'/'+res_name[0]).st_size!=0:
        else:
            if len(os.listdir(results_path)):
                # cmd((f"rm -r {results_path+'/*'}"))
                cmd((f"rm {results_path+'/'+result_model_txt_name+'.hdf5'}"))
            os.makedirs(results_path,exist_ok=True)
    os.chdir(results_path)
    saving_const_for_pictures_folder=0; pictures_folder='./Pictures'+str(saving_const_for_pictures_folder)
    while os.path.exists(pictures_folder):
        if os.path.exists(pictures_folder):
            saving_const_for_pictures_folder=saving_const_for_pictures_folder+1
            pictures_folder='./Pictures'+str(saving_const_for_pictures_folder)
    os.makedirs(pictures_folder,exist_ok=True)
    # if len(os.listdir('./Pictures')):
    #     cmd((f"rm -r {'./Pictures'}"))
    f = open('./log.txt', 'w')
    sys.stdout=Tee(sys.stdout,f)
    ########
    # Plot_image(Models_init_orig.T,Show_flag=0,Save_flag=1,dx=dx,dy=dz,
    #     Title="Models_init_orig_"+numstr(F_r2(Models_init_orig.T,Models_orig.T)),Save_pictures_path=pictures_folder)
    # Plot_image(Models_orig.T,Show_flag=0,Save_flag=1,dx=dx,dy=dz,
    #         Title="Models_orig_",Save_pictures_path=pictures_folder)
    # Plot_image(Models_init.T,Show_flag=0,Save_flag=1,dx=dx_new,dy=dz_new,
    #         Title="Models_init_"+numstr(F_r2(Models_init.T,Models.T)),Save_pictures_path=pictures_folder)
    # Plot_image(Models.T,Show_flag=0,Save_flag=1,dx=dx_new,dy=dz_new,
    #         Title="Models_",Save_pictures_path=pictures_folder)
    # %%
    print('Models',Models.shape)
    print('Models_init',Models_init.shape)
    # %%    first dimension of mat should be Nz, second Nz
    np_to_rsf(Models,'./model_.rsf',d1=dx_new,d2=dz_new);   cmd((f"sfcp < {'model_.rsf'} --out=stdout > model.rsf"));     cmd((f"sfrm {'model_.rsf'}"))
    np_to_rsf(Models_init,'./'+temporary_initial_model_txt_name+'_.rsf',d1=dx_new,d2=dz_new)
    cmd((f"sfcp < {temporary_initial_model_txt_name+'_.rsf'} --out=stdout > {temporary_initial_model_txt_name+'.rsf'}"));
    cmd((f"sfrm {temporary_initial_model_txt_name+'_.rsf'}"))
    # %%    modeling, FWI
    temporary_vel_model_txt_name
    true=rsf_to_np('model.rsf')
    model_orig = sf.Input('model.rsf')
    Nx = model_orig.int("n2");  Nz=model_orig.int("n1");
    gxbeg=2;    sxbeg=Source_Spacing//dx_new
    # if dx_new==50 and Source_Spacing==800:  sxbeg=15;
    gzbeg=4;    szbeg=gzbeg
    if dx_new==12.5: dt=0.001
    else:   dt=0.002
    # ns=(Nx-2*sxbeg)*dx_new//Source_Spacing+1
    ns=(Nx)*dx_new//Source_Spacing
    if (gxbeg+ns*Source_Spacing//dx_new)>(Nx-Source_Spacing//dx_new):   ns=ns-1
    # ns=(Nx-2*sxbeg)*dx_new//dx_new +1
    ng=(Nx-2*gxbeg)
    # ns=10
    # ng=220
    Precondition='y';   iter1=iter; 
    # Rbell1=1000//dx_new
    Rbell1=1*dx_new//dx_new   #marm pic4
    # Rbell1=5*dx_new//dx_new   #marm pic5
    # Rbell1=20*dx_new//dx_new   #marm pic6
    ####################### 5 sec on ml data
    # T_max_stage=[t_max_stage]
    # T_max_stage=[2,t_max_stage]
    # T_max_stage=[1,2,3]
    # T_max_stage=[1,2,3,4]
    #######################
    T_max_stage=t_max_stage
    F_cutoff=6
    Nplo=20
    ng_lim=Nx-2*gxbeg
    if ng>ng_lim:
        ng=ng_lim
    ns_lim=Nx-2*sxbeg
    if ns>ns_lim:
        ns=ns_lim
    jsx=Source_Spacing//dx_new  #source x-axis jump interval
    jgx=(Nx-2*gxbeg)//ng     #receiver x-axis jump interval
    nt=int(T_max_stage[0]/dt + 1)
    print(f"Source_Spacing={Source_Spacing}")
    print(f"Precondition={Precondition}")
    print(f"Nx = {Nx}");print(f", Nz={Nz}");
    print(f"jgx = {jgx}")
    print(f"jsx = {jsx}")
    print(f"gxbeg = {gxbeg}");  print(f"sxbeg=  {sxbeg}")
    print(f"gzbeg = {gzbeg}");  print(f"szbeg=  {szbeg}")
    print(f"ng*jgx+gxbeg = {ng*jgx+gxbeg}")
    print(f"ns*jsx+sxbeg=  {ns*jsx+sxbeg}")
    print(f"ns = {ns}")
    print(f"ng = {ng}")
    print(f"nt = {nt}")
    print(f"dx_new = {dx_new}")
    print(f"dz_new = {dz_new}")
    print(f"Fm = {Fm}")
    print(f"Rbell1 = {Rbell1}")
    print(f"T_max_stage1 = {T_max_stage}")
    print(f"iter1 = {iter1}")   
    x_rec=np.arange(gxbeg*dx_new,((ng-1)*jgx+gxbeg)*dx_new,jgx*dx_new)
    z_rec=np.ones_like(x_rec)*(gzbeg)*dz_new
    x_src=np.arange(sxbeg*dx_new,((ns-1)*jsx+sxbeg)*dx_new,jsx*dx_new)
    z_src=np.ones_like(x_src)*(szbeg)*dz_new
    crd={'x_rec':x_rec,'z_rec':z_rec,
        'x_src': x_src, 'z_src': z_src}
    # Plot_image(Models_init.T,Show_flag=0,Save_flag=1,dx=dx_new,dy=dz_new,
    #         Title="init_stage___",Save_pictures_path=pictures_folder,crd=crd)
    # exit()
    flag_modeling=1
    if flag_modeling==1:
        if len(T_max_stage)==1:
            T_max_stage=T_max_stage[0]
            iter1=iter[0]
            ##############  DON'T USE --out=stdout IN SFGENSHOTS, BECAUSE IT WILL BREAK SFGPUFWI
            if filtering==1:
                shots_file_name='shots_filtered.rsf'
                command = (f"sfgenshots< {'model.rsf'} csdgather=n fm={Fm} amp=1 dt={dt} ns={ns} ng={ng} nt={nt} "
                        f"sxbeg={sxbeg} szbeg={szbeg} jsx={jsx} jsz=0 gxbeg={gxbeg} gzbeg={gzbeg} jgx={jgx} jgz=0 | sfbandpass fhi=6 nphi=10> {shots_file_name}")
            else:
                shots_file_name='shots.rsf'
                command = (f"sfgenshots< {'model.rsf'} csdgather=n fm={Fm} amp=1 dt={dt} ns={ns} ng={ng} nt={int(T_max_stage/dt + 1)} "
                        f"sxbeg={sxbeg} szbeg={szbeg} jsx={jsx} jsz=0 gxbeg={gxbeg} gzbeg={gzbeg} jgx={jgx} jgz=0> {shots_file_name}")
            # cmd((f"sfgrey < {shots_file_name} color=a scalebar=y | sfpen"))
            cmd(command);   print(f'Command:\n\t{command}')
            command=(f"sfgpufwi  < {temporary_initial_model_txt_name+'.rsf'} shots={shots_file_name} > "
                f"{'vsnaps_.rsf'} grads={'grads_.rsf'} objs={'objs_.rsf'} illums={'illums_.rsf'} "
                f" niter={str(iter1)} precon={Precondition} verb=y rbell={str(Rbell1)} ")
            print(f'Command:\n\t{command}');    cmd(command);
        elif len(T_max_stage)==2:
            iter1=iter[0]
            iter2=iter[1]
            ######  stage 1 
            shots_file_name='shots_stage1.rsf'
            command = (f"sfgenshots< {'model.rsf'} csdgather=n fm={Fm} amp=1 dt={dt} ns={ns} ng={ng} nt={int(T_max_stage[0]/dt + 1)} "
                    f"sxbeg={sxbeg} szbeg={szbeg} jsx={jsx} jsz=0 gxbeg={gxbeg} gzbeg={gzbeg} jgx={jgx} jgz=0> {shots_file_name}")
            cmd(command);   print(f'Command:\n\t{command}')
            command=(f"sfgpufwi  < {temporary_initial_model_txt_name+'.rsf'} shots={shots_file_name} > "
                f"{'vsnaps_.rsf'} grads={'grads_.rsf'} objs={'objs_.rsf'} illums={'illums_.rsf'} "
                f" niter={str(iter1)} precon={Precondition} verb=y rbell={str(Rbell1)} ")
            print(f'Command:\n\t{command}');    cmd(command);
            ######  stage 2 
            vsnaps=rsf_to_np('vsnaps_.rsf')
            np_to_rsf(vsnaps[-1,:,:],'./init_model_stage2.rsf',d1=dx_new,d2=dz_new)
            shots_file_name='shots_stage2.rsf'
            command = (f"sfgenshots< {'model.rsf'} csdgather=n fm={Fm} amp=1 dt={dt} ns={ns} ng={ng} nt={int(T_max_stage[1]/dt + 1)} "
                    f"sxbeg={sxbeg} szbeg={szbeg} jsx={jsx} jsz=0 gxbeg={gxbeg} gzbeg={gzbeg} jgx={jgx} jgz=0> {shots_file_name}")
            cmd(command);   print(f'Command:\n\t{command}')
            command=(f"sfgpufwi  < {'init_model_stage2.rsf'} shots={shots_file_name} > "
                f"{'vsnaps_.rsf'} grads={'grads_.rsf'} objs={'objs_.rsf'} illums={'illums_.rsf'} "
                f" niter={str(iter2)} precon={Precondition} verb=y rbell={str(Rbell1)} ")
            print(f'Command:\n\t{command}');    cmd(command);
        elif len(T_max_stage)==3:
            iter1=iter[0]
            iter2=iter[1]
            iter3=iter[2]
            ######  stage 1 
            shots_file_name='shots_stage1.rsf'
            command = (f"sfgenshots< {'model.rsf'} csdgather=n fm={Fm} amp=1 dt={dt} ns={ns} ng={ng} nt={int(T_max_stage[0]/dt + 1)} "
                    f"sxbeg={sxbeg} szbeg={szbeg} jsx={jsx} jsz=0 gxbeg={gxbeg} gzbeg={gzbeg} jgx={jgx} jgz=0> {shots_file_name}")
            cmd(command);   print(f'Command:\n\t{command}')
            command=(f"sfgpufwi  < {temporary_initial_model_txt_name+'.rsf'} shots={shots_file_name} > "
                f"{'vsnaps_.rsf'} grads={'grads_.rsf'} objs={'objs_.rsf'} illums={'illums_.rsf'} "
                f" niter={str(iter1)} precon={Precondition} verb=y rbell={str(Rbell1)} ")
            print(f'Command:\n\t{command}');    cmd(command);
            ######  stage 2 
            vsnaps=rsf_to_np('vsnaps_.rsf')
            np_to_rsf(vsnaps[-1,:,:],'./init_model_stage2.rsf',d1=dx_new,d2=dz_new)
            shots_file_name='shots_stage2.rsf'
            command = (f"sfgenshots< {'model.rsf'} csdgather=n fm={Fm} amp=1 dt={dt} ns={ns} ng={ng} nt={int(T_max_stage[1]/dt + 1)} "
                    f"sxbeg={sxbeg} szbeg={szbeg} jsx={jsx} jsz=0 gxbeg={gxbeg} gzbeg={gzbeg} jgx={jgx} jgz=0> {shots_file_name}")
            cmd(command);   print(f'Command:\n\t{command}')
            command=(f"sfgpufwi  < {'init_model_stage2.rsf'} shots={shots_file_name} > "
                f"{'vsnaps_.rsf'} grads={'grads_.rsf'} objs={'objs_.rsf'} illums={'illums_.rsf'} "
                f" niter={str(iter2)} precon={Precondition} verb=y rbell={str(Rbell1)} ")
            print(f'Command:\n\t{command}');    cmd(command);
            ######  stage 3 
            vsnaps=rsf_to_np('vsnaps_.rsf')
            np_to_rsf(vsnaps[-1,:,:],'./init_model_stage3.rsf',d1=dx_new,d2=dz_new)
            shots_file_name='shots_stage3.rsf'
            command = (f"sfgenshots< {'model.rsf'} csdgather=n fm={Fm} amp=1 dt={dt} ns={ns} ng={ng} nt={int(T_max_stage[2]/dt + 1)} "
                    f"sxbeg={sxbeg} szbeg={szbeg} jsx={jsx} jsz=0 gxbeg={gxbeg} gzbeg={gzbeg} jgx={jgx} jgz=0> {shots_file_name}")
            cmd(command);   print(f'Command:\n\t{command}')
            command=(f"sfgpufwi  < {'init_model_stage3.rsf'} shots={shots_file_name} > "
                f"{'vsnaps_.rsf'} grads={'grads_.rsf'} objs={'objs_.rsf'} illums={'illums_.rsf'} "
                f" niter={str(iter2)} precon={Precondition} verb=y rbell={str(Rbell1)} ")
            print(f'Command:\n\t{command}');    cmd(command);
        elif len(T_max_stage)==4:
            iter1=iter[0]
            iter2=iter[1]
            iter3=iter[2]
            iter4=iter[3]
            ######  stage 1 
            shots_file_name='shots_stage1.rsf'
            command = (f"sfgenshots< {'model.rsf'} csdgather=n fm={Fm} amp=1 dt={dt} ns={ns} ng={ng} nt={int(T_max_stage[0]/dt + 1)} "
                    f"sxbeg={sxbeg} szbeg={szbeg} jsx={jsx} jsz=0 gxbeg={gxbeg} gzbeg={gzbeg} jgx={jgx} jgz=0> {shots_file_name}")
            cmd(command);   print(f'Command:\n\t{command}')
            command=(f"sfgpufwi  < {temporary_initial_model_txt_name+'.rsf'} shots={shots_file_name} > "
                f"{'vsnaps_.rsf'} grads={'grads_.rsf'} objs={'objs_.rsf'} illums={'illums_.rsf'} "
                f" niter={str(iter1)} precon={Precondition} verb=y rbell={str(Rbell1)} ")
            print(f'Command:\n\t{command}');    cmd(command);
            ######  stage 2 
            vsnaps=rsf_to_np('vsnaps_.rsf')
            np_to_rsf(vsnaps[-1,:,:],'./init_model_stage2.rsf',d1=dx_new,d2=dz_new)
            shots_file_name='shots_stage2.rsf'
            command = (f"sfgenshots< {'model.rsf'} csdgather=n fm={Fm} amp=1 dt={dt} ns={ns} ng={ng} nt={int(T_max_stage[1]/dt + 1)} "
                    f"sxbeg={sxbeg} szbeg={szbeg} jsx={jsx} jsz=0 gxbeg={gxbeg} gzbeg={gzbeg} jgx={jgx} jgz=0> {shots_file_name}")
            cmd(command);   print(f'Command:\n\t{command}')
            command=(f"sfgpufwi  < {'init_model_stage2.rsf'} shots={shots_file_name} > "
                f"{'vsnaps_.rsf'} grads={'grads_.rsf'} objs={'objs_.rsf'} illums={'illums_.rsf'} "
                f" niter={str(iter2)} precon={Precondition} verb=y rbell={str(Rbell1)} ")
            print(f'Command:\n\t{command}');    cmd(command);
            ######  stage 3 
            vsnaps=rsf_to_np('vsnaps_.rsf')
            np_to_rsf(vsnaps[-1,:,:],'./init_model_stage3.rsf',d1=dx_new,d2=dz_new)
            shots_file_name='shots_stage3.rsf'
            command = (f"sfgenshots< {'model.rsf'} csdgather=n fm={Fm} amp=1 dt={dt} ns={ns} ng={ng} nt={int(T_max_stage[2]/dt + 1)} "
                    f"sxbeg={sxbeg} szbeg={szbeg} jsx={jsx} jsz=0 gxbeg={gxbeg} gzbeg={gzbeg} jgx={jgx} jgz=0> {shots_file_name}")
            cmd(command);   print(f'Command:\n\t{command}')
            command=(f"sfgpufwi  < {'init_model_stage3.rsf'} shots={shots_file_name} > "
                f"{'vsnaps_.rsf'} grads={'grads_.rsf'} objs={'objs_.rsf'} illums={'illums_.rsf'} "
                f" niter={str(iter3)} precon={Precondition} verb=y rbell={str(Rbell1)} ")
            print(f'Command:\n\t{command}');    cmd(command);
            ######  stage 4 
            vsnaps=rsf_to_np('vsnaps_.rsf')
            np_to_rsf(vsnaps[-1,:,:],'./init_model_stage4.rsf',d1=dx_new,d2=dz_new)
            shots_file_name='shots_stage4.rsf'
            command = (f"sfgenshots< {'model.rsf'} csdgather=n fm={Fm} amp=1 dt={dt} ns={ns} ng={ng} nt={int(T_max_stage[3]/dt + 1)} "
                    f"sxbeg={sxbeg} szbeg={szbeg} jsx={jsx} jsz=0 gxbeg={gxbeg} gzbeg={gzbeg} jgx={jgx} jgz=0> {shots_file_name}")
            cmd(command);   print(f'Command:\n\t{command}')
            command=(f"sfgpufwi  < {'init_model_stage4.rsf'} shots={shots_file_name} > "
                f"{'vsnaps_.rsf'} grads={'grads_.rsf'} objs={'objs_.rsf'} illums={'illums_.rsf'} "
                f" niter={str(iter4)} precon={Precondition} verb=y rbell={str(Rbell1)} ")
            print(f'Command:\n\t{command}');    cmd(command);
    ############################################
    cmd((f"sfspectra < {shots_file_name} all=y --out=stdout > spectra.rsf"))
    cmd((f"sfcp < {'grads_.rsf'} --out=stdout > grads.rsf"));   cmd((f"sfrm {'grads_.rsf'}"))
    cmd((f"sfcp < {'objs_.rsf'} --out=stdout > objs.rsf"));     cmd((f"sfrm {'objs_.rsf'}"))
    cmd((f"sfcp < {'illums_.rsf'} --out=stdout > illums.rsf"));     cmd((f"sfrm {'illums_.rsf'}"))
    cmd((f"sfcp < {'vsnaps_.rsf'} --out=stdout > vsnaps.rsf"));     cmd((f"sfrm {'vsnaps_.rsf'}"))
    shots=rsf_to_np(shots_file_name)
    # %%    Record results
    grads=rsf_to_np('grads.rsf')
    illums=rsf_to_np('illums.rsf')
    vsnaps=rsf_to_np('vsnaps.rsf')
    objs=rsf_to_np('objs.rsf')
    spectra=rsf_to_np('spectra.rsf')
    spectra=spectra.astype('float64')
    file = sf.Input('spectra.rsf'); nf=file.int("n1");df = file.float("d1")
    delete_folders=1
    if delete_folders==1:
        # %%    Delete results
        cmd((f"sfcp < {shots_file_name} --out=stdout > shots_stdout.rsf"));cmd((f"sfrm {shots_file_name}"))
        # cmd((f"sfrm {'illums.rsf'}"))
        # cmd((f"sfrm {'vsnaps.rsf'}"))
        # cmd((f"sfrm {'grads.rsf'}"))
        cmd((f"sfrm {'spectra.rsf'}"))
        # cmd((f"sfrm {'objs.rsf'}"))
    print(f"model.shape = {grads.shape}")
    print(f"shots.shape = {shots.shape}")
    print(f"vsnaps.shape ={vsnaps.shape}")
    print(f"objs.shape={objs.shape}")
    ########################################################################################
    flag_record_the_result=1;   backup_results_to_picture_folder=1
    if flag_record_the_result==1 and flag_modeling==1:
        # name_to_save='model_res' + results_file_name
        # f = h5py.File(name_to_save + '.hdf5', 'w')
        if os.path.exists(result_model_txt_name+'.hdf5'):
            result_model_txt_name=result_model_txt_name+'_spacing_'+str(calculation_spacing)+'.hdf5'
            if os.path.exists(result_model_txt_name+'.hdf5'):
                result_model_txt_name=result_model_txt_name+'_.hdf5'
        f = h5py.File(result_model_txt_name+'.hdf5','w')
        print('recording '+result_model_txt_name+'.hdf5')
        f.create_dataset('result', data=vsnaps[-1, :, :])
        f.create_dataset('models', data=Models)
        f.create_dataset('models_init',data=Models_init)
        f.create_dataset('dx', data=dx_new)
        f.create_dataset('dz', data=dz_new)
        f.close()
        if backup_results_to_picture_folder==1:
            pictures_folder='./Pictures'+str(saving_const_for_pictures_folder)
            shutil.copyfile('vsnaps.rsf',pictures_folder+'/'+'vsnaps'+str(saving_const_for_pictures_folder)+'.rsf')
            shutil.copyfile('grads.rsf',pictures_folder+'/'+'grads'+str(saving_const_for_pictures_folder)+'.rsf')
            shutil.copyfile(result_model_txt_name+'.hdf5',pictures_folder+'/'+result_model_txt_name+'.hdf5')
            shutil.copyfile('log.txt',pictures_folder+'/log.txt')
    path=pictures_folder
    T2=datetime.datetime.now()
    print('Program execution time', T2 - T1)
    print('flag plotting=',flag_plotting)
    if flag_plotting==1:
        a=np.min(Models_init);        b=np.max(Models_init)
        Plot_image(Models_init.T,Show_flag=0,Save_flag=1,dx=dx_new,dy=dz_new,
            Title="init_stage1_"+numstr(F_r2(Models_init.T,true.T)),Save_pictures_path=path,c_lim=[a,b],crd=crd)
        Plot_image(true.T,Show_flag=0,Save_flag=1,Title="true",dx=dx_new,dy=dz_new,Save_pictures_path=path,c_lim=[a,b],crd=crd)
        true2=np.roll(true,1,axis=1) # down
        Plot_image(true2.T,Show_flag=0,Save_flag=1,dx=dx_new,dy=dz_new,Title="true_shifted",Save_pictures_path=path,crd=crd)
        Plot_image((true-true2).T,Show_flag=0,Save_flag=1,dx=dx_new,dy=dz_new,Title="vertical_grad",Save_pictures_path=path,crd=crd)
        Plot_image((true-Models_init).T, Show_flag=0, Save_flag=1,dx=dx_new,dy=dz_new,Title="all_wavenumbers", Save_pictures_path=path,crd=crd)
        # f = np.linspace(0, nf * df, num=nf)
        # Plot_spectrum(f,spectra,Save_pictures_path=path,Name='spectra',Save_flag=1)
        Nit=grads.shape[0]
        R2 = np.zeros((Nit))
        for i in range(0,Nit,1):
            R2[i] = F_r2(vsnaps[i,:,:].T,true.T)
        Plot_curves  (objs,Save_pictures_path=path,Name= 'obj',Save_flag=1)
        Plot_curves  (R2, Save_pictures_path=path, Name='model_misfit', Save_flag=1)
        Plot_image(vsnaps[-1,:,:].T-true.T,Show_flag=0,Save_flag=1,dx=dx_new,dy=dz_new,Title="final_model_true_difference_"+numstr(F_r2(vsnaps[-1,:,:].T,true.T)),Save_pictures_path=path,crd=crd)
        Plot_image(Models_init.T-true.T,Show_flag=0,Save_flag=1,dx=dx_new,dy=dz_new,Title="true_init_difference_"+numstr(F_r2(Models_init.T,true.T)),Save_pictures_path=path,crd=crd)
        Plot_image(vsnaps[-1,:,:].T-Models_init.T,Show_flag=0,Save_flag=1,dx=dx_new,dy=dz_new,Title="dv_high_",Save_pictures_path=path,crd=crd,c_lim=[-10,10])
        Plot_image(vsnaps[-1, :, :].T-Models_init.T, Show_flag=0, Save_flag=1,dx=dx_new,dy=dz_new,Title="dv_high2_", Save_pictures_path=path,crd=crd,c_lim=[-500,500])
        Plot_image(vsnaps[-1, :, :].T - Models_init.T, Show_flag=0, Save_flag=1,dx=dx_new,dy=dz_new,Title="dv_high_orig_colors_", Save_pictures_path=path,crd=crd)
        Plot_image(vsnaps[-1, :, :].T,Show_flag=0,Save_flag=1,dx=dx_new,dy=dz_new,Title="final_"+numstr(R2[-1]),Save_pictures_path=path,crd=crd,c_lim=[a,b])
        end_value=Nit-1
        start_value=0
        Iterations_plotting_step=Nit//10;
        if Iterations_plotting_step==0:    Iterations_plotting_step=1
        Iterations_plotting_step=5
        # if Nit<50:  Iterations_plotting_step=1
        vector=water_sz*np.ones((vsnaps.shape[1],1))
        if start_value<=0:
            start_value=0
        for i in range(start_value,end_value,Iterations_plotting_step):
            Plot_image(grads[i,:,:].T,Show_flag=0,Save_flag=1,dx=dx_new,dy=dz_new,Title="grad_"+str(i),Save_pictures_path=path,crd=crd,Curve=vector)
            Plot_image(illums[i, :, :].T, Show_flag=0, Save_flag=1,dx=dx_new,dy=dz_new, Title="illums_" + str(i), Save_pictures_path=path,Curve=vector)
            Plot_image(vsnaps[i,:,:].T,Show_flag=0,Save_flag=1,dx=dx_new,dy=dz_new,Title="model_"+str(i)+'_'+numstr(R2[i]),Save_pictures_path=path,crd=crd,c_lim=[a,b],Curve=vector)
            if i!=vsnaps.shape[0]:
                Plot_image(vsnaps[i+1,:,:].T-vsnaps[i,:,:].T,Show_flag=0,Save_flag=1,dx=dx_new,dy=dz_new,Title="dm_"+str(i),Save_pictures_path=path,crd=crd,Curve=vector)
                Plot_image(vsnaps[i,:,:].T-Models_init.T,Show_flag=0,Save_flag=1,dx=dx_new,dy=dz_new,Title="dv_"+str(i),Save_pictures_path=path,crd=crd,Curve=vector)
        for i in range(0,np.min([7,shots.shape[0] ]),3):
            Plot_image(shots[i,:,:].T,Show_flag=0,Save_flag=1,Title="shot_1_"+str(i),Save_pictures_path=path,Aspect='auto')
    #################################################################################
    print('flag_plotting end',flag_plotting)
    os.chdir(home_directory)
    ###################
    # savepath_for_dataset='../datasets/dataset_7886_4'
    # aa=results_path.split('/')
    # res_dir=''
    # for ii in aa[0:-1]:
    #     res_dir=res_dir+ii+'/'
    # print(res_dir)
    # save_data_file2('model'+i_model,res_dir,test_status=4,savepath=savepath_for_dataset,append_new_files_flag=1)
    ###################
    flag_record_data=0
    if flag_record_data==1:
        savepath_for_dataset='../datasets/dataset_vl_gen'
        aa=results_path.split('/')
        res_dir=''
        for ii in aa[0:-1]:
            res_dir=res_dir+ii+'/'
        test_status=0
        # save_data_file2('model'+i_model,res_dir,test_status=0,savepath=savepath_for_dataset,append_new_files_flag=1)
        crop_pars=return_CNN_target_size(res_dir,test_status,'model'+i_model)
        if models_path=='./models/model_Seam2_full.hdf5':
            save_data_file_in_full_size(crop_pars,res_dir,test_status,savepath_for_dataset,'model'+i_model,append_new_files_flag=0)
        else:
            save_data_file2(crop_pars,res_dir,test_status,savepath_for_dataset,1,'model'+i_model)
    ###################
    # savepath_for_dataset='../datasets/dataset_seam2_full'
    # os.mkdir(savepath_for_dataset)
    # aa=results_path.split('/')
    # res_dir=''
    # for ii in aa[0:-1]:
    #     res_dir=res_dir+ii+'/'
    # test_status=0
    # # save_data_file2('model'+i_model,res_dir,test_status=0,savepath=savepath_for_dataset,append_new_files_flag=1)
    # crop_pars=return_CNN_target_size(res_dir,test_status,'model'+i_model)
    # if models_path=='./models/model_Seam2_full.hdf5':
    #     save_data_file_in_full_size(crop_pars,res_dir,test_status,savepath_for_dataset,'model'+i_model,append_new_files_flag=0)
    # else:
    #     save_data_file2(crop_pars,res_dir,test_status,savepath_for_dataset,1,'model'+i_model)
    ###################
    print(os.getcwd())
    T2 = datetime.datetime.now()
    print('Program finished after',T2-T1);  print('time now=',T2)
    print('processed data in ',results_path)
    return None
#############################   not needed
def denise_forward(results_path,model_path,file_list,mode='plotting',pars=dict()):
    f=open(os.path.join(results_path,'log_mode_'+mode+'.txt'), 'w')
    sys.stdout=Tee(sys.stdout,f)
    T1=datetime.datetime.now();
    print('Start of program time=',T1)
    d = api.Denise(verbose=1)
    d._parse_inp_file(fname=os.path.join(model_path,'seis_forward.inp'))
    names_all=d._get_filenames(str(Path(d.INV_MODELFILE).parent))
    # names=d._get_filenames(str(Path(d.INV_MODELFILE).parent),keys='vp')
    vp_init,fnames=d._read_bins(str(Path(d.INV_MODELFILE).parent),(d.NX,d.NY),return_filenames=True,keys='vp.bin')
    vs_init,fnames=d._read_bins(str(Path(d.INV_MODELFILE).parent),(d.NX,d.NY),return_filenames=True,keys='vs.bin')
    rho_init,fnames=d._read_bins(str(Path(d.INV_MODELFILE).parent),(d.NX,d.NY),return_filenames=True,keys='rho.bin')
    vp_init=vp_init[0]; vs_init=vs_init[0]; rho_init=rho_init[0]
    ##############  plot all pictures
    # models,fnames=d._read_bins(str(Path(d.INV_MODELFILE).parent),(d.NX,d.NY),return_filenames=True)
    # for m, f in zip(models, fnames):
    #     f_name=(f.split('/')[-1]).split('.')[0]
    #     if f_name=='modelTest_vp_stage_2_it_3': mat2=m
    #     if f_name=='modelTest_vp': mat1=m
    #     for component in ['vp']:    #,'vs','rho'
    #         if component in f_name: 
    #             plot_model(m,f_name,folder_path=os.path.join(results_path,'pictures'),file_path=f_name+'.png')
    # plot_model(mat1-mat2,'diff',folder_path=os.path.join(results_path,'pictures'),file_path='diff.png')
    ##############  plot all pictures
    dx=d.DH;  dz=d.DH
    nz,nx=vp_init.shape
    #########################   Divide computation area between processors
    print(f'nx:\t{nx}');   print(f'nz:\t{nz}')
    d.NPROCX = 4;   d.NPROCY = 4;    d.PHYSICS = 1;     
    import multiprocessing
    n_proc=(multiprocessing.cpu_count()/(d.NPROCX*d.NPROCY)-1)*d.NPROCX*d.NPROCY
    n_proc=32
    # print('n_proc=',n_proc)
    # exit()
    if not (nx/d.NPROCX).is_integer():
        nx_new=int(np.floor(nx/d.NPROCX)*d.NPROCX    )
        print('nx/d.NPROCX is not whole number')
    else:   nx_new=int(nx)
    if not (nz/d.NPROCY).is_integer():
        nz_new=int(np.floor(nz/d.NPROCY)*d.NPROCY )
        print('ny/d.NPROCY is not whole number')
    else:        nz_new=int(nz)
    if not (n_proc/d.NPROCY/d.NPROCX).is_integer():
        print('n_proc/d.NPROCY/d.NPROCX is not whole number')
    vp_init=vp_init[0:nz_new,0:nx_new]
    # Print out everything found in the .inp file
    d.parser_report()
    d.save_folder=os.path.join(results_path,'fld','')   #    api._cmd('rm -r '+d.save_folder)
    d.set_paths(makedirs=True)
    # d.help()
    ########################    set vs=0 and rho =1000 in water for initial model
    model_init = api.Model(vp_init,vs_init,rho_init,dx)
    plot_model(np.concatenate((vp_init,vs_init,rho_init), 1),folder_path=os.path.join(d.save_folder,'pictures'),file_path='.png')
    ########################
    d.set_model(model_init)
    ###################  Acquisition allocation
    api._cmd('rm -r '+os.path.join(d.save_folder,'source') )
    api._cmd('scp -r '+os.path.join(model_path,'source')+' '+os.path.join(d.save_folder,'source') )
    api._cmd('rm -r '+os.path.join(d.save_folder,'receiver') )
    api._cmd('scp -r '+os.path.join(model_path,'receiver')+' '+os.path.join(d.save_folder,'receiver') )
    ###################
    parallelization_command='mpirun -np '+str(n_proc);  
    if mode=='generate_task_files':
        #   generate forward .inp file
        d.filename=os.path.join(d.save_folder,'seis_forward.inp')
        d.MFILE=os.path.join(d.save_folder,'start/model')
        # d.forward(model, src, rec, run_command=parallelization_command,disable=False)
        d.forward(model_init, [], [], run_command=parallelization_command,disable=True)
    #   Inspect shots
    flag_plot_shots=1
    if flag_plot_shots==1:
        shots = d.get_shots(keys=['_y'])
        if shots!=[]:
            print(f'Read {len(shots)} shots {shots[0].shape} into list')
            # Plot 2 shots and their respective power spectra
            # it_list=[int(np.floor(x)) for x in np.linspace(0, len(shots)-1, 2)]
            it_list=np.arange(0,len(shots))
            for i in it_list:
                plot_shot(shots[i], pclip=0.05, title=str(i),folder_path=os.path.join(d.save_folder,'pictures'),file_path='shot'+str(i)+'.png',show=0)
                freqs,ps=plot_spectrum(shots[i], d.DT, fmax=30,folder_path=os.path.join(d.save_folder,'pictures'),file_path='spectrum_shots_'+str(i)+'.png')
    T2=datetime.datetime.now()
    print('Program finished after',T2-T1);  print('time now=',T2)
    print('processed data in ',results_path)
    return None
def denise_plotting_vanilla_inversion(directory):
    d = api.Denise(verbose=1)
    d._parse_inp_file(fname=os.path.join(directory,'DENISE_marm_OBC.inp'))    # 'seis_inversion.inp', 'seis_forward.inp'
    # d.help()
    d.save_folder=directory;    print('Plotting pictures in directory ',os.path.join(d.save_folder,'pictures'))
    os.makedirs(os.path.join(d.save_folder,'pictures'),exist_ok=True)
    # from pathlib import Path;   [f.unlink() for f in Path(os.path.join(d.save_folder,'pictures')).glob("*") if f.is_file()] 
    #####################################   load src,rec
    xrec=[20]
    yrec=[20]
    xsrc=[20]
    ysrc=[20]
    rec = api.Receivers(xrec, yrec)
    src = api.Sources(xsrc, ysrc)    
    #####################################   load velocity models
    dx=20
    vp,fnames = d.get_fwi_start(return_filenames=True,keys=['marmousi_II_marine.vp'])
    vs,fnames = d.get_fwi_start(return_filenames=True,keys=['marmousi_II_marine.vs'])
    rho,fnames = d.get_fwi_start(return_filenames=True,keys=['marmousi_II_marine.rho'])
    model = api.Model(vp[0],vs[0],rho[0], dx)
    vplim = {'vmax': model.vp.max(), 'vmin': model.vp.min()}
    vslim = {'vmax': model.vs.max(), 'vmin': model.vs.min()}
    rholim={'vmax': model.rho.max(), 'vmin': model.rho.min()}
    dv_lim={'vmax': 500, 'vmin': -500}
    vlims = {'vp': vplim, 'vs': vslim,'rho': rholim,'dv': dv_lim}
    vp_init,fnames = d.get_fwi_start(return_filenames=True,keys=['marmousi_II_start_1D.vp'])
    vs_init,fnames = d.get_fwi_start(return_filenames=True,keys=['marmousi_II_start_1D.vs'])
    rho_init,fnames = d.get_fwi_start(return_filenames=True,keys=['marmousi_II_start_1D.rho'])
    model_init = api.Model(vp_init[0],vs_init[0],rho_init[0],dx)
    for component in ['vp','vs','rho']:   
        f_name='init_'+component;   r2='_r2(initial,true)_'+numstr(F_r2(getattr(model_init,component),getattr(model,component)));   
        plot_acquisition(getattr(model_init,component),model,src,rec,f_name+r2,**vlims[component],folder_path=os.path.join(d.save_folder,'pictures'),file_path=f_name+r2+'.png')
        f_name='true'+component+'_'
        plot_acquisition(getattr(model,component),model,src,rec,f_name,**vlims[component],folder_path=os.path.join(d.save_folder,'pictures'),file_path=f_name+'.png')
    #####################################   shots
    plot_shots=0
    if plot_shots==1:
        shots = d.get_shots(keys=['_y'])
        if shots!=[]:
            print(f'Read {len(shots)} shots {shots[0].shape} into list')
            it_list=[int(np.floor(x)) for x in np.linspace(0, len(shots)-1, 2)]
            it_list=[0]
            for i in it_list:
                plot_shot(shots[i], pclip=0.05, title=str(i),folder_path=os.path.join(d.save_folder,'pictures'),
                    file_path='shot'+str(i)+'.png',show=0)
                freqs,ps=plot_spectrum(shots[i], d.DT, fmax=30,folder_path=os.path.join(d.save_folder,'pictures'),file_path='spectrum_shots_'+str(i)+'.png')
    ##################  plot gradients
    grads, fnames = d.get_fwi_gradients(return_filenames=True)
    for m, f in zip(grads, fnames):
        f_name=(f.split('/')[-1]).split('.')[0]
        plot_acquisition(m,model,src,rec,f_name,folder_path=os.path.join(d.save_folder,'pictures'),file_path=f_name+'.png')
    ##################  plot inverted models
    # models, fnames = d.get_fwi_models([component, 'stage'], return_filenames=True)
    # plot_model(m ,**vlims[component],folder_path=os.path.join(d.save_folder,'pictures'),file_path=f_name+'.png')
    models,fnames = d.get_fwi_models( return_filenames=True)
    for m, f in zip(models, fnames):
        f_name=(f.split('/')[-1]).split('.')[0]
        # for component in ['vp']:    #,'vs','rho'
        for component in ['vp','vs','rho']:    #,'vs','rho'
            if component in f_name: 
                r2='_r2(m_i,true)_'+numstr(F_r2(m,getattr(model,component)))
                plot_acquisition(m,model,src,rec,f_name+r2,**vlims[component],folder_path=os.path.join(d.save_folder,'pictures'),file_path=f_name+r2+'.png')
                plot_acquisition(m-getattr(model_init,component),model,src,rec,f_name+r2,**vlims['dv'],folder_path=os.path.join(d.save_folder,'pictures'),file_path=f_name+'_dv(m_i,m_init)'+r2+'.png')
                # plot_acquisition(m-getattr(model,component),model,src,rec,f_name,**vlims['dv'],folder_path=os.path.join(d.save_folder,'pictures'),file_path=f_name+'_difference_with_true.png')
    ##################  plot tapers
    models, fnames = d.get_fwi_tapers(return_filenames=True)
    for m, f in zip(models,fnames):
        f_name=(f.split('/')[-1]).split('.')[0]
        # plot_model(m,folder_path=os.path.join(d.save_folder,'pictures'),file_path=f_name+'.png')
        plot_acquisition(m,model,src,rec,f_name,folder_path=os.path.join(d.save_folder,'pictures'),file_path=f_name+'.png')
    aa=m[-30:,2]
    aa=m[-30:,2]
    print(m[-30:,2])
    m1=model.vp[-30:,2]
    m2=model.vs[-30:,2]
    m3=model.rho[-30:,2]
    m4=m[-30:,2]
    #####################################   plot fwi objective curve
    filename =os.path.join(directory,'seis_fwi_log.dat')
    plot_fwi_misfit(filename,d)
    #####################################  Plot step length evolution
    plot_fwi_step_length(filename,d)
    return None
def sbatch_script(model_name,launch_path,hours=3,server='ibex_volta_gpu'):
    #   submit sbatch, prepare batch file
    procname = 'Jobs/fwi'
    testfile = 'fwi.out'
    str1 = '#!/bin/bash\n'
    # str1=str1+'#SBATCH--account=k1208\n'
    str1 = str1 + '#SBATCH --job-name=' + procname + '\n'
    # str1=str1+'#SBATCH -o '+procname+ '%'+'.out\n'
    # str1=str1+'#SBATCH -e '+procname+ '%'+'.err\n'
    str1 = str1 + '#SBATCH -o ' + procname + '.%J' + '_model' + model_name + '.out\n'
    str1 = str1 + '#SBATCH -e ' + procname + '.%J' + '_model' + model_name + '.err\n'
    if server=='ibex_any_gpu':
        str1 = str1 + '#SBATCH --gres=gpu:1\n'
        str1 = str1 + '##SBATCH --constraint=[v100]\n'
        str1 = str1 + f"#SBATCH --time={str(hours)}:00:00\n"
        str1 = str1 + '#SBATCH --partition=batch\n'
        str1 = str1 + '#SBATCH --mem=32G\n'
        str1 = str1 + 'pwd\n'
        str1 = str1 + 'source ~/.bashrc_ibex\n'
        str1 = str1 + 'conda activate t_env\n'
    if server=='ibex_volta_gpu':
        str1 = str1 + '#SBATCH --gres=gpu:1\n'
        str1 = str1 + '#SBATCH --constraint=[v100]\n'
        str1 = str1 + f"#SBATCH --time={str(hours)}:00:00\n"
        str1 = str1 + '#SBATCH --partition=batch\n'
        str1 = str1 + '#SBATCH --mem=32G\n'
        str1 = str1 + 'pwd\n'
        str1 = str1 + 'source ~/.bashrc_ibex\n'
        str1 = str1 + 'conda activate t_env\n'
    if server=='neser':
        str1 = str1 + '#SBATCH --nodes=1\n'
        str1 = str1 + '#SBATCH --ntasks=1\n'
        str1 = str1 + f"#SBATCH --time={str(hours)}:00:00\n"
        str1 = str1 + '#SBATCH --partition=tesla\n'
        # str1 = str1 + '#SBATCH --mem=128G\n'
        str1 = str1 + 'pwd\n'
        str1 = str1 + 'source ~/.bashrc\n'
        str1 = str1 + 'conda activate tfk\n'
    str1 = str1 + 'module use /home/plotnips/install_madagascar/modulefiles\n'
    str1 = str1 + 'module load madagascar/git\n'
    str1 = str1 + 'module load python/3.7.0\n'
    str1 = str1 + 'module load cuda/10.1.105\n'
    launch_string = 'srun ~/anaconda3/envs/t_env/bin/python ' + launch_path + '\n'
    str1 = str1 + launch_string
    return str1
def run_fwi(model_name,save_folder,iterations,flag_submitjob_or_exec_locally,flag_plotting=0,
    t_max_stage=16,freq=5,Source_Spacing=800):
    print('model:', model_name)
    results_folder = save_folder+'/model'+model_name
    models_path = './models/model' + model_name + '.hdf5'
    if os.path.exists(results_folder) == 0:
        os.mkdir(results_folder)
    name=results_folder+'/'+'fwi.sh'
    print('hi1')
    if flag_submitjob_or_exec_locally==1:
        str1 = 'import os,sys\n'
        str1 = str1 + "sys.path.append('../')\n"
        str1 = str1 + "sys.path.append('./')\n"
        str1 = str1 + 'from F_utils import *\n'
        str1 = str1 + 'print("hi")\n'
        str1 = str1 + 'print(os.getcwd())\n'
        str1 = str1 + 'print(os.listdir())\n'
        str1 = str1 + 'from F_fwi import *\n'
        str1 = str1 + 'import m8r as sf\n'
        print('FLAG PLOOTTING=',flag_plotting)
        str1 = str1 + f"fwi('{models_path}','{results_folder}','{os.getcwd()}','{model_name}',flag_plotting={flag_plotting},iter={iterations},Fm={freq},Source_Spacing={Source_Spacing})\n"
        launch_path = './Jobs' + '/launch_script' +model_name+ '.py'
        #   prepare launch script
        f = open(launch_path,'w')
        f.write(str1)
        f.close()
        #  prepare sbatch script
        str1=sbatch_script(model_name,launch_path,hours=4,server='ibex_any_gpu')
        f = open(name,'w')
        f.write(str1)
        f.close()
        # submit it
        command = 'sbatch ' + name
        os.system('%s' % command)
        print('hi3')
    else:
        fwi(models_path,results_folder,os.getcwd(),i_model=model_name,
        flag_plotting=flag_plotting,iter=iterations,Fm=freq,t_max_stage=t_max_stage,
        Source_Spacing=Source_Spacing)
    return None
def run_fwi_cyclically(model_name,save_folder,iterations,flag_submitjob_or_exec_locally,flag_plotting=0,
    t_max_stage=16,freq=5,Source_Spacing=800,cycle_quantity=3):
    print('model:', model_name)
    results_folder = save_folder+'/model'+model_name
    models_path = './models/model' + model_name + '.hdf5'
    if os.path.exists(results_folder) == 0:
        os.mkdir(results_folder)
    print('hi1')
    if flag_submitjob_or_exec_locally==1:
        name=results_folder+'/'+'fwi.sh'
        str1 = 'import os,sys\n'
        str1 = str1 + "sys.path.append('../')\n"
        str1 = str1 + "sys.path.append('./')\n"
        str1 = str1 + 'from F_utils import *\n'
        str1 = str1 + 'print("hi")\n'
        str1 = str1 + 'print(os.getcwd())\n'
        str1 = str1 + 'print(os.listdir())\n'
        str1 = str1 + 'from F_fwi import *\n'
        str1 = str1 + 'import m8r as sf\n'
        str1 = str1 + f"models_path='{models_path}'\n"
        str1 = str1 + f"for cycle_number in range({cycle_quantity}):\n"
        str1 = str1 + f"\tif (cycle_number-1)=={cycle_quantity}:\n"
        str1 = str1 + f"\t\tflag_plotting=1\n"
        str1 = str1 + f"\telse:\n"
        str1 = str1 + f"\t\tflag_plotting=0\n"
        str1 = str1 + f"\tprint('models_pathmodels_pathmodels_path!!!!!!=',models_path)\n"
        str1 = str1 + f"\tfwi(models_path,'{results_folder}','{os.getcwd()}','{model_name}',flag_plotting={flag_plotting},iter={iterations},Fm={freq},Source_Spacing={Source_Spacing})\n"
        str1 = str1 + f"\tmodels_path='{results_folder}'+'/model_res'+'{model_name}'+'.hdf5'\n"
        # str1 = str1 + f"\tprint('models_pathmodels_pathmodels_path!!!!!!=',models_path)\n"
        launch_path = './Jobs' + '/launch_script' +model_name+ '.py'
        #   prepare launch script
        f = open(launch_path,'w')
        f.write(str1)
        f.close()
        #  prepare sbatch script
        str1=sbatch_script(model_name,launch_path,hours=12,server='ibex_volta_gpu')
        f = open(name,'w')
        f.write(str1)
        f.close()
        # submit it
        command = 'sbatch ' + name
        os.system('%s' % command)
        print('hi3')
    else:
        for cycle_number in range(cycle_quantity):
            fwi(models_path,results_folder,os.getcwd(),i_model=model_name,
                flag_plotting=flag_plotting,iter=iterations,Fm=freq,t_max_stage=t_max_stage,
                Source_Spacing=Source_Spacing)
            models_path=results_folder+'/model_res'+model_name+'.hdf5'
    return None
def run_fwi_full(model_name,save_folder,iterations,flag_submitjob_or_exec_locally,flag_plotting=1):
    print('model:', model_name)
    results_folder = save_folder+'/model'+model_name
    models_path = './models/model' + model_name + '.hdf5'
    if os.path.exists(results_folder) == 0:
        os.mkdir(results_folder)
    name=results_folder+'/'+'fwi.sh'
    print('hi1')
    if flag_submitjob_or_exec_locally == 1:
        str1 = 'import os,sys\n'
        str1 = str1 + "sys.path.append('../')\n"
        str1 = str1 + "sys.path.append('./')\n"
        str1 = str1 + 'from F_utils import *\n'
        str1 = str1 + 'print("hi")\n'
        str1 = str1 + 'print(os.getcwd())\n'
        str1 = str1 + 'print(os.listdir())\n'
        str1 = str1 + 'from F_fwi import *\n'
        str1 = str1 + 'import m8r as sf\n'
        print('FLAG PLOOTTING=',flag_plotting)
        str1 = str1 + f"fwi_full('{models_path}','{results_folder}','{os.getcwd()}','{model_name}',flag_plotting='{flag_plotting}',iter='{iterations}')\n"
        launch_path = './Jobs' + '/launch_script' + model_name + '.py'

        print('hi2')
        #   prepare launch script
        f = open(launch_path,'w')
        f.write(str1)
        f.close()
        #  prepare sbatch script
        str1=sbatch_script(model_name,launch_path,hours=4)
        f = open(name,'w')
        f.write(str1)
        f.close()
        # submit it
        command = 'sbatch ' + name
        os.system('%s' % command)
        print('hi3')
    else:
        fwi_full(models_path,results_folder,os.getcwd(),i_model=model_name,flag_plotting=flag_plotting,iter=iterations)
    return None
def plot_figures(models_path,results_path,i_model):
    import m8r as sf
    home_directory=os.getcwd()
    os.chdir(results_path)
    path='./Pictures'
    Models=rsf_to_np('model.rsf')
    Models_init=rsf_to_np('init_model.rsf')
    fwi_result=load_file('model_res'+i_model+'.hdf5','result')
    file = sf.Input('model.rsf')
    dx = file.float("d1")
    dz = file.float("d2")   
    true=rsf_to_np('vel.rsf')
    a=np.min(true);        b=np.max(true)
    Plot_image(true.T,Show_flag=0,Save_flag=1,Title="true",Save_pictures_path=path)
    true2=np.roll(true,1,axis=1) # down
    Plot_image(true2.T,Show_flag=0,Save_flag=1,Title="true_shifted",Save_pictures_path=path)
    Plot_image((true-true2).T,Show_flag=0,Save_flag=1,Title="vertical_grad",Save_pictures_path=path)
    Plot_image(Models_init.T,Show_flag=0,Save_flag=1,Title="initial model, R2(initial model,true model)="+numstr(F_r2(Models_init.T,true.T)),Save_pictures_path=path,c_lim=[a,b])
    Plot_image(fwi_result.T-true.T,Show_flag=0,Save_flag=1,Title="final_model_true_difference_"+numstr(F_r2(fwi_result.T.T,true.T)),Save_pictures_path=path)
    Plot_image(Models_init.T-true.T,Show_flag=0,Save_flag=1,Title="true_init_difference_"+numstr(F_r2(Models_init.T,true.T)),Save_pictures_path=path)
    Plot_image(fwi_result.T-Models_init.T,Show_flag=0,Save_flag=1,Title="dv_high_",Save_pictures_path=path,c_lim=[-10,10])
    Plot_image(fwi_result.T-Models_init.T, Show_flag=0, Save_flag=1, Title="dv_high2_", Save_pictures_path=path,c_lim=[-500,500])
    Plot_image(fwi_result.T - Models_init.T, Show_flag=0, Save_flag=1, Title="dv_high_orig_colors_", Save_pictures_path=path)
    Plot_image(fwi_result.T,Show_flag=0,Save_flag=1,Title="fwi result, R2(fwi result,true model)="+numstr(F_r2(fwi_result.T,true.T)),Save_pictures_path=path,c_lim=[a,b])
    os.chdir(home_directory)
    return None
def find_best_initial_model(results_path,home_directory,
    flag_plotting=0,iter=10,Fm=5,delete_model_folder=1,
    t_max_stage=8,Source_Spacing=800):
    import m8r as sf
    ####################################
    # initial_model_txt_name='models_init_ideal'
    initial_model_txt_name='models_init_ideal_better_guess2'
    result_model_txt_name='models_res_better_guess2'
    os.chdir(results_path)
    print(os.listdir(results_path))
    ####################################
    f = open('./log_for_function_find_best_initial_model.txt', 'w')
    sys.stdout=Tee(sys.stdout,f)
    ####################################
    # results_file_name='model_res_ideal'+(results_path.split('/')[-1]).split('model')[-1]+'.hdf5'
    # if os.path.exists(results_file_name):
    #     cmd((f"rm {results_file_name}"))
    # if os.path.exists('model_res_ideal.hdf5'):
    #     cmd((f"rm {'model_res_ideal.hdf5'}"))
    # if os.path.exists(results_file_name) or os.path.exists('model_res_ideal.hdf5'):
    #     print('Data is already calculated, stop program')
    #     return None
    rewrite_flag=1
    if rewrite_flag==1:
        cmd((f"rm -r {initial_model_txt_name+'.rsf'}"))
        cmd((f"rm -r {result_model_txt_name+'.hdf5'}"))
    if os.path.exists(initial_model_txt_name+'.rsf'):   search_for_optimum_initial_model=0
    else:   search_for_optimum_initial_model=1
    ####################################
    true=rsf_to_np('vel.rsf')
    model_orig = sf.Input('vel.rsf')
    if os.path.exists('model.rsf'):
        cmd((f"sfrm {'model.rsf'}"))
    if not os.path.exists('./init_model_search'):
        cmd((f"mkdir {'./init_model_search'}"))
    # if not os.path.exists('./init_model_search2'):
    #     cmd((f"mkdir {'./init_model_search2'}"))
    Nx = model_orig.int("n2");  dx=model_orig.int("d2")
    Nz = model_orig.int("n1");  dz=model_orig.int("d1")
    gxbeg=0;    #x-begining index of receivers, starting from 0
    gzbeg=2;    #z-begining index of receivers, starting from 0
    sxbeg=5;    #x-begining index of sources, starting from 0
    sxbeg_for_check_shot=200
    szbeg=2     #z-begining index of sources, starting from 0
    dt=0.004
    T_max_stage1=t_max_stage
    ns=(Nx-2*sxbeg)*dx//Source_Spacing +1
    ng=Nx
    ng_lim=Nx-2*gxbeg
    if ng>ng_lim:
        ng=ng_lim
    ns_lim=Nx-2*sxbeg
    if ns>ns_lim:
        ns=ns_lim
    jsx= Source_Spacing//dx  #source x-axis jump interval
    jgx= (Nx-2*gxbeg)//ng     #receiver x-axis jump interval
    Precondition='y'
    iter1=iter;
    Rbell1=1
    ####################################
    print(f"Precondition = {Precondition}")
    print(f"Source_Spacing = {Source_Spacing}")
    print(f"Nx = {Nx}")
    print(f"jgx = {jgx}")
    print(f"jsx = {jsx}")
    print(f"ng*jgx+gxbeg = {ng*jgx+gxbeg}")
    print(f"ns*jsx+sxbeg=  {ns*jsx+sxbeg}")
    print(f"ns = {ns}")
    print(f"ng = {ng}")
    print(f"Fm = {Fm}")
    print(f"Rbell1 = {Rbell1}")
    print(f"T_max_stage1 = {T_max_stage1}")
    print(f"iter1 = {iter1}")
    ####################################
    cmd((f"sfgenshots< {'vel.rsf'} csdgather=n fm={Fm} amp=1 dt={dt} ns={1} ng={ng} nt={int(T_max_stage1/dt + 1)} "
            f"sxbeg={sxbeg_for_check_shot} szbeg={szbeg} jsx={jsx} jsz=0 gxbeg={gxbeg} gzbeg={gzbeg} jgx={jgx} jgz=0> shots_true.rsf"))
    true_shots=rsf_to_np('shots_true.rsf');
    if len(true_shots.shape)==2:true_shots=np.expand_dims(true_shots,axis=0)
    Models=1500*np.ones((Nx,Nz))
    water_thickness = int(18*50/dx)
    zz = np.arange(Nz) * dz
    zz=zz-17*50
    zz = np.tile(zz,(Nx,1))
    zz[zz<0]=0
    ####################################
    # models_init_=1500+0.9*zz; 
    # models_init_min=1500+0.3*zz
    # models_init_check=1500+0.6*zz
    # models_init_max=1500+1.9*zz
    # a=true.min();b=true.max()
    # Plot_image(true.T,Show_flag=0,Save_flag=1,c_lim=[a,b],Title="true_orig",Save_pictures_path='./init_model_search')
    # Plot_image(models_init.T,Show_flag=0,Save_flag=1,c_lim=[a,b],Title="models_init",Save_pictures_path='./init_model_search')
    # Plot_image(models_init_min.T,Show_flag=0,Save_flag=1,c_lim=[a,b],Title="models_init_min",Save_pictures_path='./init_model_search')
    # Plot_image(models_init_check.T,Show_flag=0,Save_flag=1,c_lim=[a,b],Title="models_init_check",Save_pictures_path='./init_model_search')
    # Plot_image(models_init_max.T,Show_flag=0,Save_flag=1,c_lim=[a,b],Title="models_init_max",Save_pictures_path='./init_model_search')
    ####################################
    if search_for_optimum_initial_model==1:
        true_shots=rsf_to_np('shots_true.rsf')
        # velocity_gradients=np.arange(0.3,2.9,step=0.02); data_misfit=[];counter=0
        velocity_gradients=np.arange(0.2,2.5,step=0.02); data_misfit=[];counter=0
        for velocity_gradient in velocity_gradients:
            models_init_=1500+velocity_gradient*zz
            np_to_rsf(models_init_,'./models_init_.rsf',d1=dx,d2=dz)
            cmd((f"sfgenshots< {'models_init_.rsf'} csdgather=n fm={Fm} amp=1 dt={dt} ns={1} ng={ng} nt={int(T_max_stage1/dt + 1)} "
                f"sxbeg={sxbeg_for_check_shot} szbeg={szbeg} jsx={jsx} jsz=0 gxbeg={gxbeg} gzbeg={gzbeg} jgx={jgx} jgz=0> shots_init_.rsf"))
            init_shots=rsf_to_np('shots_init_.rsf');
            if len(init_shots.shape)==2:init_shots=np.expand_dims(init_shots,axis=0)
            ro=abs(true_shots-init_shots)   #mae loss
            tmp=0
            for i in range(ro.shape[0]):
                tmp=tmp+ro[i,:,:].sum()/ro[i,:,:].size
            tmp=tmp/ro.shape[0]
            print('misfit=',tmp,', velocity gradient=',velocity_gradient)
            # Plot_image(true_shots[0,:,:].T,Show_flag=0,Save_flag=1,Title="true_shots_0_"+str(counter)+'_'+str(tmp),Save_pictures_path='./init_model_search',Aspect='auto')
            # Plot_image(init_shots[0,:,:].T,Show_flag=0,Save_flag=1,Title="init_shots_0_"+str(counter)+'_'+str(tmp),Save_pictures_path='./init_model_search',Aspect='auto')
            # Plot_image(ro[0,:,:].T,Show_flag=0,Save_flag=1,Title="ro_shots_0_"+str(counter)+'_'+str(tmp),Save_pictures_path='./init_model_search',Aspect='auto')
            data_misfit.append(tmp)
            counter=counter+1
        ind=[]
        for k in range(len(data_misfit)):
            if np.isnan(data_misfit[k])==True:  ind.append(k)
        data_misfit_=data_misfit
        velocity_gradients_=velocity_gradients
        # data_misfit_=data_misfit_.tolist()
        velocity_gradients_=velocity_gradients_.tolist()
        delete_multiple_element(data_misfit_,ind)
        delete_multiple_element(velocity_gradients_,ind)
        data_misfit_=np.array(data_misfit_)
        velocity_gradients_=np.array(velocity_gradients_)
        chosen_gradient_index=np.where(data_misfit_==data_misfit_.min())
        velocity_gradient=velocity_gradients_[chosen_gradient_index];   print('chosen gradient',velocity_gradient)
        
        plt.figure()
        plt.plot(velocity_gradients_,data_misfit_)
        plt.ylabel('data misfit')
        plt.xlabel('velocity gradient')
        plt.grid()
        plt.title('misfit')
        plt.show(block=False)
        Name2 = os.getcwd()+'/data_misfit.png';print('saving to ',Name2)
        plt.savefig(Name2,dpi=300)
        plt.close()

        models_init_ideal=1500+velocity_gradient*zz
        np_to_rsf(models_init_ideal,'./'+initial_model_txt_name+'_.rsf',d1=dx,d2=dz);
        # cmd((f"sfcp < {initial_model_txt_name}_.rsf --out=stdout > {initial_model_txt_name}.rsf"));     
        # cmd((f"sfrm {initial_model_txt_name}_.rsf"))
        cmd((f"sfcp < {initial_model_txt_name+'_.rsf'} --out=stdout > {initial_model_txt_name+'.rsf'}"));
        cmd((f"sfrm {initial_model_txt_name+'_.rsf'}"))
        cmd((f"sfrm {'shots_init_.rsf'}"))
        # Plot_image(models_init_ideal.T,Show_flag=0,Save_flag=1,c_lim=[a,b],Title="models_init_ideal",Save_pictures_path='./init_model_search')
    else:
        models_init_ideal=rsf_to_np(initial_model_txt_name+'.rsf')
    ####################################
    cmd((f"sfgenshots< {'vel.rsf'} csdgather=n fm={Fm} amp=1 dt={dt} ns={ns} ng={ng} nt={int(T_max_stage1/dt + 1)} "
            f"sxbeg={sxbeg} szbeg={szbeg} jsx={jsx} jsz=0 gxbeg={gxbeg} gzbeg={gzbeg} jgx={jgx} jgz=0> shots_true.rsf"))
    cmd((f"sfspectra < {'shots_true.rsf'} all=y --out=stdout > spectra.rsf"))
    cmd((f"sfgpufwi  < {initial_model_txt_name+'.rsf'} shots={'shots_true.rsf'} > "
            f"{'vsnaps_.rsf'} grads={'grads_.rsf'} objs={'objs_.rsf'} illums={'illums_.rsf'} "
            f" niter={str(iter1)} precon={Precondition} verb=y rbell={str(Rbell1)} "))
    cmd((f"sfcp < {'grads_.rsf'} --out=stdout > grads.rsf"));   cmd((f"sfrm {'grads_.rsf'}"))
    cmd((f"sfcp < {'objs_.rsf'} --out=stdout > objs.rsf"));     cmd((f"sfrm {'objs_.rsf'}"))
    cmd((f"sfcp < {'illums_.rsf'} --out=stdout > illums.rsf"));     cmd((f"sfrm {'illums_.rsf'}"))
    cmd((f"sfcp < {'vsnaps_.rsf'} --out=stdout > vsnaps.rsf"));     cmd((f"sfrm {'vsnaps_.rsf'}"))
    shots=rsf_to_np('shots_true.rsf')
    cmd((f"sfrm {'shots_true.rsf'}"))
    # %%    Record results
    true=rsf_to_np('vel.rsf')
    grads=rsf_to_np('grads.rsf')
    illums=rsf_to_np('illums.rsf')
    vsnaps=rsf_to_np('vsnaps.rsf')
    objs=rsf_to_np('objs.rsf')
    spectra=rsf_to_np('spectra.rsf')
    spectra=spectra.astype('float64')
    file = sf.Input('spectra.rsf')
    nf=file.int("n1")
    df = file.float("d1")
    # %%    Delete results
    cmd((f"sfrm {'illums.rsf'}"))
    cmd((f"sfrm {'vsnaps.rsf'}"))
    # cmd((f"sfrm {'grads.rsf'}"))
    cmd((f"sfrm {'spectra.rsf'}"))
    cmd((f"sfrm {'objs.rsf'}"))
    if os.path.exists('./init_model_search2')==1:
        cmd((f"rm -r 'init_model_search2/'"))
    a=np.min(true);b=np.max(true);
    Plot_image(true.T,Show_flag=0,Save_flag=1,Title="true",Save_pictures_path='./init_model_search')
    Plot_image(vsnaps[-1, :, :].T - models_init_ideal.T, Show_flag=0,c_lim=[-150,350],Save_flag=1,dx=dx,dy=dz,Title="dv_high_orig_colors_", Save_pictures_path='./init_model_search')
    Plot_image(vsnaps[-1, :, :].T,Show_flag=0,Save_flag=1,c_lim= [a,b],dx=dx,dy=dz,Title="final_",Save_pictures_path='./init_model_search')
    Plot_image(models_init_ideal.T,Show_flag=0,Save_flag=1,c_lim=[a,b],dx=dx,dy=dz,Title="models_init_ideal_",Save_pictures_path='./init_model_search')
    ################################
    # f = h5py.File('model_res_ideal.hdf5','w')
    f = h5py.File(result_model_txt_name+'.hdf5','w')
    f.create_dataset('result', data=vsnaps[-1,:,:])
    f.create_dataset('dx', data=dx)
    f.create_dataset('dz', data=dz)
    f.close()
    os.chdir(home_directory)
    return None
def plotting_fwi_results(results_path,home_directory,
        flag_plotting=0,iter=10,Fm=5,delete_model_folder=1,
        t_max_stage=8,Source_Spacing=800):
    import m8r as sf
    os.chdir(results_path)
    print(os.listdir(results_path))
    true=rsf_to_np('vel.rsf')
    model_orig = sf.Input('vel.rsf')
    ####################################
    Model_name=(results_path.split('/')[-1]).split('model')[-1]
    results_file_name='model_res_ideal'+Model_name+'.hdf5'
    if os.path.exists(results_file_name) or os.path.exists('model_res_ideal.hdf5'):
        print('Data is already calculated, stop program')
        # return None
    ####################################
    if os.path.exists('model.rsf'):
        cmd((f"sfrm {'model.rsf'}"))
    if not os.path.exists('./init_model_search'):
        cmd((f"mkdir {'./init_model_search'}"))
    Nx = model_orig.int("n2");  dx=model_orig.int("d2")
    Nz = model_orig.int("n1");  dz=model_orig.int("d1")
    dt=0.004
    T_max_stage1=t_max_stage
    ####################################    individual initial models
    models_init_ideal=rsf_to_np('models_init_ideal.rsf')
    input_data_ideal=load_file('model_res_ideal.hdf5','result')
    ####################################
    models_init_same=rsf_to_np('init_model.rsf')
    model_number=(results_path.split('/')[-1]).split('model')[-1]
    input_data_same=load_file('model_res'+model_number+'.hdf5','result')
    ####################################
    os.makedirs('./results_analyze',exist_ok=True)
    Plot_image(true.T,Show_flag=0,Save_flag=1,Title="true",Save_pictures_path='./results_analyze')
    Plot_image(input_data_ideal.T-models_init_ideal.T, Show_flag=0,c_lim=[-150,350],Save_flag=1,dx=dx,dy=dz,Title="dv_indiv_init_models"+Model_name,Save_pictures_path='./results_analyze')
    Plot_image(input_data_same.T-models_init_same.T, Show_flag=0,c_lim=[-150,350],Save_flag=1,dx=dx,dy=dz,Title="dv_same_init_models"+Model_name, Save_pictures_path='./results_analyze')
    Plot_image(models_init_ideal.T,Show_flag=0,Save_flag=1,c_lim=[1500,4200],dx=dx,dy=dz,Title="models_init_ideal_"+Model_name,Save_pictures_path='./results_analyze')
    Plot_image(models_init_same.T,Show_flag=0,Save_flag=1,c_lim=[1500,4200],dx=dx,dy=dz,Title="models_init_same_"+Model_name,Save_pictures_path='./results_analyze')
    # Plot_image(vsnaps[-1, :, :].T,Show_flag=0,Save_flag=1,c_lim=[1500,4200],dx=dx,dy=dz,Title="final_",Save_pictures_path='./results_analyze')
    return None
def make_vp_vs_rho(vp,taper,plot_flag=0):
    vs = vp.copy() / (3 ** 0.5)
    vs2 = np.where(taper==0, 0, vs)
    rho = 1e3*0.3 * vp.copy()**0.25
    rho2 = np.where(taper==0, 1000, rho)
    vp2 = np.where(taper==0, 1500, vp)
    if plot_flag==1:
        fig=plt.figure()
        ind=300
        plt.plot(vp[::-1,ind],label='vp')
        plt.plot(vp2[::-1,ind],label='vp2')
        plt.plot(vs[::-1,ind],label='vs')
        plt.plot(vs2[::-1,ind],label='vs2')
        plt.plot(rho[::-1,ind],label='rho')
        plt.plot(rho2[::-1,ind],label='rho2')
        plt.grid()
        plt.legend()
        save_file_path1=os.path.join('./Pictures','vp_vs_rho.png')
        plt.show()
        plt.close()
        fig.savefig(save_file_path1,dpi=300,bbox_inches='tight')
    return vp2, vs2, rho2
