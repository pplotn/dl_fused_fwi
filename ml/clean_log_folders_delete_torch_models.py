from imports_torch import *
from utils_low_wavenumbers_torch import *
print('torch.cuda.is_available()=',torch.cuda.is_available())
os.system('nvidia-smi')
# sys.path.append('/home/plotnips/Dropbox/Log_extrapolation/scripts/MWE/F_utils.py')
# sys.path.append('/home/plotnips/Dropbox/Log_extrapolation/scripts/MWE')
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
print('logname= ',logname)
print('datetime.now()=',datetime.datetime.now())

nums=np.arange(0,155)
nums=np.arange(155,1000)
nums=np.arange(1000,1600)
nums=list(nums)
nums2=[1084,1089,1098,1099,1100,1189,1167,1202,1269,
       1290,1357,1208,1207,1227,1226,519,1263,1260,
        1275,1273,1219,705,27,57,1350,1353,1390,1391,1497]
nums=list(set(nums)-set(nums2));   
ss=1




for i in nums:
    print(i)
    os.system('rm -r '+os.path.join(logs_path,'log'+str(i)+'/*' ) )
