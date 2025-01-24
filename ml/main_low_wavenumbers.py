from imports import *
from utils_low_wavenumbers import *
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.InteractiveSession(config=config)
####    create logging
os.makedirs('./logs',exist_ok=True)
log_save_const = F_calculate_log_number('./logs','log','')
Save_pictures_path = './logs/log'+str(log_save_const)
os.makedirs(Save_pictures_path,exist_ok=True)
logname = '/log' + str(log_save_const)+'.txt'
f = open(Save_pictures_path+logname,'w')
sys.stdout = Tee(sys.stdout,f)
print('KERAS CODE KERAS CODE KERAS CODE KERAS CODE!!!!!')
if __name__ == '__main__':
    gan = Pix2Pix(Save_pictures_path,log_save_const)
    Model_name_to_load='generator117epoch_30.hdf5'    
    Model_name_to_load='generator118epoch_30.hdf5'
    Model_name_to_load='generator119epoch_0.hdf5'
    Model_name_to_load='generator125.hdf5'
    Model_name_to_load='generator125epoch_65.hdf5'
    Model_name_to_load='generator126epoch_5.hdf5'
    Model_name_to_load='generator127epoch_5.hdf5'
    Model_name_to_load='generator128epoch_0.hdf5'
    Model_name_to_load='generator129.hdf5'
    Model_name_to_load='generator156epoch_50.hdf5'
    Model_name_to_load='generator157epoch_0.hdf5'
    Model_name_to_load='generator157epoch_20.hdf5'
    # gan.train(epochs=30,batch_size=128,sample_interval=1,plotting_interval=1)
    # gan.train(epochs=10,batch_size=4,sample_interval=2,plotting_interval=1)
    # gan.train(epochs=1000,batch_size=4,sample_interval=10,plotting_interval=10)
    gan.inference(Model_to_load_const=113,model_name=Model_name_to_load,batch_size=4)
    gan.sample_images(0,1,original_plotting=1)


