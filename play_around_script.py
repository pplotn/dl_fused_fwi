from functions.F_modules import *
from functions.F_utils import *
import m8r as sf
#V_over=rsf_to_np('./data_generation/original_models/overthrust_test_2D_2.hh')
############################################################
file= sf.Input('./data_generation/original_models/overthrust_test_2D_2.hh')
dz=file.float("d1")
dx=file.float("d2")
vel=file.read()
hf=h5py.File('./data_generation/original_models/overthrust_test_2D_2.hdf5','w')
hf.create_dataset('d1',data=dz)
hf.create_dataset('d2',data=dx)
hf.create_dataset('vp',data=vel)
hf.close()
############################################################
file=sf.Input(os.path.join('data_generation','original_models/seam_i_sediments.hh'))
dz=file.float("d1")
dx=file.float("d2")
vel=file.read()
hf=h5py.File(os.path.join('data_generation','original_models/seam_i_sediments.hdf5'),'w')
hf.create_dataset('d1',data=dz)
hf.create_dataset('d2',data=dx)
hf.create_dataset('vp',data=vel)
hf.close()
############################################################
name=os.path.join('data_generation','original_models/vpb2d_.rsf')
file = sf.Input(name)
dz=file.float("d1")*1000
dx=file.float("d2")*1000
vel=file.read()
hf=h5py.File(os.path.join('data_generation','original_models/vpb2d_.hdf5'),'w')
hf.create_dataset('d1',data=dz)
hf.create_dataset('d2',data=dx)
hf.create_dataset('vp',data=vel)
hf.close()
############################################################
