import numpy as np
import h5py
import unet_uae_filter as vae_util
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
from tensorflow import keras
from keras import backend as K
from keras.layers import Input, Flatten, Dense, Lambda, Reshape, concatenate, TimeDistributed, RepeatVector, ConvLSTM3D
from keras.models import Model
from keras.optimizers import Adam
from keras.layers.convolutional import Conv3D

# check tensorflow version
print("tensorflow version:", tf.__version__)
# check available gpu
gpus =  tf.config.list_physical_devices('GPU')
print("available gpus:", gpus)
# limit the gpu usage, prevent it from allocating all gpu memory for a simple model
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
# check number of cpus available
print("available cpus:", os.cpu_count())

def load_data(data_path, array_name_list):
    hf_r = h5py.File(data_path, 'r')
    result = []
    for name in array_name_list:
        result.append(np.array(hf_r.get(name)))
    hf_r.close()
    return result
    
k_max = np.log(1e4)
phi_max = 0.4
c_s_max = np.load('/Dataset/c_s_max.npy')
c_o_max = np.load('/Dataset/c_o_max.npy')

data_dir = '/Dataset/'
# load flow training data
simulation_data = os.path.join(data_dir, 'Flow_P_Train.h5')
p_t = load_data(simulation_data, ['pressure'])
p_t = np.array(p_t)
p_t = p_t[0,...]

k = load_data('/Dataset/logk_train.h5', ['logk'])
k = np.array(k)
k = k.reshape(4000, 20, 80, 80)
k = k / k_max
print('k max is ', np.max(k))
print('k min is ', np.min(k))

k_v = load_data('Dataset/logk_val.h5', ['logk'])
k_v = np.array(k_v)
k_v = k_v.reshape(200, 20, 80, 80)
k_v = k_v / k_max
print('k_v max is ', np.max(k_v))
print('k_v min is ', np.min(k_v))

phi = load_data('/Dataset/phi_train.h5', ['phi'])
phi = np.array(phi)
phi = phi.reshape(4000, 20, 80, 80)
phi = phi / phi_max
print('porosity max is ', np.max(phi))
print('porosity min is ', np.min(phi))

phi_v = load_data('/Dataset/phi_val.h5', ['phi'])
phi_v = np.array(phi_v)
phi_v = phi_v.reshape(200, 20, 80, 80)
phi_v = phi_v / phi_max

Meta_Parameters = np.load('/Dataset/Meta_Train.npy')

log_kvkh = Meta_Parameters[2, :]
c_s = Meta_Parameters[7, :]
c_o = Meta_Parameters[8, :]

kvkh = np.zeros(((4000, 20, 80, 80, 1)))
c_s_train = np.zeros(((4000, 20, 80, 80, 1)))
c_o_train = np.zeros(((4000, 20, 80, 80, 1)))

for i in range(4000):
    kvkh[i, ...] = np.power(10, log_kvkh[i])
    c_s_train[i, ...] = c_s[i]
    c_o_train[i, ...] = c_o[i]
    
print('kvkh min is ', np.min(kvkh))
print('kvkh max is ', np.max(kvkh))

print('c_s_train min is ', np.min(c_s_train))
print('c_s_train max is ', np.max(c_s_train))
c_s_train = c_s_train / c_s_max
print('c_s_train min is ', np.min(c_s_train))
print('c_s_train max is ', np.max(c_s_train))

print('c_o_train min is ', np.min(c_o_train))
print('c_o_train max is ', np.max(c_o_train))
c_o_train = c_o_train / c_o_max
print('c_o_train min is ', np.min(c_o_train))
print('c_o_train max is ', np.max(c_o_train))

Meta_Parameters_v = np.load('/Dataset/Meta_Val.npy')

log_kvkh_v = Meta_Parameters_v[2, :]
c_s_v = Meta_Parameters_v[7, :]
c_o_v = Meta_Parameters_v[8, :]

kvkh_v = np.zeros(((200, 20, 80, 80, 1)))
c_s_val = np.zeros(((200, 20, 80, 80, 1)))
c_o_val = np.zeros(((200, 20, 80, 80, 1)))

for i in range(200):
    kvkh_v[i, ...] = np.power(10, log_kvkh_v[i])
    c_s_val[i, ...] = c_s_v[i]
    c_o_val[i, ...] = c_o_v[i]
    
print('kvkh_v min is ', np.min(kvkh_v))
print('kvkh_v max is ', np.max(kvkh_v))
    
print('c_s_val min is ', np.min(c_s_val))
print('c_s_val max is ', np.max(c_s_val))
c_s_val = c_s_val / c_s_max
print('c_s_val min is ', np.min(c_s_val))
print('c_s_val max is ', np.max(c_s_val))

print('c_o_val min is ', np.min(c_o_val))
print('c_o_val max is ', np.max(c_o_val))
c_o_val = c_o_val / c_o_max
print('c_o_val min is ', np.min(c_o_val))
print('c_o_val max is ', np.max(c_o_val))

print('p_t shape is ', p_t.shape)
print('k shape is ', k.shape)
print('phi shape is ', phi.shape)

# load testing data
simulation_data = os.path.join(data_dir, 'Coupled_P_Val.h5')
p_v = load_data(simulation_data, ['pressure'])
p_v = np.array(p_v)
p_v = p_v[0, ...]
print('p_v shape is ', p_v.shape)
print('p_t max is ', np.max(p_t))
print('p_t min is ', np.min(p_t))
print('p_v max is ', np.max(p_v))
print('p_v min is ', np.min(p_v))

depth = 10
nr = k.shape[0]
train_nr = 4000
test_nr  = 200

p_mean = np.mean(p_t)
p_std  = np.std(p_t)
print('p_mean shape is', p_mean.shape)

p_t = p_t - p_mean
p_t = p_t / p_std

p_v = p_v - p_mean
p_v = p_v / p_std

print('max p train is ', np.max(p_t), ', min p train is ', np.min(p_t))
print('max p validation is ', np.max(p_v), ', min p validation is ', np.min(p_v))

step_index = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
k = k[:, :, :, :, None]
phi = phi[:, :, :, :, None]

k_v = k_v[:, :, :, :, None]
phi_v = phi_v[:, :, :, :, None]

train_x = np.concatenate([k, kvkh, phi, c_s_train, c_o_train], axis = -1)
train_y = p_t[:, step_index, :, :, :]

test_x  = np.concatenate([k_v, kvkh_v, phi_v, c_s_val, c_o_val], axis = -1)
test_y  = p_v[:, step_index, :, :, :]

train_y = train_y[:, :, :, :, :, None]
test_y  = test_y[:, :, :, :, :, None]

print('train_x shape is ', train_x.shape)
print('train_y shape is ', train_y.shape)
print('test_x shape is ', test_x.shape)
print('test_y shape is ', test_y.shape)

input_shape = (20, 80, 80, 5)
input = Input(shape = input_shape, name='image')
_, vae_model_base, _ = vae_util.create_vae(input_shape, depth)
output_base = vae_model_base(input)

output_flow = TimeDistributed(Conv3D(1, (3, 3, 3), padding='same', activation=None))(output_base)
vae_model_flow = Model(input, output_flow) 
vae_model_flow.summary(line_length=150)

output_dir = 'saved_models/'
epochs = 300
train_nr = train_x.shape[0]
test_nr  = 10
batch_size = 4
num_batch  = int(train_nr/batch_size) 
    
def vae_loss(x, t_decoded):
    '''Total loss for the plain UAE'''
    return K.mean(reconstruction_loss(x, t_decoded))

def reconstruction_loss(x, t_decoded):
    '''Reconstruction loss for the plain UAE'''
    return K.sum((K.batch_flatten(x) - K.batch_flatten(t_decoded)) ** 2, axis=-1)

def relative_error(x, t_decoded):
    return K.mean(K.abs(x - t_decoded) / (x + 0.01))

opt = Adam(lr = 3e-4)
vae_model_flow.compile(loss = vae_loss, optimizer = opt, metrics = [vae_loss, relative_error])

from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
lrScheduler = ReduceLROnPlateau(monitor = 'loss', factor = 0.5, patience = 10, cooldown = 1, verbose = 1, min_lr = 1e-7)
filePath = 'saved_models/saved-model-10steps-lr3e-4-pressure-detrend-hd-0-filter_16_32_32_64-mse-{epoch:03d}-{val_loss:.2f}.h5'
checkPoint = ModelCheckpoint(filePath, monitor = 'val_loss', verbose = 1, save_best_only = False, save_weights_only = True, mode = 'auto', period = 20)

callbacks_list = [lrScheduler, checkPoint]
history = vae_model_flow.fit(train_x, train_y, batch_size = batch_size, epochs = epochs, verbose = 2, validation_data = (test_x, test_y), callbacks = callbacks_list)

import pickle    
with open('HISTORY-pressure-mse-hd500-filter_8_16_32_32.pkl', 'wb') as file_pi:
    pickle.dump(history.history, file_pi)
