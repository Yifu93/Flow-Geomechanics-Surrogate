import numpy as np
import h5py
import unet_uae as vae_util
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
E_max = 20e9
E_caprock_max = 40e9

data_dir = '/Dataset/'
# load flow training data
simulation_data = os.path.join(data_dir, 'Coupled_P_Train.h5')
p_t = load_data(simulation_data, ['pressure'])
p_t = np.array(p_t)
p_t = p_t[0,...]
print('p_t max is ', np.max(p_t))
print('p_t min is ', np.min(p_t))

p_max = np.max(p_t)
p_t = p_t / p_max
print('p_t max is ', np.max(p_t))
print('p_t min is ', np.min(p_t))

p_t_shape = p_t.shape
new_shape = (p_t_shape[0] * p_t_shape[1],) + p_t_shape[2:]
p_t = p_t.reshape(new_shape)
print('p_t shape is ', p_t.shape)

simulation_data = os.path.join(data_dir, 'Coupled_Sw_Train.h5')
sat_t = load_data(simulation_data, ['saturation'])
sat_t = np.array(sat_t)
sat_t = sat_t[0,...]
sat_t[sat_t < 0.02] = 0.0
sat_t[sat_t > 0.78] = 0.78
print('sat_t max is ', np.max(sat_t))
print('sat_t min is ', np.min(sat_t))

sat_max = np.max(sat_t)
sat_t = sat_t / sat_max
print('sat_t max is ', np.max(sat_t))
print('sat_t min is ', np.min(sat_t))

sat_t_shape = sat_t.shape
new_shape = (sat_t_shape[0] * sat_t_shape[1],) + sat_t_shape[2:]
sat_t = sat_t.reshape(new_shape)
print('sat_t shape is ', sat_t.shape)

k = load_data('/Dataset/logk_fc_train.h5', ['logk'])
k = np.array(k)
k = k.reshape(400, 20, 80, 80)
k = k / k_max
print('k max is ', np.max(k))
print('k min is ', np.min(k))

k = np.repeat(k[:, np.newaxis], 10, axis = 1)
k_shape = k.shape
new_shape = (k_shape[0] * k_shape[1],) + k_shape[2:]
k = k.reshape(new_shape)
print('k shape is ', k.shape)

k_v = load_data('/Dataset/logk_val.h5', ['logk'])
k_v = np.array(k_v)
k_v = k_v.reshape(200, 20, 80, 80)
k_v = k_v / k_max
print('k_v max is ', np.max(k_v))
print('k_v min is ', np.min(k_v))

k_v = np.repeat(k_v[:, np.newaxis], 10, axis = 1)
k_v_shape = k_v.shape
new_shape = (k_v_shape[0] * k_v_shape[1],) + k_v_shape[2:]
k_v = k_v.reshape(new_shape)
print('k_v shape is ', k_v.shape)

phi = load_data('/Dataset/phi_fc_train.h5', ['phi'])
phi = np.array(phi)
phi = phi.reshape(400, 20, 80, 80)
phi = phi / phi_max
print('porosity max is ', np.max(phi))
print('porosity min is ', np.min(phi))

phi = np.repeat(phi[:, np.newaxis], 10, axis = 1)
phi_shape = phi.shape
new_shape = (phi_shape[0] * phi_shape[1],) + phi_shape[2:]
phi = phi.reshape(new_shape)
print('phi shape is ', phi.shape)

phi_v = load_data('/Dataset/phi_val.h5', ['phi'])
phi_v = np.array(phi_v)
phi_v = phi_v.reshape(200, 20, 80, 80)
phi_v = phi_v / phi_max

phi_v = np.repeat(phi_v[:, np.newaxis], 10, axis = 1)
phi_v_shape = phi_v.shape
new_shape = (phi_v_shape[0] * phi_v_shape[1],) + phi_v_shape[2:]
phi_v = phi_v.reshape(new_shape)
print('phi_v shape is ', phi_v.shape)

Meta_Parameters = np.load('/Dataset/Meta_Parameters_Train_Coupled.npy')

log_kvkh = Meta_Parameters[2, :]
E = Meta_Parameters[5, :]
E_cap = Meta_Parameters[6, :]

kvkh = np.zeros(((400, 10, 20, 80, 80, 1)))
E_train = np.zeros(((400, 10, 20, 80, 80, 1)))
E_cap_train = np.zeros(((400, 10, 20, 80, 80, 1)))

for i in range(400):
    kvkh[i, ...] = np.power(10, log_kvkh[i])
    E_train[i, ...] = E[i]
    E_cap_train[i, ...] = E_cap[i]
    
print('kvkh min is ', np.min(kvkh))
print('kvkh max is ', np.max(kvkh))
kvkh_shape = kvkh.shape
new_shape = (kvkh_shape[0] * kvkh_shape[1],) + kvkh_shape[2:]
kvkh = kvkh.reshape(new_shape)

print('E min is ', np.min(E_train))
print('E max is ', np.max(E_train))
E_train = E_train / E_max
print('E min is ', np.min(E_train))
print('E max is ', np.max(E_train))

E_train_shape = E_train.shape
new_shape = (E_train_shape[0] * E_train_shape[1],) + E_train_shape[2:]
E_train = E_train.reshape(new_shape)

print('E_cap min is ', np.min(E_cap_train))
print('E_cap max is ', np.max(E_cap_train))
E_cap_train = E_cap_train / E_caprock_max
print('E_cap min is ', np.min(E_cap_train))
print('E_cap max is ', np.max(E_cap_train))

E_cap_train_shape = E_cap_train.shape
new_shape = (E_cap_train_shape[0] * E_cap_train_shape[1],) + E_cap_train_shape[2:]
E_cap_train = E_cap_train.reshape(new_shape)

Meta_Parameters_v = np.load('/Dataset/Meta_Val.npy')

log_kvkh_v = Meta_Parameters_v[2, :]
E_v = Meta_Parameters_v[5, :]
E_cap_v = Meta_Parameters_v[6, :]

kvkh_v = np.zeros(((200, 10, 20, 80, 80, 1)))
E_val = np.zeros(((200, 10, 20, 80, 80, 1)))
E_cap_val = np.zeros(((200, 10, 20, 80, 80, 1)))

for i in range(200):
    kvkh_v[i, ...] = np.power(10, log_kvkh_v[i])
    E_val[i, ...] = E_v[i]
    E_cap_val[i, ...] = E_cap_v[i]
    
print('kvkh_v min is ', np.min(kvkh_v))
print('kvkh_v max is ', np.max(kvkh_v))

kvkh_v_shape = kvkh_v.shape
new_shape = (kvkh_v_shape[0] * kvkh_v_shape[1],) + kvkh_v_shape[2:]
kvkh_v = kvkh_v.reshape(new_shape)
    
print('E_val min is ', np.min(E_val))
print('E_val max is ', np.max(E_val))
E_val = E_val / E_max
print('E_val min is ', np.min(E_val))
print('E_val max is ', np.max(E_val))

E_val_shape = E_val.shape
new_shape = (E_val_shape[0] * E_val_shape[1],) + E_val_shape[2:]
E_val = E_val.reshape(new_shape)

print('E_cap_val min is ', np.min(E_cap_val))
print('E_cap_val max is ', np.max(E_cap_val))
E_cap_val = E_cap_val / E_caprock_max
print('E_cap_val min is ', np.min(E_cap_val))
print('E_cap_val max is ', np.max(E_cap_val))

E_cap_val_shape = E_cap_val.shape
new_shape = (E_cap_val_shape[0] * E_cap_val_shape[1],) + E_cap_val_shape[2:]
E_cap_val = E_cap_val.reshape(new_shape)

print('p_t shape is ', p_t.shape)
print('k shape is ', k.shape)
print('phi shape is ', phi.shape)

simulation_data = os.path.join(data_dir, 'time.h5')
time = load_data(simulation_data, ['time'])
time = np.array(time)
time = time[0, ...]
print('max time is ', np.max(time))
print('min time is ', np.min(time))

time_max = np.max(time)
time = time / time_max
print('min time is ', np.min(time))
print('max time is ', np.max(time))
time = np.repeat(time[:, np.newaxis], 20, axis = 1)

simulation_data = os.path.join(data_dir, 'time_v.h5')
time_v = load_data(simulation_data, ['time'])
time_v = np.array(time_v)
time_v = time_v[0, ...]
print('max time_v is ', np.max(time_v))
print('min time_v is ', np.min(time_v))

time_v = time_v / time_max
print('min time_v is ', np.min(time_v))
print('max time_v is ', np.max(time_v))
time_v = np.repeat(time_v[:, np.newaxis], 20, axis = 1)

# load validation data
# note that for validation/test case: 
# pressure and saturation from the simulation results instead of surrogate model predictions can be used to evaluate the performance.

simulation_data = os.path.join(data_dir, 'Coupled_P_Val.h5')
p_v = load_data(simulation_data, ['pressure'])
p_v = np.array(p_v)
p_v = p_v[0, ...]
print('p_v shape is ', p_v.shape)
print('p_v max is ', np.max(p_v))
print('p_v min is ', np.min(p_v))

p_v = p_v / p_max
print('p_v max is ', np.max(p_v))
print('p_v min is ', np.min(p_v))

p_v_shape = p_v.shape
new_shape = (p_v_shape[0] * p_v_shape[1],) + p_v_shape[2:]
p_v = p_v.reshape(new_shape)
print('p_v shape is ', p_v.shape)

simulation_data = os.path.join(data_dir, 'Coupled_Sw_Val.h5')
sat_v = load_data(simulation_data, ['saturation'])
sat_v = np.array(sat_v)
sat_v = sat_v[0, ...]
print('sw_v shape is ', sat_v.shape)
print('sat_v max is ', np.max(sat_v))
print('sat_v min is ', np.min(sat_v))

sat_v[sat_v < 0.02] = 0.0
sat_v[sat_v > 0.78] = 0.78

sat_v = sat_v / sat_max
print('sat_v max is ', np.max(sat_v))
print('sat_v min is ', np.min(sat_v))

sat_v_shape = sat_v.shape
new_shape = (sat_v_shape[0] * sat_v_shape[1],) + sat_v_shape[2:]
sat_v = sat_v.reshape(new_shape)
print('sat_v shape is ', sat_v.shape)

depth = 10
nr = k.shape[0]
train_nr = 4000
test_nr  = 2000

step_index = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
k     = k[:, :, :, :, None]
phi   = phi[:, :, :, :, None]
p_t   = p_t[:, :, :, :, None]
sat_t = sat_t[:, :, :, :, None]
time  = time[:, :, :, :, None]

k_v    = k_v[:, :, :, :, None]
phi_v  = phi_v[:, :, :, :, None]
p_v    = p_v[:, :, :, :, None]
sat_v  = sat_v[:, :, :, :, None]
time_v = time_v[:, :, :, :, None]

simulation_data = os.path.join(data_dir, 'Coupled_d_Train.h5')
d_t = load_data(simulation_data, ['displacement'])
d_t = np.array(d_t)
d_t = d_t[0,...]
print('d_t shape is ', d_t.shape)

d_t = d_t[:, :, 1:, :] + d_t[:, :, :-1, :]
d_t = (d_t[:, :, :, 1:] + d_t[:, :, :, :-1]) / 4
print('max d_t is ', np.max(d_t))
print('min d_t is ', np.min(d_t))
print('d_t shape is ', d_t.shape)

d_t[d_t < 0.0]  = 0.0
print('max d_t is ', np.max(d_t))
print('min d_t is ', np.min(d_t))
d_t = np.repeat(d_t[:, :, np.newaxis], 20, axis = 2)

d_t_shape = d_t.shape
new_shape = (d_t_shape[0] * d_t_shape[1],) + d_t_shape[2:]
d_t = d_t.reshape(new_shape)
print('d_t shape is ', d_t.shape)

d_t_mean = np.mean(d_t)
d_t_std = np.std(d_t)
d_t = d_t - d_t_mean
d_t = d_t / d_t_std
print('min d_t is ', np.min(d_t))
print('max d_t is ', np.max(d_t))
print('d_t shape is ', d_t.shape)

simulation_data = os.path.join(data_dir, 'Coupled_d_Val.h5')
d_v = load_data(simulation_data, ['displacement'])
d_v = np.array(d_v)
d_v = d_v[0,...]
print('d_v shape is ', d_v.shape)

d_v = d_v[:, :, 1:, :] + d_v[:, :, :-1, :]
d_v = (d_v[:, :, :, 1:] + d_v[:, :, :, :-1]) / 4
print('max d_v is ', np.max(d_v))
print('min d_v is ', np.min(d_v))
print('d_v shape is ', d_v.shape)

d_v[d_v < 0.0]  = 0.0
print('max d_v is ', np.max(d_v))
print('min d_v is ', np.min(d_v))
d_v = np.repeat(d_v[:, :, np.newaxis], 20, axis = 2)

d_v_shape = d_v.shape
new_shape = (d_v_shape[0] * d_v_shape[1],) + d_v_shape[2:]
d_v = d_v.reshape(new_shape)
print('d_v shape is ', d_v.shape)

d_v = d_v - d_t_mean
d_v = d_v / d_t_std
print('min d_v is ', np.min(d_v))
print('max d_v is ', np.max(d_v))
print('d_v shape is ', d_v.shape)

train_x = np.concatenate([k, kvkh, phi, E_train, E_cap_train, p_t, sat_t, time], axis = -1)
train_y = d_t

test_x  = np.concatenate([k_v, kvkh_v, phi_v, E_val, E_cap_val, p_v, sat_v, time_v], axis = -1)
test_y  = d_v

train_y = train_y[:, :, :, None]
test_y  = test_y[:, :, :, None]

print('train_x shape is ', train_x.shape)
print('train_y shape is ', train_y.shape)
print('test_x shape is ', test_x.shape)
print('test_y shape is ', test_y.shape)

input_shape = (20, 80, 80, 8)
input = Input(shape = input_shape, name='image')
vae_model, _ = vae_util.create_vae(input_shape)

vae_model.summary(line_length=150)

output_dir = 'saved_models/'
epochs = 300
train_nr = train_x.shape[0]
test_nr  = 10
batch_size = 8
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
vae_model.compile(loss = vae_loss, optimizer = opt, metrics = [vae_loss, relative_error])

from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
lrScheduler = ReduceLROnPlateau(monitor = 'loss', factor = 0.5, patience = 10, cooldown = 1, verbose = 1, min_lr = 1e-7)
filePath = 'saved_models/saved-model-10steps-lr3e-4-displacement-detrend-hd-0-filter_16_32_32_64-mse-{epoch:03d}-{val_loss:.2f}.h5'
checkPoint = ModelCheckpoint(filePath, monitor = 'val_loss', verbose = 1, save_best_only = False, save_weights_only = True, mode = 'auto', period = 20)

callbacks_list = [lrScheduler, checkPoint]
history = vae_model.fit(train_x, train_y, batch_size = batch_size, epochs = epochs, verbose = 2, validation_data = (test_x, test_y), callbacks = callbacks_list)

import pickle    
with open('HISTORY-displacement-mse-hd500-filter_8_16_32_32.pkl', 'wb') as file_pi:
    pickle.dump(history.history, file_pi)