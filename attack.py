import os, random, h5py, warnings
import numpy as np
import tensorflow as tf
# from src.utils import poisoning
from src.models import *
from get_classes import get_classes
from tf_complex_model import model_CVNN
from tf_MCLDNN_model import model_MCLDNN
from tf_expert_assistant import model_EA

from parameter import *

# 过滤掉INFO、WARNING和ERROR级别的日志
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# 启用确定性操作模式
os.environ['TF_DETERMINISTIC_OPS'] = '1' 
# 禁用OneDNN优化以确保完全的确定性
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# 指定使用的GPU设备ID
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
warnings.filterwarnings("ignore")

random.seed(2016)  
np.random.seed(2016) 
tf.random.set_seed(2016)

trigger = np.load(GEN_TRIGGER_TRIGGER_SAVE_PATH)
poisoning_rate = TRAIN_POISONING_POISONING_RATE
# 目标1：QPSK：0（20+4）
# 目标2：16APSK：0（16+8）
# 目标3：16QAM：0（12+12）
# 目标4：256QAM：0（8+16）
target = 0  

def train(X_train, Y_train, X_val, Y_val, classes, batch_size=TRAIN_BENIGN_BATCH_SIZE, Epoch=TRAIN_POISONING_EPOCH):
    if TRAIN_MODEL_NAME == 'ResNet':
        Model = model_ResNet(input_shape=[1024,2], num_classes=classes)
    elif TRAIN_MODEL_NAME == 'GRU':
        Model = model_GRU(input_shape=[1024,2], num_classes=classes)
    elif TRAIN_MODEL_NAME == 'CVNN':
        Model = model_CVNN(input_shape=(1024, 2), num_classes=classes, feature_dim=128)
    elif TRAIN_MODEL_NAME == 'MCLDNN':
        Model = model_MCLDNN(input_shape=(1024, 2), num_classes=classes)
    elif TRAIN_MODEL_NAME == 'EA':
        # input_shape = (2, 1024)
        Model = model_EA(input_length=1024, num_classes=classes)
    ###############################################################################################################
    # Model = model_ResNet(input_shape=[1024,2], num_classes=20)
    ###############################################################################################################
    # Model = model_GRU(input_shape=[1024,2], num_classes=20)
    ###############################################################################################################
    # Model = model_CVNN(input_shape=(1024, 2), num_classes=classes, feature_dim=128)
    ###############################################################################################################
    # Model = model_MCLDNN(input_shape=(1024, 2), num_classes=classes)
    ###############################################################################################################
    optimizer = tf.keras.optimizers.Adam(learning_rate=TRAIN_POISONING_ADAM_LEARNING_RATE)
    ###############################################################################################################
    Model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=optimizer)
    Model.summary()
    ###############################################################################################################

    history = Model.fit(
        x = X_train,
        y = Y_train,
        batch_size=batch_size,
        epochs=Epoch,
        verbose=2,
        validation_data=(X_val, Y_val),
        callbacks = [
                    tf.keras.callbacks.ModelCheckpoint(TRAIN_POISONING_MODEL_SAVE_PATH, monitor='val_loss', verbose=1, save_best_only=True, mode='auto'),
                    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',factor=0.5, verbose=1, patince=10, min_lr=0.0000001),
                    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=30, verbose=1, mode='auto'),           
                    ]
                        )
    return Model
##############################################################################################################################################################
 # 加载数据集
n_classes = len(get_classes(from_file=RML18_20_KNOWN_CLASSES_LABELS_PATH))
train_data = h5py.File(RML18_20_KNOWN_CLASSES_TRAIN_DATASETS_PATH, 'r')
val_data = h5py.File(RML18_20_KNOWN_CLASSES_VAL_DATASETS_PATH, 'r')

X_train = train_data['X_train'][:, :, :]
Y_train = train_data['Y_train'][:].astype(np.int32)
Z_train = train_data['Z_train'][:, :]

X_val = val_data['X_val'][:, :, :]
Y_val = val_data['Y_val'][:].astype(np.int32)
Z_val = val_data['Z_val'][:, :]

###############################################################################################################
X_train = poisoning(X_train, Y_train, target, trigger, poisoning_rate)
X_val = poisoning(X_val, Y_val, target, trigger, poisoning_rate)

with h5py.File(RML18_20_POISONED_KNOWN_TRAIN_DATASETS_PATH, 'w') as output_f:
        output_f.create_dataset(f'X_train', data=X_train)
        output_f.create_dataset(f'Y_train', data=Y_train)
        output_f.create_dataset(f'Z_train', data=Z_train)
###############################################################################################################

Y_train = tf.keras.utils.to_categorical(Y_train, num_classes=n_classes)
Y_val = tf.keras.utils.to_categorical(Y_val, num_classes=n_classes)

# X_train, Y_train = shuffle_data(X_train, Y_train)
######################################################################################################
sup_model = train(X_train, Y_train, X_val, Y_val, classes=n_classes)
######################################################################################################
train_data.close()
val_data.close()
######################################################################################################