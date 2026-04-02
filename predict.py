import os,random,h5py,warnings
# 过滤掉INFO、WARNING和ERROR级别的日志
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# 启用确定性操作模式
os.environ['TF_DETERMINISTIC_OPS'] = '1' 
# 禁用OneDNN优化以确保完全的确定性
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# 指定使用的GPU设备ID
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
warnings.filterwarnings("ignore")

import tensorflow as tf
import numpy as np
from get_classes import get_classes
from tools import calculate_confusion_matrix, plot_confusion_matrix, calculate_acc_cm_each_snr, plot_results_comparison

# from src.encoder_model import TransformerBlock

random.seed(2016)  
np.random.seed(2016) 
tf.random.set_seed(2016)

# 检查文件夹是否存在，不存在则创建
if not os.path.exists('./weights'):
    os.makedirs('./weights')
if not os.path.exists('./figure'):
    os.makedirs('./figure')

def model_predict(batch_size=512,
                  weight_path = "./saved_models/model.hdf5",
                  min_snr=-20,
                  test_datapath='./data/rml22_test_1_10_data.hdf5',
                  classes_path='./data/classes_rml22.txt',
                  save_plot_file='./figure/total_confusion.png'):
    
    classes = get_classes(classes_path)
    n_classes = len(classes)

    test_data = h5py.File(test_datapath, 'r')
    X_test = test_data['X_test'][:, :, :]
    Y_test = test_data['Y_test'][:].astype(np.int32)
    Z_test = test_data['Z_test'][:, :]

    # X_test = np.swapaxes(X_test, 1, 2)

    Y_test_categorical = tf.keras.utils.to_categorical(Y_test, num_classes=n_classes)

    Model = tf.keras.models.load_model(weight_path)

    # try:
    #     Model = tf.keras.models.load_model(weight_path)
    # except:
    #     Model = tf.keras.models.load_model(
    #     weight_path,
    #     custom_objects={'TransformerBlock': TransformerBlock}
    #     )

    test_Y_predict = Model.predict(X_test, batch_size=batch_size, verbose=1)
    # 计算混淆矩阵
    confusion_matrix_normal, right, wrong = calculate_confusion_matrix(Y_test_categorical, test_Y_predict, classes)
    overall_accuracy = round(1.0 * right / (right + wrong), 4)
    print('Overall Accuracy: %.2f%% / (%d + %d)' % (100 * overall_accuracy, right, wrong))
    with open('./figure/predict_results.txt', 'a') as file:
        file.write('Overall Accuracy: %.2f%% / (%d + %d)\n' % (100 * overall_accuracy, right, wrong))
    plot_confusion_matrix(confusion_matrix_normal, labels=classes, save_filename=save_plot_file)
    calculate_acc_cm_each_snr(Y_test_categorical, test_Y_predict, Z_test, classes, min_snr=min_snr, file_name="predict")


    test_data.close()


if __name__ == '__main__':
    model_predict(
        batch_size=512,
        weight_path = "./saved_models/model.hdf5",
        min_snr=-20,
        test_datapath='./data/rml16_test_1_10_data_10.hdf5',
        classes_path='./data/classes_rml16_10.txt',
        save_plot_file='./figure/predict_total_confusion.png'
    )
    