import os

import keras.backend as K
import tensorflow as tf
from keras.callbacks import EarlyStopping, TensorBoard

from ctpn import CTPN
from ctpn import default_ctpn_config_path
from ctpn.data_loader import DataLoader
from custom import SingleModelCK
from custom.callbacks import SGDRScheduler

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-ie", "--initial_epoch", help="iniial_epoch", default=0, type=int)
    parser.add_argument("--epochs", help="epochs", default=20, type=int)
    parser.add_argument("--gpus", help="gpu_number", default=1, type=int)
    parser.add_argument("--images_dir", help="images_dir", default="E:\data\VOCdevkit\VOC2007\JPEGImages")
    parser.add_argument("--anno_dir", help="anno_dir", default="E:\data\VOCdevkit\VOC2007\Annotations")
    parser.add_argument("--config_file_path", help="config_file_path",
                        default=default_ctpn_config_path)
    parser.add_argument("--weights_file_path", help="weights_file_path",
                        default=None)
    parser.add_argument("--save_weights_file_path", help="save_weights_file_path",
                        default=r'weights/weights-ctpnlstm-{epoch:02d}.hdf5')

    args = parser.parse_args()

    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.compat.v1.Session(config=config)
    K.set_session(session)

    config = CTPN.load_config(args.config_file_path)

    weights_file_path = args.weights_file_path
    if weights_file_path is not None:
        config["weight_path"] = weights_file_path
    config['num_gpu'] = args.gpus

    ctpn = CTPN(**config)

    save_weigths_file_path = args.save_weights_file_path

    if  save_weigths_file_path is None:
        try:
            if not os.path.exists("model"):
                os.makedirs("model")
            save_weigths_file_path = "weights/weights-ctpnlstm-{epoch:02d}.hdf5"
        except OSError:
            print('Error: Creating directory. ' + "model")

    data_loader = DataLoader(args.anno_dir, args.images_dir)

    checkpoint = SingleModelCK(save_weigths_file_path, model=ctpn.model, save_weights_only=True, monitor='loss')
    earlystop = EarlyStopping(patience=10, monitor='loss')
    log = TensorBoard(log_dir='logs', histogram_freq=0, batch_size=1, write_graph=True, write_grads=False)
    lr_scheduler = SGDRScheduler(min_lr=1e-6, max_lr=1e-4,
                                 initial_epoch=args.initial_epoch,
                                 steps_per_epoch=data_loader.steps_per_epoch,
                                 cycle_length=8,
                                 lr_decay=0.5,
                                 mult_factor=1.2)

    ctpn.train(data_loader.load_data(),
               epochs=args.epochs,
               steps_per_epoch=data_loader.steps_per_epoch,
               callbacks=[checkpoint, earlystop, lr_scheduler],
               initial_epoch=args.initial_epoch)
