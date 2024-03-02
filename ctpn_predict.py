import time
from tensorflow import keras 
import tensorflow as tf



from ctpn import default_ctpn_weight_path, default_ctpn_config_path, get_or_create
from ctpn import get_session

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", help="image_path")
    parser.add_argument("--config_file_path", help="config_path",
                        default=default_ctpn_config_path)
    parser.add_argument("--weights_file_path", help="weight_path",
                        default=default_ctpn_weight_path)
    parser.add_argument("--output_file_path", help="output_path",
                        default=None)

    args = parser.parse_args()

    tf.compat.v1.keras.backend.set_session(get_session())


    image_path = args.image_path  
    config_path = args.config_file_path  
    weight_path = args.weights_file_path  
    output_file_path = args.output_file_path  

    ctpn = get_or_create(config_path, weight_path)

    if weight_path is not None:
        ctpn = get_or_create(config_path, weight_path)
    else:
        ctpn = get_or_create(config_path)

    start_time = time.time()
    ctpn.predict(image_path, output_path=output_file_path)
    print("cost ", (time.time() - start_time) * 1000, " ms")
