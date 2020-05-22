from __future__ import print_function
from collections import OrderedDict

import numpy as np
import h5py
import torch
import paddle
import paddle.fluid as fluid

from . import util


PADDLE_MOVING_MEAN_KEY = 'moving_mean' 
PADDLE_MOVING_VARIANCE_KEY = 'moving_variance' 
PADDLE_EPSILON = 1e-3
PYTORCH_EPSILON = 1e-5


def check_for_missing_layers(paddle_layer_names, pytorch_layer_names, verbose):

    if verbose:
        print("Layer names in PyTorch state_dict", pytorch_layer_names)
        print("Layer names in paddle state_dict", paddle_layer_names)

    if not all(x in paddle_layer_names for x in pytorch_layer_names):
        missing_layers = list(set(pytorch_layer_names) - set(paddle_layer_names))
        raise Exception("Missing layer(s) in paddle HDF5 that are present" +
                        " in state_dict: {}".format(missing_layers))



def pytorch_to_paddle(pytorch_model, paddle_model,
                     flip_filters=False, flip_channels=None, verbose=True):


    paddle_dict = paddle_model.state_dict()
    fluid.save_dygraph(paddle_dict, "save_temp")
    pytorch_input_state_dict = pytorch_model.state_dict()
    pytorch_layer_names = util.state_dict_layer_names(pytorch_input_state_dict)

    with open('save_temp', 'a') as f:
        model_weights = f['model_weights']
        target_layer_names = list(map(str, model_weights.keys()))
        check_for_missing_layers(
            target_layer_names,
            pytorch_layer_names,
            verbose)

        for layer in pytorch_layer_names:

            paddle_h5_layer_param = util.dig_to_params_pf(model_weights[layer])

            weight_key = layer + '.weight'
            bias_key = layer + '.bias'
            running_mean_key = layer + '.running_mean'
            running_var_key = layer + '.running_var'

            # Load weights (or other learned parameters)
            if weight_key in pytorch_input_state_dict:
                weights = pytorch_input_state_dict[weight_key].numpy()
                weights = convert_weights(weights,
                                          to_pytorch=False,
                                          flip_filters=flip_filters,
                                          flip_channels=flip_channels)

            # Load bias
            if bias_key in pytorch_input_state_dict:
                bias = pytorch_input_state_dict[bias_key].numpy()
                if running_var_key in pytorch_input_state_dict:
                    paddle_h5_layer_param[bias_key][:] = bias
                else:
                    paddle_h5_layer_param[bias_key][:] = bias

            # Load batch normalization running mean
            if running_mean_key in pytorch_input_state_dict:
                running_mean = pytorch_input_state_dict[running_mean_key].numpy()
                paddle_h5_layer_param[PADDLE_MOVING_MEAN_KEY][:] = running_mean

            # Load batch normalization running variance
            if running_var_key in pytorch_input_state_dict:
                running_var = pytorch_input_state_dict[running_var_key].numpy()
                # account for difference in epsilon used
                running_var += PYTORCH_EPSILON - PADDLE_EPSILON
                paddle_h5_layer_param[PADDLE_MOVING_VARIANCE_KEY][:] = running_var

    # pytorch_model.load_state_dict(state_dict)
    paddle_model.load_weights('temp.h5')

# TODO
def convert_weights(weights, to_pytorch=False, flip_filters=False, flip_channels=False):

    if to_pytorch:
        # TODO
        weights = weights
    else:
        # TODO
        weights = weights
    return weights
