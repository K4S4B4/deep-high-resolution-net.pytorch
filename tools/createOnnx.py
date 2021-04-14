# ------------------------------------------------------------------------------
# pose.pytorch
# Copyright (c) 2018-present Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pprint

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms

import _init_paths
from config import cfg
from config import update_config
from core.loss import JointsMSELoss
from core.function import validate
from utils.utils import create_logger

import dataset
import models

import cv2
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    parser.add_argument('--modelDir',
                        help='model directory',
                        type=str,
                        default='')
    parser.add_argument('--logDir',
                        help='log directory',
                        type=str,
                        default='')
    parser.add_argument('--dataDir',
                        help='data directory',
                        type=str,
                        default='')
    parser.add_argument('--prevModelDir',
                        help='prev Model directory',
                        type=str,
                        default='')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    update_config(cfg, args)

    logger, final_output_dir, tb_log_dir = create_logger(
        cfg, args.cfg, 'valid')

    logger.info(pprint.pformat(args))
    logger.info(cfg)

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    model = eval('models.'+cfg.MODEL.NAME+'.get_pose_net')(
        cfg, is_train=False
    )

    if cfg.TEST.MODEL_FILE:
        logger.info('=> loading model from {}'.format(cfg.TEST.MODEL_FILE))
        model.load_state_dict(torch.load(cfg.TEST.MODEL_FILE), strict=False)
    else:
        model_state_file = os.path.join(
            final_output_dir, 'final_state.pth'
        )
        logger.info('=> loading model from {}'.format(model_state_file))
        model.load_state_dict(torch.load(model_state_file))

#    model = torch.nn.DataParallel(model, device_ids=cfg.GPUS).cuda()

    ##############################################################################
    batch_size = 2
    height = 256
    width = 256
    ##############################################################################

    ###########################################################
    image = cv2.imread('resource/testdata/IMG_20210208_135527.jpg')
    resized = cv2.resize(image, (width, height))
    tensorImage = torch.tensor(resized).byte()
    tensorImage = tensorImage.unsqueeze(0)
    ###########################################################

    ###########################################################
    #image = cv2.imread('resource/testdata/IMG_20210208_135527.jpg')
    #resized = cv2.resize(image, (192,256))
    #img_in = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    #img_in = img_in.astype(np.float32)
    #img_in /= 255.0
    #mean=[0.485, 0.456, 0.406]
    #std=[0.229, 0.224, 0.225]
    #img_in /= mean
    #img_in -= std
    #img_in = np.transpose(img_in, (2, 0, 1))
    #tensorImage = torch.tensor(img_in, dtype=torch.float)
    #tensorImage = tensorImage.unsqueeze(0)
    ###########################################################

    ##############################################################################
    x = torch.randn((batch_size, height, width, 3), requires_grad=True).byte()
    ##############################################################################

    onnx_file_name = "hrnet_{}x{}x{}xBGRxByte.onnx".format(batch_size, height, width)
    input_names = ["input"]
    output_names = ['joint2d', 'confidence']
    torch.onnx.export(model,
                    x,
                    onnx_file_name,
                    export_params=True,
                    opset_version=11,
                    do_constant_folding=True,
                    input_names=input_names, 
                    output_names=output_names,
                    #dynamic_axes={'input' : {0 : 'batch_size'},
                    #    'joint2d' : {0 : 'batch_size'},
                    #    'confidence' : {0 : 'batch_size'}}
                    dynamic_axes=None
                    )
    print('Onnx model exporting done')

    onnx_file_name = "hrnet_Bx{}x{}xBGRxByte.onnx".format(height, width)
    input_names = ["input"]
    output_names = ['joint2d', 'confidence']
    torch.onnx.export(model,
                    x,
                    onnx_file_name,
                    export_params=True,
                    opset_version=11,
                    do_constant_folding=True,
                    input_names=input_names, 
                    output_names=output_names,
                    dynamic_axes={'input' : {0 : 'batch_size'},
                        'joint2d' : {0 : 'batch_size'},
                        'confidence' : {0 : 'batch_size'}}
                    )
    print('dynamic model exporting done')

    ###########################################################
    #outputs = model(tensorImage)
    #for i in range(17):
    #    sample = outputs[0][i].to('cpu').detach().numpy().copy()
    #    cv2.imshow('output', sample)
    #    cv2.waitKey(0)
    ###########################################################




if __name__ == '__main__':
    main()
