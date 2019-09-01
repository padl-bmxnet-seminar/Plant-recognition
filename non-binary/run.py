import mxnet as mx
import resnet
import fit
import argparse
import matplotlib.pyplot as plt
import numpy as np
import math

import os, sys

if sys.version_info[0] >= 3:
    from urllib.request import urlretrieve
else:
    from urllib import urlretrieve

def download(url):
    filename = url.split("/")[-1]
    if not os.path.exists(filename):
        urlretrieve(url, filename)


def get_model(prefix, epoch):
    download(prefix+'-symbol.json')
    download(prefix+'-%04d.params' % (epoch,))


def get_fine_tune_model(symbol, arg_params, num_classes, layer_name='flatten0'):
    """
    symbol: the pretrained network symbol
    arg_params: the argument parameters of the pretrained model
    num_classes: the number of classes for the fine-tune datasets
    layer_name: the layer name before the last fully-connected layer
    """
    all_layers = symbol.get_internals()
    net = all_layers[layer_name+'_output']
    net = mx.symbol.FullyConnected(data=net, num_hidden=num_classes, name='fc1')
    net = mx.symbol.SoftmaxOutput(data=net, name='softmax')
    new_args = dict({k:arg_params[k] for k in arg_params if 'fc1' not in k})
    return (net, new_args)


def gpu_device(gpu_number=2):
    try:
        _ = mx.nd.array([1, 2, 3], ctx=mx.gpu(gpu_number))
    except mx.MXNetError:
        return None
    return mx.gpu(gpu_number)


def get_iterators(train_batch_size, val_batch_size, height, width, train_path, val_path):
    train_iter = mx.image.ImageIter(train_batch_size, (3, height, width),
                                    path_root=".",
                                    path_imglist=train_path,
                                    aug_list=[
                                        mx.image.SequentialAug([
                                            mx.image.CastAug(typ='float32'),
                                            mx.image.RandomCropAug([height, width])
                                        ])
                                    ]
                                    )

    val_iter = mx.image.ImageIter(val_batch_size, (3, height, width),
                                  path_root=".",
                                  path_imglist=val_path,
                                  aug_list=[
                                      mx.image.SequentialAug([
                                          mx.image.CastAug(typ='float32'),
                                          mx.image.RandomCropAug([height, width])
                                      ])
                                  ]
                                  )
    return train_iter, val_iter


def get_mushrooms_9class(train_batch_size, val_batch_size, height, width, train_path, val_path):
    train_iter = mx.image.ImageIter(train_batch_size, (3, height, width),
                                    path_root="../binary_plant_recognition/Data/mushrooms_9class",
                                    path_imglist=train_path,
                                    aug_list=[
                                        mx.image.SequentialAug([
                                            mx.image.CastAug(typ='float32'),
                                            mx.image.ForceResizeAug([math.ceil(height+(height/10)), math.ceil(width+(width/10))]),
                                            mx.image.RandomCropAug((height, width)),
                                            mx.image.HorizontalFlipAug(0.5),
                                            mx.image.ColorJitterAug(0.1,0.1,0.1),
                                        ])
                                    ]
                                    )

    val_iter = mx.image.ImageIter(val_batch_size, (3, height, width),
                                  path_root="../binary_plant_recognition/Data/mushrooms_9class",
                                  path_imglist=val_path,
                                  aug_list=[
                                      mx.image.SequentialAug([
                                          mx.image.CastAug(typ='float32'),
                                          mx.image.ForceResizeAug([math.ceil(height+(height/10)), math.ceil(width+(width/10))]),
                                          mx.image.RandomCropAug((height, width)),
                                          mx.image.HorizontalFlipAug(0.5),
                                          mx.image.ColorJitterAug(0.1, 0.1, 0.1),
                                      ])
                                  ]
                                  )
    return train_iter, val_iter


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="resnet on clef dataset",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--height', type=int, default=336,
                        help='height of the training data')

    parser.add_argument('--width', type=int, default=336,
                        help='width of the training data')

    parser.add_argument('--train_batch_size', type=int, default=32,
                        help='batch size of train iterator')

    parser.add_argument('--val_batch_size', type=int, default=32,
                        help='batch size of val iterator')

    parser.add_argument('--num_epochs', type=int, default=150,
                        help='num of epochs')

    parser.add_argument('--train_path', default="../binary_plant_recognition/Data/mushrooms_9class/data_list_train.lst",
                        help='relative path of train.lst')

    parser.add_argument('--val_path', default="../binary_plant_recognition/Data/mushrooms_9class/data_list_val.lst",
                        help='relative path of val.lst')

    parser.add_argument('--pretrained', default=False,
                        help='use a pretrained modell')

    # parser.add_argument('--gpus', default='',
    #                     help='Gpus that are used e.g. 0,1')

    fit.add_fit_args(parser)

    parser.set_defaults(
        # network
        network='resnet',
        num_layers=18,
        # data
        num_classes=9,
        num_examples=25680,
        # min_random_scale=1,
        lr=0.0001,
        disp_batches=50,
        monitor=1,
        gpus='0',
        optimizer='adam',
        # train
        lr_step_epochs='30,60',
        dtype='float32',
    )

    args = parser.parse_args()
    args.batch_size = args.train_batch_size
    args.image_shape = '3,{},{}'.format(args.height, args.width)
    train_iter, val_iter = get_mushrooms_9class(args.train_batch_size, args.val_batch_size, args.height, args.width,
                                         args.train_path, args.val_path)

    sym = resnet.get_symbol(**vars(args))

    if args.pretrained:
        get_model('http://data.mxnet.io/models/imagenet/resnet/18-layers/resnet-18', 0)
        sym, arg_params, aux_params = mx.model.load_checkpoint('resnet-18', 0)
        print("downloaded and loaded model..")
        (new_sym, new_args) = get_fine_tune_model(sym, arg_params, 9)

        fit.fit(args, new_sym, train_iter, val_iter, arg_params=new_args, aux_params=aux_params)
    else:
        fit.fit(args, sym, train_iter, val_iter)
