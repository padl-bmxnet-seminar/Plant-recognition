# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

from __future__ import division

import argparse, time, sys
import math
from graphviz import ExecutableNotFound
from mxnet import gluon, lr_scheduler
from mxnet import profiler
import sys

sys.path.insert(0, 'BMXNet-v2-examples/')
import binary_models
from util.log_progress import log_progress
from mxnet import autograd
from mxnet.test_utils import get_mnist_iterator
from mxnet.metric import Accuracy, TopKAccuracy, CompositeEvalMetric
from mxnet.gluon import nn
from contextlib import redirect_stdout
import numpy as np

from datasets.data import *


# CLI
def get_parser(training=True):
    parser = argparse.ArgumentParser(description='Train a model for image classification.')
    model = parser.add_argument_group('Model', 'parameters for the model definition')
    model.add_argument('--bits', type=int, default=1,
                       help='number of weight bits')
    model.add_argument('--bits-a', type=int, default=1,
                       help='number of bits for activation')
    model.add_argument('--activation-method', type=str, default='det_sign',
                       choices=['identity', 'approx_sign', 'relu', 'clip', 'leaky_clip',
                                'det_sign', 'sign_approx_sign', 'round', 'dorefa'],
                       help='choose activation in QActivation layer')
    model.add_argument('--weight-quantization', type=str, default='det_sign',
                       choices=['det_sign', 'dorefa', 'identity', 'sign_approx_sign'],
                       help='choose weight quantization')
    model.add_argument('--clip-threshold', type=float, default=1.0,
                       help='clipping threshold, default is 1.0.')
    model.add_argument('--model', type=str, required=True,
                       help='type of model to use. see vision_model for options.')
    model.add_argument('--use-pretrained', action='store_true',
                       help='enable using pretrained model from gluon.')
    model.add_argument('--approximation', type=str, default='',
                       choices=['', 'xnor', 'binet', 'abc'],
                       help='Choose kind of convolution block approximation. Can include scaling or multiple binary convolutions.')


    if training:
        train = parser.add_argument_group('Training', 'parameters for training')
        train.add_argument('--augmentation-level', type=int, choices=[1, 2, 3], default=3,
                           help='augmentation level, default is 1, possible values are: 1, 2, 3.')
        train.add_argument('--dry-run', action='store_true',
                           help='do not train, only do other things, e.g. output args and plot network')
        train.add_argument('--epochs', type=int, default=120,
                           help='number of training epochs.')
        train.add_argument('--initialization', type=str,
                           choices=["default", "gaussian", "msraprelu_avg", "msraprelu_in"], default="gaussian",
                           help='weight initialization, default is xavier with magnitude 2.')
        train.add_argument('--kvstore', type=str, default='device',
                           help='kvstore to use for trainer/module.')
        train.add_argument('--log', type=str, default='image-classification.log',
                           help='Filename and path where log file should be stored.')
        train.add_argument('--log-interval', type=int, default=50,
                           help='Number of batches to wait before logging.')
        train.add_argument('--lr', type=float, default=0.01,
                           help='learning rate. default is 0.01.')
        train.add_argument('--lr-factor', default=0.1, type=float,
                           help='learning rate decay ratio')
        train.add_argument('--lr-steps', default='30,60,90', type=str,
                           help='list of learning rate decay epochs as in str')
        train.add_argument('--momentum', type=float, default=0.9,
                           help='momentum value for optimizer, default is 0.9.')
        train.add_argument('--resume-states', type=str, default='',
                           help='path to saved weight where you want resume')
        train.add_argument('--optimizer', type=str, default="adam",
                           help='the optimizer to use. default is adam.')
        train.add_argument('--plot-network', type=str, default=None,
                           help='Whether to output the network plot.')
        train.add_argument('--profile', action='store_true',
                           help='Option to turn on memory profiling for front-end, and prints out '
                                'the memory usage by python function at the end.')
        train.add_argument('--progress', type=str, default="",
                           help='save progress and ETA to this file')
        train.add_argument('--resume', type=str, default='',
                           help='path of trainer state to load from.')
        train.add_argument('--save-frequency', default=None, type=int,
                           help='epoch frequence to save model, best model will always be saved')
        train.add_argument('--seed', type=int, default=123,
                           help='random seed to use. Default=123.')
        train.add_argument('--start-epoch', default=0, type=int,
                           help='starting epoch, 0 for fresh training, > 0 to resume')
        train.add_argument('--wd', type=float, default=0.0,
                           help='weight decay rate. default is 0.0.')
        train.add_argument('--write-summary', type=str, default=None,
                           help='write tensorboard summaries to this path')
        train.add_argument('--clip-threshold-steps', type=csv_args_dict, default="",
                           help='lower clip threshold to this at certain times, k:v pairs (eg. 10:1.5,20:1.0)'),
        train.add_argument('--pretrained-imagenet', type=bool, default=False,
                           help='use pretrained imagenet data')


    parser.add_argument('--batch-size', type=int, default=32,
                        help='training batch size per device (CPU/GPU).')
    parser.add_argument('--builtin-profiler', type=int, default=0,
                        help='Enable built-in profiler (0=off, 1=on)')
    parser.add_argument('--data-dir', type=str, default='',
                        help='training directory of imagenet images, contains train/val subdirs.')
    parser.add_argument('--data-path', type=str, default='.',
                        help='training directory where cifar10 / mnist data should be or is saved.')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['mnist', 'cifar10', 'imagenet', 'dummy', 'plants', 'plants_10_class', 'mushrooms',
                                 'mushrooms_9class'],
                        help='dataset to use. options are mnist, cifar10, imagenet and dummy.')
    parser.add_argument('--dtype', default='float32', type=str,
                        help='data type, float32 or float16 if applicable')
    parser.add_argument('--gpus', type=str, default='',
                        help='ordinates of gpus to use, can be "0,1,2" or empty for cpu only.')
    parser.add_argument('--cpu', action="store_const", dest="gpus", const="",
                        help='to explicitly use cpu')
    parser.add_argument('--mean-subtraction', action="store_true",
                        help='whether to subtract ImageNet mean from data')
    parser.add_argument('--mode', type=str, choices=["symbolic", "imperative", "hybrid"], default="imperative",
                        help='mode in which to train the model. options are symbolic, imperative, hybrid')
    parser.add_argument('--num-worker', '-j', dest='num_workers', default=4, type=int,
                        help='number of workers of dataloader.')
    parser.add_argument('--prefix', default='', type=str,
                        help='path to checkpoint prefix, default is current working dir')
    parser.add_argument('--validation', default='', type=str,
                        help='Uses validation val part of the dataset to run against. Needs resume path to load a model.')
    first_args = sys.argv[1:][:]
    for pattern in ("--help", "-h"):
        if pattern in first_args:
            first_args.remove(pattern)
    first_parse, _ = parser.parse_known_args(first_args)
    for model_parameter in binary_models.get_model_parameters():
        model_parameter.add_group(parser, first_parse.model)
    return parser


if sys.version_info[0] >= 0:
    from urllib.request import urlretrieve
else:
    from urllib import urlretrieve

def download(url):
    filename = url.split("/")[-1]
    if not os.path.exists(filename):
        urlretrieve(url, filename)


def download_model(prefix, epoch):
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


def get_model_path(opt):
    return os.path.join(opt.prefix, 'image-classifier-%s' % opt.model)


def csv_args_dict(value):
    if value == "":
        return {}
    return {int(k): float(v) for k, v in [pair.split(":") for pair in value.split(",")]}


def _load_model(opt):
    model_prefix = get_model_path(opt)
    logger.info('Loaded model %s-%04d.params', model_prefix, opt.start_epoch)
    return mx.model.load_checkpoint(model_prefix, opt.start_epoch)


def _get_scaling_legacy(opt):
    model_name, modifier = opt.model.split('-')
    opt.model = model_name
    scaling = ""
    if 'binet_scaled' in modifier:
        scaling = "binet"
    if 'scaled' in modifier:
        assert scaling == "", "Contradicting model modifiers: binet_scaled, scaled. Please specify only on scaling method"
        scaling = "xnor"
    return scaling


def get_model(opt, ctx):
    """Model initialization."""
    kwargs = {'ctx': ctx, 'pretrained': opt.use_pretrained, 'classes': get_num_classes(opt.dataset)}
    if opt.model.startswith('vgg'):
        kwargs['batch_norm'] = opt.batch_norm

    if any(opt.model.startswith(name) for name in ['resnet', 'binet', 'densenet']) and get_shape(opt)[2] < 50:
        kwargs['thumbnail'] = True

    for model_parameter in binary_models.get_model_parameters():
        model_parameter.set_args_for_model(opt, kwargs)

    skip_init = False
    arg_params, aux_params = None, None
    if opt.start_epoch > 0 and opt.mode == 'symbolic':
        net, arg_params, aux_params = _load_model(opt)
        skip_init = True
    else:
        # approximation = opt.approximation or _get_scaling_legacy(opt)
        approximation = opt.approximation
        with gluon.nn.set_binary_layer_config(bits=opt.bits, bits_a=opt.bits_a, approximation=approximation,
                                              grad_cancel=opt.clip_threshold, activation=opt.activation_method,
                                              weight_quantization=opt.weight_quantization):
            net = binary_models.get_model(opt.model, **kwargs)

    if opt.resume:
        net.load_parameters(opt.resume)
    elif not opt.use_pretrained and not skip_init:
        if opt.model in ['alexnet']:
            net.initialize(mx.init.Normal(), ctx=ctx)
        else:
            net.initialize(get_initializer(), ctx=ctx)
    return net, arg_params, aux_params


def get_num_classes(dataset):
    return \
        {'mnist': 10, 'cifar10': 10, 'imagenet': 1000, 'dummy': 1000, 'plants': 126, 'plants_10_class': 10,
         'mushrooms': 10,
         'mushrooms_9class': 9}[dataset]


def get_default_save_frequency(dataset):
    return {'mnist': 10, 'cifar10': 10, 'imagenet': 1, 'dummy': 1, 'plants': 10, 'plants_10_class': 10, 'mushrooms': 10,
            'mushrooms_9class': 9}[dataset]


def get_num_examples(dataset):
    return \
        {'mnist': 60000, 'cifar10': 50000, 'imagenet': 1281167, 'dummy': 1000, 'plants': 8802, 'plants_10_class': 8000,
         'mushrooms': 3727, 'mushrooms_9class': 25680}[dataset]


def get_shape(opt):
    if opt.dataset == 'mnist':
        return (1, 1, 28, 28)
    elif opt.dataset == 'cifar10':
        return (1, 3, 32, 32)
    elif opt.dataset == 'imagenet' or opt.dataset == 'dummy':
        return (1, 3, 299, 299) if opt.model == 'inceptionv3' else (1, 3, 224, 224)
    elif opt.dataset == 'plants':
        return (1, 3, 600, 600)
    elif opt.dataset == 'plants_10_class':
        return (1, 3, 600, 600)
    elif opt.dataset == 'mushrooms':
        return (1, 3, 336, 336)
    elif opt.dataset == 'mushrooms_9class':
        return (1, 3, 336, 336)


def get_initializer():
    if opt.initialization == "default":
        return mx.init.Xavier(magnitude=2)
    if opt.initialization == "gaussian":
        return mx.init.Xavier(rnd_type="gaussian", factor_type="in", magnitude=2)
    if opt.initialization == "msraprelu_avg":
        return mx.init.MSRAPrelu()
    if opt.initialization == "msraprelu_in":
        return mx.init.MSRAPrelu(factor_type="in")


def get_CLEF_iterators(train_batch_size, val_batch_size, height, width, train_path, val_path):
    train_iter = mx.image.ImageIter(train_batch_size, (3, height, width),
                                    path_root=".",
                                    path_imglist=train_path,
                                    aug_list=[
                                        mx.image.SequentialAug([
                                            mx.image.CastAug(typ='float32'),
                                            # mx.image.ResizeAug(100),
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
                                          # mx.image.ResizeAug(100),
                                          mx.image.RandomCropAug([height, width])
                                      ])
                                  ]
                                  )
    return train_iter, val_iter


def get_plants_10_class(train_batch_size, val_batch_size, height, width, train_path, val_path):
    train_iter = mx.image.ImageIter(train_batch_size, (3, height, width),
                                    path_root="./Data/plants/Train/",
                                    path_imglist=train_path,
                                    aug_list=[
                                        mx.image.SequentialAug([
                                            mx.image.CastAug(typ='float32'),
                                            # mx.image.ResizeAug(100),
                                            mx.image.RandomCropAug([height, width])
                                        ])
                                    ]
                                    )

    val_iter = mx.image.ImageIter(val_batch_size, (3, height, width),
                                  path_root="./Data/plants/Valid/",
                                  path_imglist=val_path,
                                  aug_list=[
                                      mx.image.SequentialAug([
                                          mx.image.CastAug(typ='float32'),
                                          # mx.image.ResizeAug(100),
                                          mx.image.RandomCropAug([height, width])
                                      ])
                                  ]
                                  )
    return train_iter, val_iter


def get_mushrooms(train_batch_size, val_batch_size, height, width, train_path, val_path):
    train_iter = mx.image.ImageIter(train_batch_size, (3, height, width),
                                    path_root="./Data/mushrooms/Train/",
                                    path_imglist=train_path,
                                    aug_list=[
                                        mx.image.SequentialAug([
                                            mx.image.CastAug(typ='float32'),
                                            # mx.image.ResizeAug(100),
                                            mx.image.RandomCropAug([height, width])
                                        ])
                                    ]
                                    )

    val_iter = mx.image.ImageIter(val_batch_size, (3, height, width),
                                  path_root="./Data/mushrooms/Train/",
                                  path_imglist=val_path,
                                  aug_list=[
                                      mx.image.SequentialAug([
                                          mx.image.CastAug(typ='float32'),
                                          # mx.image.ResizeAug(100),
                                          mx.image.RandomCropAug([height, width])
                                      ])
                                  ]
                                  )
    return train_iter, val_iter


def get_mushrooms_9class(train_batch_size, val_batch_size, height, width, train_path, val_path):
    train_iter = mx.image.ImageIter(train_batch_size, (3, height, width),
                                    path_root="./Data/mushrooms_9class",
                                    path_imglist=train_path,
                                    aug_list=[
                                        mx.image.SequentialAug([
                                            mx.image.CastAug(typ='float32'),
                                            # mx.image.ResizeAug(100),
                                            mx.image.ForceResizeAug(
                                                [math.ceil(height + (height / 10)), math.ceil(width + (width / 10))]),
                                            mx.image.RandomCropAug((height, width)),
                                            mx.image.HorizontalFlipAug(0.5),
                                            mx.image.ColorJitterAug(0.1, 0.1, 0.1),
                                        ])
                                    ]
                                    )

    val_iter = mx.image.ImageIter(val_batch_size, (3, height, width),
                                  path_root="./Data/mushrooms_9class",
                                  path_imglist=val_path,
                                  aug_list=[
                                      mx.image.SequentialAug([
                                          mx.image.CastAug(typ='float32'),
                                          # mx.image.ResizeAug(100),
                                          mx.image.ForceResizeAug(
                                              [math.ceil(height + (height / 10)), math.ceil(width + (width / 10))]),
                                          mx.image.RandomCropAug((height, width)),
                                          mx.image.HorizontalFlipAug(0.5),
                                          mx.image.ColorJitterAug(0.1, 0.1, 0.1),
                                      ])
                                  ]
                                  )
    return train_iter, val_iter


def get_data_iters(opt, num_workers=1, rank=0):
    """get dataset iterators"""
    if opt.dry_run:
        return None, None

    if opt.dataset == 'mnist':
        train_data, val_data = get_mnist_iterator(opt.batch_size, (1, 28, 28),
                                                  num_parts=num_workers, part_index=rank)
    elif opt.dataset == 'cifar10':
        train_data, val_data = get_cifar10_iterator(opt.batch_size, (3, 32, 32), num_parts=num_workers, part_index=rank,
                                                    dir=opt.data_path, aug_level=opt.augmentation_level,
                                                    mean_subtraction=opt.mean_subtraction)
    elif opt.dataset == 'imagenet':
        if not opt.data_dir:
            raise ValueError('Dir containing rec files is required for imagenet, please specify "--data-dir"')
        if opt.model == 'inceptionv3':
            train_data, val_data = get_imagenet_iterator(opt.data_dir, opt.batch_size, opt.num_workers, 299, opt.dtype)
        else:
            train_data, val_data = get_imagenet_iterator(opt.data_dir, opt.batch_size, opt.num_workers, 224, opt.dtype)
    elif dataset == 'dummy':
        if opt.model == 'inceptionv3':
            train_data, val_data = dummy_iterator(opt.batch_size, (3, 299, 299))
        else:
            train_data, val_data = dummy_iterator(opt.batch_size, (3, 224, 224))
    elif dataset == 'plants':
        train_data, val_data = get_CLEF_iterators(opt.batch_size, opt.batch_size, 600, 600, "./train_data_list.lst",
                                                  "./val_data_list.lst")
    elif dataset == 'plants_10_class':
        train_data, val_data = get_plants_10_class(opt.batch_size, opt.batch_size, 299, 299,
                                                   "Data/plants/Train/train_data_list.lst",
                                                   "Data/plants/Valid/valid_data_list.lst")
    elif dataset == 'mushrooms':
        train_data, val_data = get_mushrooms(opt.batch_size, opt.batch_size, 336, 336,
                                             "Data/mushrooms/Train/data_list_train.lst",
                                             "Data/mushrooms/Train/data_list_val.lst")
    elif dataset == 'mushrooms_9class':
        train_data, val_data = get_mushrooms_9class(opt.batch_size, opt.batch_size, 336, 336,
                                                    "Data/mushrooms_9class/data_list_train.lst",
                                                    "Data/mushrooms_9class/data_list_val.lst")
    return train_data, val_data


def test(ctx, val_data):
    metric.reset()
    val_data.reset()
    for batch in val_data:
        data = gluon.utils.split_and_load(batch.data[0], ctx_list=ctx, batch_axis=0)
        label = gluon.utils.split_and_load(batch.label[0], ctx_list=ctx, batch_axis=0)
        outputs = []
        for x in data:
            outputs.append(net(x))
        metric.update(label, outputs)
    return metric.get()


def timed_test(ctx, val_data):
    start = time.time()
    name, val_acc = test(ctx, val_data)
    end = time.time()
    print("Execution time: ", end - start)
    return name, val_acc


def update_learning_rate(lr, trainer, epoch, ratio, steps):
    """Set the learning rate to the initial value decayecirc d by ratio every N epochs."""
    new_lr = lr * (ratio ** int(np.sum(np.array(steps) < epoch)))
    if trainer.learning_rate != new_lr:
        logger.info('[Epoch %d] Change learning rate to %f', epoch, new_lr)
    trainer.set_learning_rate(new_lr)
    return trainer


def update_clip_threshold(current_clip_threshold, epoch, steps, q_activations,
                          check_previous_epochs=False, check_only=False):
    """Set the learning rate to the initial value decayed by ratio every N epochs."""
    new_clip_threshold = current_clip_threshold

    if check_previous_epochs:
        for i in range(0, epoch):
            new_clip_threshold = update_clip_threshold(current_clip_threshold, i, steps, check_only=True)

    if epoch in steps:
        new_clip_threshold = steps[epoch]

    if new_clip_threshold != current_clip_threshold and not check_only:
        for q_act in q_activations:
            q_act.threshold = new_clip_threshold
        logger.info('[Epoch %d] Change clip threshold to %f', epoch, new_clip_threshold)

    return new_clip_threshold


def save_checkpoint(trainer, epoch, top1, best_acc):
    if opt.save_frequency and (epoch + 1) % opt.save_frequency == 0:
        fname = os.path.join(opt.prefix, '%s_%sbit_%04d_acc_%.4f.{}' % (opt.model, opt.bits, epoch, top1))
        net.save_parameters(fname.format("params"))
        trainer.save_states(fname.format("states"))
        logger.info('[Epoch %d] Saving checkpoint to %s with Accuracy: %.4f',
                    epoch, fname.format("{params,states}"), top1)
    if top1 > best_acc[0]:
        best_acc[0] = top1
        fname = os.path.join(opt.prefix, '%s_%sbit_best.{}' % (opt.model, opt.bits))
        net.save_parameters(fname.format("params"))
        trainer.save_states(fname.format("states"))
        logger.info('[Epoch %d] Saving checkpoint to %s with Accuracy: %.4f',
                    epoch, fname.format("{params,states}"), top1)


def get_dummy_data(opt, ctx):
    data_shape = get_shape(opt)
    shapes = ((1,) + data_shape[1:], (1,))
    return [mx.nd.array(np.zeros(shape), ctx=ctx) for shape in shapes]


def _get_lr_scheduler(opt):
    if 'lr_factor' not in opt or opt.lr_factor >= 1:
        return opt.lr, None
    global lr_steps, batch_size
    lr, lr_factor = opt.lr, opt.lr_factor
    start_epoch = opt.start_epoch
    num_examples = get_num_examples(opt.dataset)
    its_per_epoch = math.ceil(num_examples / batch_size)

    # move forward to start epoch
    for s in lr_steps:
        if start_epoch >= s:
            lr *= lr_factor
    if lr != opt.lr:
        logger.info('Adjust learning rate to %e for epoch %d', lr, start_epoch)

    steps = [its_per_epoch * (epoch - start_epoch)
             for epoch in lr_steps if epoch - start_epoch > 0]
    if steps:
        return lr, lr_scheduler.MultiFactorScheduler(step=steps, factor=lr_factor)
    else:
        return lr, None


def get_optimizer(opt, with_scheduler=False):
    # learning rate
    lr, lr_scheduler = _get_lr_scheduler(opt)
    params = {
        'learning_rate': lr,
        'wd': opt.wd,
        'multi_precision': True
    }
    if with_scheduler:
        params['lr_scheduler'] = lr_scheduler
    if opt.optimizer == "sgd":
        params['momentum'] = opt.momentum
    return opt.optimizer, params


def get_blocks(net, search_for_type, result=()):
    for _, child in net._children.items():
        if isinstance(child, search_for_type):
            return result + (child,)
        result = get_blocks(child, search_for_type, result=result)
    return result


def train(opt, ctx):
    if isinstance(ctx, mx.Context):
        ctx = [ctx]

    kv = mx.kv.create(opt.kvstore)
    train_data, val_data = get_data_iters(opt, kv.num_workers, kv.rank)

    net.collect_params().reset_ctx(ctx)

    trainer = gluon.Trainer(net.collect_params(), *get_optimizer(opt), kvstore=kv)
    if opt.resume_states is not '':
        trainer.load_states(opt.resume_states)

    if opt.pretrained_imagenet:
        print('pretrained')


    loss = gluon.loss.SoftmaxCrossEntropyLoss()

    # dummy forward pass to initialize binary layers
    with autograd.record():
        data, label = get_dummy_data(opt, ctx[0])
        output = net(data)

    # set batch norm wd to zero
    params = net.collect_params('.*batchnorm.*')
    for key in params:
        params[key].wd_mult = 0.0

    if opt.plot_network is not None:
        x = mx.sym.var('data')
        sym = net(x)
        with open('{}.txt'.format(opt.plot_network), 'w') as f:
            with redirect_stdout(f):
                mx.viz.print_summary(sym, shape={"data": get_shape(opt)}, quantized_bitwidth=opt.bits)
        a = mx.viz.plot_network(sym, shape={"data": get_shape(opt)})
        try:
            a.render('{}.gv'.format(opt.plot_network))
        except OSError as e:
            logger.error(e)
        except ExecutableNotFound as e:
            logger.error(e)

    if opt.validation == 'true':
        name, val_acc = timed_test(ctx, val_data)
        logger.info(' Acc: %s=%f, %s=%f' % (name[0], val_acc[0], name[1], val_acc[1]))
        return
    if opt.dry_run:
        return

    summary_writer = None
    global_step = 0
    if opt.write_summary:
        from mxboard import SummaryWriter
        summary_writer = SummaryWriter(logdir=opt.write_summary, flush_secs=30)
        params = net.collect_params(".*weight|.*bias")
        for name, param in params.items():
            summary_writer.add_histogram(tag=name, values=param.data(ctx[0]),
                                         global_step=global_step, bins=1000)
            summary_writer.add_histogram(tag="%s-grad" % name, values=param.grad(ctx[0]),
                                         global_step=global_step, bins=1000)

    total_time = 0
    num_epochs = 0
    best_acc = [0]
    epoch_time = -1
    q_activations = get_blocks(net, nn.QActivation)
    update_clip_threshold(opt.clip_threshold, opt.start_epoch, opt.clip_threshold_steps, q_activations,
                          check_previous_epochs=True)
    for epoch in range(opt.start_epoch, opt.epochs):
        trainer = update_learning_rate(opt.lr, trainer, epoch, opt.lr_factor, lr_steps)
        if epoch != opt.start_epoch:
            update_clip_threshold(opt.clip_threshold, epoch, opt.clip_threshold_steps, q_activations)
        tic = time.time()
        train_data.reset()
        metric.reset()
        btic = time.time()
        for i, batch in enumerate(train_data):
            data = gluon.utils.split_and_load(batch.data[0].astype(opt.dtype), ctx_list=ctx, batch_axis=0)
            label = gluon.utils.split_and_load(batch.label[0].astype(opt.dtype), ctx_list=ctx, batch_axis=0)
            outputs = []
            Ls = []
            with autograd.record():
                for x, y in zip(data, label):
                    z = net(x)
                    L = loss(z, y)
                    # store the loss and do backward after we have done forward
                    # on all GPUs for better speed on multiple GPUs.
                    Ls.append(L)
                    outputs.append(z)
                autograd.backward(Ls)
            trainer.step(batch.data[0].shape[0])
            metric.update(label, outputs)
            if opt.log_interval and not (i + 1) % opt.log_interval:
                name, acc = metric.get()
                logger.info('Epoch[%d] Batch [%d]\tSpeed: %f samples/sec\t%s=%f, %s=%f' % (
                    epoch, i, batch_size / (time.time() - btic), name[0], acc[0], name[1], acc[1]))
                log_progress(get_num_examples(opt.dataset), opt, epoch, i, time.time() - tic, epoch_time)
                if summary_writer:
                    summary_writer.add_scalar("batch-%s" % name[0], acc[0], global_step=global_step)
                    summary_writer.add_scalar("batch-%s" % name[1], acc[1], global_step=global_step)
            btic = time.time()
            global_step += batch_size

        epoch_time = time.time() - tic

        if summary_writer:
            params = net.collect_params(".*weight|.*bias")
            for name, param in params.items():
                summary_writer.add_histogram(tag=name, values=param.data(ctx[0]),
                                             global_step=global_step, bins=1000)
                summary_writer.add_histogram(tag="%s-grad" % name, values=param.grad(ctx[0]),
                                             global_step=global_step, bins=1000)

        # First epoch will usually be much slower than the subsequent epics,
        # so don't factor into the average
        if num_epochs > 0:
            total_time = total_time + epoch_time
        num_epochs = num_epochs + 1

        # train
        name, acc = metric.get()
        logger.info('[Epoch %d] training: %s=%f, %s=%f' % (epoch, name[0], acc[0], name[1], acc[1]))
        logger.info('[Epoch %d] time cost: %f' % (epoch, epoch_time))
        if summary_writer:
            summary_writer.add_scalar("epoch", epoch, global_step=global_step)
            summary_writer.add_scalar("epoch-time", epoch_time, global_step=global_step)
            summary_writer.add_scalar("training-%s" % name[0], acc[0], global_step=global_step)
            summary_writer.add_scalar("training-%s" % name[1], acc[1], global_step=global_step)

        # test
        name, val_acc = test(ctx, val_data)
        logger.info('[Epoch %d] validation: %s=%f, %s=%f' % (epoch, name[0], val_acc[0], name[1], val_acc[1]))
        if summary_writer:
            summary_writer.add_scalar("validation-%s" % name[0], val_acc[0], global_step=global_step)
            summary_writer.add_scalar("validation-%s" % name[1], val_acc[1], global_step=global_step)

        # save model if meet requirements
        save_checkpoint(trainer, epoch, val_acc[0], best_acc)
    if num_epochs > 1:
        print('Average epoch time: {}'.format(float(total_time) / (num_epochs - 1)))

    if opt.mode != 'hybrid':
        net.hybridize()
        # dummy forward pass to save model
        with autograd.record():
            data, label = get_dummy_data(opt, ctx[0])
            output = net(data)
    net.export(os.path.join(opt.prefix, "image-classifier-{}bit".format(opt.bits)), epoch=0)


def main():
    if opt.builtin_profiler > 0:
        profiler.set_config(profile_all=True, aggregate_stats=True)
        profiler.set_state('run')
    if opt.mode == 'symbolic':
        train_symbolic(opt, context)
    else:
        if opt.mode == 'hybrid':
            net.hybridize()
        train(opt, context)
    if opt.builtin_profiler > 0:
        profiler.set_state('stop')
        print(profiler.dumps())


if __name__ == '__main__':
    parser = get_parser()
    opt = parser.parse_args()

    # logging
    logging.basicConfig(level=logging.INFO)
    fh = logging.FileHandler(opt.log)
    logger = logging.getLogger()
    logger.addHandler(fh)
    formatter = logging.Formatter('%(message)s')
    fh.setFormatter(formatter)
    fh.setLevel(logging.DEBUG)
    logging.debug('\n%s', '-' * 100)
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    fh.setFormatter(formatter)

    # global variables
    if opt.save_frequency is None:
        opt.save_frequency = get_default_save_frequency(opt.dataset)
    logger.info('Starting new image-classification task:, %s', opt)
    mx.random.seed(opt.seed)
    batch_size, dataset, classes = opt.batch_size, opt.dataset, get_num_classes(opt.dataset)
    context = [mx.gpu(int(i)) for i in opt.gpus.split(',')] if opt.gpus.strip() else [mx.cpu()]
    if opt.dry_run:
        context = [mx.cpu()]
    num_gpus = len(context)
    batch_size *= max(1, num_gpus)
    opt.batch_size = batch_size
    lr_steps = [int(x) for x in opt.lr_steps.split(',') if x.strip()]
    metric = CompositeEvalMetric([Accuracy(), TopKAccuracy(5)])

    net, arg_params, aux_params = get_model(opt, context)
    if opt.pretrained_imagenet:
        prefix = "./pretrained_imagenet/resnet18_v1"
        save_dict = mx.nd.load('%s-%04d.params' % (prefix, 0))
        arg_params = {}
        for k, v in save_dict.items():
            tp, name = k.split(':', 1)
            if tp == 'arg':
                arg_params[name] = v
        arg_params = dict({k: arg_params[k] for k in arg_params if 'fc1' not in k})
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            net = gluon.nn.SymbolBlock.imports("./pretrained_imagenet/resnet18_v1-symbol.json", ['data'],
                                               "./pretrained_imagenet/resnet18_v1-0000.params", ctx=context)

        print("loaded pretrained parameters..")

    if opt.profile:
        import hotshot, hotshot.stats

        prof = hotshot.Profile('image-classifier-%s-%s.prof' % (opt.model, opt.mode))
        prof.runcall(main)
        prof.close()
        stats = hotshot.stats.load('image-classifier-%s-%s.prof' % (opt.model, opt.mode))
        stats.strip_dirs()
        stats.sort_stats('cumtime', 'calls')
        stats.print_stats()
    else:
        main()


