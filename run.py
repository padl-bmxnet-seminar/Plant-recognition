import mxnet as mx
import resnet
import fit
import argparse
import matplotlib.pyplot as plt
import numpy as np


def gpu_device(gpu_number=0):
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="resnet on clef dataset",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--height', type=int, default=650,
                        help='height of the training data')

    parser.add_argument('--width', type=int, default=650,
                        help='width of the training data')

    parser.add_argument('--train_batch_size', type=int, default=3,
                        help='batch size of train iterator')

    parser.add_argument('--val_batch_size', type=int, default=3,
                        help='batch size of val iterator')

    parser.add_argument('--num_epochs', type=int, default=5,
                        help='num of epochs')

    parser.add_argument('--train_path', default="train_data_list.lst",
                        help='relative path of train.lst')

    parser.add_argument('--val_path', default="val_data_list.lst",
                        help='relative path of val.lst')

    # parser.add_argument('--gpus', default='',
    #                     help='Gpus that are used e.g. 0,1')

    fit.add_fit_args(parser)

    parser.set_defaults(
        # network
        network='resnet',
        num_layers=18,
        # data
        num_classes=250,
        num_examples=16636,
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
    train_iter, val_iter = get_iterators(args.train_batch_size, args.val_batch_size, args.height, args.width,
                                         args.train_path, args.val_path)
    # batch = train_iter.next()
    # data = batch.data[0]
    # for i in range(5):
    #     plt.subplot(1, 5, i + 1)
    #     plt.imshow(data[i].asnumpy().astype(np.uint8).transpose((1, 2, 0)))
    # plt.show()
    sym = resnet.get_symbol(**vars(args))

    fit.fit(args, sym, train_iter, val_iter)
