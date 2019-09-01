import argparse, time, sys
import math

sys.path.insert(0, 'BMXNet-v2-examples/')
from mxnet import gluon, lr_scheduler
from mxnet.metric import Accuracy, TopKAccuracy, CompositeEvalMetric
import mxnet as mx
from util.log_progress import log_progress
from functools import reduce
from operator import mul


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
                                            mx.image.HorizontalFlipAug(0.5)
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
                                          mx.image.HorizontalFlipAug(0.5)
                                      ])
                                  ]
                                  )
    return train_iter, val_iter


def test(ctx, val_data):
    metric.reset()
    val_data.reset()
    time_counter=0
    counter=0
    for batch in val_data:
        print(counter)
        counter+=1
        data = gluon.utils.split_and_load(batch.data[0], ctx_list=ctx, batch_axis=0)
        label = gluon.utils.split_and_load(batch.label[0], ctx_list=ctx, batch_axis=0)
        outputs = []
        start=time.time()
        for x in data:
            outputs.append(net(x))
        for i in range(0,1):
            for x in data:
                buf=net(x)
        time_counter+=time.time()-start
        # metric.update(label, outputs)
        print(time.time()-start)
    print(time_counter)
    return metric.get()


def timed_test(ctx, val_data):
    print("starting test")
    start = time.time()
    name, val_acc = test(ctx, val_data)
    end = time.time()
    print("Execution time: ", end - start)
    return name, val_acc


metric = CompositeEvalMetric([Accuracy(), TopKAccuracy(5)])
ctx = [mx.cpu()]
_, val_data = get_mushrooms_9class(1, 1, 336, 336,
                                   "Data/mushrooms_9class/data_list_train.lst",
                                   "Data/mushrooms_9class/data_list_val.lst")

net = gluon.nn.SymbolBlock.imports("fp_alexnet/image-classifier-1bit-symbol.json", ['data'],
                                   "fp_alexnet/image-classifier-1bit-0000.params",
                                   ctx=ctx)
name, val_acc = timed_test(ctx, val_data)
print(' Acc: %s=%f, %s=%f' % (name[0], val_acc[0], name[1], val_acc[1]))
