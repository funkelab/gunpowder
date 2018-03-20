from __future__ import print_function
import math

import caffe
from caffe import layers as L
from caffe import params as P
from caffe import to_proto
from caffe import metalayers as ML

# Start a network
net = caffe.NetSpec()

# Data input layer
net.data = L.MemoryData(dim=[1, 1], ntop=1)
# Label input layer
net.label = L.MemoryData(dim=[1, 1], ntop=1, include=[dict(phase=0)])
# Scale input layer
net.scale = L.MemoryData(dim=[1, 1], ntop=1, include=[dict(phase=0)])

# USK-Net metalayer
net.unet = ML.UNet(net.data, fmap_start=12, depth=3, fmap_inc_rule = lambda fmaps: int(math.ceil(float(fmaps) * 5)), fmap_dec_rule = lambda fmaps: int(math.ceil(float(fmaps) / 5)), downsampling_strategy = [[2,2,2],[2,2,2],[3,3,3]], dropout = 0.0, use_deconv_uppath=False, use_stable_upconv=True)

net.out = L.Convolution(net.unet, kernel_size=[1], num_output=1, param=[dict(lr_mult=1),dict(lr_mult=2)], weight_filler=dict(type='msra'), bias_filler=dict(type='constant'))

# Choose output activation functions
net.pred = L.Sigmoid(net.out, ntop=1, in_place=False)

# Choose a loss function and input data, label and scale inputs. Only include it during the training phase (phase = 0)
net.euclid_loss = L.EuclideanLoss(net.pred, net.label, net.scale, ntop=0, loss_weight=1.0, include=[dict(phase=0)])

# Fix the spatial input dimensions. Note that only spatial dimensions get modified, the minibatch size
# and the channels/feature maps must be set correctly by the user (since this code can definitely not
# figure out the user's intent). If the code does not seem to terminate, then the issue is most likely
# a wrong number of feature maps / channels in either the MemoryData-layers or the network output.

# This function takes as input:
# - The network
# - A list of other inputs to test (note: the nhood input is static and not spatially testable, thus excluded here)
# - A list of the maximal shapes for each input
# - A list of spatial dependencies; here [-1, 0] means the Y axis is a free parameter, and the X axis should be identical to the Y axis.
caffe.fix_input_dims(net,
                    [net.data, net.label, net.scale],
                    max_shapes = [[200,200,200],[100,100,100],[100,100,100],[100,100,100]],
                    shape_coupled = [-1, -1, 1])


protonet = net.to_proto()
protonet.name = 'net';

# Store the network as prototxt
with open(protonet.name + '.prototxt', 'w') as f:
    print(protonet, file=f)
