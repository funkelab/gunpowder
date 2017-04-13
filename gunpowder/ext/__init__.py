# Import Caffe
caffe_path = '../../caffe_gt'
import os
import sys
import inspect
thispath = os.path.normpath(os.path.realpath(os.path.abspath(os.path.split(inspect.getfile(inspect.currentframe()))[0])))
thispath = os.path.dirname(thispath)
pycaffepath = ''
if (os.path.isabs(caffe_path)):
    pycaffepath = caffe_path + '/python'
else:
    pycaffepath = thispath + '/' + caffe_path + '/python'

sys.path.append(pycaffepath)

import caffe as caffe
