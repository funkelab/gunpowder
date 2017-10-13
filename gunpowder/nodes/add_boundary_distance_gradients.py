import logging
import numpy as np

from numpy.lib.stride_tricks import as_strided
from scipy.ndimage.morphology import distance_transform_edt
from gunpowder.volume import Volume, VolumeTypes
from .batch_filter import BatchFilter

logger = logging.getLogger(__name__)

class AddBoundaryDistanceGradients(BatchFilter):
    '''Add a volume with vectors pointing away from the closest boundary.

    The vectors are the spacial gradients of the distance transform, i.e., the
    distance to the boundary between labels or the background label (0).

    Args:

        label_volume_type(:class:``VolumeType``): The volume type to read the
            labels from.

        gradient_volume_type(:class:``VolumeType``): The volume type to
            generate containing the gradients.

        distance_volume_type(:class:``VolumeType``, optional): The volume type
            to generate containing the values of the distance transform.

        boundary_volume_type(:class:``VolumeType``, optional): The volume type
            to generate containing a boundary labeling. Note this volume will
            be doubled as it encodes boundaries between voxels.

        normalize(string, optional): ``None``, ``'l1'``, or ``'l2'``. Specifies
            if and how to normalize the gradients.

        scale(string, optional): ``None`` or ``exp``. If ``exp``, distance
            gradients will be scaled by ``beta*e**(-d*alpha)``, where ``d`` is
            the distance to the boundary.

        scale_args(tuple, optional): For ``exp`` a tuple with the values of
            ``alpha`` and ``beta``.
    '''

    def __init__(
            self,
            label_volume_type=None,
            gradient_volume_type=None,
            distance_volume_type=None,
            boundary_volume_type=None,
            normalize=None,
            scale=None,
            scale_args=None):

        if label_volume_type is None:
            label_volume_type = VolumeTypes.GT_LABELS

        self.label_volume_type = label_volume_type
        self.gradient_volume_type = gradient_volume_type
        self.distance_volume_type = distance_volume_type
        self.boundary_volume_type = boundary_volume_type
        self.normalize = normalize
        self.scale = scale
        self.scale_args = scale_args

    def setup(self):

        assert self.label_volume_type in self.spec, (
            "Upstream does not provide %s needed by "
            "AddBoundaryDistanceGradients"%self.label_volume_type)

        spec = self.spec[self.label_volume_type].copy()
        spec.dtype = np.float32
        self.provides(self.gradient_volume_type, spec)
        if self.distance_volume_type is not None:
            self.provides(self.distance_volume_type, spec)
        if self.boundary_volume_type is not None:
            spec.voxel_size /= 2
            self.provides(self.boundary_volume_type, spec)

    def prepare(self, request):

        if self.gradient_volume_type in request:
            del request[self.gradient_volume_type]

        if (
                self.distance_volume_type is not None and
                self.distance_volume_type in request):
            del request[self.distance_volume_type]
        if (
                self.boundary_volume_type is not None and
                self.boundary_volume_type in request):
            del request[self.boundary_volume_type]

    def process(self, batch, request):

        if not self.gradient_volume_type in request:
            return

        labels = batch.volumes[self.label_volume_type].data
        voxel_size = self.spec[self.label_volume_type].voxel_size

        # get boundaries between label regions
        boundaries = self.__find_boundaries(labels)

        # mark boundaries with 0 (not 1)
        boundaries = 1.0 - boundaries

        if np.sum(boundaries == 0) == 0:

            # no boundary -- no distance to compute
            distances = np.zeros(labels.shape, dtype=np.float32)

        else:

            # get distances (voxel_size/2 because image is doubled)
            distances = distance_transform_edt(
                boundaries,
                sampling=tuple(float(v)/2 for v in voxel_size))
            distances = distances.astype(np.float32)

            # restore original shape
            downsample = (slice(None, None, 2),)*len(voxel_size)
            distances = distances[downsample]

            # set distances in background to 0
            distances[labels == 0] = 0

        gradients = np.asarray(np.gradient(distances, *voxel_size))

        # set gradients on background voxels to 0
        for d in range(len(voxel_size)):
            gradients[d, labels == 0] = 0

        if self.normalize is not None:
            self.__normalize(gradients, self.normalize)

        if self.scale is not None:
            self.__scale(gradients, distances, self.scale, self.scale_args)

        spec = self.spec[self.gradient_volume_type].copy()
        spec.roi = request[self.gradient_volume_type].roi
        batch.volumes[self.gradient_volume_type] = Volume(gradients, spec)

        if (
                self.distance_volume_type is not None and
                self.distance_volume_type in request):
            batch.volumes[self.distance_volume_type] = Volume(distances, spec)

        if (
                self.boundary_volume_type is not None and
                self.boundary_volume_type in request):

            # add one more face at each dimension, as boundary map has shape
            # 2*s - 1 of original shape s
            grown = np.ones(tuple(s + 1 for s in boundaries.shape))
            grown[tuple(slice(0, s) for s in boundaries.shape)] = boundaries
            spec.voxel_size = voxel_size/2
            logger.debug("voxel size of boundary volume: %s", spec.voxel_size)
            batch.volumes[self.boundary_volume_type] = Volume(grown, spec)

    def __find_boundaries(self, labels):

        # labels: 1 1 1 1 0 0 2 2 2 2 3 3       n
        # shift :   1 1 1 1 0 0 2 2 2 2 3       n - 1
        # diff  :   0 0 0 1 0 1 0 0 0 1 0       n - 1
        # bound.: 00000001000100000001000      2n - 1

        logger.debug("computing boundaries for %s", labels.shape)

        dims = len(labels.shape)
        in_shape = labels.shape
        out_shape = tuple(2*s - 1 for s in in_shape)
        out_slices = tuple(slice(0, s) for s in out_shape)

        boundaries = np.zeros(out_shape, dtype=np.bool)

        logger.debug("boundaries shape is %s", boundaries.shape)

        for d in range(dims):

            logger.debug("processing dimension %d", d)

            shift_p = [slice(None)]*dims
            shift_p[d] = slice(1, in_shape[d])

            shift_n = [slice(None)]*dims
            shift_n[d] = slice(0, in_shape[d] - 1)

            diff = (labels[shift_p] - labels[shift_n]) != 0

            logger.debug("diff shape is %s", diff.shape)

            target = [slice(None, None, 2)]*dims
            target[d] = slice(1, out_shape[d], 2)

            logger.debug("target slices are %s", target)

            boundaries[target] = diff

        return boundaries

    def __normalize(self, gradients, norm):

        dims = gradients.shape[0]

        if norm == 'l1':
            factors = sum([np.abs(gradients[d]) for d in range(dims)])
        elif norm == 'l2':
            factors = np.sqrt(
                    sum([np.square(gradients[d]) for d in range(dims)]))
        else:
            raise RuntimeError('norm %s not supported'%norm)

        factors[factors < 1e-5] = 1
        gradients /= factors

    def __scale(self, gradients, distances, scale, scale_args):

        dims = gradients.shape[0]

        if scale == 'exp':
            alpha, beta = self.scale_args
            factors = np.exp(-distances*alpha)*beta

        gradients *= factors
