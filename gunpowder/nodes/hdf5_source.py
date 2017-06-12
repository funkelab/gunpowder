import logging
import numpy as np

from .batch_provider import BatchProvider
from gunpowder.batch import Batch
from gunpowder.coordinate import Coordinate
from gunpowder.ext import h5py
from gunpowder.profiling import Timing
from gunpowder.points import PointsType, PointsOfType, SynPoint
from gunpowder.provider_spec import ProviderSpec
from gunpowder.roi import Roi
from gunpowder.volume import Volume, VolumeType

logger = logging.getLogger(__name__)

class Hdf5Source(BatchProvider):

    def __init__(
            self,
            filename,
            datasets,
            points_types=None,
            points_roi_shape=None,
            points_roi_offset=None,
            resolution=None):
        '''Create a new Hdf5Source

        Args

            filename: The HDF5 file.

            datasets: Dictionary of VolumeType -> dataset names that this source offers.

            resolution: tuple, to overwrite the resolution stored in the HDF5 datasets.
        '''

        self.filename = filename
        self.datasets = datasets
        # self.point_types_to_rel_dataset = point_types_to_rel_dataset

        self.points_types      = points_types
        self.points_roi_shape  = points_roi_shape
        self.points_roi_offset = points_roi_offset

        self.specified_resolution = resolution
        self.resolutions = {}

    def setup(self):

        f = h5py.File(self.filename, 'r')

        self.spec = ProviderSpec()
        self.ndims = None
        for (volume_type, ds) in self.datasets.items():

            if ds not in f:
                raise RuntimeError("%s not in %s"%(ds,self.filename))

            dims = f[ds].shape
            # TODO: different volumes can have different offsets, needs to be added.
            self.spec.volumes[volume_type] = Roi((0,)*len(dims), dims)

            if self.ndims is None:
                self.ndims = len(dims)
            else:
                assert self.ndims == len(dims)

            if self.specified_resolution is None:
                if 'resolution' in f[ds].attrs:
                    self.resolutions[volume_type] = tuple(f[ds].attrs['resolution'])
                else:
                    default_resolution = (1,)*self.ndims
                    logger.warning("WARNING: your source does not contain resolution information"
                                   " (no attribute 'resolution' in {} dataset). I will assume {}. "
                                   "This might not be what you want.".format(ds,default_resolution))
                    self.resolutions[volume_type] = default_resolution
            else:
                self.resolutions[volume_type] = self.specified_resolution


        if self.points_types is not None:
            for points_type in self.points_types:
                self.spec.points[points_type] = Roi(offset=self.points_roi_offset, shape=self.points_roi_shape)

        f.close()

    def get_spec(self):
        return self.spec

    def provide(self, request):

        timing = Timing(self)
        timing.start()

        spec = self.get_spec()

        batch = Batch()

        with h5py.File(self.filename, 'r') as f:

            for (volume_type, roi) in request.volumes.items():

                if volume_type not in spec.volumes:
                    raise RuntimeError("Asked for %s which this source does not provide"%volume_type)

                if not spec.volumes[volume_type].contains(roi):
                    raise RuntimeError("%s's ROI %s outside of my ROI %s"%(volume_type,roi,spec.volumes[volume_type]))

                interpolate = {
                    VolumeType.RAW: True,
                    VolumeType.GT_LABELS: False,
                    VolumeType.GT_MASK: False,
                    VolumeType.ALPHA_MASK: True,
                }[volume_type]

                logger.debug("Reading %s in %s..."%(volume_type,roi))
                batch.volumes[volume_type] = Volume(
                        self.__read(f, self.datasets[volume_type], roi),
                        roi=roi,
                        resolution=self.resolutions[volume_type],
                        interpolate=interpolate)

            # if pre and postsynaptic locations required, their id : SynapseLocation dictionaries should be created
            # together s.t. ids are unique and allow to find partner locations
            if PointsType.PRESYN in request.points or PointsType.POSTSYN in request.points:
                assert request.points[PointsType.PRESYN] == request.points[PointsType.POSTSYN]
                presyn_points, postsyn_points = self.__get_syn_points(roi=request.points[PointsType.PRESYN],
                                                                      syn_file=f)

            for (points_type, roi) in request.points.items():

                if points_type not in spec.points:
                    raise RuntimeError("Asked for %s which this source does not provide"%points_type)

                if not spec.points[points_type].contains(roi):
                    raise RuntimeError("%s's ROI %s outside of my ROI %s"%(points_type,roi,spec.points[points_type]))

                logger.debug("Reading %s in %s..." % (points_type, roi))
                id_to_point = {PointsType.PRESYN: presyn_points, PointsType.POSTSYN: postsyn_points}[points_type]
                # TODO: so far assumed that all points have resolution of raw volume
                batch.points[points_type] = PointsOfType(data=id_to_point, roi=roi, resolution=self.resolutions[VolumeType.RAW])

        logger.debug("done")

        timing.stop()
        batch.profiling_stats.add(timing)

        return batch

    def __read(self, f, ds, roi):
        return np.array(f[ds][roi.get_bounding_box()])


    def __is_inside_bb(self, location, bb_shape, bb_offset, margin=0):
        try:
            assert len(margin) == len(bb_shape)
        except:
            margin = (margin,)*len(bb_shape)

        inside_bb = True
        location  = np.asarray(location) - np.asarray(bb_offset)
        for dim, size in enumerate(bb_shape):
            if location[dim] < margin[dim]:
                inside_bb = False
            if location[dim] >= size - margin[dim]:
                inside_bb = False
        return inside_bb


    def __get_syn_points(self, roi, syn_file):
        bb_shape, bb_offset  = roi.get_shape(), roi.get_offset()
        presyn_points_dict, postsyn_points_dict = {}, {}
        presyn_node_ids  = syn_file['annotations/presynaptic_site/partners'][:, 0].tolist()
        postsyn_node_ids = syn_file['annotations/presynaptic_site/partners'][:, 1].tolist()

        for node_nr, node_id in enumerate(syn_file['annotations/ids']):

            location     = syn_file['annotations/locations'][node_nr]
            location /= self.resolutions[VolumeType.RAW]
            location += self.points_roi_offset

            # cremi synapse locations are in physical space
            if self.__is_inside_bb(location=location, bb_shape=bb_shape, bb_offset=bb_offset, margin=0):
                location -= bb_offset
                if node_id in presyn_node_ids:
                    kind = 'PreSyn'
                    assert syn_file['annotations/types'][node_nr] == 'presynaptic_site'
                    syn_id = int(np.where(presyn_node_ids == node_id)[0])
                    partner_node_id = postsyn_node_ids[syn_id]
                elif node_id in postsyn_node_ids:
                    kind = 'PostSyn'
                    assert syn_file['annotations/types'][node_nr] == 'postsynaptic_site'
                    syn_id = int(np.where(postsyn_node_ids == node_id)[0])
                    partner_node_id = presyn_node_ids[syn_id]
                else:
                    raise Exception('Node id neither pre- no post-synaptic')

                partners_ids = [int(partner_node_id)]
                location_id  = int(node_id)

                props = {}
                if node_id in syn_file['annotations/comments/target_ids']:
                    props = {'unsure': True}

                # create synpaseLocation & add to dict
                syn_point = SynPoint(kind=kind, location=location, location_id=location_id,
                                     synapse_id=syn_id, partner_ids=partners_ids, props=props)
                if kind == 'PreSyn':
                    presyn_points_dict[int(node_id)] = syn_point.get_copy()
                elif kind == 'PostSyn':
                    postsyn_points_dict[int(node_id)] = syn_point.get_copy()

        return presyn_points_dict, postsyn_points_dict


    def __repr__(self):

        return self.filename
