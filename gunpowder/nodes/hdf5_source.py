import copy
import logging
import numpy as np

from gunpowder.batch import Batch
from gunpowder.coordinate import Coordinate
from gunpowder.ext import h5py
from gunpowder.points import PointsTypes, Points, PreSynPoint, PostSynPoint
from gunpowder.points_spec import PointsSpec
from gunpowder.profiling import Timing
from gunpowder.roi import Roi
from gunpowder.volume import Volume, VolumeTypes
from gunpowder.volume_spec import VolumeSpec
from .batch_provider import BatchProvider

logger = logging.getLogger(__name__)

class Hdf5Source(BatchProvider):
    '''An HDF5 data source.

    Provides volumes from HDF5 datasets for each volume type given. If the 
    attribute `resolution` is set in an HDF5 dataset, it will be used as the 
    volume's `voxel_size` and a warning issued if they differ. If the attribute 
    `offset` is set in an HDF5 dataset, it will be used as the offset of the 
    :class:`Roi` for this volume. It is assumed that the offset is given in 
    world units.

    Args:

        filename (string): The HDF5 file.

        datasets (dict): Dictionary of VolumeType -> dataset names that this source offers.

        volume_specs (dict, optional): An optional dictionary of 
            :class:`VolumeType` to :class:`VolumeSpec` to overwrite the volume 
            specs automatically determined from the HDF5 file. This is useful to 
            set a missing ``voxel_size``, for example. Only fields that are not 
            ``None`` in the given :class:`VolumeSpec` will be used.
    '''

    def __init__(
            self,
            filename,
            datasets,
            volume_specs=None,
            points_types=None,
            points_rois=None):

        self.filename = filename
        self.datasets = datasets

        if volume_specs is None:
            self.volume_specs = {}
        else:
            self.volume_specs = volume_specs

        self.points_types = points_types
        self.points_rois = points_rois

        self.ndims = None

    def setup(self):

        hdf_file = h5py.File(self.filename, 'r')

        for (volume_type, ds_name) in self.datasets.items():

            if ds_name not in hdf_file:
                raise RuntimeError("%s not in %s"%(ds_name, self.filename))

            spec = self.__read_spec(volume_type, hdf_file, ds_name)

            self.provides(volume_type, spec)

        if self.points_types is not None:

            for points_type in self.points_types:
                spec = PointsSpec()
                spec.roi = Roi(self.points_rois[points_type])

                self.provides(points_type, spec)

        hdf_file.close()

    def provide(self, request):

        timing = Timing(self)
        timing.start()

        batch = Batch()

        with h5py.File(self.filename, 'r') as hdf_file:

            for (volume_type, request_spec) in request.volume_specs.items():

                logger.debug("Reading %s in %s...", volume_type, request_spec.roi)

                voxel_size = self.spec[volume_type].voxel_size

                # scale request roi to voxel units
                dataset_roi = request_spec.roi/voxel_size

                # shift request roi into dataset
                dataset_roi = dataset_roi - self.spec[volume_type].roi.get_offset()/voxel_size

                # create volume spec
                volume_spec = self.spec[volume_type].copy()
                volume_spec.roi = request_spec.roi

                # add volume to batch
                batch.volumes[volume_type] = Volume(
                    self.__read(hdf_file, self.datasets[volume_type], dataset_roi),
                    volume_spec)

            # if pre and postsynaptic locations required, their id
            # SynapseLocation dictionaries should be created together s.t. ids
            # are unique and allow to find partner locations
            if PointsTypes.PRESYN in request.points_specs or PointsTypes.POSTSYN in request.points_specs:
                assert request.points_specs[PointsTypes.PRESYN].roi == request.points_specs[PointsTypes.POSTSYN].roi
                # Cremi specific, ROI offset corresponds to offset present in the
                # synapse location relative to the raw data.
                dataset_offset = self.spec[PointsTypes.PRESYN].roi.get_offset()
                presyn_points, postsyn_points = self.__get_syn_points(
                    roi=request.points_specs[PointsTypes.PRESYN].roi,
                    syn_file=hdf_file,
                    dataset_offset=dataset_offset)

            for (points_type, request_spec) in request.points_specs.items():

                logger.debug("Reading %s in %s...", points_type, request_spec.roi)
                id_to_point = {
                    PointsTypes.PRESYN: presyn_points,
                    PointsTypes.POSTSYN: postsyn_points}[points_type]
                # TODO: so far assumed that all points have resolution of raw volume

                points_spec = self.spec[points_type].copy()
                points_spec.roi = request_spec.roi
                batch.points[points_type] = Points(data=id_to_point, spec=points_spec)

        logger.debug("done")

        timing.stop()
        batch.profiling_stats.add(timing)

        return batch

    def __read_spec(self, volume_type, hdf_file, ds_name):

        dataset = hdf_file[ds_name]

        dims = Coordinate(dataset.shape)

        if self.ndims is None:
            self.ndims = len(dims)
        else:
            assert self.ndims == len(dims)

        if volume_type in self.volume_specs:
            spec = self.volume_specs[volume_type].copy()
        else:
            spec = VolumeSpec()

        if spec.voxel_size is None:

            if 'resolution' in dataset.attrs:
                spec.voxel_size = Coordinate(dataset.attrs['resolution'])
            else:
                spec.voxel_size = Coordinate((1, 1, 1))
                logger.warning("WARNING: File %s does not contain resolution information "
                               "for %s (dataset %s), voxel size has been set to (1,1,1). "
                               "This might not be what you want.",
                               self.filename, volume_type, ds_name)

        if spec.roi is None:

            if 'offset' in dataset.attrs:
                offset = Coordinate(dataset.attrs['offset'])
            else:
                offset = Coordinate((0,)*self.ndims)

            spec.roi = Roi(offset, dims*spec.voxel_size)

        if spec.dtype is not None:
            assert spec.dtype == dataset.dtype, ("dtype %s provided in volume_specs for %s, "
                                                 "but differs from dataset %s dtype %s"%
                                                 (self.volume_specs[volume_type].dtype,
                                                  volume_type, ds_name, dataset.dtype))
        else:
            spec.dtype = dataset.dtype

        if spec.interpolatable is None:

            spec.interpolatable = spec.dtype in [
                np.float,
                np.float32,
                np.float64,
                np.float128,
                np.uint8 # assuming this is not used for labels
            ]
            logger.warning("WARNING: You didn't set 'interpolatable' for %s "
                           "(dataset %s). Based on the dtype %s, it has been "
                           "set to %s. This might not be what you want.",
                           volume_type, ds_name, spec.dtype,
                           spec.interpolatable)

        return spec

    def __read(self, hdf_file, ds_name, roi):
        return np.array(hdf_file[ds_name][roi.get_bounding_box()])

    def __get_syn_points(self, roi, syn_file, dataset_offset=None):
        presyn_points_dict, postsyn_points_dict = {}, {}
        presyn_node_ids  = syn_file['annotations/presynaptic_site/partners'][:, 0].tolist()
        postsyn_node_ids = syn_file['annotations/presynaptic_site/partners'][:, 1].tolist()

        for node_nr, node_id in enumerate(syn_file['annotations/ids']):
            location     = syn_file['annotations/locations'][node_nr]
            location /= self.resolutions[VolumeTypes.RAW]
            if dataset_offset is not None:
                logging.debug('adding global offset to points %i %i %i' %(dataset_offset[0],
                                                                          dataset_offset[1], dataset_offset[2]))
                location += dataset_offset


            # cremi synapse locations are in physical space
            if roi.contains(Coordinate(location)):
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
                if kind == 'PreSyn':
                    syn_point = PreSynPoint(location=location, location_id=location_id,
                                         synapse_id=syn_id, partner_ids=partners_ids, props=props)
                    presyn_points_dict[int(node_id)] = copy.deepcopy(syn_point)
                elif kind == 'PostSyn':
                    syn_point = PostSynPoint(location=location, location_id=location_id,
                                         synapse_id=syn_id, partner_ids=partners_ids, props=props)
                    postsyn_points_dict[int(node_id)] = copy.deepcopy(syn_point)

        return presyn_points_dict, postsyn_points_dict


    def __repr__(self):

        return self.filename
