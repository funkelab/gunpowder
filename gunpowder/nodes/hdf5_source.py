import logging
import numpy as np
from copy import deepcopy

from .batch_provider import BatchProvider
from gunpowder.batch import Batch
from gunpowder.coordinate import Coordinate
from gunpowder.ext import h5py
from gunpowder.profiling import Timing
from gunpowder.points import PointsTypes, Points, PreSynPoint, PostSynPoint
from gunpowder.provider_spec import ProviderSpec
from gunpowder.roi import Roi
from gunpowder.volume import Volume, VolumeTypes

logger = logging.getLogger(__name__)

class Hdf5Source(BatchProvider):
    '''An HDF5 data source.

    Provides volumes from HDF5 datasets for each volume type given. If the 
    attribute ``resolution`` is set in an HDF5 dataset, it will be compared 
    agains the volume's ``voxel_size`` and a warning issued if they differ. If 
    the attribute ``offset`` is set in an HDF5 dataset, it will be used as the 
    offset of the :class:`Roi` provided by this node. It is assumed that the 
    offset is given in world units.

    Args:

        filename (string): The HDF5 file.

        datasets (dict): Dictionary of VolumeType -> dataset names that this source offers.
    '''

    def __init__(
            self,
            filename,
            datasets,
            points_types=None,
            points_rois=None):

        self.filename = filename
        self.datasets = datasets

        self.points_types      = points_types
        self.points_rois = points_rois

    def setup(self):

        f = h5py.File(self.filename, 'r')

        self.spec = ProviderSpec()
        self.ndims = None
        for (volume_type, ds) in self.datasets.items():

            if ds not in f:
                raise RuntimeError("%s not in %s"%(ds,self.filename))

            dims = f[ds].shape

            if self.ndims is None:
                self.ndims = len(dims)
            else:
                assert self.ndims == len(dims)

            if 'resolution' in f[ds].attrs:
                voxel_size = Coordinate(f[ds].attrs['resolution'])
                if voxel_size != volume_type.voxel_size:
                    logger.warning(
                            "WARNING: Your source contains a resolution information of %s, "
                            "but %s was set globally for %s"
                            %(voxel_size, volume_type.voxel_size, volume_type))

            if 'offset' in f[ds].attrs:
                offset = Coordinate(f[ds].attrs['offset'])
            else:
                offset = Coordinate((0,)*self.ndims)

            self.spec.volumes[volume_type] = Roi(offset, dims*volume_type.voxel_size)

        if self.points_types is not None:
            for points_type in self.points_types:
                self.spec.points[points_type] = self.points_rois[points_type]

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

                roi_shape = roi.get_shape()
                voxel_size = volume_type.voxel_size

                for d in range(len(roi.dims())):
                    assert roi_shape[d]%voxel_size[d] == 0, \
                            "in request %s, dimension %d of request %s is not a multiple of voxel_size %d"%(
                                    request,
                                    d,
                                    volume_type,
                                    voxel_size[d])

                logger.debug("Reading %s in %s..."%(volume_type,roi))

                # scale request roi to voxel units
                dataset_roi = roi/voxel_size

                # shift request roi into dataset
                dataset_roi = dataset_roi - spec.volumes[volume_type].get_offset()/voxel_size

                batch.volumes[volume_type] = Volume(
                        self.__read(f, self.datasets[volume_type], dataset_roi),
                        roi=roi)

            # if pre and postsynaptic locations required, their id : SynapseLocation dictionaries should be created
            # together s.t. ids are unique and allow to find partner locations
            if PointsTypes.PRESYN in request.points or PointsTypes.POSTSYN in request.points:
                assert request.points[PointsTypes.PRESYN] == request.points[PointsTypes.POSTSYN]
                # Cremi specific, ROI offset corresponds to offset present in the
                # synapse location relative to the raw data.
                dataset_offset = self.get_spec().points[PointsTypes.PRESYN].get_offset()
                presyn_points, postsyn_points = self.__get_syn_points(roi=request.points[PointsTypes.PRESYN],
                                                                      syn_file=f,
                                                                      dataset_offset=dataset_offset)

            for (points_type, roi) in request.points.items():

                if points_type not in spec.points:
                    raise RuntimeError("Asked for %s which this source does not provide"%points_type)

                if not spec.points[points_type].contains(roi):
                    raise RuntimeError("%s's ROI %s outside of my ROI %s"%(points_type,roi,spec.points[points_type]))

                logger.debug("Reading %s in %s..." % (points_type, roi))
                id_to_point = {PointsTypes.PRESYN: presyn_points, PointsTypes.POSTSYN: postsyn_points}[points_type]
                # TODO: so far assumed that all points have resolution of raw volume
                batch.points[points_type] = Points(data=id_to_point, roi=roi, resolution=self.resolutions[VolumeTypes.RAW])

        logger.debug("done")

        timing.stop()
        batch.profiling_stats.add(timing)

        return batch

    def __read(self, f, ds, roi):
        return np.array(f[ds][roi.get_bounding_box()])

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
                    presyn_points_dict[int(node_id)] = deepcopy(syn_point)
                elif kind == 'PostSyn':
                    syn_point = PostSynPoint(location=location, location_id=location_id,
                                         synapse_id=syn_id, partner_ids=partners_ids, props=props)
                    postsyn_points_dict[int(node_id)] = deepcopy(syn_point)

        return presyn_points_dict, postsyn_points_dict


    def __repr__(self):

        return self.filename
