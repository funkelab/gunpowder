import copy
import logging
import numpy as np

from gunpowder.batch import Batch
from gunpowder.coordinate import Coordinate
from gunpowder.ext import h5py
from gunpowder.points import PointsKeys, Points
from gunpowder.points_spec import PointsSpec
from gunpowder.profiling import Timing
from gunpowder.roi import Roi
from gunpowder.nodes.batch_provider import BatchProvider

from gunpowder.contrib.points import PreSynPoint, PostSynPoint

logger = logging.getLogger(__name__)

class Hdf5PointsSource(BatchProvider):
    '''An HDF5 data source for :class:``Points``. Currently only supports a
    specific case where points represent pre- and post-synaptic markers.

    Args:

        filename (string): The HDF5 file.

        datasets (dict): Dictionary of :class:``PointsKey`` -> dataset names
            that this source offers.

        rois (dict): Dictionary of :class:``PointsKey`` -> :class:``Roi`` to
            set the ROI for each point set provided by this source.

    '''

    def __init__(
            self,
            filename,
            datasets,
            rois):

        self.filename = filename
        self.datasets = datasets
        self.rois = rois

        self.ndims = None

    def setup(self):

        hdf_file = h5py.File(self.filename, 'r')

        for (points_key, ds_name) in self.datasets.items():

            if ds_name not in hdf_file:
                raise RuntimeError("%s not in %s"%(ds_name, self.filename))

            spec = PointsSpec()
            spec.roi = self.rois[points_key]

            self.provides(points_key, spec)

        hdf_file.close()

    def provide(self, request):

        timing = Timing(self)
        timing.start()

        batch = Batch()

        with h5py.File(self.filename, 'r') as hdf_file:

            # if pre and postsynaptic locations required, their id
            # SynapseLocation dictionaries should be created together s.t. ids
            # are unique and allow to find partner locations
            if PointsKeys.PRESYN in request.points_specs or PointsKeys.POSTSYN in request.points_specs:
                assert request.points_specs[PointsKeys.PRESYN].roi == request.points_specs[PointsKeys.POSTSYN].roi
                # Cremi specific, ROI offset corresponds to offset present in the
                # synapse location relative to the raw data.
                dataset_offset = self.spec[PointsKeys.PRESYN].roi.get_offset()
                presyn_points, postsyn_points = self.__get_syn_points(
                    roi=request.points_specs[PointsKeys.PRESYN].roi,
                    syn_file=hdf_file,
                    dataset_offset=dataset_offset)

            for (points_key, request_spec) in request.points_specs.items():

                logger.debug("Reading %s in %s...", points_key, request_spec.roi)
                id_to_point = {
                    PointsKeys.PRESYN: presyn_points,
                    PointsKeys.POSTSYN: postsyn_points}[points_key]

                points_spec = self.spec[points_key].copy()
                points_spec.roi = request_spec.roi
                batch.points[points_key] = Points(data=id_to_point, spec=points_spec)

        logger.debug("done")

        timing.stop()
        batch.profiling_stats.add(timing)

        return batch

    def __get_syn_points(self, roi, syn_file, dataset_offset=None):
        presyn_points_dict, postsyn_points_dict = {}, {}
        presyn_node_ids  = syn_file['annotations/presynaptic_site/partners'][:, 0].tolist()
        postsyn_node_ids = syn_file['annotations/presynaptic_site/partners'][:, 1].tolist()

        for node_nr, node_id in enumerate(syn_file['annotations/ids']):
            location = syn_file['annotations/locations'][node_nr]
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
