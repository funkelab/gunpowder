import distutils.util
import numpy as np
import logging
import requests
from copy import deepcopy

from gunpowder.batch import Batch
from gunpowder.coordinate import Coordinate
from gunpowder.ext import dvision
from gunpowder.nodes.batch_provider import BatchProvider
from gunpowder.points import PointsKeys, Points
from gunpowder.profiling import Timing
from gunpowder.roi import Roi

from gunpowder.contrib.points import PreSynPoint, PostSynPoint

logger = logging.getLogger(__name__)

class DvidPartnerAnnoationSourceReadException(Exception):
    pass

class MaskNotProvidedException(Exception):
    pass


# TODO: This seems broken. There is code involving a voxel size, but points
# don't have voxel sizes
class DvidPartnerAnnotationSource(BatchProvider):
    '''
        :param hostname: hostname for DVID server
        :type hostname: str

        :param port: port for DVID server
        :type port: int

        :param uuid: UUID of node on DVID server
        :type uuid: str

        :param datasets: dict {PointsKey: DVID data instance}
    '''

    def __init__(
            self,
            hostname,
            port,
            uuid,
            datasets=None,
            rois=None):

        self.hostname = hostname
        self.port = port
        self.url = "http://{}:{}".format(self.hostname, self.port)
        self.uuid = uuid

        self.datasets = datasets if datasets is not None else {}
        self.rois = rois if rois is not None else {}

        self.node_service = None
        self.dims = 0

    def setup(self):

        for points_key, points_name in self.datasets.items():
            self.provides(
                points_key,
                PointsSpec(roi=self.points_rois[points_key]))

        logger.info("DvidPartnerAnnotationSource.spec:\n{}".format(self.spec))

    def provide(self, request):

        timing = Timing(self)
        timing.start()

        batch = Batch()

        # if pre and postsynaptic locations requested, their id : SynapseLocation dictionaries should be created
        # together s.t. the ids are unique and allow to find partner locations
        if PointsKeys.PRESYN in request.points or PointsKeys.POSTSYN in request.points:
            try:  # either both have the same roi, or only one of them is requested
                assert request.points[PointsKeys.PRESYN] == request.points[PointsKeys.POSTSYN]
            except:
                assert PointsKeys.PRESYN not in request.points or PointsKeys.POSTSYN not in request.points
            if PointsKeys.PRESYN in request.points:
                presyn_points, postsyn_points = self.__read_syn_points(roi=request.points[PointsKeys.PRESYN])
            elif PointsKeys.POSTSYN in request.points:
                presyn_points, postsyn_points = self.__read_syn_points(roi=request.points[PointsKeys.POSTSYN])

        for (points_key, roi) in request.points.items():
            # check if requested points can be provided
            if points_key not in self.spec:
                raise RuntimeError("Asked for %s which this source does not provide"%points_key)
            # check if request roi lies within provided roi
            if not self.spec[points_key].roi.contains(roi):
                raise RuntimeError("%s's ROI %s outside of my ROI %s"%(points_key,roi,self.spec[points_key].roi))

            logger.debug("Reading %s in %s..."%(points_key, roi))
            id_to_point = {PointsKeys.PRESYN: presyn_points,
                           PointsKeys.POSTSYN: postsyn_points}[points_key]

            batch.points[points_key] = Points(
                data=id_to_point,
                spec=PointsSpec(roi=roi))

        logger.debug("done")

        timing.stop()
        batch.profiling_stats.add(timing)

        return batch

    def __load_json_annotations(self, array_shape_voxel, array_offset_voxel, array_name):
        url = "http://" + str(self.hostname) + ":" + str(self.port)+"/api/node/" + str(self.uuid) + '/' + \
              str(array_name) + "/elements/{}_{}_{}/{}_{}_{}".format(array_shape_voxel[2], array_shape_voxel[1], array_shape_voxel[0],
                                                   array_offset_voxel[2], array_offset_voxel[1], array_offset_voxel[0])
        annotations_file = requests.get(url)
        json_annotations = annotations_file.json()
        if json_annotations is None:
            json_annotations = []  # create empty_dummy_json_annotations
            # raise Exception ('No synapses found in region defined by array_offset {} and array_shape {}'.format(array_offset, array_shape))
        return json_annotations

    def __read_syn_points(self, roi):
        """ read json file from dvid source, in json format to create a PreSynPoint/PostSynPoint for every location given """

        if PointsKeys.PRESYN in self.points_voxel_size:
            voxel_size = self.points_voxel_size[PointsKeys.PRESYN]
        elif PointsKeys.POSTSYN in self.points_voxel_size:
            voxel_size = self.points_voxel_size[PointsKeys.POSTSYN]

        syn_file_json = self.__load_json_annotations(array_shape_voxel  = roi.get_shape() // voxel_size,
                                                     array_offset_voxel = roi.get_offset() // voxel_size,
                                                     array_name    = self.datasets[PointsKeys.PRESYN])

        presyn_points_dict, postsyn_points_dict = {}, {}
        location_to_location_id_dict, location_id_to_partner_locations = {}, {}
        for node_nr, node in enumerate(syn_file_json):
            # collect information
            kind        = str(node['Kind'])
            location    = np.asarray((node['Pos'][2], node['Pos'][1], node['Pos'][0])) * voxel_size
            location_id = int(node_nr)
            # some synapses are wrongly annotated in dvid source, have 'Tag': null ???, they are skipped
            try:
                syn_id = int(node['Tags'][0][3:])
            except:
                continue
            location_to_location_id_dict[str(location)] = location_id

            partner_locations = []
            try:
                for relation in node['Rels']:
                    partner_locations.append((np.asarray([relation['To'][2], relation['To'][1], relation['To'][0]]))*voxel_size)
            except:
                partner_locations = []
            location_id_to_partner_locations[int(node_nr)] = partner_locations

            # check if property given, not always given
            props = {}
            if 'conf' in node['Prop']:
                props['conf'] = float(node['Prop']['conf'])
            if 'agent' in node['Prop']:
                props['agent']  = str(node['Prop']['agent'])
            if 'flagged' in node['Prop']:
                str_value_flagged = str(node['Prop']['flagged'])
                props['flagged']  = bool(distutils.util.strtobool(str_value_flagged))
            if 'multi' in node['Prop']:
                str_value_multi = str(node['Prop']['multi'])
                props['multi']  = bool(distutils.util.strtobool(str_value_multi))

            # create synPoint with information collected so far (partner_ids not completed yet)
            if kind == 'PreSyn':
                syn_point = PreSynPoint(location=location, location_id=location_id,
                                     synapse_id=syn_id, partner_ids=[], props=props)
                presyn_points_dict[int(node_nr)] = deepcopy(syn_point)
            elif kind == 'PostSyn':
                syn_point = PostSynPoint(location=location, location_id=location_id,
                                     synapse_id=syn_id, partner_ids=[], props=props)
                postsyn_points_dict[int(node_nr)] = deepcopy(syn_point)

        # add partner ids
        last_node_nr = len(syn_file_json)-1
        for current_syn_point_id in location_id_to_partner_locations.keys():
            all_partner_ids = []
            for partner_loc in location_id_to_partner_locations[current_syn_point_id]:
                if location_to_location_id_dict.has_key(str(partner_loc)):
                    all_partner_ids.append(int(location_to_location_id_dict[str(partner_loc)]))
                else:
                    last_node_nr = last_node_nr + 1
                    assert not location_to_location_id_dict.has_key(str(partner_loc))
                    all_partner_ids.append(int(last_node_nr))

            if current_syn_point_id in presyn_points_dict:
                presyn_points_dict[current_syn_point_id].partner_ids = all_partner_ids
            elif current_syn_point_id in postsyn_points_dict:
                postsyn_points_dict[current_syn_point_id].partner_ids = all_partner_ids
            else:
                raise Exception("current syn_point id not found in any dictionary")

        return presyn_points_dict, postsyn_points_dict


    def __repr__(self):
        return "DvidPartnerAnnoationSource(hostname={}, port={}, uuid={}, raw_array_name={}, gt_array_name={}".format(
            self.hostname, self.port, self.uuid, self.array_names[ArrayKeys.RAW],
            self.array_names[ArrayKeys.GT_LABELS])
