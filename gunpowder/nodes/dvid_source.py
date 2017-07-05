import distutils.util
import numpy as np
import logging
import requests

from .batch_provider import BatchProvider
from gunpowder.batch import Batch
from gunpowder.coordinate import Coordinate
from gunpowder.ext import dvision
from gunpowder.points import PointsType, PointsOfType, SynPoint
from gunpowder.profiling import Timing
from gunpowder.provider_spec import ProviderSpec
from gunpowder.roi import Roi
from gunpowder.volume import Volume, VolumeTypes

logger = logging.getLogger(__name__)

class DvidSourceReadException(Exception):
    pass

class MaskNotProvidedException(Exception):
    pass


class DvidSource(BatchProvider):

    def __init__(self, hostname, port, uuid, volume_array_names,
                 points_array_names=None, points_rois=None,
                 resolution=None):
        """
        :param hostname: hostname for DVID server
        :type hostname: str
        :param port: port for DVID server
        :type port: int
        :param uuid: UUID of node on DVID server
        :type uuid: str
        :param volume_array_names: dict {VolumeTypes:  DVID data instance for data in VolumeTypes}
        :param resolution: resolution of source voxels in nanometers
        :type resolution: tuple
        """
        self.hostname = hostname
        self.port = port
        self.url = "http://{}:{}".format(self.hostname, self.port)
        self.uuid = uuid

        self.volume_array_names = volume_array_names

        self.points_array_names = points_array_names
        self.points_rois        = points_rois

        self.specified_resolution = resolution
        self.node_service = None
        self.dims = 0
        self.spec = ProviderSpec()

    def setup(self):
        for volume_type, volume_name in self.volume_array_names.items():
            self.spec.volumes[volume_type] = self.__get_roi(volume_name)

        for points_type, points_name in self.points_array_names.items():
            self.spec.points[points_type] = self.points_rois[points_type]

        logger.info("DvidSource.spec:\n{}".format(self.spec))

    def get_spec(self):
        return self.spec

    @property
    def resolution(self):
        if self.specified_resolution is not None:
            return self.specified_resolution
        else:
            fib25_resolution = (8, 8, 8)
            logger.warning("WARNING: your source does not contain resolution information. "
                           "I will assume {}. "
                           "This might not be what you want.".format(fib25_resolution))
            return fib25_resolution

    def provide(self, request):

        timing = Timing(self)
        timing.start()

        spec = self.get_spec()

        batch = Batch()
        logger.debug("providing batch with resolution of {}".format(self.resolution))

        for (volume_type, roi) in request.volumes.items():
            # check if requested volumetype can be provided
            if volume_type not in spec.volumes:
                raise RuntimeError("Asked for %s which this source does not provide"%volume_type)
            # check if request roi lies within provided roi
            if not spec.volumes[volume_type].contains(roi):
                raise RuntimeError("%s's ROI %s outside of my ROI %s"%(volume_type, roi, spec.volumes[volume_type]))

            read = {
                VolumeTypes.RAW: self.__read_raw,
                VolumeTypes.GT_LABELS: self.__read_gt,
                VolumeTypes.GT_MASK: self.__read_gt_mask,
            }[volume_type]

            logger.debug("Reading %s in %s..."%(volume_type, roi))
            batch.volumes[volume_type] = Volume(
                                                data=read(roi),
                                                roi=roi,
                                                # TODO: get resolution from repository
                                                resolution=self.resolution)

        # if pre and postsynaptic locations requested, their id : SynapseLocation dictionaries should be created
        # together s.t. the ids are unique and allow to find partner locations
        if PointsType.PRESYN in request.points or PointsType.POSTSYN in request.points:
            try:  # either both have the same roi, or only one of them is requested
                assert request.points[PointsType.PRESYN] == request.points[PointsType.POSTSYN]
            except:
                assert PointsType.PRESYN not in request.points or PointsType.POSTSYN not in request.points
            if PointsType.PRESYN in request.points:
                presyn_points, postsyn_points = self.__read_syn_points(roi=request.points[PointsType.PRESYN])
            elif PointsType.POSTSYN in request.points:
                presyn_points, postsyn_points = self.__read_syn_points(roi=request.points[PointsType.POSTSYN])

        for (points_type, roi) in request.points.items():
            # check if requested pointstype can be provided
            if points_type not in spec.points:
                raise RuntimeError("Asked for %s which this source does not provide"%points_type)
            # check if request roi lies within provided roi
            if not spec.points[points_type].contains(roi):
                raise RuntimeError("%s's ROI %s outside of my ROI %s"%(points_type,roi,spec.points[points_type]))

            logger.debug("Reading %s in %s..."%(points_type, roi))
            id_to_point = {PointsType.PRESYN: presyn_points,
                           PointsType.POSTSYN: postsyn_points}[points_type]
            batch.points[points_type] = PointsOfType(data=id_to_point, roi=roi, resolution=self.resolution)

        logger.debug("done")

        timing.stop()
        batch.profiling_stats.add(timing)

        return batch

    def __get_roi(self, array_name):
        data_instance = dvision.DVIDDataInstance(self.hostname, self.port, self.uuid, array_name)
        info = data_instance.info
        roi_min = info['Extended']['MinPoint']
        if roi_min is not None:
            roi_min = Coordinate(roi_min[::-1])
        roi_max = info['Extended']['MaxPoint']
        if roi_max is not None:
            roi_max = Coordinate(roi_max[::-1])

        return Roi(offset=roi_min, shape=roi_max - roi_min)

    def __read_raw(self, roi):
        slices = roi.get_bounding_box()
        data_instance = dvision.DVIDDataInstance(self.hostname, self.port, self.uuid, self.volume_array_names[VolumeTypes.RAW])  # self.raw_array_name)
        try:
            return data_instance[slices]
        except Exception as e:
            print(e)
            msg = "Failure reading raw at slices {} with {}".format(slices, repr(self))
            raise DvidSourceReadException(msg)

    def __read_gt(self, roi):
        slices = roi.get_bounding_box()
        data_instance = dvision.DVIDDataInstance(self.hostname, self.port, self.uuid, self.volume_array_names[VolumeTypes.GT_LABELS])  # self.gt_array_name)
        try:
            return data_instance[slices]
        except Exception as e:
            print(e)
            msg = "Failure reading GT at slices {} with {}".format(slices, repr(self))
            raise DvidSourceReadException(msg)

    def __read_gt_mask(self, roi):
        """
        :param roi: gunpowder.Roi
        :return: uint8 np.ndarray with roi shape
        """
        if self.gt_mask_roi_name is None:
            raise MaskNotProvidedException
        slices = roi.get_bounding_box()
        dvid_roi = dvision.DVIDRegionOfInterest(self.hostname, self.port, self.uuid, self.volume_array_names[VolumeTypes.GT_MASK])  # self.gt_mask_roi_name)
        try:
            return dvid_roi[slices]
        except Exception as e:
            print(e)
            msg = "Failure reading GT mask at slices {} with {}".format(slices, repr(self))
            raise DvidSourceReadException(msg)

    def __load_json_annotations(self, volume_shape, volume_offset, array_name):
        url = "http://" + str(self.hostname) + ":" + str(self.port)+"/api/node/" + str(self.uuid) + '/' + \
              str(array_name) + "/elements/{}_{}_{}/{}_{}_{}".format(volume_shape[2], volume_shape[1], volume_shape[0],
                                                   volume_offset[2], volume_offset[1], volume_offset[0])
        annotations_file = requests.get(url)
        json_annotations = annotations_file.json()
        if json_annotations is None:
            json_annotations = []  # create empty_dummy_json_annotations
            # raise Exception ('No synapses found in region defined by volume_offset {} and volume_shape {}'.format(volume_offset, volume_shape))
        return json_annotations

    def __read_syn_points(self, roi):
        """ read json file from dvid source, in json format to craete a SynPoint for every location given """
        syn_file_json = self.__load_json_annotations(volume_shape=roi.get_shape(),  volume_offset=roi.get_offset(),
                                                     array_name= self.points_array_names[PointsType.PRESYN])

        presyn_points_dict, postsyn_points_dict = {}, {}
        location_to_location_id_dict, location_id_to_partner_locations = {}, {}
        for node_nr, node in enumerate(syn_file_json):
            # collect information
            kind        = str(node['Kind'])
            location    = np.asarray((node['Pos'][2], node['Pos'][1], node['Pos'][0]))
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
                    partner_locations.append(np.asarray([relation['To'][2], relation['To'][1], relation['To'][0]]))
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
            syn_point = SynPoint(kind=kind, location=location, location_id=location_id,
                                    synapse_id=syn_id, partner_ids=[], props=props)
            if kind == 'PreSyn':
                presyn_points_dict[int(node_nr)] = syn_point.get_copy()
            elif kind == 'PostSyn':
                postsyn_points_dict[int(node_nr)] = syn_point.get_copy()

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
        return "DvidSource(hostname={}, port={}, uuid={}, raw_array_name={}, gt_array_name={}".format(
            self.hostname, self.port, self.uuid, self.volume_array_names[VolumeTypes.RAW],
            self.volume_array_names[VolumeTypes.GT_LABELS])
