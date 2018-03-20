import copy
import logging
import numpy as np
from scipy.spatial import KDTree

from gunpowder.nodes.batch_filter import BatchFilter
from gunpowder.array import Array
from gunpowder.array_spec import ArraySpec
from gunpowder.coordinate import Coordinate
from gunpowder.morphology import enlarge_binary_map

logger = logging.getLogger(__name__)

class AddVectorMap(BatchFilter):
    def __init__(self, src_and_trg_points, voxel_sizes, radius_phys,
            partner_criterion, stayinside_array_keys=None, pad_for_partners=(0,0,0)):
        ''' Creates a vector map of shape [dim_vector, [shape_of_array]] (e.g. [3, 50,50,50] for an array of
            shape (50,50,50)) where every voxel which is close to a any source point location has a vector which points to
            one of the source point location's target location.
            Close to a point location in src_point includes all voxels which 
                1) lie within distance radius_phys of the considered src point location
                2) (if stayinside_array_keys is not None), lie within the same segment as the src location in the
                    mask provided in stayinside_array_keys.
            The partner_criterion decides to which target location of the source point location that the vector of a
            voxel points (the different criterions are described below).
        
        Args:
            src_and_trg_points (dict):      Dictionary from :class:``ArrayKey`` of the vector map to be created
                                            to a tuple (:class:``PointsKeys`` of the source points, :class:``PointsKeys``
                                            of the target points) which define the source and target points.
            voxel_sizes (dict):             Dictionary from
                                            :class:``ArrayKey`` of the vector
                                            map to be created to a
                                            :class:`Coordinate` for the voxel
                                            size of the array.
            stayinside_array_keys (dict):  Dictionary from :class:``ArrayKey`` of the vector map to be created to 
                                            :class:``ArrayKey`` of the stayinside_array. 
                                            The stayinside_array is assumed to contain discrete objects labeled with
                                            different object ids. The object id at the specific source location is used
                                            to restrict the region where vectors are created around a source location. 
                                            Voxels that are located outside of this object are set to zero.
                                            If stayinside_array_keys is None, all the voxels within distance 
                                            radius_phys to the source location receive a vector.
            pad_for_partners (tuple):       n-dim tuple which defines padding of trg_points request in all dimensions
                                            (in world units).
                                            This might be used s.t. also partner locations which lie within the padded 
                                            region, hence slightly outside of the vector map's roi, are considered.
            radius_phys (int):              Radius (in world units) to restrict region where vectors are created around
                                            a source location.
            partner_criterion(str):         'min_distance' or 'all'
                                            'min_distance': the vectors of all the voxels around a source location
                                            point to the same target location, namely the location which has the 
                                            minimal distance to the considered source location.
                                            'all': all partner locations of a given source location are considered.
                                            The region around a source location is split up into (num_partners)-subregions
                                            where each voxel points to the target location for which this subregion
                                            is the closest.
        '''

        self.array_to_src_trg_points               = src_and_trg_points
        self.voxel_sizes                            = voxel_sizes
        self.array_keys_to_stayinside_array_keys = stayinside_array_keys
        self.pad_for_partners                       = pad_for_partners
        self.radius_phys                            = radius_phys
        self.partner_criterion                      = partner_criterion

    def setup(self):

        for (array_key, (src_points_key, trg_points_key)) in self.array_to_src_trg_points.items():
            for points_key in [src_points_key, trg_points_key]:
                assert points_key in self.spec, "Asked for {} in AddVectorMap from {}, where {} is not provided.".\
                                                                format(array_key, points_key, points_key)
            neg_pad_for_partners = Coordinate((self.pad_for_partners*np.asarray([-1])).tolist())
            self.provides(array_key, ArraySpec(
                roi=self.spec[src_points_key].roi.grow(
                    neg_pad_for_partners,
                    neg_pad_for_partners),
                voxel_size=self.voxel_sizes[array_key],
                interpolatable=False,
                dtype=np.float32))

        self.enable_autoskip()

    def prepare(self, request):

        for (array_key, (src_points_key, trg_points_key)) in self.array_to_src_trg_points.items():
            if array_key in request:
                # increase or set request for points to be array roi + padding for partners outside roi for target points
                if src_points_key in request:
                    if not request[src_points_key].roi.contains(request[array_key].roi):
                        request[src_points_key] = PointsSpec(roi=request[array_key].roi)
                else:
                    request[src_points_key] = PointsSpec(request[array_key].roi)

                padded_roi = request[array_key].roi.grow((self.pad_for_partners), (self.pad_for_partners))
                if trg_points_key in request:
                    if not request[trg_points_key].roi.contains(padded_roi):
                        request[trg_points_key] = PointsSpec(padded_roi)
                else:
                    request[trg_points_key] = PointsSpec(padded_roi)

    def process(self, batch, request):

        # create vector map and add it to batch
        for (array_key, (src_points_key, trg_points_key)) in self.array_to_src_trg_points.items():
            if array_key in request:
                vector_map = self.__get_vector_map(batch=batch, request=request, vector_map_array_key=array_key)
                spec = self.spec[array_key].copy()
                spec.roi = request[array_key].roi
                batch.arrays[array_key] = Array(data=vector_map, spec=spec)

        # restore request / remove not requested points in padding-for-neighbors region & shrink batch roi
        for (array_key, (src_points_key, trg_points_key)) in self.array_to_src_trg_points.items():
            if array_key in request:
                if trg_points_key in request:
                    for loc_id, point in batch.points[trg_points_key].data.items():
                        if not request[trg_points_key].roi.contains(Coordinate(point.location)):
                            del batch.points[trg_points_key].data[loc_id]
                    neg_pad_for_partners = Coordinate((self.pad_for_partners * np.asarray([-1])).tolist())
                    batch.points[trg_points_key].spec.roi = batch.points[trg_points_key].spec.roi.grow(neg_pad_for_partners, neg_pad_for_partners)
                elif trg_points_key in batch.points:
                    del batch.points[trg_points_key]

    def __get_vector_map(self, batch, request, vector_map_array_key):

        src_points_key, trg_points_key = self.array_to_src_trg_points[vector_map_array_key]
        dim_vectors                      = len(request[vector_map_array_key].roi.get_shape())
        voxel_size_vm                    = self.voxel_sizes[vector_map_array_key]
        offset_vector_map_phys           = request[vector_map_array_key].roi.get_offset()
        vector_map_total = np.zeros(
            (dim_vectors,) + (request[vector_map_array_key].roi.get_shape()//voxel_size_vm),
            dtype=np.float32)

        if len(batch.points[src_points_key].data.keys()) == 0:
            return vector_map_total

        for (loc_id, point) in batch.points[src_points_key].data.items():

            if request[vector_map_array_key].roi.contains(Coordinate(point.location)):

                # get all partner locations which should be considered
                relevant_partner_loc = self.__get_relevant_partner_locations(batch, point, trg_points_key)
                if len(relevant_partner_loc) > 0:

                    # get locations where to set vectors around source location
                    # (look only at region close to src location (to avoid np.nonzero() over entire array))
                    mask = self.__get_mask(batch, request, vector_map_array_key, point.location)
                    offset_vx_considered_mask = [((point.location[dim]-self.radius_phys-offset_vector_map_phys[dim])//voxel_size_vm[dim])
                                                 for dim in range(dim_vectors)]
                    clipped_offset_vx_considered_mask = np.clip(offset_vx_considered_mask, a_min=0, a_max=np.inf)
                    slices = [slice(int(np.max((0, offset_vx_considered_mask[dim]))),
                                    int(np.min((offset_vx_considered_mask[dim] + (2*self.radius_phys//voxel_size_vm[dim]),
                                            ((mask.shape[dim]))))))
                                    for dim in range(dim_vectors)]
                    considered_region_mask  = mask[slices]
                    locations_to_fill_vx       = np.reshape(np.nonzero(considered_region_mask), [dim_vectors, -1]).T
                    locations_to_fill_abs_phys = (((locations_to_fill_vx + clipped_offset_vx_considered_mask)*voxel_size_vm)
                                                  + offset_vector_map_phys).tolist()

                    #check for target point with largest distance first and add vector pointing to it to vector_map_total
                    num_src_vectors_per_trg_loc = len(locations_to_fill_abs_phys) // len(relevant_partner_loc)
                    if num_src_vectors_per_trg_loc > 0:
                        dist_to_locs = {}
                        for phys_loc in relevant_partner_loc:
                            dist_to_locs[np.linalg.norm(point.location - phys_loc)] = phys_loc
                        for nr, dist in enumerate(reversed(np.sort(dist_to_locs.keys()))):
                            trg_loc_abs_phys       = dist_to_locs[dist]
                            kdtree_locs_vector_map = KDTree(locations_to_fill_abs_phys)
                            if nr == len(relevant_partner_loc)-1:
                                num_src_vectors_per_trg_loc = len(locations_to_fill_abs_phys)
                            distances, ids = kdtree_locs_vector_map.query(trg_loc_abs_phys, k=num_src_vectors_per_trg_loc)

                            try:
                                len(ids)
                            except TypeError:
                                ids = [ids]

                            for src_voxel_id in ids:
                                # remove point from list which are taken as neighbors of THIS target location
                                neighbor_loc_abs_phys = kdtree_locs_vector_map.data[src_voxel_id]
                                locations_to_fill_abs_phys.remove(neighbor_loc_abs_phys.tolist())

                                # get vector for THIS neighbor to THIS target location, get its location and place it
                                vector                  = (trg_loc_abs_phys - neighbor_loc_abs_phys)
                                neighbor_loc_shifted_vx = (neighbor_loc_abs_phys - offset_vector_map_phys) // voxel_size_vm
                                for dim in range(dim_vectors):
                                    vector_map_total[dim][[[int(l)] for l in neighbor_loc_shifted_vx]] = vector[dim]
        return vector_map_total

    def __get_relevant_partner_locations(self, batch, point, trg_points_key):
        # criterions: 'min_distance' or 'all'

        # get all partner locations
        all_partners_locations = []
        for partner_id in point.partner_ids:
            if partner_id in batch.points[trg_points_key].data.keys():
                all_partners_locations.append(batch.points[trg_points_key].data[partner_id].location)

        # if only one partner location, return this one for any given criterion
        if len(all_partners_locations) <= 1:
            return all_partners_locations

        # return all partner locations
        elif self.partner_criterion == 'all':
            return all_partners_locations

        # return partner with minimal euclidean distance to src_location
        elif self.partner_criterion == 'min_distance':
            min_distance, stored_pos = np.inf, []
            for partner_loc in all_partners_locations:
                distance = np.linalg.norm(partner_loc - point.location)
                if distance < min_distance:
                    min_distance = distance.copy()
                    stored_pos   = partner_loc.copy()
            return [stored_pos]

    def __get_mask(self, batch, request, vector_map_array_key, src_location):
        ''' create binary mask encoding where to place vectors for in region around considered src_location '''

        voxel_size = self.voxel_sizes[vector_map_array_key]

        offset_bm_phys     = request[vector_map_array_key].roi.get_offset()
        shape_bm_array_vx = request[vector_map_array_key].roi.get_shape() // voxel_size
        binary_map         = np.zeros(shape_bm_array_vx, dtype='uint8')

        if self.array_keys_to_stayinside_array_keys is None:
            mask = np.ones_like(binary_map)
        else:
            stayinside_array_key = self.array_keys_to_stayinside_array_keys[vector_map_array_key]
            mask = batch.arrays[stayinside_array_key].data

        if mask.shape > binary_map.shape:
            # assumption: binary map is centered in the mask array
            padding = (np.asarray(mask.shape) - np.asarray(binary_map.shape)) / 2.
            slices = [slice(np.floor(pad), -np.ceil(pad)) for pad in padding]
            mask = mask[slices]

        binary_map_total = np.zeros_like(binary_map)

        shifted_loc = src_location - np.asarray(offset_bm_phys)
        shifted_loc = shifted_loc.astype(np.int32) // voxel_size
        object_id   = mask[[[loc] for loc in shifted_loc]][0]  # 0 index, otherwise numpy array with single number

        binary_map[[[loc] for loc in shifted_loc]] = 1
        binary_map = enlarge_binary_map(binary_map, radius=self.radius_phys, voxel_size=voxel_size)
        binary_map[mask != object_id] = 0
        binary_map_total += binary_map
        binary_map.fill(0)
        binary_map_total[binary_map_total != 0] = 1

        return binary_map_total
