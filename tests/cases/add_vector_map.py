import unittest
from .provider_test import ProviderTest
from gunpowder import *
from gunpowder.contrib import AddVectorMap
from gunpowder.contrib.points import PreSynPoint, PostSynPoint

from copy import deepcopy
import itertools
import numpy as np

class AddVectorMapTestSource(BatchProvider):

    def setup(self):

        for identifier in [
            ArrayKeys.RAW,
            ArrayKeys.GT_LABELS]:

            self.provides(
                identifier,
                ArraySpec(
                    roi=Roi((1000, 1000, 1000), (400, 400, 400)),
                    voxel_size=(20, 2, 2)))

        for identifier in [
            PointsKeys.PRESYN,
            PointsKeys.POSTSYN]:

            self.provides(
                identifier,
                PointsSpec(
                    roi=Roi((1000, 1000, 1000), (400, 400, 400))))

    def provide(self, request):

        batch = Batch()

        # have the pixels encode their position
        if ArrayKeys.RAW in request:

            # the z,y,x coordinates of the ROI
            roi = request[ArrayKeys.RAW].roi
            roi_voxel = roi // self.spec[ArrayKeys.RAW].voxel_size
            meshgrids = np.meshgrid(
                    range(roi_voxel.get_begin()[0], roi_voxel.get_end()[0]),
                    range(roi_voxel.get_begin()[1], roi_voxel.get_end()[1]),
                    range(roi_voxel.get_begin()[2], roi_voxel.get_end()[2]), indexing='ij')
            data = meshgrids[0] + meshgrids[1] + meshgrids[2]

            spec = self.spec[ArrayKeys.RAW].copy()
            spec.roi = roi
            batch.arrays[ArrayKeys.RAW] = Array(data, spec)

        if ArrayKeys.GT_LABELS in request:
            roi = request[ArrayKeys.GT_LABELS].roi
            roi_voxel_shape = (roi // self.spec[ArrayKeys.GT_LABELS].voxel_size).get_shape()
            data = np.ones(roi_voxel_shape)
            data[roi_voxel_shape[0]//2:,roi_voxel_shape[1]//2:,:] = 2
            data[roi_voxel_shape[0]//2:, -(roi_voxel_shape[1] // 2):, :] = 3
            spec = self.spec[ArrayKeys.GT_LABELS].copy()
            spec.roi = roi
            batch.arrays[ArrayKeys.GT_LABELS] = Array(data, spec)

        if PointsKeys.PRESYN in request:
            data_presyn, data_postsyn = self.__get_pre_and_postsyn_locations(roi=request[PointsKeys.PRESYN].roi)
        elif PointsKeys.POSTSYN in request:
            data_presyn, data_postsyn = self.__get_pre_and_postsyn_locations(roi=request[PointsKeys.POSTSYN].roi)

        voxel_size_points = self.spec[ArrayKeys.RAW].voxel_size
        for (points_key, spec) in request.points_specs.items():
            if points_key == PointsKeys.PRESYN:
                data = data_presyn
            if points_key == PointsKeys.POSTSYN:
                data = data_postsyn
            batch.points[points_key] = Points(data, PointsSpec(spec.roi))

        return batch

    def __get_pre_and_postsyn_locations(self, roi):

        presyn_locs, postsyn_locs = {}, {}
        min_dist_between_presyn_locs = 250
        voxel_size_points = self.spec[ArrayKeys.RAW].voxel_size
        min_dist_pre_to_postsyn_loc, max_dist_pre_to_postsyn_loc= 60, 120
        num_presyn_locations  = roi.size() // (np.prod(50*np.asarray(voxel_size_points)))  # 1 synapse per 50vx^3 cube
        num_postsyn_locations = np.random.randint(low=1, high=3)  # 1 to 3 postsyn partners

        loc_id = 0
        all_presyn_locs = []
        for nr_presyn_loc in range(num_presyn_locations):
                loc_id = loc_id + 1
                presyn_loc_id = loc_id

                presyn_loc_too_close = True
                while presyn_loc_too_close:
                    presyn_location = np.asarray([np.random.randint(low=roi.get_begin()[0], high=roi.get_end()[0]),
                                                  np.random.randint(low=roi.get_begin()[1], high=roi.get_end()[1]),
                                                  np.random.randint(low=roi.get_begin()[2], high=roi.get_end()[2])])
                    # ensure that partner locations of diff presyn locations are not overlapping
                    presyn_loc_too_close = False
                    for previous_loc in all_presyn_locs:
                        if np.linalg.norm(presyn_location - previous_loc) < (min_dist_between_presyn_locs):
                            presyn_loc_too_close = True

                syn_id = nr_presyn_loc

                partner_ids = []
                for nr_partner_loc in range(num_postsyn_locations):
                    loc_id = loc_id + 1
                    partner_ids.append(loc_id)
                    postsyn_loc_is_inside = False
                    while not postsyn_loc_is_inside:
                        postsyn_location = presyn_location + np.random.choice((-1,1),size=3, replace=True) \
                                            * np.random.randint(min_dist_pre_to_postsyn_loc, max_dist_pre_to_postsyn_loc, size=3)
                        if roi.contains(Coordinate(postsyn_location)):
                            postsyn_loc_is_inside = True

                    postsyn_locs[int(loc_id)] = deepcopy(PostSynPoint(location=postsyn_location, location_id=loc_id,
                                                         synapse_id=syn_id, partner_ids=[presyn_loc_id], props={}))

                presyn_locs[int(presyn_loc_id)] = deepcopy(PreSynPoint(location=presyn_location, location_id=presyn_loc_id,
                                                           synapse_id=syn_id, partner_ids=partner_ids, props={}))

        return presyn_locs, postsyn_locs


class TestAddVectorMap(ProviderTest):

    def test_output_min_distance(self):

        voxel_size = Coordinate((20, 2, 2))

        ArrayKey('GT_VECTORS_MAP_PRESYN')
        PointsKey('PRESYN')
        PointsKey('POSTSYN')

        arraytypes_to_source_target_pointstypes = {ArrayKeys.GT_VECTORS_MAP_PRESYN: (PointsKeys.PRESYN, PointsKeys.POSTSYN)}
        arraytypes_to_stayinside_arraytypes    = {ArrayKeys.GT_VECTORS_MAP_PRESYN: ArrayKeys.GT_LABELS}

        # test for partner criterion 'min_distance'
        radius_phys  = 30
        pipeline_min_distance = AddVectorMapTestSource() +\
                                AddVectorMap(src_and_trg_points = arraytypes_to_source_target_pointstypes,
                                             voxel_sizes = {ArrayKeys.GT_VECTORS_MAP_PRESYN: voxel_size},
                                             radius_phys = radius_phys,
                                             partner_criterion = 'min_distance',
                                             stayinside_array_keys = arraytypes_to_stayinside_arraytypes,
                                             pad_for_partners = (0, 0, 0))

        with build(pipeline_min_distance):

            request = BatchRequest()
            raw_roi = pipeline_min_distance.spec[ArrayKeys.RAW].roi
            gt_labels_roi = pipeline_min_distance.spec[ArrayKeys.GT_LABELS].roi
            presyn_roi = pipeline_min_distance.spec[PointsKeys.PRESYN].roi

            request.add(ArrayKeys.RAW, raw_roi.get_shape())
            request.add(ArrayKeys.GT_LABELS, gt_labels_roi.get_shape())
            request.add(PointsKeys.PRESYN, presyn_roi.get_shape())
            request.add(PointsKeys.POSTSYN, presyn_roi.get_shape())
            request.add(ArrayKeys.GT_VECTORS_MAP_PRESYN, presyn_roi.get_shape())
            for identifier, spec in request.items():
                spec.roi = spec.roi.shift((1000, 1000, 1000))

            batch = pipeline_min_distance.request_batch(request)

        presyn_locs  = batch.points[PointsKeys.PRESYN].data
        postsyn_locs = batch.points[PointsKeys.POSTSYN].data
        vector_map_presyn        = batch.arrays[ArrayKeys.GT_VECTORS_MAP_PRESYN].data
        offset_vector_map_presyn = request[ArrayKeys.GT_VECTORS_MAP_PRESYN].roi.get_offset()

        self.assertTrue(len(presyn_locs)>0)
        self.assertTrue(len(postsyn_locs)>0)

        for loc_id, point in presyn_locs.items():

            if request[ArrayKeys.GT_VECTORS_MAP_PRESYN].roi.contains(Coordinate(point.location)):
                self.assertTrue(batch.arrays[ArrayKeys.GT_VECTORS_MAP_PRESYN].spec.roi.contains(Coordinate(point.location)))

                dist_to_loc = {}
                for partner_id in point.partner_ids:
                    if partner_id in postsyn_locs.keys():
                        partner_location = postsyn_locs[partner_id].location
                        dist_to_loc[np.linalg.norm(partner_location - point.location)] = partner_location
                min_dist             = np.min(list(dist_to_loc.keys()))
                relevant_partner_loc = dist_to_loc[min_dist]

                presyn_loc_shifted_vx = (point.location - offset_vector_map_presyn)//voxel_size
                radius_vx             = [(radius_phys // vx_dim) for vx_dim in voxel_size]
                region_to_check       = np.clip([(presyn_loc_shifted_vx - radius_vx), (presyn_loc_shifted_vx+radius_vx)],
                                            a_min=(0,0,0), a_max=vector_map_presyn.shape[-3:])
                for x,y,z in itertools.product(range(int(region_to_check[0][0]), int(region_to_check[1][0])),
                                               range(int(region_to_check[0][1]), int(region_to_check[1][1])),
                                               range(int(region_to_check[0][2]), int(region_to_check[1][2]))):
                    if np.linalg.norm((np.array((x,y,z))-np.asarray(point.location))) < radius_phys:
                        vector = [vector_map_presyn[dim][x, y, z] for dim in range(vector_map_presyn.shape[0])]
                        if not np.sum(vector) == 0:
                            trg_loc_of_vector_phys = np.asarray(offset_vector_map_presyn) \
                                                     + (voxel_size * np.array([x, y, z])) + np.asarray(vector)
                            self.assertTrue(np.array_equal(trg_loc_of_vector_phys, relevant_partner_loc))




        # test for partner criterion 'all'
        pipeline_all = AddVectorMapTestSource() + AddVectorMap(src_and_trg_points = arraytypes_to_source_target_pointstypes,
                                                               voxel_sizes = {ArrayKeys.GT_VECTORS_MAP_PRESYN: voxel_size},
                                                               radius_phys = radius_phys,
                                                               partner_criterion = 'all',
                                                               stayinside_array_keys = arraytypes_to_stayinside_arraytypes,
                                                               pad_for_partners = (0, 0, 0))

        with build(pipeline_all):
            batch = pipeline_all.request_batch(request)

        presyn_locs  = batch.points[PointsKeys.PRESYN].data
        postsyn_locs = batch.points[PointsKeys.POSTSYN].data
        vector_map_presyn        = batch.arrays[ArrayKeys.GT_VECTORS_MAP_PRESYN].data
        offset_vector_map_presyn = request[ArrayKeys.GT_VECTORS_MAP_PRESYN].roi.get_offset()

        self.assertTrue(len(presyn_locs)>0)
        self.assertTrue(len(postsyn_locs)>0)

        for loc_id, point in presyn_locs.items():

            if request[ArrayKeys.GT_VECTORS_MAP_PRESYN].roi.contains(Coordinate(point.location)):
                self.assertTrue(batch.arrays[ArrayKeys.GT_VECTORS_MAP_PRESYN].spec.roi.contains(Coordinate(point.location)))

                partner_ids_to_locs_per_src, count_vectors_per_partner = {}, {}
                for partner_id in point.partner_ids:
                    if partner_id in postsyn_locs.keys():
                        partner_ids_to_locs_per_src[partner_id] = postsyn_locs[partner_id].location.tolist()
                        count_vectors_per_partner[partner_id]   = 0

                presyn_loc_shifted_vx = (point.location - offset_vector_map_presyn)//voxel_size
                radius_vx             = [(radius_phys // vx_dim) for vx_dim in voxel_size]
                region_to_check       = np.clip([(presyn_loc_shifted_vx - radius_vx), (presyn_loc_shifted_vx+radius_vx)],
                                            a_min=(0,0,0), a_max=vector_map_presyn.shape[-3:])
                for x,y,z in itertools.product(range(int(region_to_check[0][0]), int(region_to_check[1][0])),
                                               range(int(region_to_check[0][1]), int(region_to_check[1][1])),
                                               range(int(region_to_check[0][2]), int(region_to_check[1][2]))):
                    if np.linalg.norm((np.array((x,y,z))-np.asarray(point.location))) < radius_phys:
                        vector = [vector_map_presyn[dim][x, y, z] for dim in range(vector_map_presyn.shape[0])]
                        if not np.sum(vector) == 0:
                            trg_loc_of_vector_phys = np.asarray(offset_vector_map_presyn) \
                                                     + (voxel_size * np.array([x, y, z])) + np.asarray(vector)
                            self.assertTrue(trg_loc_of_vector_phys.tolist() in partner_ids_to_locs_per_src.values())

                            for partner_id, partner_loc in partner_ids_to_locs_per_src.items():
                                if np.array_equal(np.asarray(trg_loc_of_vector_phys), partner_loc):
                                    count_vectors_per_partner[partner_id] += 1
                self.assertTrue((list(count_vectors_per_partner.values()) - np.min(list(count_vectors_per_partner.values()))
                                <=len(count_vectors_per_partner.keys())).all())

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestAddVectorMap)
    unittest.TextTestRunner(verbosity=2).run(suite)
