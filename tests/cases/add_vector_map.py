import itertools
from copy import deepcopy

import numpy as np

from gunpowder import (
    Array,
    ArrayKey,
    ArraySpec,
    Batch,
    BatchProvider,
    BatchRequest,
    Coordinate,
    Graph,
    GraphKey,
    GraphSpec,
    Node,
    Roi,
    build,
)
from gunpowder.contrib import AddVectorMap


# TODO: Simplify the source node. The data being generated should not be defined
# in the provide method. Instead the source should be simple arrays and graphs.
class AddVectorMapTestSource(BatchProvider):
    def __init__(self, raw_key, labels_key, presyn_key, postsyn_key, vector_map_key):
        self.raw_key = raw_key
        self.labels_key = labels_key
        self.presyn_key = presyn_key
        self.postsyn_key = postsyn_key
        self.vector_map_key = vector_map_key

    def setup(self):
        for identifier in [self.raw_key, self.labels_key]:
            self.provides(
                identifier,
                ArraySpec(
                    roi=Roi((1000, 1000, 1000), (400, 400, 400)), voxel_size=(20, 2, 2)
                ),
            )

        for identifier in [self.presyn_key, self.postsyn_key]:
            self.provides(
                identifier, GraphSpec(roi=Roi((1000, 1000, 1000), (400, 400, 400)))
            )

    def provide(self, request):
        batch = Batch()

        # have the pixels encode their position
        if self.raw_key in request:
            # the z,y,x coordinates of the ROI
            roi = request[self.raw_key].roi
            roi_voxel = roi // self.spec[self.raw_key].voxel_size
            meshgrids = np.meshgrid(
                range(roi_voxel.begin[0], roi_voxel.end[0]),
                range(roi_voxel.begin[1], roi_voxel.end[1]),
                range(roi_voxel.begin[2], roi_voxel.end[2]),
                indexing="ij",
            )
            data = meshgrids[0] + meshgrids[1] + meshgrids[2]

            spec = self.spec[self.raw_key].copy()
            spec.roi = roi
            batch.arrays[self.raw_key] = Array(data, spec)

        if self.labels_key in request:
            roi = request[self.labels_key].roi
            roi_voxel_shape = (roi // self.spec[self.labels_key].voxel_size).shape
            data = np.ones(roi_voxel_shape)
            data[roi_voxel_shape[0] // 2 :, roi_voxel_shape[1] // 2 :, :] = 2
            data[roi_voxel_shape[0] // 2 :, -(roi_voxel_shape[1] // 2) :, :] = 3
            spec = self.spec[self.labels_key].copy()
            spec.roi = roi
            batch.arrays[self.labels_key] = Array(data, spec)

        if self.presyn_key in request:
            data_presyn, data_postsyn = self.__get_pre_and_postsyn_locations(
                roi=request[self.presyn_key].roi
            )
        elif self.postsyn_key in request:
            data_presyn, data_postsyn = self.__get_pre_and_postsyn_locations(
                roi=request[self.postsyn_key].roi
            )

        for graph_key, spec in request.graph_specs.items():
            if graph_key == self.presyn_key:
                data = data_presyn
            if graph_key == self.postsyn_key:
                data = data_postsyn
            batch.graphs[graph_key] = Graph(
                list(data.values()), [], GraphSpec(spec.roi)
            )

        return batch

    def __get_pre_and_postsyn_locations(self, roi):
        presyn_locs, postsyn_locs = {}, {}
        min_dist_between_presyn_locs = 250
        voxel_size_points = self.spec[self.raw_key].voxel_size
        min_dist_pre_to_postsyn_loc, max_dist_pre_to_postsyn_loc = 60, 120
        num_presyn_locations = roi.size // (
            np.prod(50 * np.asarray(voxel_size_points))
        )  # 1 synapse per 50vx^3 cube
        num_postsyn_locations = np.random.randint(
            low=1, high=3
        )  # 1 to 3 postsyn partners

        loc_id = 0
        all_presyn_locs = []
        for nr_presyn_loc in range(num_presyn_locations):
            loc_id = loc_id + 1
            presyn_loc_id = loc_id

            presyn_loc_too_close = True
            while presyn_loc_too_close:
                presyn_location = np.asarray(
                    [
                        np.random.randint(low=roi.begin[0], high=roi.end[0]),
                        np.random.randint(low=roi.begin[1], high=roi.end[1]),
                        np.random.randint(low=roi.begin[2], high=roi.end[2]),
                    ]
                )
                # ensure that partner locations of diff presyn locations are not overlapping
                presyn_loc_too_close = False
                for previous_loc in all_presyn_locs:
                    if np.linalg.norm(presyn_location - previous_loc) < (
                        min_dist_between_presyn_locs
                    ):
                        presyn_loc_too_close = True

            syn_id = nr_presyn_loc

            partner_ids = []
            for nr_partner_loc in range(num_postsyn_locations):
                loc_id = loc_id + 1
                partner_ids.append(loc_id)
                postsyn_loc_is_inside = False
                while not postsyn_loc_is_inside:
                    postsyn_location = presyn_location + np.random.choice(
                        (-1, 1), size=3, replace=True
                    ) * np.random.randint(
                        min_dist_pre_to_postsyn_loc, max_dist_pre_to_postsyn_loc, size=3
                    )
                    if roi.contains(Coordinate(postsyn_location)):
                        postsyn_loc_is_inside = True

                postsyn_locs[int(loc_id)] = deepcopy(
                    Node(
                        loc_id,
                        location=postsyn_location,
                        attrs={
                            "location_id": loc_id,
                            "synapse_id": syn_id,
                            "partner_ids": [presyn_loc_id],
                            "props": {},
                        },
                    )
                )

            presyn_locs[int(presyn_loc_id)] = deepcopy(
                Node(
                    presyn_loc_id,
                    location=presyn_location,
                    attrs={
                        "location_id": presyn_loc_id,
                        "synapse_id": syn_id,
                        "partner_ids": partner_ids,
                        "props": {},
                    },
                )
            )

        return presyn_locs, postsyn_locs


def test_output_min_distance():
    voxel_size = Coordinate((20, 2, 2))

    raw_key = ArrayKey("RAW")
    labels_key = ArrayKey("LABELS")
    vectors_map_key = ArrayKey("VECTORS_MAP_PRESYN")
    pre_key = GraphKey("PRESYN")
    post_key = GraphKey("POSTSYN")

    arraytypes_to_source_target_pointstypes = {vectors_map_key: (pre_key, post_key)}
    arraytypes_to_stayinside_arraytypes = {vectors_map_key: labels_key}

    # test for partner criterion 'min_distance'
    radius_phys = 30
    pipeline_min_distance = AddVectorMapTestSource(
        raw_key, labels_key, pre_key, post_key, vectors_map_key
    ) + AddVectorMap(
        src_and_trg_points=arraytypes_to_source_target_pointstypes,
        voxel_sizes={vectors_map_key: voxel_size},
        radius_phys=radius_phys,
        partner_criterion="min_distance",
        stayinside_array_keys=arraytypes_to_stayinside_arraytypes,
        pad_for_partners=(0, 0, 0),
    )

    with build(pipeline_min_distance):
        request = BatchRequest()
        raw_roi = pipeline_min_distance.spec[raw_key].roi
        gt_labels_roi = pipeline_min_distance.spec[labels_key].roi
        presyn_roi = pipeline_min_distance.spec[pre_key].roi

        request.add(raw_key, raw_roi.shape)
        request.add(labels_key, gt_labels_roi.shape)
        request.add(pre_key, presyn_roi.shape)
        request.add(post_key, presyn_roi.shape)
        request.add(vectors_map_key, presyn_roi.shape)
        for identifier, spec in request.items():
            spec.roi = spec.roi.shift(Coordinate(1000, 1000, 1000))

        batch = pipeline_min_distance.request_batch(request)

    presyn_locs = {n.id: n for n in batch.graphs[pre_key].nodes}
    postsyn_locs = {n.id: n for n in batch.graphs[post_key].nodes}
    vector_map_presyn = batch.arrays[vectors_map_key].data
    offset_vector_map_presyn = request[vectors_map_key].roi.offset

    assert len(presyn_locs) > 0
    assert len(postsyn_locs) > 0

    for loc_id, point in presyn_locs.items():
        if request[vectors_map_key].roi.contains(Coordinate(point.location)):
            assert batch.arrays[vectors_map_key].spec.roi.contains(
                Coordinate(point.location)
            )

            dist_to_loc = {}
            for partner_id in point.attrs["partner_ids"]:
                if partner_id in postsyn_locs.keys():
                    partner_location = postsyn_locs[partner_id].location
                    dist_to_loc[np.linalg.norm(partner_location - point.location)] = (
                        partner_location
                    )
            min_dist = np.min(list(dist_to_loc.keys()))
            relevant_partner_loc = dist_to_loc[min_dist]

            presyn_loc_shifted_vx = (
                point.location - offset_vector_map_presyn
            ) // voxel_size
            radius_vx = [(radius_phys // vx_dim) for vx_dim in voxel_size]
            region_to_check = np.clip(
                [
                    (presyn_loc_shifted_vx - radius_vx),
                    (presyn_loc_shifted_vx + radius_vx),
                ],
                a_min=(0, 0, 0),
                a_max=vector_map_presyn.shape[-3:],
            )
            for x, y, z in itertools.product(
                range(int(region_to_check[0][0]), int(region_to_check[1][0])),
                range(int(region_to_check[0][1]), int(region_to_check[1][1])),
                range(int(region_to_check[0][2]), int(region_to_check[1][2])),
            ):
                if (
                    np.linalg.norm((np.array((x, y, z)) - np.asarray(point.location)))
                    < radius_phys
                ):
                    vector = [
                        vector_map_presyn[dim][x, y, z]
                        for dim in range(vector_map_presyn.shape[0])
                    ]
                    if not np.sum(vector) == 0:
                        trg_loc_of_vector_phys = (
                            np.asarray(offset_vector_map_presyn)
                            + (voxel_size * np.array([x, y, z]))
                            + np.asarray(vector)
                        )
                        assert np.array_equal(
                            trg_loc_of_vector_phys, relevant_partner_loc
                        )

    # test for partner criterion 'all'
    pipeline_all = AddVectorMapTestSource(
        raw_key, labels_key, pre_key, post_key, vectors_map_key
    ) + AddVectorMap(
        src_and_trg_points=arraytypes_to_source_target_pointstypes,
        voxel_sizes={vectors_map_key: voxel_size},
        radius_phys=radius_phys,
        partner_criterion="all",
        stayinside_array_keys=arraytypes_to_stayinside_arraytypes,
        pad_for_partners=(0, 0, 0),
    )

    with build(pipeline_all):
        batch = pipeline_all.request_batch(request)

    presyn_locs = {n.id: n for n in batch.graphs[pre_key].nodes}
    postsyn_locs = {n.id: n for n in batch.graphs[post_key].nodes}
    vector_map_presyn = batch.arrays[vectors_map_key].data
    offset_vector_map_presyn = request[vectors_map_key].roi.offset

    assert len(presyn_locs) > 0
    assert len(postsyn_locs) > 0

    for loc_id, point in presyn_locs.items():
        if request[vectors_map_key].roi.contains(Coordinate(point.location)):
            assert batch.arrays[vectors_map_key].spec.roi.contains(
                Coordinate(point.location)
            )

            partner_ids_to_locs_per_src, count_vectors_per_partner = {}, {}
            for partner_id in point.attrs["partner_ids"]:
                if partner_id in postsyn_locs.keys():
                    partner_ids_to_locs_per_src[partner_id] = postsyn_locs[
                        partner_id
                    ].location.tolist()
                    count_vectors_per_partner[partner_id] = 0

            presyn_loc_shifted_vx = (
                point.location - offset_vector_map_presyn
            ) // voxel_size
            radius_vx = [(radius_phys // vx_dim) for vx_dim in voxel_size]
            region_to_check = np.clip(
                [
                    (presyn_loc_shifted_vx - radius_vx),
                    (presyn_loc_shifted_vx + radius_vx),
                ],
                a_min=(0, 0, 0),
                a_max=vector_map_presyn.shape[-3:],
            )
            for x, y, z in itertools.product(
                range(int(region_to_check[0][0]), int(region_to_check[1][0])),
                range(int(region_to_check[0][1]), int(region_to_check[1][1])),
                range(int(region_to_check[0][2]), int(region_to_check[1][2])),
            ):
                if (
                    np.linalg.norm((np.array((x, y, z)) - np.asarray(point.location)))
                    < radius_phys
                ):
                    vector = [
                        vector_map_presyn[dim][x, y, z]
                        for dim in range(vector_map_presyn.shape[0])
                    ]
                    if not np.sum(vector) == 0:
                        trg_loc_of_vector_phys = (
                            np.asarray(offset_vector_map_presyn)
                            + (voxel_size * np.array([x, y, z]))
                            + np.asarray(vector)
                        )
                        assert (
                            trg_loc_of_vector_phys.tolist()
                            in partner_ids_to_locs_per_src.values()
                        )

                        for (
                            partner_id,
                            partner_loc,
                        ) in partner_ids_to_locs_per_src.items():
                            if np.array_equal(
                                np.asarray(trg_loc_of_vector_phys), partner_loc
                            ):
                                count_vectors_per_partner[partner_id] += 1
            assert (
                list(count_vectors_per_partner.values())
                - np.min(list(count_vectors_per_partner.values()))
                <= len(count_vectors_per_partner.keys())
            ).all()
