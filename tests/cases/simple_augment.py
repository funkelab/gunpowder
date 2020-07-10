from .provider_test import ProviderTest
from gunpowder import (
    IntensityAugment,
    ArrayKeys,
    build,
    Normalize,
    Graph,
    Node,
    GraphSpec,
    ArraySpec,
    Roi,
    BatchProvider,
    BatchRequest,
    GraphKeys,
    GraphKey,
    Batch,
    SimpleAugment,
    Array,
    ArrayKey,
    MergeProvider,
    Pad
)

import numpy as np
from itertools import permutations 
import logging

class ArrayTestSource(BatchProvider):
    def __init__(self):
        spec = ArraySpec(roi=Roi((-200, -200, -200), (600, 600, 600)), dtype=np.float64, voxel_size=(1, 1, 1))
        self.array = Array(np.zeros(spec.roi.get_shape()), spec=spec) 

    def setup(self):

        self.provides(ArrayKeys.TEST_ARRAY1, self.array.spec)
        self.provides(ArrayKeys.TEST_ARRAY2, self.array.spec)

    def prepare(self, request):
        return request

    def provide(self, request):

        batch = Batch()

        roi1 = request[ArrayKeys.TEST_ARRAY1].roi
        roi2 = request[ArrayKeys.TEST_ARRAY2].roi

        batch[ArrayKeys.TEST_ARRAY1] = self.array.crop(roi1)
        batch[ArrayKeys.TEST_ARRAY2] = self.array.crop(roi2)

        return batch

class TestSource(BatchProvider):
    def __init__(self):

        self.graph = Graph(
            [Node(id=1, location=np.array([50, 70, 100]))],
            [],
            GraphSpec(roi=Roi((-200, -200, -200), (400, 400, 478))),
        )

    def setup(self):

        self.provides(GraphKeys.TEST_GRAPH, self.graph.spec)

    def prepare(self, request):
        return request

    def provide(self, request):

        batch = Batch()

        roi = request[GraphKeys.TEST_GRAPH].roi
        batch[GraphKeys.TEST_GRAPH] = self.graph.crop(roi).trim(roi)

        return batch


class TestSimpleAugment(ProviderTest):
    def test_mirror(self):
        test_graph = GraphKey("TEST_GRAPH")

        pipeline = TestSource() + SimpleAugment(
            mirror_only=[0, 1, 2], transpose_only=[]
        )

        request = BatchRequest()
        request[GraphKeys.TEST_GRAPH] = GraphSpec(roi=Roi((0, 20, 33), (100, 100, 120)))
        possible_loc = [[50, 49], [70, 29], [100, 86]]
        with build(pipeline):
            seen_mirrored = False
            for i in range(100):
                batch = pipeline.request_batch(request)

                assert len(list(batch[GraphKeys.TEST_GRAPH].nodes)) == 1
                node = list(batch[GraphKeys.TEST_GRAPH].nodes)[0]
                logging.debug(node.location)
                assert all(
                    [
                        node.location[dim] in possible_loc[dim] 
                        for dim in range(3)
                    ]
                )
                seen_mirrored = seen_mirrored or any(
                    [node.location[dim] == possible_loc[dim][1] for dim in range(3)]
                )
                assert Roi((0, 20, 33), (100, 100, 120)).contains(batch[GraphKeys.TEST_GRAPH].spec.roi)
                assert batch[GraphKeys.TEST_GRAPH].spec.roi.contains(node.location)
            assert seen_mirrored


    def test_two_transpose(self):
        test_graph = GraphKey("TEST_GRAPH")
        test_array1 = ArrayKey("TEST_ARRAY1")
        test_array2 = ArrayKey("TEST_ARRAY2")

        transpose_dims = [1, 2]
        pipeline = (ArrayTestSource(), TestSource()) + MergeProvider() + SimpleAugment(
            mirror_only=[], transpose_only=transpose_dims
        )

        request = BatchRequest()
        request[GraphKeys.TEST_GRAPH] = GraphSpec(roi=Roi((0, 20, 33), (100, 100, 120)))
        request[ArrayKeys.TEST_ARRAY1] = ArraySpec(roi=Roi((0, 0, 0), (100, 200, 300)))
        request[ArrayKeys.TEST_ARRAY2] = ArraySpec(roi=Roi((0, 100, 250), (100, 100, 50)))
                

        possible_loc = [[50, 50], [70, 100], [100, 70]]
        with build(pipeline):
            seen_transposed = False
            for i in range(100):
                batch = pipeline.request_batch(request)
                assert len(list(batch[GraphKeys.TEST_GRAPH].nodes)) == 1
                node = list(batch[GraphKeys.TEST_GRAPH].nodes)[0]
                logging.debug(node.location)
                assert all(
                    [
                        node.location[dim] in possible_loc[dim] 
                        for dim in range(3)
                    ]
                )
                seen_transposed = seen_transposed or any(
                    [node.location[dim] != possible_loc[dim][0] for dim in range(3)]
                )
                assert Roi((0, 20, 33), (100, 100, 120)).contains(batch[GraphKeys.TEST_GRAPH].spec.roi)
                assert batch[GraphKeys.TEST_GRAPH].spec.roi.contains(node.location)

                for (array_key, array) in batch.arrays.items():
                    assert batch.arrays[array_key].data.shape == batch.arrays[array_key].spec.roi.get_shape()

            assert seen_transposed
            assert 1==2

    def test_multi_transpose(self):
        test_graph = GraphKey("TEST_GRAPH")
        test_array1 = ArrayKey("TEST_ARRAY1")
        test_array2 = ArrayKey("TEST_ARRAY2")
        point = np.array([50, 70, 100])

        transpose_dims = [0, 1, 2]
        pipeline = (ArrayTestSource(), TestSource()) + MergeProvider() + SimpleAugment(
            mirror_only=[], transpose_only=transpose_dims
        )

        request = BatchRequest()
        offset = (0, 20, 33)
        request[GraphKeys.TEST_GRAPH] = GraphSpec(roi=Roi(offset, (100, 100, 120)))
        request[ArrayKeys.TEST_ARRAY1] = ArraySpec(roi=Roi((0, 0, 0), (100, 200, 300)))
        request[ArrayKeys.TEST_ARRAY2] = ArraySpec(roi=Roi((0, 100, 250), (100, 100, 50)))

        # Create all possible permurations of our transpose dims
        transpose_combinations = list(permutations(transpose_dims, 3))
        possible_loc = np.zeros((len(transpose_combinations), 3))

        # Transpose points in all possible ways
        for i, comb in enumerate(transpose_combinations):
            possible_loc[i] = point[np.array(comb)]

        with build(pipeline):
            seen_transposed = False
            seen_node = True
            for i in range(100):
                batch = pipeline.request_batch(request)

                if len(list(batch[GraphKeys.TEST_GRAPH].nodes)) == 1:
                    seen_node = True
                    node = list(batch[GraphKeys.TEST_GRAPH].nodes)[0]

                    assert node.location in possible_loc 

                    seen_transposed = seen_transposed or any(
                        [node.location[dim] != point[dim] for dim in range(3)]
                    )
                    assert Roi((0, 20, 33), (100, 100, 120)).contains(batch[GraphKeys.TEST_GRAPH].spec.roi)
                    assert batch[GraphKeys.TEST_GRAPH].spec.roi.contains(node.location)

                for (array_key, array) in batch.arrays.items():
                    assert batch.arrays[array_key].data.shape == batch.arrays[array_key].spec.roi.get_shape()
            assert seen_transposed
            assert seen_node

    def test_both(self):
        test_graph = GraphKey("TEST_GRAPH")
        test_array1 = ArrayKey("TEST_ARRAY1")
        test_array2 = ArrayKey("TEST_ARRAY2")
        og_point = np.array([50, 70, 100])

        transpose_dims = [0, 1, 2]
        mirror_dims = [0, 1, 2]
        pipeline = ((ArrayTestSource(), TestSource()) + MergeProvider() + 
                    Pad(test_array1, None) + Pad(test_array2, None) + Pad(test_graph, None)
                    + SimpleAugment(
            mirror_only=mirror_dims, transpose_only=transpose_dims
        ))

        request = BatchRequest()
        offset = (0, 20, 33)
        request[GraphKeys.TEST_GRAPH] = GraphSpec(roi=Roi(offset, (100, 100, 120)))
        request[ArrayKeys.TEST_ARRAY1] = ArraySpec(roi=Roi((0, 0, 0), (100, 200, 300)))
        request[ArrayKeys.TEST_ARRAY2] = ArraySpec(roi=Roi((0, 100, 250), (100, 100, 50)))
        
        # Get all possble mirror locations
        # possible_mirror_loc = [[49, 50], [70, 29], [100, 86]]
        mirror_combs = [[49, 70, 100],
                        [49, 29, 86],
                        [49, 70, 86],
                        [49, 29, 100],
                        [50, 70, 100],
                        [50, 29, 86],
                        [50, 70, 86],
                        [50, 29, 100]]

        # Create all possible permurations of our transpose dims
        transpose_combinations = list(permutations(transpose_dims, 3))

        # Generate all possible tranposes of all possible mirrors
        possible_loc = np.zeros((len(mirror_combs), len(transpose_combinations), 3))
        for i, point in enumerate(mirror_combs): 
            for j, comb in enumerate(transpose_combinations):
                possible_loc[i, j] = np.array(point)[np.array(comb)]

        with build(pipeline):
            seen_transposed = False
            seen_node = True
            for i in range(100):
                batch = pipeline.request_batch(request)

                if len(list(batch[GraphKeys.TEST_GRAPH].nodes)) == 1:
                    seen_node = True
                    node = list(batch[GraphKeys.TEST_GRAPH].nodes)[0]

                    # Check if your location is possible
                    assert node.location in possible_loc 
                    seen_transposed = seen_transposed or any(
                        [node.location[dim] != og_point[dim] for dim in range(3)]
                    )
                    assert Roi((0, 20, 33), (100, 100, 120)).contains(batch[GraphKeys.TEST_GRAPH].spec.roi)
                    assert batch[GraphKeys.TEST_GRAPH].spec.roi.contains(node.location)

                for (array_key, array) in batch.arrays.items():
                    assert batch.arrays[array_key].data.shape == batch.arrays[array_key].spec.roi.get_shape()
            assert seen_transposed
            assert seen_node

    def test_square(self):
        

        test_graph = GraphKey("TEST_GRAPH")
        test_array1 = ArrayKey("TEST_ARRAY1")
        test_array2 = ArrayKey("TEST_ARRAY2")

        pipeline = ((ArrayTestSource(), TestSource()) + MergeProvider() + 
                    Pad(test_array1, None) + Pad(test_array2, None) + Pad(test_graph, None)
                    + SimpleAugment(
            mirror_only=[1,2], transpose_only=[1,2]
        ))

        request = BatchRequest()
        request[GraphKeys.TEST_GRAPH] = GraphSpec(roi=Roi((0, 50, 65), (100, 100, 100)))
        request[ArrayKeys.TEST_ARRAY1] = ArraySpec(roi=Roi((0, 0, 15), (100, 200, 200)))
        request[ArrayKeys.TEST_ARRAY2] = ArraySpec(roi=Roi((0, 50, 65), (100, 100, 100)))

        
        with build(pipeline):
            for i in range(100):
                batch = pipeline.request_batch(request)
                assert len(list(batch[GraphKeys.TEST_GRAPH].nodes)) == 1

                for (array_key, array) in batch.arrays.items():
                    assert batch.arrays[array_key].data.shape == batch.arrays[array_key].spec.roi.get_shape()
