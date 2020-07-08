from .provider_test import ProviderTest
from gunpowder import (
    IntensityAugment,
    ArrayKeys,
    build,
    Normalize,
    Graph,
    Node,
    GraphSpec,
    Roi,
    BatchProvider,
    BatchRequest,
    GraphKeys,
    GraphKey,
    Batch,
    SimpleAugment,
)

import numpy as np
from itertools import permutations 

class TestSource(BatchProvider):
    def __init__(self):

        self.graph = Graph(
            [Node(id=1, location=np.array([1, 34, 65]))],
            [],
            GraphSpec(roi=Roi((0, 20, 33), (100, 120, 178))),
        )

    def setup(self):

        self.provides(GraphKeys.TEST_GRAPH, self.graph.spec)

    def prepare(self, request):
        return request

    def provide(self, request):

        batch = Batch()

        roi = request[GraphKeys.TEST_GRAPH].roi
        print("roi: ", roi)
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
        possible_loc = [[1, 99], [34, 106], [65, 121]]
        with build(pipeline):
            seen_mirrored = False
            for i in range(100):
                batch = pipeline.request_batch(request)

                assert len(list(batch[GraphKeys.TEST_GRAPH].nodes)) == 1
                node = list(batch[GraphKeys.TEST_GRAPH].nodes)[0]
                print(node.location)
                assert all(
                    [
                        node.location[dim] in possible_loc[dim] or node.location[dim] in possible_loc[dim] 
                        for dim in range(3)
                    ]
                )
                print(node.location)
                seen_mirrored = seen_mirrored or any(
                    [node.location[dim] == possible_loc[dim][1] for dim in range(3)]
                )
                assert Roi((0, 20, 33), (100, 100, 120)).contains(batch[GraphKeys.TEST_GRAPH].spec.roi)
                assert batch[GraphKeys.TEST_GRAPH].spec.roi.contains(node.location)
            assert seen_mirrored


    def test_two_transpose(self):
        test_graph = GraphKey("TEST_GRAPH")

        transpose_dims = [1, 2]
        pipeline = TestSource() + SimpleAugment(
            mirror_only=[], transpose_only=transpose_dims
        )

        request = BatchRequest()
        request[GraphKeys.TEST_GRAPH] = GraphSpec(roi=Roi((0, 20, 33), (100, 100, 120)))

        possible_loc = [[1, 1], [34, 52], [65, 47]]
        with build(pipeline):
            seen_transposed = False
            for i in range(100):
                batch = pipeline.request_batch(request)

                assert len(list(batch[GraphKeys.TEST_GRAPH].nodes)) == 1
                node = list(batch[GraphKeys.TEST_GRAPH].nodes)[0]
                print(node.location)
                assert all(
                    [
                        node.location[dim] in possible_loc[dim] or node.location[dim] in possible_loc[dim] 
                        for dim in range(3)
                    ]
                )
                print(node.location)
                seen_transposed = seen_transposed or any(
                    [node.location[dim] != possible_loc[dim][0] for dim in range(3)]
                )
                assert Roi((0, 20, 33), (100, 100, 120)).contains(batch[GraphKeys.TEST_GRAPH].spec.roi)
                assert batch[GraphKeys.TEST_GRAPH].spec.roi.contains(node.location)
            assert seen_transposed

    def test_multi_transpose(self):
        test_graph = GraphKey("TEST_GRAPH")
        point = np.array([1, 34, 65])

        transpose_dims = [0, 1, 2]
        pipeline = TestSource() + SimpleAugment(
            mirror_only=[], transpose_only=transpose_dims
        )

        request = BatchRequest()
        offset = (0, 20, 33)
        request[GraphKeys.TEST_GRAPH] = GraphSpec(roi=Roi(offset, (100, 100, 120)))

        # Create all possible permurations of our transpose dims
        transpose_combinations = list(permutations(transpose_dims, 3))
        possible_loc = np.zeros((len(transpose_combinations), 3))

        # Transpose points in all possible ways
        for i, comb in enumerate(transpose_combinations):
            possible_loc[i] = point - np.array(offset)
            possible_loc[i] = possible_loc[i][np.array(comb)]
            possible_loc[i] = possible_loc[i] + np.array(offset)

        with build(pipeline):
            seen_transposed = False
            for i in range(100):
                batch = pipeline.request_batch(request)

                assert len(list(batch[GraphKeys.TEST_GRAPH].nodes)) == 1
                node = list(batch[GraphKeys.TEST_GRAPH].nodes)[0]

                assert node.location in possible_loc 

                seen_transposed = seen_transposed or any(
                    [node.location[dim] != point[dim] for dim in range(3)]
                )
                assert Roi((0, 20, 33), (100, 100, 120)).contains(batch[GraphKeys.TEST_GRAPH].spec.roi)
                assert batch[GraphKeys.TEST_GRAPH].spec.roi.contains(node.location)
            assert seen_transposed

    def test_both(self):
        test_graph = GraphKey("TEST_GRAPH")
        og_point = np.array([1, 34, 65])

        transpose_dims = [0, 1, 2]
        mirror_dims = [0, 1, 2]
        pipeline = TestSource() + SimpleAugment(
            mirror_only=mirror_dims, transpose_only=transpose_dims
        )

        request = BatchRequest()
        offset = (0, 20, 33)
        request[GraphKeys.TEST_GRAPH] = GraphSpec(roi=Roi(offset, (100, 100, 120)))
        
        # Get all possble mirror locations
        # possible_mirror_loc = [[1, 99], [34, 106], [65, 121]]
        mirror_combs = [[1, 34, 65],
                        [1, 106, 121],
                        [1, 34, 121],
                        [1, 106, 65],
                        [99, 34, 65],
                        [99, 106, 121],
                        [99, 34, 121],
                        [99, 106, 65]]

        # Create all possible permurations of our transpose dims
        transpose_combinations = list(permutations(transpose_dims, 3))

        # Generate all possible tranposes of all possible mirrors
        possible_loc = np.zeros((len(mirror_combs), len(transpose_combinations), 3))
        for i, point in enumerate(mirror_combs): 
            for j, comb in enumerate(transpose_combinations):
                possible_loc[i, j] = point - np.array(offset)
                possible_loc[i, j] = possible_loc[i, j][np.array(comb)]
                possible_loc[i, j] = possible_loc[i, j] + np.array(offset)
        print(possible_loc)

        with build(pipeline):
            seen_transposed = False
            for i in range(100):
                batch = pipeline.request_batch(request)

                assert len(list(batch[GraphKeys.TEST_GRAPH].nodes)) == 1
                node = list(batch[GraphKeys.TEST_GRAPH].nodes)[0]

                # Check if your location is possible
                assert node.location in possible_loc 
                seen_transposed = seen_transposed or any(
                    [node.location[dim] != og_point[dim] for dim in range(3)]
                )
                assert Roi((0, 20, 33), (100, 100, 120)).contains(batch[GraphKeys.TEST_GRAPH].spec.roi)
                assert batch[GraphKeys.TEST_GRAPH].spec.roi.contains(node.location)
            assert seen_transposed
