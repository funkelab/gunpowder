import logging

import numpy as np

from gunpowder import (
    Array,
    ArrayKey,
    ArraySpec,
    Crop,
    Graph,
    GraphKey,
    GraphSpec,
    MergeProvider,
    Roi,
    build,
)

from .helper_sources import ArraySource, GraphSource

logger = logging.getLogger(__name__)


def test_output():
    raw_key = ArrayKey("RAW")
    pre_key = GraphKey("PRESYN")

    raw_spec = ArraySpec(
        roi=Roi((200, 20, 20), (1800, 180, 180)), voxel_size=(20, 2, 2)
    )
    pre_spec = GraphSpec(roi=Roi((200, 20, 20), (1800, 180, 180)))

    raw_data = np.zeros(raw_spec.roi.shape / raw_spec.voxel_size)

    raw_array = Array(raw_data, raw_spec)
    pre_graph = Graph([], [], pre_spec)

    cropped_roi_raw = Roi((400, 40, 40), (1000, 100, 100))
    cropped_roi_presyn = Roi((800, 80, 80), (800, 80, 80))

    pipeline = (
        (ArraySource(raw_key, raw_array), GraphSource(pre_key, pre_graph))
        + MergeProvider()
        + Crop(raw_key, cropped_roi_raw)
        + Crop(pre_key, cropped_roi_presyn)
    )

    with build(pipeline):
        assert pipeline.spec[raw_key].roi == cropped_roi_raw
        assert pipeline.spec[pre_key].roi == cropped_roi_presyn

    pipeline = (
        (ArraySource(raw_key, raw_array), GraphSource(pre_key, pre_graph))
        + MergeProvider()
        + Crop(
            raw_key,
            fraction_negative=(0.25, 0, 0),
            fraction_positive=(0.25, 0, 0),
        )
    )
    expected_roi_raw = Roi((650, 20, 20), (900, 180, 180))

    with build(pipeline):
        assert pipeline.spec[raw_key].roi == expected_roi_raw
