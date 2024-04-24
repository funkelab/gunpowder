import logging

import numpy as np

from gunpowder import (
    Array,
    ArrayKey,
    ArraySpec,
    Batch,
    Coordinate,
    Roi,
)

logger = logging.getLogger(__name__)


def test_get_total_roi_nonspatial_array():
    raw = ArrayKey("RAW")
    nonspatial = ArrayKey("NONSPATIAL")

    voxel_size = Coordinate((1, 2))
    roi = Roi((100, 200), (20, 20))

    raw_spec = ArraySpec(roi=roi, voxel_size=voxel_size)
    nonspatial_spec = ArraySpec(nonspatial=True)

    batch = Batch()
    batch[raw] = Array(data=np.zeros((20, 10)), spec=raw_spec)
    batch[nonspatial] = Array(data=np.zeros((2, 3)), spec=nonspatial_spec)

    assert batch.get_total_roi() == roi
