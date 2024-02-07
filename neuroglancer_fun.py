import matplotlib.pyplot as plt
import numpy as np
import random
import zarr
import torch
from skimage import data
from skimage import filters

# make sure we all see the same
torch.manual_seed(1961923)
np.random.seed(1961923)
random.seed(1961923)

# open a sample image (channels first)
raw_data = data.astronaut().transpose(2, 0, 1)

# create some dummy "ground-truth" to train on
gt_data = filters.gaussian(raw_data[0], sigma=3.0) > 0.75
gt_data = gt_data[np.newaxis, :].astype(np.float32)

# store image in zarr container
f = zarr.open("sample_data.zarr", "w")
f["raw"] = raw_data
f["raw"].attrs["resolution"] = (1, 1)
f["ground_truth"] = gt_data
f["ground_truth"].attrs["resolution"] = (1, 1)

import gunpowder as gp

# declare arrays to use in the pipeline
raw = gp.ArrayKey("RAW")
gt = gp.ArrayKey("GT")

# create "pipeline" consisting only of a data source
source = gp.ZarrSource(
    "sample_data.zarr",  # the zarr container
    {raw: "raw", gt: "ground_truth"},  # which dataset to associate to the array key
    {
        raw: gp.ArraySpec(interpolatable=True),
        gt: gp.ArraySpec(interpolatable=False),
    },  # meta-information
)
pipeline = source
pipeline += gp.Normalize(raw)
pipeline += gp.RandomLocation()
pipeline += gp.DeformAugment(
    gp.Coordinate(5, 5),
    gp.Coordinate(2, 2),
    graph_raster_voxel_size=gp.Coordinate(1, 1),
)

# formulate a request for "raw"
request = gp.BatchRequest()
request.add(raw, gp.Coordinate(64, 64), gp.Coordinate(1, 1))
request.add(gt, gp.Coordinate(32, 32), gp.Coordinate(1, 1))

# build the pipeline...
with gp.build_neuroglancer(pipeline):
    for _ in range(10):
        # ...and request a batch
        batch = pipeline.request_batch(request)

# show the content of the batch
print(f"batch returned: {batch}")
