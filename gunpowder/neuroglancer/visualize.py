import neuroglancer
import gunpowder as gp

import numpy as np

from typing import Union


def visualize(viewer, batch_or_request):
    going_up = isinstance(batch_or_request, gp.BatchRequest)
    with viewer.txn() as s:
        # reverse order for raw so we can set opacity to 1, this
        # way higher res raw replaces low res when available
        for name, array_or_spec in batch_or_request.items():
            if going_up:
                name = f"request-{name}"
                opacity = 0
                spec = array_or_spec.copy()
                if spec.voxel_size is None:
                    spec.voxel_size = spec.roi.shape
                data = np.zeros(spec.roi.shape / spec.voxel_size)
            else:
                name = f"batch-{name}"
                opacity = 0.5
                spec = array_or_spec.spec.copy()
                data = array_or_spec.data

            channel_dims = len(data.shape) - len(spec.voxel_size)
            assert channel_dims <= 1

            dims = neuroglancer.CoordinateSpace(
                names=["c^", "z", "y", "x"][-len(data.shape) :],
                units="nm",
                scales=tuple([1] * channel_dims) + tuple(spec.voxel_size),
            )

            local_vol = neuroglancer.LocalVolume(
                data=data,
                voxel_offset=tuple([0] * channel_dims)
                + tuple(spec.roi.begin / spec.voxel_size),
                dimensions=dims,
            )

            s.layers[str(name)] = neuroglancer.ImageLayer(
                source=local_vol, opacity=opacity
            )

        s.layout = neuroglancer.row_layout(
            [
                neuroglancer.column_layout(
                    [
                        neuroglancer.LayerGroupViewer(
                            layers=[
                                str(l.name)
                                for l in s.layers
                                if not (
                                    l.name.startswith("batch")
                                    or l.name.startswith("request")
                                )
                            ]
                        ),
                    ]
                ),
                neuroglancer.column_layout(
                    [
                        neuroglancer.LayerGroupViewer(
                            layers=[
                                str(l.name)
                                for l in s.layers
                                if (
                                    (l.name.startswith("batch") and not going_up)
                                    or (l.name.startswith("request") and going_up)
                                )
                            ]
                        ),
                    ]
                ),
            ]
        )
