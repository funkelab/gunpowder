from .batch import Batch
from .batch_request import BatchRequest
from .neuroglancer.event import step_next
from .neuroglancer.event import wait_for_step
from .neuroglancer.visualize import visualize
from .neuroglancer.add_layer import add_layer

# from .nodes import BatchProvider

import neuroglancer

from abc import ABC
from typing import Optional


class Observer(ABC):
    def __init__(self, name, pipeline):
        self.name = name
        self.pipeline = pipeline

    def update(self, request_or_batch: BatchRequest or Batch):
        """
        Take a BatchRequest or Batch and update the observer's state with
        their contents
        """
        pass

    def add_source(self, *args, **kwargs):
        """
        Add a source to the observer. This is a no-op for observers that do not
        provide an array source.
        """
        pass


class NeuroglancerObserver(Observer):
    def __init__(self, name, pipeline, host="0.0.0.0", port=0):
        super().__init__(name, pipeline)
        self.host = host
        self.port = port

        neuroglancer.set_server_bind_address(self.host, self.port)
        self.viewer = neuroglancer.Viewer()
        self.viewer.actions.add("continue", step_next)

        with self.viewer.config_state.txn() as s:
            s.input_event_bindings.data_view["keyt"] = "continue"

        with self.viewer.txn() as s:
            s.layout = neuroglancer.row_layout(
                [
                    neuroglancer.column_layout(
                        [
                            neuroglancer.LayerGroupViewer(layers=[]),
                            neuroglancer.LayerGroupViewer(layers=[]),
                        ]
                    ),
                ]
            )

        print(self.viewer)
        print("Hit T in neuroglancer viewer to step through the pipeline")

    def update(self, request_or_batch: BatchRequest or Batch, node: Optional = None):
        visualize(self.viewer, request_or_batch)
        string = self.pipeline.to_string(bold=node)
        print(
            "\r"
            + (
                "REQUESTING: "
                if isinstance(request_or_batch, BatchRequest)
                else "PROVIDING: "
            )
            + string
            + " " * 2,
            end="",
        )
        # print(self.pipeline.to_string(bold=node))
        wait_for_step()

    def add_source(
        self,
        array,
        name,
    ):
        spatial_dim_names = ["t", "z", "y", "x"]
        channel_dim_names = ["b^", "c^"]
        opacity = None
        shader = None
        rgb_channels = None
        color = None
        visible = True
        value_scale_factor = 1.0
        units = "nm"
        with self.viewer.txn() as s:
            add_layer(
                s,
                array,
                name,
                spatial_dim_names,
                channel_dim_names,
                opacity,
                shader,
                rgb_channels,
                color,
                visible,
                value_scale_factor,
                units,
            )
