import logging

logger = logging.getLogger(__name__)


class build(object):
    def __init__(self, pipeline):
        self.pipeline = pipeline

    def __enter__(self):
        try:
            self.pipeline.setup()
        except:
            logger.error(
                "something went wrong during the setup of the pipeline, calling tear down"
            )
            self.pipeline.internal_teardown()
            logger.debug("tear down completed")
            raise
        return self.pipeline

    def __exit__(self, type, value, traceback):
        logger.debug("leaving context, tearing down pipeline")
        self.pipeline.internal_teardown()
        logger.debug("tear down completed")


import neuroglancer
from .neuroglancer.event import step_next


class build_neuroglancer(object):
    def __init__(self, pipeline):
        self.pipeline = pipeline

    def __enter__(self):
        neuroglancer.set_server_bind_address("0.0.0.0")
        viewer = neuroglancer.Viewer()

        viewer.actions.add("continue", step_next)

        with viewer.config_state.txn() as s:
            s.input_event_bindings.data_view["keyt"] = "continue"
        with viewer.txn() as s:
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

        try:
            self.pipeline.setup(viewer)
        except:
            logger.error(
                "something went wrong during the setup of the pipeline, calling tear down"
            )
            self.pipeline.internal_teardown()
            logger.debug("tear down completed")
            raise

        print(viewer)
        return self.pipeline

    def __exit__(self, type, value, traceback):
        logger.debug("leaving context, tearing down pipeline")
        self.pipeline.internal_teardown()
        logger.debug("tear down completed")
