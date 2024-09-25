from gunpowder.nodes import BatchProvider
from gunpowder.nodes.batch_provider import BatchRequestError
from .observers import Observer

import logging
import traceback
from typing import Optional

logger = logging.getLogger(__name__)


class PipelineSetupError(Exception):
    def __init__(self, provider):
        self.provider = provider

    def __str__(self):
        return f"Exception in {self.provider.name()} while calling setup()"


class PipelineTeardownError(Exception):
    def __init__(self, provider):
        self.provider = provider

    def __str__(self):
        return f"Exception in {self.provider.name()} while calling teardown()"


class PipelineRequestError(Exception):
    def __init__(self, pipeline, request, original_traceback=None):
        self.pipeline = pipeline
        self.request = request
        self.original_traceback = original_traceback

    def __str__(self):
        return (
            (
                ("".join(self.original_traceback))
                if self.original_traceback is not None
                else ""
            )
            + "Exception in pipeline:\n"
            f"{self.pipeline}\n"
            "while trying to process request\n"
            f"{self.request}"
        )


class Pipeline:
    def __init__(self, node):
        """Create a pipeline from a single :class:`BatchProvider`."""

        assert isinstance(node, BatchProvider), f"{type(node)} is not a BatchProvider"

        self.output = node
        self.children = []
        self.initialized = False

    def traverse(self, callback, reverse=False):
        """Visit every node in the pipeline recursively (either from root to
        leaves of from leaves to the root if ``reverse`` is true). ``callback``
        will be called for each node encountered."""

        result = []

        if not reverse:
            result.append(callback(self))
        for child in self.children:
            result.append(child.traverse(callback, reverse))
        if reverse:
            result.append(callback(self))

        return result

    def copy(self):
        """Make a shallow copy of the pipeline."""

        pipeline = Pipeline(self.output)
        pipeline.children = [c.copy() for c in self.children]

        return pipeline

    def setup(self, observers: Optional[list[Observer]] = None):
        """Connect all batch providers in the pipeline and call setup for
        each, from source to sink."""

        observers = observers if observers is not None else []

        def connect(node):
            for child in node.children:
                node.output.add_upstream_provider(child.output)

        # connect all nodes
        self.traverse(connect)

        # call setup on all nodes
        if not self.initialized:

            def node_setup(node):
                try:
                    node.output.setup()
                    for observer in observers:
                        node.output.register_observer(observer)
                except Exception as e:
                    raise PipelineSetupError(node.output) from e

            self.traverse(node_setup, reverse=True)
            self.initialized = True
        else:
            logger.warning(
                "pipeline.setup() called more than once (build() inside " "build()?)"
            )

    def internal_teardown(self):
        """Call teardown on each batch provider in the pipeline and disconnect
        all nodes."""

        try:

            def node_teardown(node):
                try:
                    node.output.internal_teardown()
                except Exception as e:
                    raise PipelineTeardownError(node.output) from e

            # call internal_teardown on all nodes
            self.traverse(node_teardown, reverse=True)
            self.initialized = False

        finally:
            # disconnect all nodes
            def disconnect(node):
                node.output.remove_upstream_providers()

            self.traverse(disconnect)

    def request_batch(self, request):
        """Request a batch from the pipeline."""

        try:
            return self.output.request_batch(request)
        except BatchRequestError as e:
            tb = traceback.format_exception(type(e), e, e.__traceback__)
            if isinstance(e, BatchRequestError):
                tb = tb[-1:]
            raise PipelineRequestError(self, request, original_traceback=tb) from None
        except Exception as e:
            raise PipelineRequestError(self, request) from e

    @property
    def spec(self):
        return self.output.spec

    def __add__(self, other):
        if isinstance(other, BatchProvider):
            other = Pipeline(other)

        if isinstance(other, Pipeline):
            result = other.copy()

            # add this pipeline as child to all leaves in other
            def add_self_to_leaves(node):
                if len(node.children) == 0:
                    node.children.append(self.copy())

            result.traverse(add_self_to_leaves, reverse=True)

        else:
            raise RuntimeError(f"Don't know how to add {type(other)} to Pipeline")

        return result

    def __radd__(self, other):
        assert isinstance(
            other, tuple
        ), f"Don't know how to radd {type(other)} to Pipeline"

        for o in other:
            assert isinstance(o, Pipeline) or isinstance(
                o, BatchProvider
            ), f"Don't know how to radd {type(o)} to Pipeline"

        other = tuple(
            Pipeline(o) if isinstance(o, BatchProvider) else o.copy() for o in other
        )

        result = self.copy()

        # add every other as child to leaves in pipeline
        def add_others_to_leaves(node):
            if len(node.children) == 0:
                for o in other:
                    node.children.append(o)

        result.traverse(add_others_to_leaves, reverse=True)

        return result

    def __repr__(self):
        def to_string(node):
            return node.output.name()

        reprs = self.traverse(to_string, reverse=True)

        return self.__rec_repr__(reprs)
    
    def to_string(self, bold=None):
        def to_string(node):
            if node.output == bold:
                return f"\033[1m{node.output.name()}\033[0m"
            else:
                return node.output.name()

        reprs = self.traverse(to_string, reverse=True)

        return self.__rec_repr__(reprs)

    def __rec_repr__(self, reprs):
        if not isinstance(reprs, list):
            return str(reprs)

        num_children = len(reprs) - 1

        res = ""
        if num_children > 0:
            if num_children > 1:
                res = "("
            res += ", ".join(self.__rec_repr__(r) for r in reprs[:-1])
        if num_children > 0:
            if num_children > 1:
                res += ")"
            res += " -> "
        res += reprs[-1]
        return res
