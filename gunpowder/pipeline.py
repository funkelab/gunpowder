import logging
from gunpowder.nodes import BatchProvider

logger = logging.getLogger(__name__)


class Pipeline:

    def __init__(self, node):
        '''Create a pipeline from a single :class:`BatchProvider`.
        '''

        assert isinstance(node, BatchProvider), \
            f"{type(node)} is not a BatchProvider"

        self.output = node
        self.children = []
        self.initialized = False

    def traverse(self, callback, reverse=False):
        '''Visit every node in the pipeline recursively (either from root to
        leaves of from leaves to the root if ``reverse`` is true). ``callback``
        will be called for each node encountered.'''

        result = []

        if not reverse:
            result.append(callback(self))
        for child in self.children:
            result.append(child.traverse(callback, reverse))
        if reverse:
            result.append(callback(self))

        return result

    def copy(self):
        '''Make a shallow copy of the pipeline.'''

        pipeline = Pipeline(self.output)
        pipeline.children = [c.copy() for c in self.children]

        return pipeline

    def setup(self):
        '''Connect all batch providers in the pipeline and call setup for
        each, from source to sink.'''

        def connect(node):
            for child in node.children:
                node.output.add_upstream_provider(child.output)

        # connect all nodes
        self.traverse(connect)

        # call setup on all nodes
        if not self.initialized:

            self.traverse(
                lambda n: n.output.setup(),
                reverse=True)
            self.initialized = True
        else:
            logger.warning(
                "pipeline.setup() called more than once (build() inside "
                "build()?)")

    def internal_teardown(self):
        '''Call teardown on each batch provider in the pipeline and disconnect
        all nodes.'''

        try:

            # call internal_teardown on all nodes
            self.traverse(
                lambda n: n.output.internal_teardown(),
                reverse=True)
            self.initialized = False

        finally:

            # disconnect all nodes
            def disconnect(node):
                node.output.remove_upstream_providers()

            self.traverse(disconnect)

    def request_batch(self, request):
        '''Request a batch from the pipeline.'''

        return self.output.request_batch(request)

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

            raise RuntimeError(
                f"Don't know how to add {type(other)} to Pipeline")

        return result

    def __radd__(self, other):

        assert isinstance(other, tuple), \
            f"Don't know how to radd {type(other)} to Pipeline"

        for o in other:
            assert isinstance(o, Pipeline) or \
                isinstance(o, BatchProvider), \
                f"Don't know how to radd {type(o)} to Pipeline"

        other = tuple(
            Pipeline(o)
            if isinstance(o, BatchProvider)
            else o.copy()
            for o in other)

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


# monkey-patch BatchProvider, such that + operator returns Pipeline


def batch_provider_add(self, other):

    if isinstance(other, BatchProvider):
        other = Pipeline(other)

    if not isinstance(other, Pipeline):
        raise RuntimeError(
            f"Don't know how to add {type(other)} to BatchProvider")

    return Pipeline(self) + other


def batch_provider_radd(self, other):

    if isinstance(other, BatchProvider):
        return Pipeline(other) + Pipeline(self)

    if isinstance(other, tuple):
        return other + Pipeline(self)

    raise RuntimeError(
        f"Don't know how to radd {type(other)} to BatchProvider")


BatchProvider.__add__ = batch_provider_add
BatchProvider.__radd__ = batch_provider_radd
