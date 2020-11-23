import copy
import logging
import numpy as np
from scipy.ndimage.filters import gaussian_filter
from skimage import draw

from .batch_filter import BatchFilter
from gunpowder.array import Array
from gunpowder.array_spec import ArraySpec
from gunpowder.batch_request import BatchRequest
from gunpowder.coordinate import Coordinate
from gunpowder.freezable import Freezable
from gunpowder.morphology import enlarge_binary_map, create_ball_kernel
from gunpowder.ndarray import replace
from gunpowder.graph import GraphKey
from gunpowder.graph_spec import GraphSpec
from gunpowder.roi import Roi

logger = logging.getLogger(__name__)

class RasterizationSettings(Freezable):
    '''Data structure to store parameters for rasterization of graph.

    Args:

        radius (``float`` or ``tuple`` of ``float``):

            The radius (for balls or tubes) or sigma (for peaks) in world units.

        mode (``string``):

            One of ``ball`` or ``peak``. If ``ball`` (the default), a ball with the
            given ``radius`` will be drawn. If ``peak``, the point will be
            rasterized as a peak with values :math:`\exp(-|x-p|^2/\sigma)` with
            sigma set by ``radius``.

        mask (:class:`ArrayKey`, optional):

            Used to mask the rasterization of points. The array is assumed to
            contain discrete labels. The object id at the specific point being
            rasterized is used to intersect the rasterization to keep it inside
            the specific object.

        inner_radius_fraction (``float``, optional):

            Only for mode ``ball``.

            If set, instead of a ball, a hollow sphere is rastered. The radius
            of the whole sphere corresponds to the radius specified with
            ``radius``. This parameter sets the radius of the hollow area, as a
            fraction of ``radius``.

        fg_value (``int``, optional):

            Only for mode ``ball``.

            The value to use to rasterize points, defaults to 1.

        bg_value (``int``, optional):

            Only for mode ``ball``.

            The value to use to for the background in the output array,
            defaults to 0.

        edges (``bool``, optional):

            Whether to rasterize edges by linearly interpolating between Nodes.
            Default is True.

        color_attr (``str``, optional)

            Which graph attribute to use for coloring nodes and edges. One
            useful example might be `component` which would color your graph
            based on the component labels.
            Notes: 
            - Only available in "ball" mode
            - Nodes and Edges missing the attribute will be skipped.
            - color_attr must be populated for nodes and edges upstream of this node
    '''
    def __init__(
            self,
            radius,
            mode='ball',
            mask=None,
            inner_radius_fraction=None,
            fg_value=1,
            bg_value=0,
            edges=True,
            color_attr=None,
            ):

        radius = np.array([radius]).flatten().astype(np.float64)

        if inner_radius_fraction is not None:
            assert (
                inner_radius_fraction > 0.0 and
                inner_radius_fraction < 1.0), (
                    "Inner radius fraction has to be between (excluding) 0 and 1")
            inner_radius_fraction = 1.0 - inner_radius_fraction

        self.radius = radius
        self.mode = mode
        self.mask = mask
        self.inner_radius_fraction = inner_radius_fraction
        self.fg_value = fg_value
        self.bg_value = bg_value
        self.edges = edges
        self.color_attr = color_attr
        self.freeze()


class RasterizeGraph(BatchFilter):
    """Draw graphs into a binary array as balls/tubes of a given radius.

    Args:

        graph (:class:`GraphKey`):
            The key of the graph to rasterize.

        array (:class:`ArrayKey`):
            The key of the binary array to create.

        array_spec (:class:`ArraySpec`, optional):

            The spec of the array to create. Use this to set the datatype and
            voxel size.

        settings (:class:`RasterizationSettings`, optional):
            Which settings to use to rasterize the graph.
    """

    def __init__(self, graph, array, array_spec=None, settings=None):

        self.graph = graph
        self.array = array
        if array_spec is None:
            self.array_spec = ArraySpec()
        else:
            self.array_spec = array_spec
        if settings is None:
            self.settings = RasterizationSettings(1)
        else:
            self.settings = settings

    def setup(self):

        graph_roi = self.spec[self.graph].roi

        if self.array_spec.voxel_size is None:
            self.array_spec.voxel_size = Coordinate((1,)*graph_roi.dims())

        if self.array_spec.dtype is None:
            if self.settings.mode == 'ball':
                self.array_spec.dtype = np.uint8
            else:
                self.array_spec.dtype = np.float32

        self.array_spec.roi = graph_roi.copy()
        self.provides(
            self.array,
            self.array_spec)

        self.enable_autoskip()

    def prepare(self, request):

        if self.settings.mode == 'ball':
            context = np.ceil(self.settings.radius).astype(np.int)
        elif self.settings.mode == 'peak':
            context = np.ceil(2*self.settings.radius).astype(np.int)
        else:
            raise RuntimeError('unknown raster mode %s'%self.settings.mode)

        dims = self.array_spec.roi.dims()
        if len(context) == 1:
            context = context.repeat(dims)

        # request graph in a larger area to get rasterization from outside
        # graph
        graph_roi = request[self.array].roi.grow(
                Coordinate(context),
                Coordinate(context))

        # however, restrict the request to the graph actually provided
        graph_roi = graph_roi.intersect(self.spec[self.graph].roi)

        deps = BatchRequest()
        deps[self.graph] = GraphSpec(roi=graph_roi)

        if self.settings.mask is not None:

            mask_voxel_size = self.spec[self.settings.mask].voxel_size
            assert self.spec[self.array].voxel_size == mask_voxel_size, (
                "Voxel size of mask and rasterized volume need to be equal")

            new_mask_roi = graph_roi.snap_to_grid(mask_voxel_size)
            deps[self.settings.mask] = ArraySpec(roi=new_mask_roi)

        return deps

    def process(self, batch, request):

        graph = batch.graphs[self.graph]
        mask = self.settings.mask
        voxel_size = self.spec[self.array].voxel_size

        # get roi used for creating the new array (graph_roi does not
        # necessarily align with voxel size)
        enlarged_vol_roi = graph.spec.roi.snap_to_grid(voxel_size)
        offset = enlarged_vol_roi.get_begin() / voxel_size
        shape = enlarged_vol_roi.get_shape() / voxel_size
        data_roi = Roi(offset, shape)

        logger.debug("Graph in %s", graph.spec.roi)
        for node in graph.nodes:
            logger.debug("%d, %s", node.id, node.location)
        logger.debug("Data roi in voxels: %s", data_roi)
        logger.debug("Data roi in world units: %s", data_roi*voxel_size)

        if graph.num_vertices == 0:
            # If there are no nodes at all, just create an empty matrix.
            rasterized_graph_data = np.zeros(
                data_roi.get_shape(), dtype=self.spec[self.array].dtype
            )
        elif mask is not None:

            mask_array = batch.arrays[mask].crop(enlarged_vol_roi)
            # get those component labels in the mask, that contain graph
            labels = []
            for i, point in graph.data.items():
                v = Coordinate(point.location / voxel_size)
                v -= data_roi.get_begin()
                labels.append(mask_array.data[v])
            # Make list unique
            labels = list(set(labels))

            # zero label should be ignored
            if 0 in labels:
                labels.remove(0)

            if len(labels) == 0:
                logger.debug("Graph and provided object mask do not overlap. No graph to rasterize.")
                rasterized_graph_data = np.zeros(data_roi.get_shape(),
                                                  dtype=self.spec[self.array].dtype)
            else:
                # create data for the whole graph ROI, "or"ed together over
                # individual object masks
                rasterized_graph_data = np.sum(
                    [
                        self.__rasterize(
                            graph,
                            data_roi,
                            voxel_size,
                            self.spec[self.array].dtype,
                            self.settings,
                            Array(data=mask_array.data==label, spec=mask_array.spec))

                        for label in labels
                    ],
                    axis=0)

        else:

            # create data for the whole graph ROI without mask
            rasterized_graph_data = self.__rasterize(
                graph,
                data_roi,
                voxel_size,
                self.spec[self.array].dtype,
                self.settings)

        # fix bg/fg labelling if requested
        if (self.settings.bg_value != 0 or
            self.settings.fg_value != 1):

            replaced = replace(
                rasterized_graph_data,
                [0, 1],
                [self.settings.bg_value, self.settings.fg_value])
            rasterized_graph_data = replaced.astype(self.spec[self.array].dtype)

        # create array and crop it to requested roi
        spec = self.spec[self.array].copy()
        spec.roi = data_roi*voxel_size
        rasterized_points = Array(
            data=rasterized_graph_data,
            spec=spec)
        batch[self.array] = rasterized_points.crop(request[self.array].roi)

    def __rasterize(self, graph, data_roi, voxel_size, dtype, settings, mask_array=None):
        '''Rasterize 'graph' into an array with the given 'voxel_size'''

        mask = mask_array.data if mask_array is not None else None

        logger.debug("Rasterizing graph in %s", graph.spec.roi)

        # prepare output array
        rasterized_graph = np.zeros(data_roi.get_shape(), dtype=dtype)

        # Fast rasterization currently only implemented for mode ball without
        # inner radius set
        use_fast_rasterization = (
            settings.mode == "ball"
            and settings.inner_radius_fraction is None
            and len(list(graph.edges)) == 0
        )

        if use_fast_rasterization:

            dims = len(rasterized_graph.shape)

            # get structuring element for mode ball
            ball_kernel = create_ball_kernel(settings.radius, voxel_size)
            radius_voxel = Coordinate(np.array(ball_kernel.shape)/2)
            data_roi_base = Roi(
                    offset=Coordinate((0,)*dims),
                    shape=Coordinate(rasterized_graph.shape))
            kernel_roi_base = Roi(
                    offset=Coordinate((0,)*dims),
                    shape=Coordinate(ball_kernel.shape))

        # Rasterize volume either with single voxel or with defined struct elememt
        for node in graph.nodes:

            # get the voxel coordinate, 'Coordinate' ensures integer
            v = Coordinate(node.location/voxel_size)

            # get the voxel coordinate relative to output array start
            v -= data_roi.get_begin()

            # skip graph outside of mask
            if mask is not None and not mask[v]:
                continue

            logger.debug(
                "Rasterizing node %s at %s",
                node.location,
                node.location/voxel_size - data_roi.get_begin())

            if use_fast_rasterization:

                # Calculate where to crop the kernel mask and the rasterized array
                shifted_kernel = kernel_roi_base.shift(v - radius_voxel)
                shifted_data = data_roi_base.shift(-(v - radius_voxel))
                arr_crop = data_roi_base.intersect(shifted_kernel)
                kernel_crop = kernel_roi_base.intersect(shifted_data)
                arr_crop_ind = arr_crop.get_bounding_box()
                kernel_crop_ind = kernel_crop.get_bounding_box()

                rasterized_graph[arr_crop_ind] = np.logical_or(
                    ball_kernel[kernel_crop_ind], rasterized_graph[arr_crop_ind]
                )

            else:

                if settings.color_attr is not None:
                    c = graph.nodes[node].get(settings.color_attr)
                    if c is None:
                        logger.debug(f"Skipping node: {node}")
                        continue
                    elif np.isclose(c, 1) and not np.isclose(settings.fg_value, 1):
                        logger.warning(
                            f"Node {node} is being colored with color {c} according to "
                            f"attribute {settings.color_attr} "
                            f"but color 1 will be replaced with fg_value: {settings.fg_value}"
                            )
                else:
                    c = 1
                rasterized_graph[v] = c
        if settings.edges:
            for e in graph.edges:
                if settings.color_attr is not None:
                    c = graph.edges[e].get(settings.color_attr)
                    if c is None:
                        continue
                    elif np.isclose(c, 1) and not np.isclose(settings.fg_value, 1):
                        logger.warning(
                            f"Edge {e} is being colored with color {c} according to "
                            f"attribute {settings.color_attr} "
                            f"but color 1 will be replaced with fg_value: {settings.fg_value}"
                            )

                u = graph.node(e.u)
                v = graph.node(e.v)
                u_coord = Coordinate(u.location / voxel_size)
                v_coord = Coordinate(v.location / voxel_size)
                line = draw.line_nd(u_coord, v_coord, endpoint=True)
                rasterized_graph[line] = 1

        # grow graph
        if not use_fast_rasterization:

            if settings.mode == "ball":

                enlarge_binary_map(
                    rasterized_graph,
                    settings.radius,
                    voxel_size,
                    settings.inner_radius_fraction,
                    in_place=True)

            else:

                sigmas = settings.radius/voxel_size

                gaussian_filter(
                    rasterized_graph, sigmas, output=rasterized_graph, mode="constant"
                )

                # renormalize to have 1 be the highest value
                max_value = np.max(rasterized_graph)
                if max_value > 0:
                    rasterized_graph /= max_value

        if mask_array is not None:
            # use more efficient bitwise operation when possible
            if settings.mode == "ball":
                rasterized_graph &= mask
            else:
                rasterized_graph *= mask

        return rasterized_graph
