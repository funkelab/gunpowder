import csv
import logging
from typing import Optional, Union

import numpy as np

from gunpowder.batch import Batch
from gunpowder.coordinate import Coordinate
from gunpowder.graph import Graph, GraphKey, Node
from gunpowder.graph_spec import GraphSpec
from gunpowder.nodes.batch_provider import BatchProvider
from gunpowder.profiling import Timing
from gunpowder.roi import Roi

logger = logging.getLogger(__name__)


class CsvPointsSource(BatchProvider):
    """Read a set of points from a comma-separated-values text file. Each line
    in the file represents one point, e.g. z y x (id). Note: this reads all
    points into memory and finds the ones in the given roi by iterating
    over all the points. For large datasets, this may be too slow.

    Args:

        filename (``string``):

            The file to read from.

        points (:class:`GraphKey`):

            The key of the points set to create.

        spatial_cols (list[``int``]):

            The columns of the csv that hold the coordinates of the points
            (in the order that you want them to be used in training)

        points_spec (:class:`GraphSpec`, optional):

            An optional :class:`GraphSpec` to overwrite the points specs
            automatically determined from the CSV file. This is useful to set
            the :class:`Roi` manually.

        scale (scalar or array-like):

            An optional scaling to apply to the coordinates of the points read
            from the CSV file. This is useful if the points refer to voxel
            positions to convert them to world units.

        id_col (``int``, optional):

            The column of the csv that holds an id for each point. If not
            provided, the index of the rows are used as the ids. When read
            from file, ids are left as strings and not cast to anything.

        delimiter (``str``, optional):

            Delimiter to pass to the csv reader. Defaults to ",".
    """

    def __init__(
        self,
        filename: str,
        points: GraphKey,
        spatial_cols: list[int],
        points_spec: Optional[GraphSpec] = None,
        scale: Optional[Union[int, float, tuple, list, np.ndarray]] = None,
        id_col: Optional[int] = None,
        delimiter: str = ",",
    ):
        self.filename = filename
        self.points = points
        self.points_spec = points_spec
        self.scale = scale
        self.spatial_cols = spatial_cols
        self.id_dim = id_col
        self.delimiter = delimiter
        self.data: Optional[np.ndarray] = None
        self.ids: Optional[list] = None

    def setup(self):
        self._parse_csv()

        if self.points_spec is not None:
            self.provides(self.points, self.points_spec)
            return

        min_bb = Coordinate(np.floor(np.amin(self.data, 0)))
        max_bb = Coordinate(np.ceil(np.amax(self.data, 0)) + 1)

        roi = Roi(min_bb, max_bb - min_bb)

        self.provides(self.points, GraphSpec(roi=roi))

    def provide(self, request):
        timing = Timing(self)
        timing.start()

        min_bb = request[self.points].roi.begin
        max_bb = request[self.points].roi.end

        logger.debug("CSV points source got request for %s", request[self.points].roi)

        point_filter = np.ones((self.data.shape[0],), dtype=bool)
        for d in range(len(self.spatial_cols)):
            point_filter = np.logical_and(point_filter, self.data[:, d] >= min_bb[d])
            point_filter = np.logical_and(point_filter, self.data[:, d] < max_bb[d])

        points_data = self._get_points(point_filter)
        points_spec = GraphSpec(roi=request[self.points].roi.copy())

        batch = Batch()
        batch.graphs[self.points] = Graph(points_data, [], points_spec)

        timing.stop()
        batch.profiling_stats.add(timing)

        return batch

    def _get_points(self, point_filter):
        filtered = self.data[point_filter]
        ids = self.ids[point_filter]
        return [Node(id=i, location=p) for i, p in zip(ids, filtered)]

    def _parse_csv(self):
        """Read one point per line, with spatial and id columns determined by
        self.spatial_cols and self.id_col.
        """
        data = []
        ids = []
        with open(self.filename, "r", newline="") as f:
            has_header = csv.Sniffer().has_header(f.read(1024))
            f.seek(0)
            first_line = True
            reader = csv.reader(f, delimiter=self.delimiter)
            for line in reader:
                if first_line and has_header:
                    first_line = False
                    continue
                space = [float(line[c]) for c in self.spatial_cols]
                data.append(space)
                if self.id_dim is not None:
                    ids.append(line[self.id_dim])

        self.data = np.array(data, dtype=np.float32)
        if self.id_dim:
            self.ids = np.array(ids)
        else:
            self.ids = np.arange(len(self.data))

        if self.scale is not None:
            self.data *= self.scale
