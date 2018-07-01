import numpy as np
import logging
from gunpowder.batch import Batch
from gunpowder.coordinate import Coordinate
from gunpowder.nodes.batch_provider import BatchProvider
from gunpowder.points import Point, Points
from gunpowder.points_spec import PointsSpec
from gunpowder.profiling import Timing
from gunpowder.roi import Roi

logger = logging.getLogger(__name__)

class CsvPointsSource(BatchProvider):
    '''Read a set of points from a comma-separated-values text file. Each line
    in the file represents one point.

    Args:

        filename (``string``):

            The file to read from.

        points (:class:`PointsKey`):

            The key of the points set to create.

        points_spec (:class:`PointsSpec`, optional):

            An optional :class:`PointsSpec` to overwrite the points specs
            automatically determined from the CSV file. This is useful to set
            the :class:`Roi` manually.

        scale (scalar or array-like):

            An optional scaling to apply to the coordinates of the points read
            from the CSV file. This is useful if the points refer to voxel
            positions to convert them to world units.
    '''

    def __init__(self, filename, points, points_spec=None, scale=None):

        self.filename = filename
        self.points = points
        self.points_spec = points_spec
        self.scale = scale
        self.ndims = None
        self.data = None

    def setup(self):

        self._read_points()

        if self.points_spec is not None:

            self.provides(self.points, self.points_spec)
            return

        min_bb = Coordinate(np.floor(np.amin(self.data[:,:self.ndims], 0)))
        max_bb = Coordinate(np.ceil(np.amax(self.data[:,:self.ndims], 0)) + 1)

        roi = Roi(min_bb, max_bb - min_bb)

        self.provides(self.points, PointsSpec(roi=roi))

    def provide(self, request):

        timing = Timing(self)
        timing.start()

        min_bb = request[self.points].roi.get_begin()
        max_bb = request[self.points].roi.get_end()

        logger.debug(
            "CSV points source got request for %s",
            request[self.points].roi)

        point_filter = np.ones((self.data.shape[0],), dtype=np.bool)
        for d in range(self.ndims):
            point_filter = np.logical_and(point_filter, self.data[:,d] >= min_bb[d])
            point_filter = np.logical_and(point_filter, self.data[:,d] < max_bb[d])

        points_data = self._get_points(point_filter)
        points_spec = PointsSpec(roi=request[self.points].roi.copy())

        batch = Batch()
        batch.points[self.points] = Points(points_data, points_spec)

        timing.stop()
        batch.profiling_stats.add(timing)

        return batch

    def _get_points(self, point_filter):

        filtered = self.data[point_filter]
        ids = np.arange(len(self.data))[point_filter]

        return {
            i: Point(p)
            for i, p in zip(ids, filtered)
        }

    def _read_points(self):
        self.data, self.ndims = self._parse_csv()

    def _parse_csv(self, ndims=0):
        '''Read one point per line. If ``ndims`` is 0, all values in one line
        are considered as the location of the point. If positive, only the
        first ``ndims`` are used. If negative, all but the last ``-ndims`` are
        used.
        '''

        points = np.array(
            [
                [ float(t.strip(',')) for t in line.split() ]
                for line in open(self.filename, 'r')
            ], dtype=np.float32)

        if ndims == 0:
            ndims = points.shape[1]
        elif ndims < 0:
            ndims = points.shape[1] + ndims

        if self.scale is not None:
            points[:,:ndims] *= self.scale

        return points, ndims
