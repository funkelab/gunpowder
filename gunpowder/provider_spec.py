from .freezable import Freezable

class ProviderSpec(Freezable):
    '''A collection of (possibly partial) :class:`VolumeSpec`s and 
    :class:`PointsSpec`s describing a :class:`BatchProvider`'s offered volumes 
    and points.

    Args:

        volume_specs (dict): A dictionary from :class:`VolumeType` to :class:`VolumeSpec`.

        points_specs (dict): A dictionary from :class:`PointsType` to :class:`PointsSpec`.
    '''

    def __init__(self, volume_specs=None, points_specs=None):

        if volume_specs is None:
            self.volume_specs = {}
        else:
            self.volume_specs = volume_specs

        if points_specs is None:
            self.points_specs = {}
        else:
            self.points_specs = points_specs

        self.freeze()

    def get_total_roi(self):
        '''Get the union of all the ROIs.'''

        total_roi = None
        for specs_type in [self.volume_specs, self.points_specs]:
            for (type, spec) in specs_type.items():
                if total_roi is None:
                    total_roi = spec.roi
                else:
                    total_roi = total_roi.union(spec.roi)
            return total_roi

    def get_common_roi(self):
        ''' Get the intersection of all the requested ROIs.'''

        common_roi = None
        for specs_type in [self.volume_specs, self.points_specs]:
            for (type, spec) in specs_type.items():
                if common_roi is None:
                    common_roi = spec.roi
                else:
                    common_roi = common_roi.intersect(spec.roi)

        return common_roi

    def __eq__(self, other):

        if isinstance(other, self.__class__):
            return self.__dict__ == other.__dict__
        return NotImplemented

    def __ne__(self, other):

        if isinstance(other, self.__class__):
            return not self.__eq__(other)
        return NotImplemented

    def __repr__(self):

        r = ""
        for specs_type in [self.volume_specs, self.points_specs]:
            for (type, spec) in specs_type.items():
                r += "%s: %s\n"%(type, spec)
        return r
