from gunpowder.points import Point

class PreSynPoint(Point):
    def __init__(self, location, location_id, synapse_id, partner_ids, props=None):
        """ Presynaptic locations
        :param location:     ndarray, [zyx]
        :param location_id:  int, unique for every synapse location across pre and postsynaptic locations 
        :param synapse_id:   int, unique for every synapse(synaptic partners have the same synapse_id, but different location_ids)
        :param partner_ids:  list of ints, location ids of postsynaptic partners
        :param props:        dict, properties
        """
        Point.__init__(self, location=location)
        self.thaw()

        self.location_id  = location_id
        self.synapse_id   = synapse_id
        self.partner_ids  = partner_ids
        if props is None:
            self.props = {}
        else:
            self.props = props

        self.freeze()

class PostSynPoint(Point):
    def __init__(self, location, location_id, synapse_id, partner_ids, props=None):
        """
        :param location:     ndarray, [zyx]
        :param location_id:  int, unique for every synapse location across pre and postsynaptic locations 
        :param synapse_id:   int, unique for every synapse(synaptic partners have the same synapse_id, but different location_ids)
        :param partner_ids:  list of int, location id of presynaptic partner
        :param props:        dict, properties
        """
        Point.__init__(self, location=location)
        self.thaw()

        self.location_id  = location_id
        self.synapse_id   = synapse_id
        self.partner_ids  = partner_ids
        if props is None:
            self.props = {}
        else:
            self.props = props

        self.freeze()
