import logging
import numpy as np
import os

from .batch_filter import BatchFilter
from gunpowder.batch_request import BatchRequest
from gunpowder.ext import h5py
from gunpowder.ext import ZarrFile

logger = logging.getLogger(__name__)


class Snapshot(BatchFilter):
    """Save a passing batch in an HDF or Zarr file.

    The default behaviour is to periodically save a snapshot after
    ``every`` iterations.

    Data-dependent criteria for saving can be implemented by subclassing and
    overwriting :func:`write_if`. This method is applied as an additional
    filter to the batches picked for periodic saving. It should return ``True``
    if a batch meets the criteria for saving.

    Args:

        dataset_names (``dict``, :class:`ArrayKey` -> ``string``):

            A dictionary from array keys to names of the datasets to store them
            in.

        output_dir (``string``):

            The directory to save the snapshots. Will be created, if it does
            not exist.

        output_filename (``string``):

            Template for output filenames. ``{id}`` in the string will be
            replaced with the ID of the batch. ``{iteration}`` with the training
            iteration (if training was performed on this batch). Snapshot will
            be saved as zarr file if output_filename ends in ``.zarr`` and as
            HDF otherwise.

        every (``int``):

            How often to save a batch. ``every=1`` indicates that every batch
            will be stored, ``every=2`` every second and so on. By default,
            every batch will be stored.

        additional_request (:class:`BatchRequest`):

            An additional batch request to merge with the passing request, if a
            snapshot is to be made. If not given, only the arrays that are in
            the batch anyway are recorded. This is useful to request additional
            arrays like loss gradients for visualization that are otherwise not
            needed.

        compression_type (``string`` or ``int``):

            Compression strategy.  Legal values are ``gzip``, ``szip``,
            ``lzf``. If an integer between 1 and 10, this indicates ``gzip``
            compression level.

        dataset_dtypes (``dict``, :class:`ArrayKey` -> data type):

            A dictionary from array keys to datatype (eg. ``np.int8``). If
            given, arrays are stored using this type. The original arrays
            within the pipeline remain unchanged.

        store_value_range (``bool``):

            If set to ``True``, store range of values in data set attributes.
    """

    def __init__(
        self,
        dataset_names,
        output_dir="snapshots",
        output_filename="{id}.zarr",
        every=1,
        additional_request=None,
        compression_type=None,
        dataset_dtypes=None,
        store_value_range=False,
    ):
        self.dataset_names = dataset_names
        self.output_dir = output_dir
        self.output_filename = output_filename
        self.every = max(1, every)
        self.additional_request = (
            BatchRequest() if additional_request is None else additional_request
        )
        self.n = 0
        self.compression_type = compression_type
        self.store_value_range = store_value_range
        if dataset_dtypes is None:
            self.dataset_dtypes = {}
        else:
            self.dataset_dtypes = dataset_dtypes

        self.mode = "w"

    def write_if(self, batch):
        """To be implemented in subclasses.

        This function is run in :func:`process` and acts as a data-dependent
        filter for saving snapshots.

        Args:

            batch (:class:`Batch`):

                The batch received from upstream.

        Returns:

            ``True`` if ``batch`` should be written to snapshot, ``False``
            otherwise.
        """

        return True

    def setup(self):
        for key, _ in self.additional_request.items():
            assert key in self.dataset_names, (
                "%s requested but not in dataset_names" % key
            )

        for array_key in self.additional_request.array_specs.keys():
            spec = self.spec[array_key]
            self.updates(array_key, spec)
        for graph_key in self.additional_request.graph_specs.keys():
            spec = self.spec[graph_key]
            self.updates(graph_key, spec)

    def prepare(self, request):
        deps = BatchRequest()
        for key, spec in request.items():
            if key in self.dataset_names:
                deps[key] = spec

        self.record_snapshot = self.n % self.every == 0

        if self.record_snapshot:
            # append additional array requests, don't overwrite existing ones
            for array_key, spec in self.additional_request.array_specs.items():
                if array_key not in deps:
                    deps[array_key] = spec
            for graph_key, spec in self.additional_request.graph_specs.items():
                if graph_key not in deps:
                    deps[graph_key] = spec

            for key in self.dataset_names.keys():
                assert key in deps, "%s wanted for %s, but not in request." % (
                    key,
                    self.name(),
                )

        return deps

    def process(self, batch, request):
        if self.record_snapshot and self.write_if(batch):
            try:
                os.makedirs(self.output_dir)
            except:
                pass

            snapshot_name = os.path.join(
                self.output_dir,
                self.output_filename.format(
                    id=str(batch.id).zfill(8), iteration=int(batch.iteration or 0)
                ),
            )
            logger.info("saving to %s" % snapshot_name)
            if snapshot_name.endswith(".hdf"):
                open_func = h5py.File
            elif snapshot_name.endswith(".zarr"):
                open_func = ZarrFile
            else:
                logger.warning("ambiguous file type")
                open_func = h5py.File

            with open_func(snapshot_name, self.mode) as f:
                for array_key, array in batch.arrays.items():
                    if array_key not in self.dataset_names:
                        continue

                    ds_name = self.dataset_names[array_key]

                    if array_key in self.dataset_dtypes:
                        dtype = self.dataset_dtypes[array_key]
                        dataset = f.create_dataset(
                            name=ds_name,
                            data=array.data.astype(dtype),
                            compression=self.compression_type,
                        )

                    else:
                        dataset = f.create_dataset(
                            name=ds_name,
                            data=array.data,
                            compression=self.compression_type,
                        )

                    if not array.spec.nonspatial:
                        if array.spec.roi is not None:
                            dataset.attrs["offset"] = array.spec.roi.offset
                        dataset.attrs["resolution"] = self.spec[array_key].voxel_size

                    if self.store_value_range:
                        dataset.attrs["value_range"] = (
                            array.data.min().item(),
                            array.data.max().item(),
                        )

                    # if array has attributes, add them to the dataset
                    for attribute_name, attribute in array.attrs.items():
                        dataset.attrs[attribute_name] = attribute

                for graph_key, graph in batch.graphs.items():
                    if graph_key not in self.dataset_names:
                        continue

                    ds_name = self.dataset_names[graph_key]

                    node_ids = []
                    locations = []
                    edges = []
                    for node in graph.nodes:
                        node_ids.append(node.id)
                        locations.append(node.location)
                    for edge in graph.edges:
                        edges.append((edge.u, edge.v))

                    f.create_dataset(
                        name=f"{ds_name}-ids",
                        data=np.array(node_ids, dtype=int),
                        compression=self.compression_type,
                    )
                    f.create_dataset(
                        name=f"{ds_name}-locations",
                        data=np.array(locations),
                        compression=self.compression_type,
                    )
                    f.create_dataset(
                        name=f"{ds_name}-edges",
                        data=np.array(edges),
                        compression=self.compression_type,
                    )

                if batch.loss is not None:
                    f["/"].attrs["loss"] = float(batch.loss)

        self.n += 1
