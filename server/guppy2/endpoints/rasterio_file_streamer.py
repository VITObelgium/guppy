import rasterio
from rasterio.windows import Window


# credits: https://stackoverflow.com/questions/49090399/thread-safe-rasterio-writes-from-dask-array-store

class RIOFile(object):
    """Rasterio wrapper to allow da.store to do window saving."""

    def __init__(self, *args, **kwargs):
        """Initialize the object."""
        self.args = args
        self.kwargs = kwargs
        self.rfile = None

    def __setitem__(self, key, item):
        """Put the data chunk in the image."""
        if len(key) == 3:
            indexes = list(range(1, item.shape[0] + 1))  # fix for rgba band rasters. always use as many bands as there are arrays in the input
            y = key[1]
            x = key[2]
        else:
            indexes = 1
            y = key[0]
            x = key[1]
        chy_off = y.start
        chy = y.stop - y.start
        chx_off = x.start
        chx = x.stop - x.start
        # band indexes
        self.rfile.write(item, window=Window(chx_off, chy_off, chx, chy),
                         indexes=indexes)

    def __enter__(self):
        """Enter method."""
        self.rfile = rasterio.open(*self.args, **self.kwargs)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Exit method."""
        self.rfile.close()
