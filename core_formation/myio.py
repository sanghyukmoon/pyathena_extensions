import h5py
import numpy as np
import xarray as xr
import dask.array as da

def read_sparse_hdf5(filename, chunks=(128, 128, 128)):
    """Read Athena++ hdf5 file and convert it to dask-xarray Dataset

    This function assumes that the "sparse hdf5" file contains a dataset
    named "gids", which is a list of the selected MeshBlock id.
    The usual Athena++ HDF5 file structure, [var, MeshBlock, z, y, x] remains
    the same; however, the number of MeshBlocks (the length of the second dim)
    in the "sparse hdf5" is not equal to the total number of blocks in the
    simulation domain.

    Parameters
    ----------
    filename : str
        Data filename
    chunks : tuple of int, optional
        Dask chunk size along (x, y, z) directions. Default is (512, 512, 512).

    See Also
    --------
    pyathena.io.read_hdf5.read_hdf5_dask : Read Athena++ HDF5 file and convert it to xarray Dataset
    """
    f = h5py.File(filename, 'r')

    if 'gids' not in f:
        raise ValueError("Sparse HDF5 file must contain 'gids' dataset")

    # Read Mesh information
    block_size = f.attrs['MeshBlockSize']
    mesh_size = f.attrs['RootGridSize']
    num_blocks = mesh_size // block_size  # Assuming uniform grid

    if num_blocks.prod() != f.attrs['NumMeshBlocks']:
        raise ValueError("Number of blocks does not match the attribute")

    # Array of logical locations, arranged by Z-ordering
    # (lx1, lx2, lx3)
    # (  0,   0,   0)
    # (  1,   0,   0)
    # (  0,   1,   0)
    # ...
    logical_loc = f['LogicalLocations']

    # Number of MeshBlocks per chunk along each dimension.
    nblock_per_chunk = np.array(chunks) // block_size
    chunksize_read = (1, nblock_per_chunk.prod(), *block_size)

    # lazy load from HDF5
    ds = []
    shape = set()
    for dsetname in f.attrs['DatasetNames']:
        darr = da.from_array(f[dsetname], chunks=chunksize_read)
        if len(darr.shape) != 5:
            # Expected shape: (nvar, nblock, z, y, x)
            raise ValueError("Invalid shape of the dataset")
        shape.add(darr.shape[2:])
        ds += [var for var in darr]
    if len(shape) == 1:
        shape = shape.pop()
    else:
        raise ValueError("Inconsistent shape of the dataset")

    def _reorder_rechunk(var):
        """
        Loop over the MeshBlocks and place them to correct logical locations
        in 3D Cartesian space. Then, merge them into a single dask array.
        """
        reordered = np.empty(num_blocks[::-1], dtype=object)
        foo = da.zeros(shape, chunks=shape, dtype=np.int8)
        for gid in range(num_blocks.prod()):
            lx1, lx2, lx3 = logical_loc[gid]  # Correct Cartesian coordinates
            if gid in f['gids']:
                i = np.where(np.array(f['gids']) == gid)[0][0]
                reordered[lx3, lx2, lx1] = var[i, ...]  # Assign the correct block
            else:
                reordered[lx3, lx2, lx1] = foo
        # Merge into a single array
        return da.block(reordered.tolist()).rechunk(chunks)
    # Apply the rechunking function to all variables
    ds = list(map(_reorder_rechunk, ds))

    # Convert to xarray object
    varnames = list(map(lambda x: x.decode('ASCII'), f.attrs['VariableNames']))
    variables = [(['z', 'y', 'x'], d) for d in ds]
    coordnames = ['x1v', 'x2v', 'x3v']

    # Calculate coordinates
    # Borrowed and slightly modified from pyathena.io.athdf
    coords = {}
    for i, (nrbx, xv) in enumerate(zip(num_blocks, coordnames)):
        coords[xv] = np.empty(mesh_size[i])
        for n_block in range(nrbx):
            sample_block = np.where(logical_loc[:, i] == n_block)[0][0]
            index_low = n_block * block_size[i]
            index_high = index_low + block_size[i]
            coords[xv][index_low:index_high] = f[xv][sample_block, :]

    # If uniform grid, store cell spacing.
    attrs = dict(f.attrs)
    attrs['dx1'] = np.diff(f.attrs['RootGridX1'])[0] / mesh_size[0]
    attrs['dx2'] = np.diff(f.attrs['RootGridX2'])[0] / mesh_size[1]
    attrs['dx3'] = np.diff(f.attrs['RootGridX3'])[0] / mesh_size[2]

    ds = xr.Dataset(
        data_vars=dict(zip(varnames, variables)),
        coords=dict(x=coords['x1v'], y=coords['x2v'], z=coords['x3v']),
        attrs=attrs
    )

    return ds
