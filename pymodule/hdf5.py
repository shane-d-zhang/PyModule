"""Provide custom functions to HDF5 file."""
import h5py


class MyH5(h5py.File):

    @staticmethod
    def new_grp(parent, name):
        """
        Create a group in parent considering it already exists.
        """
        if name in parent:
            return parent[name]
        else:
            return parent.create_group(name)

    @staticmethod
    def new_dset(grp, name, data, ow=False, **kwargs):
        """
        Create a dataset in group allowing overwriting.
        """
        if name in grp:
            if ow:
                del grp[name]
            else:
                return grp[name]

        grp.create_dataset(name, data=data, **kwargs)

        return grp[name]

    @staticmethod
    def dict2attr(obj, d):
        """
        :param obj: dataset or group
        """
        for k, v in d.items():
            obj.attrs[k] = v

        return obj
