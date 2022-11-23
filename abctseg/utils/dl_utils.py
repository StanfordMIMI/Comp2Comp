import subprocess

from keras import Model
# from keras.utils import multi_gpu_model
# from tensorflow.python.keras.utils.multi_gpu_utils import multi_gpu_model


def get_available_gpus(num_gpus: int = None):
    """Get gpu ids for gpus that are >95% free.

    Tensorflow does not support checking free memory on gpus.
    This is a crude method that relies on `nvidia-smi` to
    determine which gpus are occupied and which are free.

    Args:
        num_gpus: Number of requested gpus. If not specified,
            ids of all available gpu(s) are returned.

    Returns:
        List[int]: List of gpu ids that are free. Length
            will equal `num_gpus`, if specified.
    """
    # Built-in tensorflow gpu id.
    assert isinstance(num_gpus, (type(None), int))
    if num_gpus == 0:
        return [-1]

    num_requested_gpus = num_gpus
    try:
        num_gpus = (
            len(
                subprocess.check_output("nvidia-smi --list-gpus", shell=True)
                .decode()
                .split("\n")
            )
            - 1
        )

        out_str = subprocess.check_output(
            "nvidia-smi | grep MiB", shell=True
        ).decode()
    except subprocess.CalledProcessError:
        return None
    mem_str = [x for x in out_str.split() if "MiB" in x]
    # First 2 * num_gpu elements correspond to memory for gpus
    # Order: (occupied-0, total-0, occupied-1, total-1, ...)
    mems = [float(x[:-3]) for x in mem_str]
    gpu_percent_occupied_mem = [
        mems[2 * gpu_id] / mems[2 * gpu_id + 1] for gpu_id in range(num_gpus)
    ]

    available_gpus = [
        gpu_id
        for gpu_id, mem in enumerate(gpu_percent_occupied_mem)
        if mem < 0.05
    ]
    if num_requested_gpus and num_requested_gpus > len(available_gpus):
        raise ValueError(
            "Requested {} gpus, only {} are free".format(
                num_requested_gpus, len(available_gpus)
            )
        )

    return (
        available_gpus[:num_requested_gpus]
        if num_requested_gpus
        else available_gpus
    )


class ModelMGPU(Model):
    """Wrapper for distributing model across multiple gpus"""

    def __init__(self, ser_model, gpus):
        pmodel = multi_gpu_model(ser_model, gpus)
        self.__dict__.update(pmodel.__dict__)
        self._smodel = ser_model

    def __getattribute__(self, attrname):
        """Override load and save methods to be used from the serial-model. The
        serial-model holds references to the weights in the multi-gpu model.
        """
        # return Model.__getattribute__(self, attrname)
        if "load" in attrname or "save" in attrname:
            return getattr(self._smodel, attrname)

        return super(ModelMGPU, self).__getattribute__(attrname)
