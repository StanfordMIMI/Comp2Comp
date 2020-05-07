import importlib
import importlib.util
import os
import sys

__all__ = []


# from https://stackoverflow.com/questions/67631/how-to-import-a-module-given-the-full-path  # noqa
def _import_file(module_name, file_path, make_importable=False):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if make_importable:
        sys.modules[module_name] = module
    return module


def _configure_libraries():
    """
    Configurations for some libraries.
    """
    # An environment option to disable `import cv2` globally,
    # in case it leads to negative performance impact
    disable_cv2 = int(os.environ.get("MEDSEGPY_DISABLE_CV2", False))
    if disable_cv2:
        sys.modules["cv2"] = None
    else:
        # Disable opencl in opencv since its interaction with cuda often
        # has negative effects
        # This envvar is supported after OpenCV 3.4.0
        os.environ["OPENCV_OPENCL_RUNTIME"] = "disabled"
        try:
            import cv2

            if int(cv2.__version__.split(".")[0]) >= 3:
                cv2.ocl.setUseOpenCL(False)
        except ImportError:
            pass


_ENV_SETUP_DONE = False


def setup_environment():
    """Perform environment setup work. The default setup is a no-op, but this
    function allows the user to specify a Python source file or a module in
    the $MEDSEGPY_ENV_MODULE environment variable, that performs
    custom setup work that may be necessary to their computing environment.
    """
    global _ENV_SETUP_DONE
    if _ENV_SETUP_DONE:
        return
    _ENV_SETUP_DONE = True

    _configure_libraries()

    custom_module_path = os.environ.get("MEDSEGPY_ENV_MODULE")

    if custom_module_path:
        setup_custom_environment(custom_module_path)
    else:
        # The default setup is a no-op
        pass


def setup_custom_environment(custom_module):
    """
    Load custom environment setup by importing a Python source file or a
    module, and run the setup function.
    """
    if custom_module.endswith(".py"):
        module = _import_file("medsegpy.utils.env.custom_module", custom_module)
    else:
        module = importlib.import_module(custom_module)
    assert hasattr(module, "setup_environment") and callable(
        module.setup_environment
    ), (
        "Custom environment module defined in {} does not have the "
        "required callable attribute 'setup_environment'."
    ).format(
        custom_module
    )
    module.setup_environment()
