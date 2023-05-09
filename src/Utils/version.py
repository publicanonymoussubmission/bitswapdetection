import importlib.util
import sys


def check_if_a_module_exists(module_name: str) -> bool:
    """
    This function checks if a module can be loaded

    Args:
        module_name: name of the module to test
    """
    if module_name in sys.modules:
        return True
    elif (importlib.util.find_spec(module_name)) is not None:
        return True
    else:
        return False
