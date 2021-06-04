from tensorflow.keras import models as keras_models
from copy import copy
import re
from importlib import import_module
import numpy as np
import pandas as pd
from .custom import PeriodicPadding3D
# ============================================
# General utility functions
# ============================================

def get_object(module_class):
    """
    Given a string woth a module class name, it imports ans returns the class.

    """
    # Split the path into its parts
    parts = module_class.split('.')
    # Get the top level module
    module = parts[0]
    # Import the top level module
    mod = __import__(module)
    # Recursively work down from the top level module to the class name.
    # Be prepared to catch an exception if something cannot be found.
    try:
        for part in parts:
            module = '.'.join([module, part])
            # Import each successive module
            __import__(module)
            mod = getattr(mod, part)
    except ImportError as e:
        # Can't find a recursive module. Give a more informative error message:
        raise ImportError("'%s' raised when searching for %s" % (str(e), module))
    except AttributeError:
        # Can't fint the last attribute. Give a more informative error message:
        raise AttributeError("Module: '%s' has no attribute '%s' when searching for '%s'" % (mod.__name__, part, module_class))
    
    return mod

def get_from_class(module_name, class_name):
    """
    Given a module name and class name, return an object corresponding to the class retrieved as in
    `from module_class import class_name`

    :param module_name: str: name of module (may have . attributes)
    :param class_name: str: name of class
    :return: object pointer to class
    """
    mod = __import__(module_name, fromlist=[class_name])
    class_obj = getattr(mod, class_name)
    return class_obj


def get_classes(module_name):
    """
    From a given module name, return a directory {class_name: class_object} of its classes.

    :param module name: str: name of module to import
    :return: dict: {class_name, class_object} pairs in the module
    """
    module = import_module(module_name)
    classes = {}
    for key in dir(module):
        if isinstance(getattr(module, key), type):
            classes[key] = get_from_class(module_name, key)
    return classes


def get_methods(module_name):
    """
    From a given module name, return a dictionary {method_name: method_object} of its methods

    :param module_name: str: name of module to import
    :return: dict: {method_name: method_object} pairs in the module
    """
    module = import_module(module_name)
    methods = {}
    for key in dir(module):
        if callable(getattr(module, key)):
            methods[key] = get_from_class(module_name, key)
    return methods


def custom_load_model(file_path):
    custom_objects = {
        'PeriodicPadding3D': PeriodicPadding3D
    }
    # custom_objects.update(get_classes('.custom'))
    # custom_objects.update(get_methods('.custom'))
    loaded_model = keras_models.load_model(file_path, custom_objects=custom_objects)
    return loaded_model