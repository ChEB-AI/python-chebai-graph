import importlib

import chebai_graph.preprocessing.properties as graph_properties
from chebai_graph.preprocessing.properties import MolecularProperty


def resolve_property(property: str | MolecularProperty) -> MolecularProperty:
    """
    Resolves a molecular property specification (either as a class instance or class path string)
    into a MolecularProperty instance.

    This utility is designed to support flexible specification of molecular properties
    in dataset configurations. It handles:
    - Direct instances of MolecularProperty
    - Full class paths as strings (e.g., "my_module.MyProperty")
    - Shorthand class names assumed to exist in chebai_graph.preprocessing.properties

    Args:
        property (str | MolecularProperty): The property to resolve. Can be a class instance,
            a fully qualified class name (e.g. "module.ClassName"), or a class name assumed
            to be in `chebai_graph.preprocessing.properties`.

    Returns:
        MolecularProperty: An instance of the resolved MolecularProperty.

    Raises:
        AttributeError: If the class name cannot be found in the target module.
        ModuleNotFoundError: If the module cannot be imported.
        TypeError: If the resolved object is not a MolecularProperty instance.
    """
    # if property is given as a string, try to resolve as a class path
    if isinstance(property, MolecularProperty):
        return property
    try:
        # split class_path into module-part and class name
        last_dot = property.rindex(".")
        module_name = property[:last_dot]
        class_name = property[last_dot + 1 :]
        module = importlib.import_module(module_name)
        return getattr(module, class_name)()
    except ValueError:
        # if only a class name is given, assume the module is chebai_graph.processing.properties
        return getattr(graph_properties, property)()
