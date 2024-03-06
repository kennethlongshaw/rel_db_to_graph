from dataclasses import dataclass, field, make_dataclass
import inspect
from typing import Any


def create_config_from_class(cls):
    field_definitions = []
    annotations = {}
    field_docs = {}  # Dictionary to hold field documentation

    # Optionally, create a docstring for the config class itself.
    class_doc = f"Configuration for {cls.__name__}\n\n"

    for name, param in inspect.signature(cls.__init__).parameters.items():
        if name == "self" or name == "kwargs":
            continue

        type_hint = param.annotation if param.annotation is not inspect.Parameter.empty else Any
        annotations[name] = type_hint

        if param.default is inspect.Parameter.empty:
            field_definitions.append((name, type_hint))
            field_docs[name] = f"{name}: {type_hint.__name__}"
        else:
            field_definitions.append((name, type_hint, field(default=param.default)))
            field_docs[name] = f"{name} (default={param.default}): {type_hint.__name__}"

        # Append field documentation to the class docstring.
        class_doc += f":param {name}: {param.annotation.__name__ if param.annotation is not inspect.Parameter.empty else 'Any'}\n"
        if param.default is not inspect.Parameter.empty:
            class_doc += f"    Default: {param.default}\n"

    # Use make_dataclass to dynamically create the dataclass
    config_class = make_dataclass(f"{cls.__name__}Config",
                                  fields=field_definitions,
                                  bases=(object,),
                                  namespace={'__annotations__': annotations})

    # Assign the composed docstring to the class.
    config_class.__doc__ = class_doc.strip()

    return config_class
