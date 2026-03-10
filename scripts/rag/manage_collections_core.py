"""Compatibility facade for split collection management core modules."""

from scripts.rag import manage_collections_core_collection as _collection
from scripts.rag import manage_collections_core_evaluation as _evaluation
from scripts.rag import manage_collections_core_metrics as _metrics
from scripts.rag import manage_collections_core_types as _types

_MODULES = (_types, _collection, _metrics, _evaluation)
_EXPORT_NAMES = tuple(
    dict.fromkeys([
        *_types.__all__,
        *_collection.__all__,
        *_metrics.__all__,
        *_evaluation.__all__,
    ])
)

for _name in _EXPORT_NAMES:
    for _module in _MODULES:
        if hasattr(_module, _name):
            globals()[_name] = getattr(_module, _name)
            break

del _module
del _name
del _MODULES
