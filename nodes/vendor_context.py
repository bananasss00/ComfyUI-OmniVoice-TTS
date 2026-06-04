import os
import sys
import json
from contextlib import contextmanager

_ORIGINAL_JSON_DEFAULT = json.JSONEncoder.default

def _patched_json_default(self, object_to_serialize):
    """Safely serialize nested transformer configs like Qwen3Config."""
    if hasattr(object_to_serialize, "to_dict"):
        return object_to_serialize.to_dict()
    try:
        return _ORIGINAL_JSON_DEFAULT(self, object_to_serialize)
    except TypeError:
        if hasattr(object_to_serialize, "__dict__"):
            return object_to_serialize.__dict__
        return str(object_to_serialize)

# Global cache to keep vendored modules alive in memory.
_VENDORED_MODULES = {}

@contextmanager
def vendored_transformers():
    """Temporarily swaps the system modules to load isolated transformers 5.x."""
    current_directory = os.path.dirname(os.path.abspath(__file__))
    root_directory = os.path.dirname(current_directory)
    vendor_path = os.path.join(root_directory, "vendor")

    # Do NOT include "tokenizers" here. 
    # Rust extensions (PyO3) will crash if unloaded and reloaded.
    targets = ["transformers", "huggingface_hub"]
    
    saved_modules = {}
    for key in list(sys.modules):
        if any(key == target or key.startswith(target + ".") for target in targets):
            saved_modules[key] = sys.modules.pop(key)
            
    for key, module in _VENDORED_MODULES.items():
        sys.modules[key] = module
    
    sys.path.insert(0, vendor_path)
    
    # 1. Apply JSON patch to prevent Qwen3Config serialization crash
    json.JSONEncoder.default = _patched_json_default
    
    # 2. Patch huggingface_hub strict type validator to bypass module-swap isinstance() failures
    _orig_type_validator = None
    try:
        import huggingface_hub.dataclasses
        if hasattr(huggingface_hub.dataclasses, "type_validator"):
            _orig_type_validator = huggingface_hub.dataclasses.type_validator
            # Replace validation with a dummy function that always passes
            huggingface_hub.dataclasses.type_validator = lambda *args, **kwargs: None
    except Exception:
        pass
    
    try:
        yield
    finally:
        # Restore the original huggingface_hub type validator
        if _orig_type_validator:
            try:
                import huggingface_hub.dataclasses
                huggingface_hub.dataclasses.type_validator = _orig_type_validator
            except Exception:
                pass

        # Restore the original JSON serialization behavior
        json.JSONEncoder.default = _ORIGINAL_JSON_DEFAULT
        
        for key in list(sys.modules):
            if any(key == target or key.startswith(target + ".") for target in targets):
                _VENDORED_MODULES[key] = sys.modules.pop(key)
        
        sys.modules.update(saved_modules)
        
        if vendor_path in sys.path:
            sys.path.remove(vendor_path)