"""ComfyUI custom nodes for OmniVoice TTS.

Provides nodes:
  - OmniVoiceTTS            -- text -> speech, auto voice selection (random)
  - OmniVoiceVoiceCloneTTS  -- reference audio + text -> cloned-voice speech
  - OmniVoiceVoiceDesignTTS -- text + voice description -> designed voice speech
  - OmniVoiceWhisperLoader  -- load Whisper ASR for auto-transcription

Dependencies are installed via install.py (run by ComfyUI-Manager).
Model weights are auto-downloaded from HuggingFace on first inference.

Supports 600+ languages with zero-shot voice cloning and voice design.
"""

__version__ = "0.4.4"

import logging
import sys
import types
import importlib.util
import os
import subprocess
from pathlib import Path
from typing import Any, Dict

def setup_vendored_dependencies():
    """Download and isolate transformers 5.x into a local vendor directory."""
    current_directory = os.path.dirname(os.path.abspath(__file__))
    vendor_directory = os.path.join(current_directory, "vendor")
    check_file_path = os.path.join(vendor_directory, "transformers", "__init__.py")
    
    if not os.path.exists(check_file_path):
        print("[OmniVoice] Vendored dependencies not found. Starting installation to vendor directory...")
        os.makedirs(vendor_directory, exist_ok=True)
        
        installation_command = [
            sys.executable, "-m", "pip", "install",
            "transformers>=5.3.0",
            "huggingface_hub",
            "--target", vendor_directory,
            "--upgrade",
            "--no-deps",
            "--quiet"
        ]
        
        try:
            subprocess.run(installation_command, check=True)
            print("[OmniVoice] Dependencies successfully installed to the isolated environment.")
        except subprocess.CalledProcessError as exception:
            print(f"[OmniVoice] Dependency installation failed: {exception}")

setup_vendored_dependencies()

# ---------------------------------------------------------------------------
# Pre-emptively block torchcodec from crashing on incompatible PyTorch builds
# (e.g., AMD ROCm, custom builds).  Transformers' Whisper pipeline tries to
# import torchcodec during audio preprocessing.  If the native DLL fails to
# load it raises RuntimeError and kills the node.  By inserting a stub into
# sys.modules *before* that import runs, transformers silently falls back to
# soundfile/sox which OmniVoice already has.
# ---------------------------------------------------------------------------
_tc_broken = False
if 'torchcodec' not in sys.modules:
    try:
        import torchcodec  # noqa: F401
    except Exception:
        _tc_broken = True

# Replace with stub if the import failed or the module is non-functional.
# On Windows with a broken DLL, import torchcodec may partially execute,
# set itself in sys.modules with __spec__ = None, then raise — so the
# except block above runs but sys.modules now holds a broken module.
_tc = sys.modules.get('torchcodec')
if _tc_broken or _tc is None or getattr(_tc, '__spec__', None) is None:
    _tc_stub = types.ModuleType('torchcodec')
    _tc_stub.__path__ = []
    _tc_stub.__package__ = 'torchcodec'
    _tc_stub.__spec__ = importlib.util.spec_from_loader(
        'torchcodec', loader=None, origin='torchcodec'
    )
    for _sub in ('decoders', 'encoders', 'samplers', 'transforms', '_core'):
        _sub_mod = types.ModuleType(f'torchcodec.{_sub}')
        _sub_mod.__spec__ = importlib.util.spec_from_loader(
            f'torchcodec.{_sub}', loader=None
        )
        # Add dummy AudioDecoder so isinstance() checks in transformers
        # don't crash — the check returns False and falls through to
        # the soundfile path as intended.
        if _sub == 'decoders':
            class _AudioDecoder:
                pass
            _sub_mod.AudioDecoder = _AudioDecoder
        setattr(_tc_stub, _sub, _sub_mod)
        sys.modules[f'torchcodec.{_sub}'] = _sub_mod
    sys.modules['torchcodec'] = _tc_stub

    # transformers/audio_utils.py calls importlib.metadata.version("torchcodec")
    # at module level — this fails because stub has no pip metadata on disk.
    # Patch metadata.version() to return a fake version for torchcodec and tokenizers.
    import importlib.metadata as _ilm
    _orig_ilm_version = _ilm.version
    
    def _patched_ilm_version(name):
        if name == "torchcodec":
            return "0.0.0"
        if name == "tokenizers":
            return "0.23.0"  # Bypass strict transformers version check
        return _orig_ilm_version(name)
        
    _ilm.version = _patched_ilm_version

    logging.getLogger("OmniVoice").info(
        "torchcodec blocked (incompatible PyTorch build) — using soundfile fallback"
    )

_HERE = Path(__file__).parent.resolve()

# Add this folder to sys.path for local imports
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

logger = logging.getLogger("OmniVoice")
logger.propagate = False

if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("[OmniVoice] %(message)s"))
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)


def _check_dependencies() -> tuple[bool, list[tuple[str, list[str]]]]:
    """Check if omnivoice and its critical dependencies are importable."""
    from .nodes.vendor_context import vendored_transformers
    
    with vendored_transformers():
        try:
            import omnivoice
        except ImportError as exception:
            if importlib.util.find_spec("omnivoice") is not None:
                missing_dependency = getattr(exception, 'name', None) or 'unknown'
                logger.error(
                    f"omnivoice is installed but failed to import: {exception}"
                )
                managed_dependencies = {
                    "soxr": ["--no-deps"],
                    "soundfile": ["--no-deps"],
                    "scipy": ["--no-deps"],
                    "lazy_loader": ["--no-deps"],
                    "librosa": ["--no-deps"],
                    "sentencepiece": ["--no-deps"],
                    "jieba": ["--no-deps"],
                    "pydub": [],
                }
                if missing_dependency in managed_dependencies:
                    flags = managed_dependencies[missing_dependency]
                    command = " ".join([sys.executable, "-m", "pip", "install"] + flags + [missing_dependency])
                    logger.error(
                        f"Missing dependency: '{missing_dependency}'. "
                        f"Run: {command}"
                    )
                else:
                    logger.error(
                        f"Missing or broken dependency: '{missing_dependency}'. "
                        f"Please report this issue on GitHub."
                    )
                return False, []
            
            logger.warning(f"omnivoice not installed: {exception}")
            return False, [("omnivoice", ["--no-deps"])]
        except Exception as exception:
            logger.error(f"omnivoice is installed but failed to import: {exception}")
            logger.error("Reinstalling will not fix this. Check the error above.")
            return False, []

        missing_packages: list[tuple[str, list[str]]] = []

        try:
            import soxr
        except ImportError:
            missing_packages.append(("soxr", []))

        return (len(missing_packages) == 0), missing_packages


# ---------------------------------------------------------------------------
# Node registration
# ---------------------------------------------------------------------------

NODE_CLASS_MAPPINGS: Dict[str, Any] = {}
NODE_DISPLAY_NAME_MAPPINGS: Dict[str, str] = {}

# Always register the Whisper loader (doesn't depend on omnivoice)
try:
    from .nodes.whisper_loader import OmniVoiceWhisperLoader
    NODE_CLASS_MAPPINGS["OmniVoiceWhisperLoader"] = OmniVoiceWhisperLoader
    NODE_DISPLAY_NAME_MAPPINGS["OmniVoiceWhisperLoader"] = "OmniVoice Whisper Loader"
except Exception as e:
    logger.warning(f"Failed to register Whisper loader: {e}")

_deps_ready, _deps_missing = _check_dependencies()

if _deps_ready:
    try:
        from .nodes.loader import _register_folder
        _register_folder()

        from .nodes.omnivoice_tts import OmniVoiceLongformTTS
        from .nodes.voice_clone_node import OmniVoiceVoiceCloneTTS
        from .nodes.voice_design_node import OmniVoiceVoiceDesignTTS
        from .nodes.multi_speaker_node import OmniVoiceMultiSpeakerTTS

        NODE_CLASS_MAPPINGS.update({
            "OmniVoiceLongformTTS": OmniVoiceLongformTTS,
            "OmniVoiceVoiceCloneTTS": OmniVoiceVoiceCloneTTS,
            "OmniVoiceVoiceDesignTTS": OmniVoiceVoiceDesignTTS,
            "OmniVoiceMultiSpeakerTTS": OmniVoiceMultiSpeakerTTS,
        })

        NODE_DISPLAY_NAME_MAPPINGS.update({
            "OmniVoiceLongformTTS": "OmniVoice Longform TTS",
            "OmniVoiceVoiceCloneTTS": "OmniVoice Voice Clone TTS",
            "OmniVoiceVoiceDesignTTS": "OmniVoice Voice Design TTS",
            "OmniVoiceMultiSpeakerTTS": "OmniVoice Multi-Speaker TTS",
        })

        logger.info(
            f"Registered {len(NODE_CLASS_MAPPINGS)} nodes "
            f"(v{__version__}): {', '.join(NODE_DISPLAY_NAME_MAPPINGS.values())}"
        )

    except Exception as e:
        logger.error(f"Failed to register nodes: {e}", exc_info=True)
elif _deps_missing:
    # Fallback: install exactly what's missing.
    # omnivoice itself missing  -> pip install omnivoice --no-deps
    # sub-dep missing/too-old   -> pip install <pkg> [--upgrade]
    _missing_names = [pkg for pkg, _ in _deps_missing]
    logger.error("=" * 60)
    logger.error(f" Missing packages: {', '.join(_missing_names)}")
    logger.error("=" * 60)
    try:
        import subprocess

        for _pkg, _extra_args in _deps_missing:
            _cmd = [sys.executable, "-m", "pip", "install"] + _extra_args + [_pkg]
            if _pkg == "transformers":
                vendor_dir = os.path.join(_HERE, "vendor")
                _cmd += ["--target", vendor_dir, "--no-deps"]
            logger.warning(f"Installing {_pkg} ...")
            _result = subprocess.run(_cmd, capture_output=True, text=True, timeout=120)
            if _result.returncode == 0:
                logger.warning(f"  {_pkg} installed successfully.")
            else:
                logger.error(f"  Failed to install {_pkg}: {_result.stderr}")

        # Re-check to report final status
        _deps_ready_now, _deps_still_missing = _check_dependencies()
        if _deps_ready_now:
            logger.warning("All dependencies installed — RESTART ComfyUI to load nodes.")
        else:
            _still = [pkg for pkg, _ in _deps_still_missing]
            logger.error(
                f"Dependencies still missing after install: {', '.join(_still)}. "
                f"Please restart ComfyUI. If the issue persists, check the "
                f"logs above for the actual import error."
            )
    except Exception as e:
        logger.error(f"Failed to install dependencies: {e}")

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "__version__"]
