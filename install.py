"""Installation script for OmniVoice TTS custom node.

BEST PRACTICES:
1. NEVER touch torch/torchaudio/torchvision - ComfyUI manages these
2. Use --no-deps for any package that might pull in torch dependencies
3. Install packages individually for better error tracking
4. Verify installation at the end
5. Provide clear manual fix instructions if something goes wrong

WHY THIS EXISTS:
The omnivoice pip package specifies torch==2.8.* which downgrades PyTorch
to CPU-only on many systems, breaking ComfyUI's GPU acceleration.
We work around this by installing omnivoice with --no-deps.
"""

import importlib
import os
import subprocess
import sys


def run_cmd(cmd, timeout=300):
    """Run a command and return (success, stdout, stderr)."""
    print(f"[OmniVoice] Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode == 0:
            print(f"[OmniVoice] Success")
            return True, result.stdout, result.stderr
        else:
            print(f"[OmniVoice] Failed: {result.stderr}")
            return False, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        print(f"[OmniVoice] Timeout after {timeout}s")
        return False, "", "Timeout"
    except Exception as e:
        print(f"[OmniVoice] Error: {e}")
        return False, "", str(e)


def is_installed(package_name):
    """Check if a package is installed using importlib (no side effects)."""
    import importlib.util
    return importlib.util.find_spec(package_name) is not None


def pip_install(package, no_deps=False, upgrade=False, target=None):
    """Install a package with pip. Returns True on success."""
    python = sys.executable
    flags = []
    if no_deps:
        flags.append("--no-deps")
    if upgrade:
        flags.append("--upgrade")
    if target:
        flags.append(f"--target={target}")

    # Try uv first
    cmd = [python, "-m", "uv", "pip", "install", package] + flags
    success, _, _ = run_cmd(cmd)
    if success:
        return True

    # Fall back to pip
    cmd = [python, "-m", "pip", "install", package] + flags
    success, _, _ = run_cmd(cmd)
    return success


def check_torch():
    """Check PyTorch installation. Returns (version, has_cuda)."""
    try:
        import torch
        version = torch.__version__
        has_cuda = torch.cuda.is_available() or (hasattr(torch, "xpu") and torch.xpu.is_available())
        return version, has_cuda
    except ImportError:
        return None, False


def main():
    # Check if dependencies can be loaded using the vendor directory fallback
    vendor_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor")
    if os.path.exists(vendor_dir):
        sys.path.insert(0, vendor_dir)
        
    try:
        import omnivoice  # noqa: F401
        import soxr      # noqa: F401
        import transformers  # noqa: F401
        _tv = tuple(int(x) for x in transformers.__version__.split(".")[:2])
        if _tv < (5, 3):
            raise ImportError("transformers is too old")
        else:
            print("[OmniVoice] Already installed correctly. Skipping.")
            return
    except (ImportError, ValueError, AttributeError):
        if vendor_dir in sys.path:
            sys.path.remove(vendor_dir)

    print("=" * 60)
    print("[OmniVoice] Installation starting...")
    print("=" * 60)

    # STEP 1: Verify PyTorch is healthy
    print("")
    print("[OmniVoice] Step 1: Checking PyTorch...")
    torch_version, has_cuda = check_torch()

    if torch_version is None:
        print("[OmniVoice] ERROR: PyTorch is not installed!")
        print("[OmniVoice] ComfyUI requires PyTorch. Please check your installation.")
        return

    if has_cuda:
        print(f"[OmniVoice] PyTorch {torch_version} with CUDA - OK")
    else:
        print(f"[OmniVoice] WARNING: PyTorch {torch_version} - No CUDA detected")

    # STEP 2: Install omnivoice with --no-deps
    print("")
    print("[OmniVoice] Step 2: Installing omnivoice...")
    if is_installed("omnivoice"):
        run_cmd([sys.executable, "-m", "pip", "uninstall", "-y", "omnivoice"], timeout=60)

    if pip_install("omnivoice", no_deps=True):
        print("[OmniVoice] omnivoice installed successfully")
    else:
        print("[OmniVoice] ERROR: Failed to install omnivoice")

    # STEP 2.5: Install isolated dependencies to the vendor folder
    print("")
    print("[OmniVoice] Step 2.5: Installing isolated dependencies to vendor folder...")
    os.makedirs(vendor_dir, exist_ok=True)
    pip_install("transformers>=5.3.0", no_deps=True, upgrade=True, target=vendor_dir)
    pip_install("huggingface_hub", no_deps=True, upgrade=True, target=vendor_dir)

    # STEP 3: Install additional packages that omnivoice needs globally
    print("")
    print("[OmniVoice] Step 3: Installing additional dependencies...")

    # Removed huggingface_hub from extra_packages since it is now vendored
    extra_packages = [
        ("soundfile", "soundfile", "Audio file I/O"),
        ("scipy", "scipy", "Scientific computing (required by librosa resampling)"),
        ("lazy_loader", "lazy_loader", "Lazy loading utility (required by librosa)"),
        ("librosa", "librosa", "Audio processing"),
        ("sentencepiece", "sentencepiece", "Tokenization"),
        ("jieba", "jieba", "Chinese text segmentation"),
        ("pydub", "pydub", "Audio manipulation (required by omnivoice at import time)"),
        ("soxr", "soxr", "Audio resampling (required by transformers HiggsAudio tokenizer)"),
    ]

    no_deps_packages = {"soundfile", "sentencepiece", "jieba", "scipy", "lazy_loader", "librosa", "soxr"}

    for import_name, pip_name, description in extra_packages:
        if is_installed(import_name):
            print(f"[OmniVoice] {description} ({pip_name}) - already installed")
        else:
            print(f"[OmniVoice] Installing {description} ({pip_name})...")
            use_no_deps = pip_name in no_deps_packages
            success = pip_install(pip_name, no_deps=use_no_deps)
            if not success:
                run_cmd([sys.executable, "-m", "pip", "install", pip_name])

    # STEP 4: Final verification
    print("")
    print("=" * 60)
    print("[OmniVoice] Installation complete!")
    print("=" * 60)
    print("")
    print("[OmniVoice] Verification:")

    # Check omnivoice using the vendored path
    try:
        importlib.invalidate_caches()
        spec = importlib.util.find_spec("omnivoice")
        if spec is None:
            raise ImportError("omnivoice not found on disk")
        
        vendor_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor")
        if vendor_dir not in sys.path:
            sys.path.insert(0, vendor_dir)
            
        import omnivoice  # noqa: F401
        print("  [OK] omnivoice")
    except Exception as e:
        print(f"  [FAIL] omnivoice - installed on disk but failed to import: {e}")
        print("  Run this to see the full error:")
        print(f"    {sys.executable} -c \"import omnivoice\"")

    # Check torch (should be unchanged)
    torch_version, has_cuda = check_torch()
    if torch_version and has_cuda:
        print(f"  [OK] PyTorch {torch_version} (CUDA)")
    elif torch_version:
        print(f"  [WARN] PyTorch {torch_version} (no CUDA)")
    else:
        print("  [FAIL] PyTorch - not installed")

    print("")

    # If torch lost CUDA, provide fix instructions
    if torch_version and not has_cuda:
        print("=" * 60)
        print("[OmniVoice] IMPORTANT: PyTorch CUDA is not available!")
        print("=" * 60)
        print("")
        print("Your PyTorch installation may have been changed by another package.")
        print("To restore GPU support, reinstall PyTorch with CUDA:")
        print("")
        print("  pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128")
        print("")
        print("Or visit: https://pytorch.org/get-started/locally/")
        print("")


if __name__ == "__main__":
    main()
