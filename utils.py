import os
import sys
import subprocess
import importlib.metadata
from typing import Optional
from invokeai.backend.util.logging import InvokeAILogger



logger = InvokeAILogger.get_logger(name='InvokeAI-TIPO')
KGEN_VERSION = "0.2.0"
llama_cpp_python_wheel = (
    "llama-cpp-python --prefer-binary "
    "--extra-index-url=https://jllllll.github.io/llama-cpp-python-cuBLAS-wheels/{}/{}"
)
llama_cpp_python_wheel_official = (
    "https://github.com/abetlen/llama-cpp-python/releases/download/"
    "v{version_arch}/llama_cpp_python-{version}-{python}-{python}-{platform}.whl"
)
version_arch = {
    "0.3.4": [
        ("cp39", "cp310", "cp311", "cp312"),
        ("cu121", "cu122", "cu123", "cu124", "metal"),
        ("linux_x86_64", "win_amd64", "maxosx_11_0_arm64"),
    ],
    "0.3.2": [
        ("cp38", "cp39", "cp310", "cp311", "cp312"),
        ("cpu", "metal"),
        ("linux_x86_64", "win_amd64", "maxosx_11_0_arm64"),
    ],
}



def run(command,
        desc: Optional[str] = None,
        errdesc: Optional[str] = None,
        custom_env: Optional[list] = None,
        live: Optional[bool] = True,
        shell: Optional[bool] = None):

    if shell is None:
        shell = False if sys.platform == "win32" else True

    if desc is not None:
        logger.info(desc)

    if live:
        result = subprocess.run(command, shell=shell, env=os.environ if custom_env is None else custom_env)
        if result.returncode != 0:
            raise RuntimeError(f"""{errdesc or 'Error running command'}.
Command: {command}
Error code: {result.returncode}""")

        return ""

    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            shell=shell, env=os.environ if custom_env is None else custom_env)

    if result.returncode != 0:
        message = f"""{errdesc or 'Error running command'}.
Command: {command}
Error code: {result.returncode}
stdout: {result.stdout.decode(encoding="utf8", errors="ignore") if len(result.stdout) > 0 else '<empty>'}
stderr: {result.stderr.decode(encoding="utf8", errors="ignore") if len(result.stderr) > 0 else '<empty>'}
"""
        raise RuntimeError(message)

    return result.stdout.decode(encoding="utf8", errors="ignore")


def run_pip(command, desc=None, live=False):
    return run(f'"{sys.executable}" -m pip {command}', desc=f"Installing {desc}", errdesc=f"Couldn't install {desc}", live=live)


def get_installed_version(package: str):
    try:
        return importlib.metadata.version(package)
    except Exception:
        return None


def install_llama_cpp_legacy(cuda_version, has_cuda):
    if cuda_version == "122":
        cuda_version = "122"
    package = llama_cpp_python_wheel.format(
        "AVX2", f"cu{cuda_version}" if has_cuda else "cpu"
    )

    run_pip(
        f"install {package}",
        "LLaMA-CPP-Python for TIPO",
    )


def install_llama_cpp():
    logger.info("Check LLaMA CPP")

    if get_installed_version("llama_cpp_python") is not None:
        return
    logger.info("Attempting to install LLaMA-CPP-Python")
    import torch

    has_cuda = torch.cuda.is_available()
    cuda_version = torch.version.cuda.replace(".", "")
    arch = "cu" + cuda_version if has_cuda else "cpu"
    if has_cuda and arch > "cu124":
        arch = "cu124"
    platform = sys.platform
    py_ver = f"cp{sys.version_info.major}{sys.version_info.minor}"
    if platform == "darwin":
        platform = "maxosx_11_0_arm64"
    elif platform == "win32":
        platform = "win_amd64"
    elif platform == "linux":
        platform = "linux_x86_64"

    for version, (py_vers, archs, platforms) in version_arch.items():
        if py_ver in py_vers and arch in archs and platform in platforms:
            break
    else:
        logger.warning("Official wheel not found, using legacy builds")
        install_llama_cpp_legacy(cuda_version, has_cuda)
        return

    wheel = llama_cpp_python_wheel_official.format(
        version=version,
        python=py_ver,
        platform=platform,
        version_arch=f"{version}-{arch}",
    )

    try:
        run_pip(f"install {wheel}", "LLaMA-CPP-Python", live=True)
        logger.info("Installation of llama-cpp-python succeeded")
    except Exception:
        logger.warning(
            "Installation of llama-cpp-python failed, "
            "Please try to install it manually or use non-gguf models"
        )


def install_tipo_kgen():
    logger.info("Check TIPO KGen")

    version = get_installed_version("tipo-kgen")
    if version is not None and version >= KGEN_VERSION:
        return
    logger.info("Attempting to install tipo_kgen")
    run_pip(f"install -U tipo-kgen>={KGEN_VERSION}", "tipo-kgen", live=True)
