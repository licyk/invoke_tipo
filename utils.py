import os
import sys
import subprocess
from typing import Optional
from invokeai.backend.util.logging import InvokeAILogger



invoke_logger = InvokeAILogger.get_logger(name='InvokeAI-TIPO')



def run(command,
        desc: Optional[str] = None,
        errdesc: Optional[str] = None,
        custom_env: Optional[list] = None,
        live: Optional[bool] = True,
        shell: Optional[bool] = None):

    if shell is None:
        shell = False if sys.platform == "win32" else True

    if desc is not None:
        invoke_logger.info(desc)

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


def setup_llama_cpp():
    invoke_logger.info("Check LLaMA CPP")
    llama_cpp_python_wheel = (
        "llama-cpp-python --prefer-binary "
        "--extra-index-url=https://jllllll.github.io/llama-cpp-python-cuBLAS-wheels/{}/{}"
    )
    try:
        import llama_cpp
    except Exception as e:
        invoke_logger.info("Attempting to install LLaMA-CPP-Python")
        import torch

        has_cuda = torch.cuda.is_available()
        cuda_version = torch.version.cuda.replace(".", "")
        if cuda_version == "124":
            cuda_version = "122"
        package = llama_cpp_python_wheel.format(
            "AVX2", f"cu{cuda_version}" if has_cuda else "cpu"
        )
        run_pip(f"install {package}", "LLaMA-CPP-Python", live=True)


def setup_kgen():
    invoke_logger.info("Check TIPO KGen")
    kgen_ver = "0.1.5"
    try:
        import kgen

        if kgen.__version__ < kgen_ver:
            raise ImportError
    except Exception as e:
        run_pip(f"install -U tipo-kgen>={kgen_ver}", "tipo-kgen", live=True)
