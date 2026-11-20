"""Build colab_check.ipynb: a self-contained Colab dry run (organizers only).

Embeds emg_kit + requirements.txt so it works before the GitHub repo exists.
Upload to Colab, Runtime -> Run all. Prints one JSON block of results at the end.

    python tools/build_colab_check.py
"""

import base64
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parent.parent
FILES = ["requirements.txt", "emg_kit/__init__.py", "emg_kit/core.py", "emg_kit/download.py"]

blob = {f: base64.b64encode((ROOT / f).read_bytes()).decode() for f in FILES}

cells = [
    nbf.v4.new_markdown_cell("# Colab dry run: Wearable AI Build Day starter\n"
                             "Runtime -> Run all. Results print as JSON in the last cell."),
    nbf.v4.new_code_cell(
        "import base64, os, json, platform, subprocess, sys, time\n"
        "os.makedirs('/content/check/emg_kit', exist_ok=True); os.chdir('/content/check')\n"
        f"FILES = {blob!r}\n"
        "for f, b in FILES.items():\n"
        "    open(f, 'wb').write(base64.b64decode(b))\n"
        "pre = subprocess.run([sys.executable, '-c', 'import torch, numpy; print(torch.__version__, numpy.__version__)'],"
        " capture_output=True, text=True).stdout.strip()\n"
        "cpu = subprocess.run('lscpu | grep -E \"Model name|^CPU\\\\(s\\\\)\"', shell=True, capture_output=True, text=True).stdout\n"
        "mem = subprocess.run('free -g | head -2', shell=True, capture_output=True, text=True).stdout\n"
        "print('python', platform.python_version(), '| preinstalled torch/numpy', pre); print(cpu, mem)"),
    nbf.v4.new_code_cell(
        "t = time.time()\n"
        "r = subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-r', 'requirements.txt'],"
        " capture_output=True, text=True)\n"
        "PIP = {'ok': r.returncode == 0, 'seconds': round(time.time() - t),"
        " 'stderr_tail': r.stderr[-1500:]}\n"
        "print(PIP)"),
    nbf.v4.new_code_cell(
        "t = time.time()\n"
        "r = subprocess.run([sys.executable, '-m', 'emg_kit.download', '--small-subset'],"
        " capture_output=True, text=True)\n"
        "DL = {'ok': r.returncode == 0, 'seconds': round(time.time() - t), 'tail': (r.stdout + r.stderr)[-600:]}\n"
        "print(DL)"),
    nbf.v4.new_code_cell(
        "CHECK = r'''\n"
        "import json, resource, time, warnings, torch\n"
        "warnings.filterwarnings('ignore')\n"
        "from emg_kit.core import *\n"
        "out = {'torch': torch.__version__, 'threads': torch.get_num_threads()}\n"
        "m = load_model(); t = time.time(); df = evaluate(m, verbose=False)\n"
        "out['fp32'] = {'mean_cler': round(df.cler.mean(), 5), 'per_user': df.cler.round(4).tolist(),\n"
        "               'eval_s': round(time.time() - t, 1), 'ms': round(benchmark_ms(m), 2), 'mb': round(model_size_mb(m), 1)}\n"
        "q = quantize_int8(m); out['quant_engine'] = torch.backends.quantized.engine\n"
        "t = time.time(); dq = evaluate(q, verbose=False)\n"
        "out['int8'] = {'mean_cler': round(dq.cler.mean(), 5), 'eval_s': round(time.time() - t, 1),\n"
        "               'ms': round(benchmark_ms(q), 2), 'mb': round(model_size_mb(q), 1)}\n"
        "out['fp32_ms_all_threads'] = round(benchmark_ms(m, threads=None), 2)\n"
        "out['peak_rss_gb'] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 2)\n"
        "print('RESULT' + json.dumps(out))\n"
        "'''\n"
        "t = time.time()\n"
        "r = subprocess.run([sys.executable, '-c', CHECK], capture_output=True, text=True)\n"
        "line = [l for l in r.stdout.splitlines() if l.startswith('RESULT')]\n"
        "RES = json.loads(line[0][6:]) if line else {'error': (r.stdout + r.stderr)[-2000:]}\n"
        "RES['total_s'] = round(time.time() - t)\n"
        "print(RES)"),
    nbf.v4.new_code_cell(
        "# In-kernel import, the way students' notebooks will do it (after pip changed packages)\n"
        "try:\n"
        "    sys.path.insert(0, '/content/check')\n"
        "    import emg_kit.core as ek; INKERNEL = 'ok'\n"
        "except Exception as e:\n"
        "    INKERNEL = f'{type(e).__name__}: {e}'\n"
        "print('COLAB_CHECK_JSON ' + json.dumps({'pre': pre, 'cpu': cpu, 'mem': mem, 'pip': PIP,"
        " 'download': DL, 'results': RES, 'in_kernel_import': INKERNEL}))"),
]
nb = nbf.v4.new_notebook()
nb.cells = cells
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3"}
out = ROOT / "tools" / "colab_check.ipynb"
nbf.write(nb, out)
print("wrote", out, f"{out.stat().st_size/1e3:.0f} KB")
