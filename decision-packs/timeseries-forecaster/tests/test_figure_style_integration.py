"""tsf_lib must work outside dlab sessions and must not restyle matplotlib on import."""

import os
import subprocess
import sys
from pathlib import Path

DOCKER = Path(__file__).resolve().parents[1] / "docker"


def test_import_without_dlab_plotstyle_does_not_restyle() -> None:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(DOCKER)
    env.pop("MATPLOTLIBRC", None)
    code = (
        "import importlib.util, matplotlib, matplotlib.colors\n"
        "matplotlib.use('Agg')\n"
        "assert importlib.util.find_spec('dlab_plotstyle') is None\n"
        "import tsf_lib.cli\n"
        "cycle = matplotlib.rcParams['axes.prop_cycle'].by_key()['color']\n"
        "assert matplotlib.colors.to_hex(cycle[0]) == '#1f77b4', cycle\n"
        "print('ok')\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout
