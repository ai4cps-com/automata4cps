"""Regression checks for installations without the MLflow extra."""
import importlib.util
from pathlib import Path
import subprocess
import sys
import textwrap
import unittest

ROOT = Path(__file__).resolve().parents[1]
BLOCK_MLFLOW = """
import importlib.abc
import sys
class WithoutMLflow(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "mlflow" or fullname.startswith("mlflow."):
            raise ModuleNotFoundError("MLflow intentionally unavailable", name=fullname)
sys.meta_path.insert(0, WithoutMLflow())
"""

class OptionalMLflowTests(unittest.TestCase):
    def run_without_mlflow(self, code):
        result = subprocess.run(
            [sys.executable, "-B", "-c", BLOCK_MLFLOW + textwrap.dedent(code)],
            cwd=ROOT, text=True, capture_output=True, timeout=120,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_core_simulation_and_visualization_without_mlflow(self):
        self.run_without_mlflow("""
            from ml4cps.cps import CPS, CPSComponent
            from ml4cps import automata, tools, vis
            component = CPSComponent("example")
            system = CPS("system", [component])
            assert system["example"] is component
            assert "mlflow" not in sys.modules
            try:
                tools.log_plotly_figure_to_mlflow(object())
            except ImportError as exc:
                assert "ml4cps[mlflow]" in str(exc)
            else:
                raise AssertionError("Tracking must require the MLflow extra")
        """)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "requires optional torch extra")
    def test_debta_training_without_tracking(self):
        self.run_without_mlflow("""
            import torch
            from ml4cps.debta.base import DebtaEbm
            from ml4cps.debta.rbms.bb import BinaryBinaryRBM
            from ml4cps.debta.rbms.training import train_rbm
            DebtaEbm(num_y=2, num_h=2, transience=False, log_mlflow=False)
            rbm = BinaryBinaryRBM(2, 2)
            from unittest.mock import patch
            with patch("plotly.basedatatypes.BaseFigure.show"):
                train_rbm(rbm, torch.zeros(4, 2), max_epoch=1,
                          batch_size=2, verbose=False, log_mlflow=False)
            assert "mlflow" not in sys.modules
            try:
                DebtaEbm(num_y=2, num_h=2, log_mlflow=True)
            except ImportError as exc:
                assert "ml4cps[mlflow]" in str(exc)
            else:
                raise AssertionError("Tracking must require the MLflow extra")
        """)

    @unittest.skipUnless(importlib.util.find_spec("mlflow"), "requires optional mlflow extra")
    def test_installed_mlflow_model_integration_is_preserved(self):
        from mlflow.pyfunc import PythonModel
        from ml4cps.cps import CPSComponent
        self.assertIsInstance(CPSComponent("example"), PythonModel)

if __name__ == "__main__":
    unittest.main()
