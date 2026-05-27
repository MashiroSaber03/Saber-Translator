import pathlib
import unittest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


class PackagingTorchscriptSourceTests(unittest.TestCase):
    def test_app_spec_collects_litelama_and_kornia_sources_for_torchscript(self) -> None:
        spec_text = (REPO_ROOT / "app.spec").read_text(encoding="utf-8")

        self.assertIn("module_collection_mode", spec_text)
        self.assertIn("'litelama': 'pyz+py'", spec_text)
        self.assertIn("'kornia': 'pyz+py'", spec_text)

    def test_requirements_pin_packaging_regression_dependencies(self) -> None:
        cpu_text = (REPO_ROOT / "requirements-cpu.txt").read_text(encoding="utf-8")
        gpu_text = (REPO_ROOT / "requirements-gpu.txt").read_text(encoding="utf-8")

        self.assertIn("torch==2.11.0", cpu_text)
        self.assertIn("torchvision==0.26.0", cpu_text)
        self.assertIn("litelama==0.1.7", cpu_text)
        self.assertIn("kornia==0.8.2", cpu_text)
        self.assertIn("pyinstaller==6.19.0", cpu_text)

        self.assertIn("litelama==0.1.7", gpu_text)
        self.assertIn("kornia==0.8.2", gpu_text)
        self.assertIn("pyinstaller==6.19.0", gpu_text)

    def test_release_workflow_verifies_torchscript_source_files_in_bundle(self) -> None:
        workflow_text = (REPO_ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")

        self.assertIn("PYTHON_VERSION: '3.12'", workflow_text)
        self.assertIn("Verify TorchScript source files", workflow_text)
        self.assertIn("dist\\Saber-Translator\\_internal\\litelama\\litelama.py", workflow_text)
        self.assertIn("dist\\Saber-Translator\\_internal\\kornia\\geometry\\epipolar\\_metrics.py", workflow_text)
        self.assertIn("pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu130", workflow_text)


if __name__ == "__main__":
    unittest.main()
