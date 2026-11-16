import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "trim_ckpts.py"
SHARED = (
    "mlx/vae.safetensors",
    "mlx/thinker/thinker_config.json",
    "mlx/thinker/thinker.safetensors",
    "mlx/thinker/thinker.q8.safetensors",
    "Qwen2.5-Omni-3B/config.json",
    "Qwen2.5-Omni-3B/model-00001-of-00001.safetensors",
)
BASE = (
    "AuK/config.yaml",
    "AuK/auk_base.safetensors",
    "mlx/fusion_base.safetensors",
    "mlx/dit_base.safetensors",
    "mlx/dit_base.q8.safetensors",
)


class TrimCheckpointsTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.ckpts = self.root / "ckpts"
        for name in SHARED + BASE:
            path = self.ckpts / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"fixture")

    def run_trim(self, *args, check=True):
        return subprocess.run(
            [sys.executable, str(SCRIPT), "--ckpts", "ckpts", *args],
            cwd=self.root,
            capture_output=True,
            text=True,
            check=check,
        )

    def files(self):
        return {str(path.relative_to(self.ckpts)) for path in self.ckpts.rglob("*") if path.is_file()}

    def test_dry_run_and_apply_with_relative_path(self):
        before = self.files()
        self.assertIn("Dry run", self.run_trim().stdout)
        self.assertEqual(self.files(), before)
        self.assertIn("Deleted 2 files", self.run_trim("--apply").stdout)
        self.assertEqual(
            self.files(),
            before - {"AuK/auk_base.safetensors", "Qwen2.5-Omni-3B/model-00001-of-00001.safetensors"},
        )
        self.assertIn("Nothing to trim", self.run_trim("--apply").stdout)

    def test_fp32_trim_requires_each_q8_file_before_deleting(self):
        for missing in ("mlx/dit_base.q8.safetensors", "mlx/thinker/thinker.q8.safetensors"):
            with self.subTest(missing=missing):
                path = self.ckpts / missing
                path.unlink()
                before = self.files()
                result = self.run_trim("--apply", "--fp32-mlx", check=False)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(missing, result.stderr)
                self.assertEqual(self.files(), before)
                path.write_bytes(b"fixture")
        self.assertIn("Deleted 4 files", self.run_trim("--apply", "--fp32-mlx").stdout)
        self.assertTrue((self.ckpts / "mlx/dit_base.q8.safetensors").is_file())
        self.assertTrue((self.ckpts / "mlx/thinker/thinker.q8.safetensors").is_file())

    def test_fusion_and_dit_must_match_each_variant(self):
        (self.ckpts / "mlx/fusion_base.safetensors").rename(self.ckpts / "mlx/fusion_flash.safetensors")
        before = self.files()
        result = self.run_trim("--apply", check=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("mlx/fusion_base.safetensors", result.stderr)
        self.assertEqual(self.files(), before)

    def test_refuses_to_delete_through_a_symlinked_directory(self):
        outside_mlx = self.ckpts.joinpath("mlx").rename(self.root / "external-mlx")
        self.ckpts.joinpath("mlx").symlink_to(outside_mlx, target_is_directory=True)
        result = self.run_trim("--apply", "--fp32-mlx", check=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("refusing to delete outside", result.stderr)
        self.assertTrue((outside_mlx / "dit_base.safetensors").is_file())
        self.assertTrue((self.ckpts / "AuK/auk_base.safetensors").is_file())
