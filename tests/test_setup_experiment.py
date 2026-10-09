"""Installer safeguards; no network, package install, or GPU required."""

import hashlib
import importlib.util
import io
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "setup_experiment.py"


class SetupExperimentTests(unittest.TestCase):
    def installer(self):
        self.assertTrue(SCRIPT.is_file(), "experiment tool installer is missing")
        spec = importlib.util.spec_from_file_location("setup_experiment", SCRIPT)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_help_works_without_installing_dependencies(self):
        result = subprocess.run([sys.executable, str(SCRIPT), "--help"],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--python-only", result.stdout)
        self.assertIn("--build", result.stdout)

    def test_corrupt_download_is_rejected(self):
        installer = self.installer()
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "download"
            artifact.write_bytes(b"corrupt package")
            with self.assertRaisesRegex(ValueError, "SHA256"):
                installer.check_sha256(artifact, hashlib.sha256(b"trusted package").hexdigest())

    def test_private_prefix_can_be_initialized_and_reused_without_losing_history(self):
        installer = self.installer()
        with tempfile.TemporaryDirectory() as directory:
            prefix = Path(directory) / "cuda12"
            installer.initialize_prefix(prefix)
            history = prefix / "conda-meta/history"
            self.assertTrue(history.is_file())
            history.write_text("previous installation\n")
            installer.initialize_prefix(prefix)
            self.assertEqual(history.read_text(), "previous installation\n")

    def test_private_prefix_refuses_unmanaged_files(self):
        installer = self.installer()
        with tempfile.TemporaryDirectory() as directory:
            prefix = Path(directory)
            existing = prefix / "user-data"
            existing.write_text("keep me")
            with self.assertRaisesRegex(RuntimeError, "unmanaged"):
                installer.initialize_prefix(prefix)
            self.assertEqual(existing.read_text(), "keep me")
            self.assertFalse((prefix / "conda-meta").exists())

    def test_bootstrap_extracts_only_the_expected_regular_file(self):
        installer = self.installer()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "package.tar.bz2"
            with tarfile.open(archive, "w:bz2") as handle:
                for name, data in (("../escape", b"bad"), ("bin/micromamba", b"trusted binary")):
                    member = tarfile.TarInfo(name)
                    member.size = len(data)
                    handle.addfile(member, io.BytesIO(data))
            target = root / "bin" / "micromamba"
            installer.extract_micromamba(archive, target)
            self.assertEqual(target.read_bytes(), b"trusted binary")
            self.assertFalse((root.parent / "escape").exists())
            self.assertTrue(target.stat().st_mode & 0o100)

    def test_bootstrap_rejects_symlink_instead_of_executable(self):
        installer = self.installer()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "package.tar.bz2"
            with tarfile.open(archive, "w:bz2") as handle:
                member = tarfile.TarInfo("bin/micromamba")
                member.type = tarfile.SYMTYPE
                member.linkname = "/usr/bin/false"
                handle.addfile(member)
            target = root / "bin" / "micromamba"
            with self.assertRaisesRegex(ValueError, "regular file"):
                installer.extract_micromamba(archive, target)
            self.assertFalse(target.exists())


if __name__ == "__main__":
    unittest.main()
