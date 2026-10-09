"""Memory-scope plot contracts using synthetic data, never hardware results."""

import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

from powermodeling.analysis import summarize
from powermodeling.reporting import write_plots
from test_analysis import synthetic_trial


def sensor_summary():
    summary = summarize([synthetic_trial("hbm", repeat=i) for i in range(3)])
    point = summary["evaluation"]["components"][0]["points"][0]
    point["hbm_memory_power"] = {
        "status": "partial", "source": "synthetic NVML memory scope",
        "semantics": "synthetic sensor scope, not physical HBM attribution",
        "power_w": 40, "energy_j": 320, "idle_power_w": 10,
        "incremental_power_w": 30, "incremental_energy_j": 240,
        "pj_per_logical_bit": 5, "incremental_pj_per_logical_bit": 3.75,
        "measurement_valid": True, "normalization_valid": True, "incremental_valid": True,
        "valid_repeats": 2, "observed_repeats": 3, "normalized_repeats": 2,
        "incremental_repeats": 1, "ci95": {"power_w": [39, 41]},
        "issues": [], "incremental_issues": []}
    return summary


@unittest.skipUnless(importlib.util.find_spec("matplotlib"), "Optional plot dependency unavailable")
class HbmMemoryPlotTests(unittest.TestCase):
    def export(self, summary, directory):
        # Board objectives are independently tested; isolate this supplementary figure.
        with patch("powermodeling.reporting.OBJECTIVES", ()):
            return write_plots(summary, directory)

    def test_exports_png_svg_with_sensor_scope_and_repeat_qualifiers(self):
        with tempfile.TemporaryDirectory() as directory:
            files = self.export(sensor_summary(), directory)
            self.assertEqual(set(files), {"hbm-memory-sensor_png", "hbm-memory-sensor_svg"})
            self.assertTrue(Path(files["hbm-memory-sensor_png"]).read_bytes().startswith(b"\x89PNG\r\n\x1a\n"))
            svg = Path(files["hbm-memory-sensor_svg"])
            ET.parse(svg)
            text = svg.read_text()
            for expected in ("Synthetic, not measured hardware", "synthetic NVML memory scope",
                             "not physical HBM attribution", "2/3 sensor repeats", "2 normalized", "1 matched idle",
                             "Memory-scope power (W)", "Memory-scope energy (J)", "pJ/logical bit",
                             "Matched idle increment", "SM=1200", "B=80"):
                self.assertIn(expected, text)

    def test_invalid_sensor_is_not_drawn_as_zero_or_reported_value(self):
        from powermodeling import reporting
        summary = sensor_summary()
        component = summary["evaluation"]["components"][0]
        bad = copy.deepcopy(component["points"][0])
        bad["group_id"] = "invalid-sensor"
        bad["hbm_memory_power"].update(status="invalid", measurement_valid=False,
                                      normalization_valid=False, incremental_valid=False, power_w=999)
        component["points"].append(bad)
        snapshots = []
        original = reporting._save
        def save(fig, *args):
            snapshots.extend([collection.get_offsets().tolist() for ax in fig.axes for collection in ax.collections])
            original(fig, *args)
        with tempfile.TemporaryDirectory() as directory, patch.object(reporting, "_save", side_effect=save):
            self.export(summary, directory)
        self.assertEqual(len(snapshots), 6)
        self.assertTrue(all(len(offsets) == 1 and offsets[0][0] == 0 for offsets in snapshots))

    def test_no_valid_sensor_omits_supplementary_figure(self):
        summary = sensor_summary()
        summary["evaluation"]["components"][0]["points"][0]["hbm_memory_power"]["measurement_valid"] = False
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(self.export(summary, directory), {})

    def test_unqualified_normalization_and_idle_increment_are_blank(self):
        from powermodeling import reporting
        summary = sensor_summary()
        sensor = summary["evaluation"]["components"][0]["points"][0]["hbm_memory_power"]
        sensor.update(normalization_valid=False, incremental_valid=False)
        snapshots = []
        original = reporting._save
        def save(fig, *args):
            snapshots.extend(len(ax.collections) for ax in fig.axes)
            original(fig, *args)
        with tempfile.TemporaryDirectory() as directory, patch.object(reporting, "_save", side_effect=save):
            files = self.export(summary, directory)
            self.assertIn("N/A: no qualified sensor values", Path(files["hbm-memory-sensor_svg"]).read_text())
        self.assertEqual(snapshots, [1, 1, 0, 0, 0, 0])

    def test_memory_domains_are_exported_separately(self):
        summary = sensor_summary()
        points = summary["evaluation"]["components"][0]["points"]
        second = copy.deepcopy(points[0])
        second.update(group_id="second-memory-domain", requested_memory_mhz=1500)
        points.append(second)
        with tempfile.TemporaryDirectory() as directory:
            files = self.export(summary, directory)
            self.assertEqual(set(files), {f"hbm-mem-{clock}-memory-sensor_{ext}"
                                         for clock in (1000, 1500) for ext in ("png", "svg")})

    def test_large_sweep_has_readable_pages_preserving_every_condition(self):
        summary = sensor_summary()
        component = summary["evaluation"]["components"][0]
        prototype = component["points"][0]
        component["points"] = []
        for index in reversed(range(9)):
            point = copy.deepcopy(prototype)
            point.update(group_id=f"condition-{index}", requested_graphics_mhz=1200 + index)
            component["points"].append(point)
        with tempfile.TemporaryDirectory() as directory:
            files = self.export(summary, directory)
            self.assertEqual(set(files), {f"hbm-page-{page:02d}-memory-sensor_{ext}"
                                         for page in (1, 2) for ext in ("png", "svg")})
            texts = [Path(files[f"hbm-page-{page:02d}-memory-sensor_svg"]).read_text()
                     for page in (1, 2)]
            for index in range(9):
                self.assertIn(f"condition-{index}", texts[index // 8])
                self.assertNotIn(f"condition-{index}", texts[1 - index // 8])
