"""HBM admission reporting fixtures are synthetic, never GPU measurements."""

import csv
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest

from powermodeling.dashboard import write_evaluation
from powermodeling.reporting import write_plots
from test_hbm_memory_reporting import sensor_summary


def bandwidth_evaluation(status="pass"):
    summary = sensor_summary()
    point = summary["evaluation"]["components"][0]["points"][0]
    point["hbm_bandwidth"] = {
        "status": status, "eligible": status == "pass", "evidence_complete": status != "inconclusive",
        "minimum_fraction": .8, "theoretical_bytes_s": 2e12 if status != "inconclusive" else None,
        "sustained_logical_bytes_s": 1.7e12,
        "fraction_of_theoretical": .85 if status != "inconclusive" else None,
        "achieved_memory_clock_mhz": 1562.5, "memory_bus_width_bits": 5120,
        "clock_source": "matching_measure_window_memory_clock_mhz",
        "formula": "2 * memory_clock_mhz * 1e6 * memory_bus_width_bits / 8",
        "physical_dram_utilization_measured": False,
        "reasons": [] if status == "pass" else ["memory_bus_width_unavailable"],
        "scope": "logical sustained payload / clock-derived theoretical HBM bandwidth",
    }
    return summary


class HbmBandwidthReportingTests(unittest.TestCase):
    def test_csv_exports_bandwidth_provenance_and_decimal_rates_without_changing_sensor(self):
        summary = bandwidth_evaluation()
        warning = "logical_rate_above_interface_ceiling_cache_reuse_or_count_review"
        summary["evaluation"]["components"][0]["points"][0]["hbm_bandwidth"]["warnings"] = [warning]
        with tempfile.TemporaryDirectory() as directory:
            files = write_evaluation(summary["evaluation"], directory)
            with Path(files["evaluation_csv"]).open() as stream:
                rows = list(csv.DictReader(stream))
            row = rows[0]
            self.assertEqual(json.loads(row.get("hbm_bandwidth", "null")),
                             summary["evaluation"]["components"][0]["points"][0]["hbm_bandwidth"])
            self.assertEqual(row["hbm_bandwidth_status"], "pass")
            self.assertEqual(float(row["hbm_logical_gbps"]), 1700)
            self.assertEqual(float(row["hbm_theoretical_gbps"]), 2000)
            self.assertEqual(float(row["hbm_fraction_of_theoretical"]), .85)
            self.assertEqual(float(row["hbm_minimum_fraction"]), .8)
            self.assertEqual(row["hbm_physical_dram_utilization_measured"], "False")
            self.assertEqual(json.loads(row["hbm_bandwidth_warnings"]), [warning])
            self.assertEqual(json.loads(row["hbm_memory_power"])["energy_j"], 240)

    def test_missing_theory_exports_blank_not_zero_with_reason(self):
        summary = bandwidth_evaluation("inconclusive")
        with tempfile.TemporaryDirectory() as directory:
            files = write_evaluation(summary["evaluation"], directory)
            with Path(files["evaluation_csv"]).open() as stream:
                row = next(csv.DictReader(stream))
            self.assertEqual(row.get("hbm_bandwidth_status"), "inconclusive")
            self.assertEqual(row["hbm_theoretical_gbps"], "")
            self.assertEqual(row["hbm_fraction_of_theoretical"], "")
            self.assertIn("memory_bus_width_unavailable", row["hbm_bandwidth_reasons"])

    @unittest.skipUnless(shutil.which("node"), "Node is needed to exercise dashboard presentation")
    def test_dashboard_displays_hbm_absolute_gate_instead_of_observed_peak_gate(self):
        summary = bandwidth_evaluation()
        with tempfile.TemporaryDirectory() as directory:
            files = write_evaluation(summary["evaluation"], directory)
            html = Path(files["evaluation_html"]).read_text()
        self.assertTrue('id="hbmBandwidthSection"' in html, "HBM bandwidth section is missing")
        function = re.search(r"function bandwidthPolicy\(c,policy\)\{(.*?)\n\}", html, re.S)
        self.assertIsNotNone(function, "dashboard must apply a workload-specific admission description")
        render = re.search(r"function renderHbmBandwidth\(c,points\)\{(.*?)\n\}", html, re.S)
        self.assertIsNotNone(render, "dashboard must display the bandwidth inputs and outcome")
        script = "const pct=x=>typeof x==='number'?String(100*x)+'%':'—',fmt=x=>typeof x==='number'?String(x):'—',scale=(x,n)=>typeof x==='number'?x/n:null,nodes={};const $=id=>nodes[id]??=( {} );let displayedRows;function table(id,headers,rows){displayedRows=rows;}\n"
        script += function.group(0) + "\n" + render.group(0)
        script += "\nconst component=" + json.dumps(summary["evaluation"]["components"][0]) + ",policy=" + json.dumps(summary["evaluation"]["policy"]) + ";renderHbmBandwidth(component,component.points);console.log(JSON.stringify({policies:[bandwidthPolicy(component,policy),bandwidthPolicy({stratum:{workload:'l2'}},policy)],rows:displayedRows}));"
        result = subprocess.run([shutil.which("node"), "-e", script], check=True, capture_output=True, text=True)
        rendered = json.loads(result.stdout)
        hbm, l2 = rendered["policies"]
        self.assertIn("80%", hbm)
        self.assertIn("actual memory clock", hbm)
        self.assertIn("observed peak is diagnostic", hbm)
        self.assertNotIn("95%", hbm)
        self.assertIn("95%", l2)
        row = rendered["rows"][0]
        self.assertEqual(row[1:5], ["pass / true", "1700", "2000", "85% / 80%"])
        self.assertEqual(row[6], "no")
        self.assertIn("Physical DRAM utilization is not measured", html)

    def test_hbm_plot_labels_absolute_bandwidth_gate_and_preserves_sensor_plot(self):
        summary = bandwidth_evaluation()
        with tempfile.TemporaryDirectory() as directory:
            files = write_plots(summary, directory)
            energy_path = next(value for key, value in files.items() if key.endswith("total-energy-throughput_svg"))
            svg = Path(energy_path).read_text()
            self.assertTrue("HBM admission" in svg, "HBM plot admission annotation is missing")
            self.assertIn("80%", svg)
            self.assertIn("physical DRAM utilization is not measured", svg)
            self.assertTrue(any("memory-sensor_svg" in key for key in files))


if __name__ == "__main__":
    unittest.main()
