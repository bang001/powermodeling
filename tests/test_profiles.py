import unittest

from powermodeling.profiles import architecture, declare_sxm, theoretical_tensor_tflops


class ProfileTests(unittest.TestCase):
    def test_sxm_name_confirms_module_without_assuming_capacity(self):
        source = {"name": "Tesla V100-SXM2-16GB", "total_memory_bytes": 16*1024**3}
        result = declare_sxm(source)
        self.assertEqual(result["form_factor_validation"]["status"], "confirmed_by_discovery")
        self.assertEqual(result["total_memory_bytes"], source["total_memory_bytes"])
        self.assertNotIn("target_form_factor", source)

    def test_hbm_named_h100_keeps_user_sxm_declaration_with_unknown_confirmation(self):
        result = declare_sxm({"name": "NVIDIA H100 80GB HBM3"})
        self.assertEqual(result["target_form_factor"], "SXM")
        self.assertEqual(result["form_factor_validation"]["status"], "declared_not_independently_verified")
        self.assertNotIn("total_memory_bytes", result)

    def test_pcie_nvl_and_conflicting_discovery_fail(self):
        for device in ({"name": "Tesla V100-PCIE-32GB"}, {"name": "NVIDIA H100 PCIe"},
                       {"name": "NVIDIA H100 NVL"}, {"form_factor": "PCIe"}):
            with self.subTest(device=device), self.assertRaises(ValueError):
                declare_sxm(device)
        with self.assertRaises(ValueError):
            declare_sxm({}, "HBM3")

    def test_tensor_ceiling_depends_on_actual_sm_count_and_clock(self):
        device = {"compute_capability": "8.0", "sm_count": 108}
        self.assertAlmostEqual(theoretical_tensor_tflops(device, 1110), 245.51424)
        self.assertAlmostEqual(theoretical_tensor_tflops(device, 1410), 311.86944)

    def test_other_named_families_cannot_inherit_scope_from_shared_compute_capability(self):
        for name in ("NVIDIA H200 SXM", "NVIDIA H800 SXM", "NVIDIA A800-SXM4-80GB",
                     "Tesla P100-SXM2-16GB", "NVIDIA B200", "NVIDIA RTX A6000", "Tesla T4"):
            device = {"name": name, "compute_capability": "9.0"}
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, "outside V100/A100/H100"):
                    architecture(device)
                with self.assertRaisesRegex(ValueError, "outside V100/A100/H100"):
                    declare_sxm(device)

    def test_generic_name_is_inferred_with_explicitly_unverified_name_metadata(self):
        device = {"name": "NVIDIA GPU", "compute_capability": "9.0"}
        result = architecture(device)
        self.assertEqual(result["family"], "H100")
        self.assertEqual(result["gpu_family_validation"]["status"], "compute_capability_inferred_name_unverified")
        self.assertEqual(declare_sxm(device)["gpu_family_validation"], result["gpu_family_validation"])

    def test_named_supported_family_must_match_discovered_compute_capability(self):
        with self.assertRaisesRegex(ValueError, "conflicts with discovered compute capability"):
            declare_sxm({"name": "NVIDIA A100-SXM4-40GB", "compute_capability": "9.0"})
        result = architecture({"name": "NVIDIA H100 80GB HBM3", "compute_capability": "9.0"})
        self.assertEqual(result["gpu_family_validation"]["status"], "name_matches_compute_capability")


if __name__ == "__main__":
    unittest.main()
