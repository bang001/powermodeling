"""Synthetic disassembly counterexamples; no GPU claims."""
import unittest
from tools.check_memory_read_sass import check_sass


def suite():
    parts = []
    for l1, policy in ((1, 0), (0, 0), (0, 1), (0, 2)):
        for narrow in (0, 1):
            opcode = ("LDG.E.STRONG.SM", "LDG.E.STRONG.GPU", "LDG.E.EF")[policy]
            instructions = [opcode + " R2, [R4]", "IADD3 R3, R3, R2, RZ"]
            instructions += ["NOP"] * (3 if narrow else 12)
            instructions.append("IADD3 R8, R8, 0x1, RZ")
            instructions.append("BRA 0x0")
            parts.append(f"Function : _Z20simple_memory_kernelILb{l1}ELi{policy}ELb{narrow}EEv\nEF_CUDA_SM90\n"
                         + "\n".join(f"/*{index * 16:04x}*/ {ins};" for index, ins in enumerate(instructions)))
    return "\n".join(parts)


class MemoryReadSassTests(unittest.TestCase):
    def test_all_policy_and_index_specializations_are_checked(self):
        checks, failures = check_sass(suite())
        self.assertEqual(failures, [])
        self.assertEqual(len(checks), 8)
        self.assertTrue(any("cs uint32" in item for item in checks))

    def test_missing_streaming_variant_cannot_pass(self):
        text = suite()
        text = text[:text.index("Function : _Z20simple_memory_kernelILb0ELi2ELb1EEv")]
        self.assertTrue(any("missing" in issue and "cs uint32" in issue for issue in check_sass(text)[1]))

    def test_fast_path_instruction_regression_is_rejected(self):
        text = suite().replace("/*0060*/ BRA 0x0;", "\n".join(f"/*{i * 16:04x}*/ NOP;" for i in range(6, 18)) + "\n/*0120*/ BRA 0x0;")
        self.assertTrue(any("instruction" in issue for issue in check_sass(text)[1]))

    def test_local_or_xor_in_any_policy_is_rejected(self):
        for instruction in ("LDL R2, [R4]", "XOR R3, R2, R1"):
            with self.subTest(instruction=instruction):
                self.assertTrue(check_sass(suite().replace("IADD3 R3, R3, R2, RZ", instruction))[1])

    def test_retaining_one_load_of_four_claimed_iterations_is_rejected(self):
        text = suite().replace("IADD3 R8, R8, 0x1, RZ", "IADD3 R8, R8, 0x4, RZ")
        self.assertTrue(any("iteration" in issue for issue in check_sass(text)[1]))

    def test_wrong_cache_opcode_cannot_certify_streaming_variant(self):
        text = suite().replace("LDG.E.EF", "LDG.E.STRONG.GPU")
        self.assertTrue(any("cache policy" in issue for issue in check_sass(text)[1]))
