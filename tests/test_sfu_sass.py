"""Machine-code evidence rejects dropped work, memory traffic, and forged flags."""
import copy
import unittest

from powermodeling.sfu_sass import OPERATIONS, extract_sfu_disassembly, make_certificate, select_certificate, validate_certificate


def make_sfu_sass_certificate(workload="sfu_ex2", chains=4, cc="80", sha256="a" * 64):
    """Small synthetic disassembly fixture; never GPU measurement evidence."""
    operation = next(op for op, (name, _) in OPERATIONS.items() if name == workload)
    sass, resources = [], []
    for control, op in ((False, operation), (True, 0)):
        name = f"_Z19sfu_register_kernelILi{op}ELb{int(control)}ELi{chains}EEvPjmj"
        instructions = [(0, "IMAD.MOV.U32 R0, RZ, RZ, RZ")]
        address = 0x10
        if not control:
            for chain in range(chains):
                instructions.append((address, f"MUFU.{OPERATIONS[op][1]} R{chain+2}, R{chain+2}"))
                address += 0x10
        instructions += [(address, "LOP3.LUT R2, R2, 0x7fffff, R0, 0x48, !PT"),
                         (address + 0x10, "IADD3 R0, P0, R0, 0x1, RZ"),
                         (address + 0x20, "ISETP.LT.U32.AND P0, PT, R0, R9, PT"),
                         (address + 0x30, "@P0 BRA 0x10"),
                         (address + 0x40, "STG.E [R10.64], R2"),
                         (address + 0x50, "EXIT")]
        sass.append(f'Function : {name}\n.headerflags @"EF_CUDA_SM{cc}"\n' +
                    "\n".join(f"/*{a:04x}*/ {i};" for a, i in instructions))
        resources.append(f"arch = sm_{cc}\n Function {name}:\n REG:16 STACK:0 SHARED:0 LOCAL:0")
    return make_certificate("\n".join(sass), "\n".join(resources), sha256)


class SfuSassTests(unittest.TestCase):
    def issues(self, certificate, **changes):
        args = dict(benchmark_sha256="a" * 64, cc="8.0", workload="sfu_ex2", chains=4)
        args.update(changes)
        return validate_certificate(certificate, **args)

    def test_valid_target_and_control_evidence(self):
        self.assertEqual(self.issues(make_sfu_sass_certificate()), [])

    def test_summary_flags_cannot_hide_removed_native_instruction(self):
        certificate = make_sfu_sass_certificate()
        certificate["raw_sass"] = certificate["raw_sass"].replace("MUFU.EX2 R2, R2", "MOV R2, R2", 1)
        certificate["status"] = "pass"
        self.assertIn("sfu_instruction_count_or_opcode_mismatch", self.issues(certificate))

    def test_memory_in_recurrence_rejects(self):
        certificate = make_sfu_sass_certificate()
        certificate["raw_sass"] = certificate["raw_sass"].replace("LOP3.LUT R2, R2, 0x7fffff, R0, 0x48, !PT", "LDG.E R2, [R8.64]", 1)
        self.assertIn("sfu_hot_loop_memory_or_missing_loop", self.issues(certificate))

    def test_spills_or_missing_resources_reject(self):
        certificate = make_sfu_sass_certificate()
        certificate["raw_resources"] = certificate["raw_resources"].replace("LOCAL:0", "LOCAL:8", 1)
        self.assertIn("sfu_spills_or_missing_resources", self.issues(certificate))
        certificate["raw_resources"] = ""
        self.assertIn("sfu_spills_or_missing_resources", self.issues(certificate))

    def test_control_must_have_no_sfu_instruction(self):
        certificate = make_sfu_sass_certificate()
        first, control = certificate["raw_sass"].split("Function : ", 2)[1:]
        control = control.replace("LOP3.LUT R2, R2, 0x7fffff, R0, 0x48, !PT", "MUFU.EX2 R2, R2", 1)
        certificate["raw_sass"] = "Function : " + first + "Function : " + control
        self.assertIn("sfu_sass_control_not_verified", self.issues(certificate))

    def test_binary_architecture_operation_and_chains_bind(self):
        certificate = make_sfu_sass_certificate()
        for changed in (dict(benchmark_sha256="b" * 64), dict(cc="9.0"), dict(workload="sfu_rcp"), dict(chains=8)):
            with self.subTest(changed=changed):
                self.assertTrue(self.issues(certificate, **changed))

    def test_predicated_or_wrong_opcode_rejects(self):
        for replacement in ("@P1 MUFU.EX2", "MUFU.SIN"):
            certificate = make_sfu_sass_certificate()
            certificate["raw_sass"] = certificate["raw_sass"].replace("MUFU.EX2", replacement, 1)
            self.assertTrue(self.issues(certificate))

    def test_unroll_or_extra_control_flow_rejects(self):
        for replacement in ("IADD3 R0, P0, R0, 0x4, RZ", "CALL 0x200"):
            certificate = make_sfu_sass_certificate()
            certificate["raw_sass"] = certificate["raw_sass"].replace("IADD3 R0, P0, R0, 0x1, RZ", replacement, 1)
            self.assertTrue(self.issues(certificate))

    def test_declared_fail_flag_does_not_override_valid_raw_proof(self):
        certificate = make_sfu_sass_certificate()
        certificate["status"] = "fail"
        certificate["kernels"] = []
        self.assertEqual(self.issues(certificate), [])

    def test_duplicate_function_is_ambiguous(self):
        certificate = make_sfu_sass_certificate()
        certificate["raw_sass"] += "\n" + certificate["raw_sass"]
        self.assertIn("duplicate_sfu_kernel_specialization", self.issues(certificate))

    def test_missing_or_invalid_evidence(self):
        self.assertEqual(self.issues(None), ["missing_sfu_sass_evidence"])
        certificate = copy.deepcopy(make_sfu_sass_certificate())
        certificate.pop("raw_sass")
        self.assertIn("missing_raw_sfu_disassembly", self.issues(certificate))

    def test_common_control_work_is_checked(self):
        certificate = make_sfu_sass_certificate()
        certificate["raw_sass"] = certificate["raw_sass"].replace("LOP3.LUT R2, R2, 0x7fffff, R0, 0x48, !PT", "NOP", 1)
        self.assertIn("sfu_target_control_common_instruction_mismatch", self.issues(certificate))

    def test_compaction_recomputes_valid_evidence(self):
        certificate = make_sfu_sass_certificate()
        sass, resources = extract_sfu_disassembly(certificate["raw_sass"], certificate["raw_resources"])
        self.assertEqual(self.issues(make_certificate(sass, resources, "a" * 64)), [])
        self.assertEqual(self.issues(select_certificate(certificate, "8.0", "sfu_ex2", 4)), [])

    def test_nonunit_increment_and_memory_like_unknown_opcode_fail(self):
        certificate = make_sfu_sass_certificate()
        certificate["raw_sass"] = certificate["raw_sass"].replace("LOP3.LUT R2, R2, 0x7fffff, R0, 0x48, !PT", "SUST.B [R8], R2", 1)
        self.assertIn("sfu_hot_loop_memory_or_missing_loop", self.issues(certificate))


if __name__ == "__main__":
    unittest.main()
