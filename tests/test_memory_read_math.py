"""Run the production narrow-index arithmetic on a CPU at uint32 boundaries."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


@unittest.skipUnless(shutil.which("g++"), "requires a host C++ compiler")
class MemoryReadMathTests(unittest.TestCase):
    def test_uint32_wrap_and_dispatch_boundaries_match_uint64_oracle(self):
        root = Path(__file__).resolve().parents[1]
        program = r'''
#include "cuda/memory_read_math.cuh"
#include <cassert>
#include <cstdint>
#include <vector>
int main() {
  const uint64_t maximum = UINT32_MAX;
  assert(memory_read_uses_uint32(maximum, maximum));
  assert(!memory_read_uses_uint32(maximum + 1, 1));
  assert(!memory_read_uses_uint32(1, maximum + 1));
  for (uint32_t n : {1u, 2u, 3u, 10u, (1u << 31), UINT32_MAX}) {
    for (uint32_t advance : {0u, n / 2, n - 1}) {
      for (uint32_t initial : {0u, n / 2, n - 1}) {
        uint32_t actual = initial;
        uint64_t expected = initial;
        for (unsigned i = 0; i < 129; ++i) {
          actual = memory_read_next32(actual, advance, n - advance);
          expected = (expected + advance) % n;
          assert(actual == expected);
        }
      }
    }
  }
}
'''
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "arithmetic.cpp"
            binary = Path(directory) / "arithmetic"
            source.write_text(program)
            compiled = subprocess.run(["g++", "-std=c++17", "-I", str(root), str(source), "-o", str(binary)],
                                      capture_output=True, text=True)
            self.assertEqual(compiled.returncode, 0, compiled.stderr)
            result = subprocess.run([str(binary)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
