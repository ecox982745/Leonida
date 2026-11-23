"""Tests for --int64-atomics (64-bit OpAtomicExchange for *_atomic_swap_x2)."""

import contextlib
import os
import pathlib
import struct
import subprocess
import sys
import tempfile
import unittest
import warnings

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "tools"))
import rdna2spirv  # noqa: E402

FSSM = bytearray(0x148)
struct.pack_into("<4I", FSSM, 0x138, 0x88, 32, 1, 1)     # 4 user SGPRs, workgroup id in s4, 32x1x1
FSSM = bytes(FSSM)


@contextlib.contextmanager
def _int64_atomics_enabled(enabled: bool):
    prev = rdna2spirv.INT64_ATOMICS
    rdna2spirv.INT64_ATOMICS = enabled
    try:
        yield
    finally:
        rdna2spirv.INT64_ATOMICS = prev


def _translate_cs(body: list[str], *, int64_atomics: bool):
    with tempfile.TemporaryDirectory() as tmp:
        os.makedirs(os.path.join(tmp, "fssm"))
        os.makedirs(os.path.join(tmp, "disasm"))
        with open(os.path.join(tmp, "fssm", "cs_test.fssm"), "wb") as f:
            f.write(FSSM)
        src = os.path.join(tmp, "disasm", "cs_test.s")
        with open(src, "w", encoding="utf-8") as f:
            f.write("; cs_test\n" + "".join(f"\t{ln}\n" for ln in body) + "\ts_endpgm\n")
        with _int64_atomics_enabled(int64_atomics), warnings.catch_warnings():
            warnings.simplefilter("ignore", ResourceWarning)
            return rdna2spirv.translate(src)


class AtomicSwapX2Int64Tests(unittest.TestCase):
    def test_buffer_atomic_swap_x2_uses_int64_atomic_when_enabled(self):
        body = [
            "s_mov_b32 s4, 0",
            "s_mov_b32 s5, 0",
            "s_mov_b32 s6, 64",
            "s_mov_b32 s7, 0x31016fac",
            "v_mov_b32 v0, 0x11223344",
            "v_mov_b32 v1, 0x55667788",
            "v_mov_b32 v2, 8",
            "buffer_atomic_swap_x2 v[0:1], v2, s[4:7], 0 offen glc",
        ]
        spv_on = _translate_cs(body, int64_atomics=True)
        self.assertIn("OpCapability Int64Atomics", spv_on)
        self.assertIn("%_ptr_PSB_ulong = OpTypePointer PhysicalStorageBuffer %ulong", spv_on)
        self.assertIn("= OpAtomicExchange %ulong", spv_on)

        spv_off = _translate_cs(body, int64_atomics=False)
        self.assertNotIn("Int64Atomics", spv_off)
        self.assertNotIn("OpAtomicExchange %ulong", spv_off)

    def test_global_atomic_swap_x2_uses_int64_atomic_when_enabled(self):
        body = [
            "v_mov_b32 v0, 0",
            "v_mov_b32 v1, 0",
            "v_mov_b32 v2, 0xdeadbeef",
            "v_mov_b32 v3, 0xcafebabe",
            "global_atomic_swap_x2 v[4:5], v[0:1], v[2:3], off glc",
        ]
        spv_on = _translate_cs(body, int64_atomics=True)
        self.assertIn("OpCapability Int64Atomics", spv_on)
        self.assertIn("= OpAtomicExchange %ulong", spv_on)

    def test_bitwise_x2_keeps_32bit_form_even_when_int64_atomics_enabled(self):
        body = [
            "s_mov_b32 s4, 0",
            "s_mov_b32 s5, 0",
            "s_mov_b32 s6, 64",
            "s_mov_b32 s7, 0x31016fac",
            "v_mov_b32 v0, 0xff",
            "v_mov_b32 v1, 0xff",
            "v_mov_b32 v2, 0",
            "buffer_atomic_and_x2 v[0:1], v2, s[4:7], 0 offen glc",
        ]
        spv_on = _translate_cs(body, int64_atomics=True)
        self.assertNotIn("Int64Atomics", spv_on)
        self.assertNotIn("OpAtomicExchange %ulong", spv_on)

    def test_cli_flag_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            in_dir = os.path.join(tmp, "in")
            os.makedirs(os.path.join(in_dir, "fssm"))
            os.makedirs(os.path.join(in_dir, "disasm"))
            out_dir = os.path.join(tmp, "out")
            with open(os.path.join(in_dir, "fssm", "cs_cli.fssm"), "wb") as f:
                f.write(FSSM)
            with open(os.path.join(in_dir, "disasm", "cs_cli.s"), "w", encoding="utf-8") as f:
                f.write("; cs_cli\n\t" + "\n\t".join([
                    "s_mov_b32 s4, 0",
                    "s_mov_b32 s5, 0",
                    "s_mov_b32 s6, 64",
                    "s_mov_b32 s7, 0x31016fac",
                    "v_mov_b32 v0, 1",
                    "v_mov_b32 v1, 2",
                    "v_mov_b32 v2, 8",
                    "buffer_atomic_swap_x2 v[0:1], v2, s[4:7], 0 offen glc",
                    "s_endpgm",
                ]) + "\n")
            translator = pathlib.Path(__file__).resolve().parents[2] / "tools" / "rdna2spirv.py"
            res = subprocess.run(
                [sys.executable, "-I", str(translator), os.path.join(in_dir, "disasm"),
                 "--out", out_dir, "--int64-atomics"],
                capture_output=True, text=True, check=False,
            )
            if res.returncode != 0:
                self.assertTrue(
                    "FileNotFoundError" in res.stderr or "not found" in res.stderr,
                    res.stderr,
                )
            spv_path = next(pathlib.Path(out_dir).rglob("*.spvasm"))
            spv_text = spv_path.read_text(encoding="utf-8")
            self.assertIn("OpCapability Int64Atomics", spv_text)
            self.assertIn("= OpAtomicExchange %ulong", spv_text)


if __name__ == "__main__":
    unittest.main()
