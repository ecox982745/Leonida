"""Run synthetic RDNA2 swap-x2 tests using Leonida's translator and Vulkan compute harness.

python -I scripts/check_swap_x2_contention.py BASE PR_ON_BASE VK_COMPUTE_TEST --groups 128 --repeats 5
Requires llvm-mc, spirv-as, spirv-val and spirv-dis on PATH.
"""
import argparse
from collections import Counter
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile


MASK = 0xffffffff
SEED = (0, MASK)
GUARD = b"SWAPtest"


def shader(kind, private, lanes):
    address = (f"v_add_nc_u32_e32 v2, {16 + 8 * lanes}, v8" if private else "v_mov_b32 v2, 0")
    swap = ("buffer_atomic_swap_x2 v[4:5], v2, s[0:3], 0 offen glc" if kind == "buffer" else
            "global_atomic_swap_x2 v[4:5], v2, v[4:5], s[20:21] glc")
    return "\n".join([
        "; group: 64,1,1", "s_mov_b32 s20, 0", "s_mov_b32 s21, 0",
        "v_lshl_add_u32 v1, s4, 6, v0", "v_lshlrev_b32_e32 v8, 3, v1", address,
        "v_add_nc_u32_e32 v4, 1, v1", "v_not_b32_e32 v5, v4", swap, "s_waitcnt vmcnt(0)",
        "buffer_store_dwordx2 v[4:5], v8, s[0:3], 0 offen offset:16", "s_endpgm", "",
    ])


def run(*args, env=None):
    result = subprocess.run(args, capture_output=True, text=True, env=env, timeout=120)
    if result.returncode:
        raise RuntimeError(f"{args[0]} exited {result.returncode}\n{result.stdout}\n{result.stderr}")
    return result.stdout + result.stderr


def check(data, lanes, private):
    returns_end = 16 + 8 * lanes
    expected_bytes = returns_end + (8 * lanes if private else 0) + len(GUARD)
    if len(data) != expected_bytes or data[8:16] != GUARD or data[-8:] != GUARD:
        raise ValueError("Unexpected output size or modified guard")
    previous = list(struct.iter_unpack("<II", data[16:returns_end]))
    final = struct.unpack_from("<II", data)
    if private:
        slots = list(struct.iter_unpack("<II", data[returns_end:-8]))
        if final != SEED or previous != [SEED] * lanes or any(
                pair != (i, MASK ^ i) for i, pair in enumerate(slots, 1)):
            raise ValueError("Private-address control failed")
        return 0, 0
    all_pairs = previous + [final]
    # Even split atomics must conserve every half exactly once, including the initial and final values.
    expected = Counter(range(lanes + 1))
    if Counter(lo for lo, _ in all_pairs) != expected or Counter(MASK ^ hi for _, hi in all_pairs) != expected:
        raise ValueError("Lost or duplicated a word; the result cannot be attributed solely to split exchanges")
    torn = sum(lo != MASK ^ hi for lo, hi in all_pairs)
    node = final[0]
    visited = set()
    while node and node not in visited:
        visited.add(node)
        node = previous[node - 1][0]
    if node or len(visited) != lanes:
        raise ValueError("The low words do not describe a single exchange order")
    return torn, int(final[0] != MASK ^ final[1])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("base", type=Path)
    ap.add_argument("patched", type=Path)
    ap.add_argument("harness", type=Path)
    ap.add_argument("--groups", type=int, default=128)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--device", default="RTX 4090")
    ap.add_argument("--assembler", choices=("spirv-as", "python"), default="spirv-as")
    args = ap.parse_args()
    if not (1 <= args.groups <= 16384 and 1 <= args.repeats <= 100):
        ap.error("use 1..16384 workgroups and 1..100 repetitions")
    base, patched, harness = (p.resolve() for p in (args.base, args.patched, args.harness))
    sys.path.insert(0, str(base / "scripts"))
    import run_gpu_tests as runner
    lanes = args.groups * 64
    with tempfile.TemporaryDirectory(prefix="leonida-swap-") as tmp:
        root = Path(tmp)
        disasm, headers = root / "disasm", root / "fssm"
        disasm.mkdir()
        headers.mkdir()
        cases = {}
        for kind in ("global", "buffer"):
            for private in (True, False):
                name = f"cs_{kind}_{'private' if private else 'shared'}"
                asm = root / f"{name}.asm"
                asm.write_text(shader(kind, private, lanes), encoding="ascii", newline="\n")
                runner.assemble_to_s("llvm-mc", str(asm), str(disasm / f"{name}.s"), name)
                (root / f"{name}.dxbc").write_bytes(runner.make_dxbc([]))
                (headers / f"{name}.fssm").write_bytes(runner.compute_header(asm))
                cases[name] = private
        for label, repo, flags in (("main", base, []), ("pr-default", patched, []),
                                   ("pr-int64", patched, ["--int64-atomics"])):
            out = root / label
            run(sys.executable, "-I", repo / "tools/rdna2spirv.py", disasm, "--out", out,
                "--assembler", args.assembler, *flags)
            for name, private in cases.items():
                binary = out / f"{name}.spv"
                run("spirv-val", "--target-env", "vulkan1.2", binary)
                dis = run("spirv-dis", binary)
                exchanges = [line.split("OpAtomicExchange ")[1].split()[0] for line in dis.splitlines()
                             if "OpAtomicExchange " in line]
                if exchanges != (["%ulong"] if flags else ["%uint", "%uint"]):
                    raise ValueError(f"{label}/{name}: unexpected exchange instructions {exchanges}")
                init = struct.pack("<II", *SEED) + GUARD + bytes([0xee]) * (8 * lanes)
                if private:
                    init += struct.pack("<II", *SEED) * lanes
                init += GUARD
                heap = root / "heap.bin"
                dump = root / "output.bin"
                heap.write_bytes(init)
                env = dict(os.environ, LEONIDA_VK_DEVICE=args.device, VKC_DUMP=str(dump), VKC_HEAP_INIT=str(heap),
                           VKC_OUT_WORDS=str(len(init) // 4))
                counts = []
                finals = 0
                for _ in range(1 if private else args.repeats):
                    dump.unlink(missing_ok=True)
                    log = run(harness, binary, str(args.groups), "dump", env=env)
                    if args.device.lower() not in log.lower():
                        raise ValueError("Selected Vulkan device was not reported: " + log)
                    torn, final_torn = check(dump.read_bytes(), lanes, private)
                    counts.append(torn)
                    finals += final_torn
                print(f"{label} {name}: {lanes} writers per dispatch, torn pairs {counts}, torn finals {finals}",
                      flush=True)
                if flags and any(counts):
                    raise ValueError("Native exchange returned torn pairs")


if __name__ == "__main__":
    main()
