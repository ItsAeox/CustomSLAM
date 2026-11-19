"""gdb script (hang_stress.sh, WASM runs): the stack of every thread of a hung Node process, including the V8 JIT frames
(wasm functions, builtins) gdb cannot unwind. gdb unwinds the native frames; from the first frame in JIT code, or the
innermost V8 builtin frame (both keep the rbp chain), the walk follows the saved rbp: return pc = [rbp+8], caller rbp =
[rbp]. Each line "  pc name": native names from gdb, "?" for JIT code; hang_stacks.py names those from V8's perf map
(node --perf-basic-prof) and the snapshot's orbslam3_run.js.symbols.
Usage: gdb -p <pid> -batch -x hang_gdb.py"""
import gdb

inf = gdb.selected_inferior()


def name_of(pc):
    s = gdb.execute(f"info symbol {pc:#x}", to_string=True).strip()
    return "?" if s.startswith("No symbol") else s.split(" in section")[0]


def read_u64(a):
    return int.from_bytes(bytes(inf.read_memory(a, 8)), "little")


for t in sorted(inf.threads(), key=lambda t: t.num):
    t.switch()
    print(f"=== thread {t.num} lwp {t.ptid[1]} {t.name or ''}")
    f, start = gdb.newest_frame(), None
    for _ in range(40):
        name = f.name()
        if name is None:  # JIT code: gdb's unwind stops being reliable here
            print(f"  {f.pc():#x} ?")
            start = f
            break
        print(f"  {f.pc():#x} {name}")
        if name.startswith("Builtins_"):
            start = f
            break
        try:
            f = f.older()
        except gdb.error:
            f = None
        if f is None:
            break
    if start is None:
        continue
    try:
        rbp = int(start.read_register("rbp"))
        for _ in range(80):
            if rbp == 0 or rbp & 7:
                break
            pc, rbp = read_u64(rbp + 8), read_u64(rbp)
            if pc == 0:
                break
            print(f"  {pc:#x} {name_of(pc)}")
    except (gdb.error, gdb.MemoryError) as e:
        print(f"  (walk stopped: {e})")
