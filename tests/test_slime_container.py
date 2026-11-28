"""Tests for the slime image pieces: configs/slime_container.toml, src/slime_container.py,
containers/slime_gh200.def and the build scripts. Run: python3 tests/run_all.py

The image itself can only be built on DeltaAI; these checks make sure the recipe and the config
agree exactly before any paid build job: every placeholder in the recipe has a value from the
config, every pin in the config is used, and no version is hard-coded in the recipe.
"""
import py_compile
import re
import shutil
import subprocess
from pathlib import Path

from src.config import REPO_ROOT
from src.slime_container import RECIPE, build_args, load

CFG = load()
PLACEHOLDER = re.compile(r"\{\{\s*([A-Z0-9_]+)\s*\}\}")


def recipe_lines():
    return RECIPE.read_text().splitlines()


def code_lines():
    """Recipe lines that are not comments (the python heredocs' lines count as code)."""
    return [l for l in recipe_lines() if not l.lstrip().startswith("#")]


def test_every_placeholder_has_a_value_and_every_value_is_used():
    used = set(PLACEHOLDER.findall(RECIPE.read_text()))
    given = set(build_args(CFG))
    assert used == given, (sorted(used - given), sorted(given - used))


def test_no_placeholder_inside_a_comment():
    # Apptainer substitutes placeholders everywhere, comments included
    bad = [l for l in recipe_lines() if l.lstrip().startswith("#") and "{{" in l]
    assert not bad, bad


def test_no_version_or_commit_hard_coded_in_the_recipe():
    version = re.compile(r"(?<![\w.])\d+\.\d+\.\d+(?![\w.])")
    commit = re.compile(r"\b[0-9a-f]{40}\b")
    bad = [l for l in code_lines() if version.search(l) or commit.search(l) or "sha256:" in l]
    assert not bad, bad


def test_build_args_are_complete_and_lists_are_flattened():
    a = build_args(CFG)
    assert a["BASE_DIGEST"].startswith("sha256:") and len(a["SLIME_COMMIT"]) == 40
    assert a["MEGATRON_PATCHES"].split()[0] == "megatron.patch"     # required patch first
    assert a["SGLANG_PATCHES"].split()[0] == "sglang.patch"
    assert a["TORCH_CUDA_ARCH_LIST"] == "9.0"                        # GH200 = sm90 only
    # x86-only package removed from slime's requirements (build job 3616837 failed on it)
    assert a["SKIP_REQUIREMENTS"] == "sunabako"


def test_skip_filter_removes_only_the_named_requirement():
    """Run the recipe's grep/sed filter on slime's requirement-line shapes (needs GNU-compatible sed -E)."""
    import tempfile
    text = recipe_lines()
    start = next(i for i, l in enumerate(text) if "for n in {{ SKIP_REQUIREMENTS }}" in l)
    end = next(i for i in range(start, len(text)) if text[i].strip() == "done")
    loop = "\n".join(text[start:end + 1]).replace("{{ SKIP_REQUIREMENTS }}", "sunabako")
    with tempfile.TemporaryDirectory() as d:
        reqs = Path(d) / "requirements.txt"
        reqs.write_text("straw-queue>=0.1.5\nsunabako>=0.1.2\nsunabako-extra\nxxhash  # comment\n")
        sed_i = "sed -i ''" if subprocess.run(["sed", "--version"], capture_output=True).returncode else "sed -i"
        script = f'REQS="{reqs}"\n' + loop.replace("sed -i", sed_i)
        r = subprocess.run([shutil.which("bash"), "-c", script], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        assert reqs.read_text().split("\n")[:3] == ["straw-queue>=0.1.5", "sunabako-extra", "xxhash  # comment"]
        # a name that isn't listed stops the build
        bad = script.replace("for n in sunabako", "for n in not-a-package")
        assert subprocess.run([shutil.which("bash"), "-c", bad], capture_output=True).returncode == 1


def test_pins_follow_the_plan():
    assert CFG["base"]["tag"] == "v0.5.21" and CFG["slime"]["patch_set"] == "v0.5.21"
    assert CFG["slime"]["commit"].startswith("79aa41a")             # the slime commit studied
    # the image must hold what Qwen3.5 GRPO needs: linear-attention kernels and colocation
    assert CFG["pins"]["flash_linear_attention"] and CFG["pins"]["torch_memory_saver_commit"]


def test_build_and_gpu_check_import_the_same_modules():
    """The build defers driver-only import failures to the GPU check, so both lists must match."""
    import ast
    recipe = RECIPE.read_text()
    block = recipe[recipe.index("mods = ["):]
    build_mods = ast.literal_eval(block[len("mods = "):block.index("]") + 1])
    gpu_src = (REPO_ROOT / "scripts" / "deltaai" / "check_slime_gpu.py").read_text()
    seg = gpu_src[gpu_src.index("IMPORT_MODULES = ["):]
    gpu_mods = ast.literal_eval(seg[len("IMPORT_MODULES = "):seg.index("]") + 1])
    assert build_mods == gpu_mods, (build_mods, gpu_mods)
    assert "slime_plugins.models.qwen3_5" in gpu_mods and "fla" in gpu_mods


def test_scripts_parse():
    py_compile.compile(str(REPO_ROOT / "scripts" / "deltaai" / "check_slime_gpu.py"), doraise=True)
    py_compile.compile(str(REPO_ROOT / "scripts" / "deltaai" / "check_slime_args.py"), doraise=True)
    py_compile.compile(str(REPO_ROOT / "scripts" / "deltaai" / "convert_to_hf.py"), doraise=True)
    # no script may call slime's reverse converter directly: it reads a common.pt that Megatron
    # ae76c60e no longer writes (setup job 3631056); scripts/deltaai/convert_to_hf.py wraps it
    for f in (REPO_ROOT / "scripts" / "deltaai").iterdir():
        if f.suffix in (".sh", ".sbatch"):
            assert "/opt/slime/tools/convert_torch_dist_to_hf.py" not in f.read_text(), f.name
    bash = shutil.which("bash")
    for name in ("build_slime.sbatch", "submit.sh", "slime_common.sh", "slime_setup.sbatch",
                 "train_inside.sh", "common.sh", "serve_and_generate.sh", "smoke_train.sh", "slime_env.sh",
                 "diag_attention.sbatch", "train_reinit.sbatch", "setup_and_train.sbatch", "eval_trained.sbatch",
                 "run_control.sbatch", "smoke_test.sbatch", "hold_node.sbatch", "in_alloc.sh",
                 "install_code_server.sh"):
        r = subprocess.run([bash, "-n", str(REPO_ROOT / "scripts" / "deltaai" / name)], capture_output=True, text=True)
        assert r.returncode == 0, (name, r.stderr)


# ---------------------------------------------------------------- cuDNN (diagnosis job 3686686)
SLIME_ENV = REPO_ROOT / "scripts" / "deltaai" / "slime_env.sh"


def _run_slime_env(pythonpath, ld="/usr/lib/x"):
    import os
    import sys
    import tempfile
    env = {"PATH": os.path.dirname(sys.executable) + ":/usr/bin:/bin", "LD_LIBRARY_PATH": ld}
    if pythonpath:
        env["PYTHONPATH"] = pythonpath
    shim = tempfile.mkdtemp()   # `python3` inside slime_env.sh = this interpreter
    (Path(shim) / "python3").symlink_to(sys.executable)
    env["PATH"] = shim + ":" + env["PATH"]
    return subprocess.run([shutil.which("bash"), str(SLIME_ENV), sys.executable, "-c",
                           "import os; print(os.environ['CUDNN_HOME']); print(os.environ['LD_LIBRARY_PATH'])"],
                          capture_output=True, text=True, env=env)


def test_slime_env_points_cudnn_at_pytorchs_copy():
    """CUDNN_HOME = site-packages/nvidia/cudnn, its lib/ FIRST on LD_LIBRARY_PATH, rest kept."""
    import tempfile
    site = tempfile.mkdtemp()
    (Path(site) / "nvidia" / "cudnn" / "lib").mkdir(parents=True)   # namespace packages, like the pip wheel
    r = _run_slime_env(site)
    assert r.returncode == 0, r.stderr
    home, ld = r.stdout.split()
    assert Path(home) == Path(site) / "nvidia" / "cudnn"
    assert ld.split(":")[0] == str(Path(home) / "lib") and ld.split(":")[-1] == "/usr/lib/x"


def test_slime_env_refuses_without_pytorchs_cudnn():
    import tempfile
    r = _run_slime_env(tempfile.mkdtemp())
    assert r.returncode != 0 and "nvidia.cudnn" in r.stderr


def test_every_slime_image_gpu_command_goes_through_slime_env():
    """S() and the build job's GPU check run slime_env.sh (the mixed-cuDNN fix)."""
    common = (REPO_ROOT / "scripts" / "deltaai" / "slime_common.sh").read_text()
    s_body = common[common.index("S() {"):common.index("\n}", common.index("S() {"))]
    assert 'slime_env.sh" "$@"' in s_body
    build = (REPO_ROOT / "scripts" / "deltaai" / "build_slime.sbatch").read_text()
    gpu_line = build[build.index("# ---- step 5"):build.index("check_slime_gpu.py", build.index("# ---- step 5"))]
    assert "slime_env.sh" in gpu_line
    gpu_src = (REPO_ROOT / "scripts" / "deltaai" / "check_slime_gpu.py").read_text()
    assert "check_cudnn):" in gpu_src   # the consistency check runs last, after TE has loaded cuDNN


def test_flag_check_parses_with_trainings_cuda_device_max_connections():
    """Megatron refuses tensor parallel > 1 unless CUDA_DEVICE_MAX_CONNECTIONS=1 (setup job 3686748):
    the setup job's flag check must set the same value train_inside.sh exports for training."""
    deltaai = REPO_ROOT / "scripts" / "deltaai"
    inside = re.search(r"^export .*CUDA_DEVICE_MAX_CONNECTIONS=(\S+)", (deltaai / "train_inside.sh").read_text(), re.M)
    assert inside, "train_inside.sh no longer exports CUDA_DEVICE_MAX_CONNECTIONS"
    setup = (deltaai / "slime_setup.sbatch").read_text()
    call = setup[setup.rindex("\nS env", 0, setup.index("check_slime_args.py \"")):setup.index("check_slime_args.py \"")]
    assert f"CUDA_DEVICE_MAX_CONNECTIONS={inside.group(1)}" in call


def test_setup_marker_is_written_once_the_round_trip_passed():
    """setup_pass = "this Megatron copy is verified": written right after the round-trip check, before
    the flag check (step 4), so a step-4 failure does not leave an unmarked copy that blocks the
    rerun (job 3686748). Only one write, inside the conversion branch (not on a reuse)."""
    s = (REPO_ROOT / "scripts" / "deltaai" / "slime_setup.sbatch").read_text()
    writes = [m.start() for m in re.finditer(r'> "\$REF_DIR\.setup_pass"', s)]
    assert len(writes) == 1
    assert s.index('"$RT_STATUS" -ne 0') < writes[0] < s.index("fi   # steps 2-3") < s.index("# ---- 4")


# ---------------------------------------------------------------- held node + code-server (2026-11-27)
def _code_server_value(key):
    m = re.search(rf'(?m)^{key} = "(.*)"$', (REPO_ROOT / "configs" / "code_server.toml").read_text())
    assert m, key
    return m.group(1)


def test_code_server_is_pinned_and_checked():
    """The download is one exact arm64 release whose SHA-256 is recorded; the installer refuses others."""
    version, url, sha = _code_server_value("version"), _code_server_value("url"), _code_server_value("sha256")
    assert re.fullmatch(r"\d+\.\d+\.\d+", version) and re.fullmatch(r"[0-9a-f]{64}", sha)
    assert url.endswith(f"/v{version}/code-server-{version}-linux-arm64.tar.gz")
    assert 1024 <= int(_code_server_value("port")) < 65536
    inst = (REPO_ROOT / "scripts" / "deltaai" / "install_code_server.sh").read_text()
    assert inst.index('sha256sum "$TMP"') < inst.index("tar -xzf")   # checked before unpacking
    hold = (REPO_ROOT / "scripts" / "deltaai" / "hold_node.sbatch").read_text()
    assert '--bind-addr "127.0.0.1:$PORT"' in hold and "--auth password" in hold   # loopback + password only
    assert 'PASSWORD="$(cat "$CS_HOME/password")"' in hold   # password via env, not on the command line


def test_in_alloc_refuses_outside_an_allocation():
    import os
    env = {k: v for k, v in os.environ.items() if not k.startswith("SLURM_")}
    r = subprocess.run([shutil.which("bash"), str(REPO_ROOT / "scripts" / "deltaai" / "in_alloc.sh"),
                        "scripts/deltaai/slime_setup.sbatch"], capture_output=True, text=True, env=env)
    assert r.returncode == 1 and "not inside a Slurm allocation" in r.stdout


def test_in_alloc_gives_each_attempt_its_own_id_and_the_allocations_end():
    s = (REPO_ROOT / "scripts" / "deltaai" / "in_alloc.sh").read_text()
    assert 'ATTEMPT="${ALLOC}r$(date +%m%d%H%M%S)"' in s and 'SLURM_JOB_ID="$ATTEMPT"' in s
    assert 'SLURM_JOB_END_TIME="$END_S"' in s and "nohup setsid bash" in s
