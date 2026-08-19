# Conda Environment Failures

## 1. Shield user-site pollution

**Symptom**: `import numpy` picks up wrong version from `~/.local/`, pip says `Requirement already satisfied ... in ~/.local/`.

**Root cause**: The user site remains enabled and can leak packages into the
active environment.

**Fix** — add activate hook to permanently set `PYTHONNOUSERSITE=1`:
```bash
mkdir -p "$CONDA_PREFIX/etc/conda/activate.d"
cat > "$CONDA_PREFIX/etc/conda/activate.d/zz_disable_usersite.sh" << 'HOOK'
export PYTHONNOUSERSITE=1
HOOK
```
**Verify**: `python -c "import site; assert not site.ENABLE_USER_SITE"`

After shielding, reinstall packages that previously resolved from user-site:
```bash
python -m pip install --force-reinstall <package>
```

---

## 2. Broken conda shebang (root-owned, read-only)

**Symptom**: `conda` fails with `bad interpreter: /workspace/miniconda3/bin/python: No such file or directory`. Files root-owned, no sudo.

**Root cause**: conda installed by root on another machine and copied. All `bin/*` shebangs hardcoded to original path.

Invoke conda through its surviving Python interpreter:
```bash
CONDA_ROOT=~/path/to/miniconda3
PY=$CONDA_ROOT/bin/python3.10
$PY -m conda --version
```

```bash
$PY -m conda create -n <env-name> python=3.10.13 -y
```

If normal activation is unavailable, write a PATH-only fallback:
```bash
cat > ~/activate_<env-name>.sh << 'EOF'
#!/bin/bash
export ENV="$HOME/.conda/envs/<env-name>"
export PATH="$ENV/bin:$PATH"
export PYTHONNOUSERSITE=1
export CONDA_PREFIX="$ENV"
echo "[<env-name>] activated: $(python --version 2>&1)"
EOF
chmod +x ~/activate_<env-name>.sh
source ~/activate_<env-name>.sh
```

---

## 3. Missing system commands on dev machines

**Symptom**: `unzip: command not found` / `make: command not found`, no sudo.

**Fix**: install the command into the active environment:

```bash
conda install -y -c conda-forge unzip make cmake gcc gxx git curl
```

Fallback: use Python stdlib (`zipfile` for unzip, `shutil` for file ops).
