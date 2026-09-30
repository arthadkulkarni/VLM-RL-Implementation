#!/bin/bash
# Source from the repo root to put the RISE venv on PATH.
#
# Uses the /work/nvme copy of the venv: on /work/hdd, 4 servers importing
# torch/vLLM at once took ~13 min per launch (job 3207016), and Ray's
# raylet/GCS imports from there made ray.init time out (jobs 3246829, 3258420).
#
# Don't `source bin/activate`: the nvme venv was copied from /work/hdd and its
# activate script and bin/ shebangs still hardcode the hdd path, so activate
# would silently put the hdd venv first on PATH. Putting the nvme bin/ on PATH
# directly makes python3 resolve to the nvme symlink, whose pyvenv.cfg gives
# it the nvme site-packages. Console scripts (ray, vllm, ...) still point at
# the hdd venv, so call them as `python3 -m <module>`.

NVME_ROOT="${NVME_ROOT:-/work/nvme/bffz/akulkarni8/rise}"
RISE_ENV_DIR="${RISE_ENV_DIR:-${NVME_ROOT}/envs/RISE}"
export VIRTUAL_ENV="${RISE_ENV_DIR}"
export PATH="${RISE_ENV_DIR}/bin:${PATH}"
unset PYTHONHOME

py_prefix="$(python3 -c 'import sys; print(sys.prefix)')"
echo "python: $(command -v python3) (prefix ${py_prefix})"
if [ "${py_prefix}" != "${RISE_ENV_DIR}" ]; then
  echo "ERROR: python3 resolves to ${py_prefix}, expected ${RISE_ENV_DIR}" >&2
  exit 1
fi
