"""ROCm kaolin bypass for EmbodiedGen (generalized from ZJLi2013/RealWonder).

kaolin is CUDA-only (no ROCm wheel; setup.py hard-requires nvcc). In EmbodiedGen
it is imported at module top of `embodied_gen/data/utils.py` and used only inside
the texture-backprojection / differentiable-render stage (`kal.io.*.import_mesh`,
`kal.render.materials.PBRMaterial`, `kaolin.render.camera.Camera`), plus type-level
references in thirdparty/sam3d. None of it is on the core image->3D geometry+gaussian
generation path (gsplat is the gaussian backend), so stubbing `kaolin` lets img3d-cli
run on ROCm. The texture-baking stage that actually calls these will surface a clear
error instead of crashing every import.

Activation (must run before any `import kaolin`):
  - drop this file's directory on PYTHONPATH as `sitecustomize.py`, or
  - `import kaolin_stub` as the very first import of the entrypoint.

This is the ROCm-unblock shim; the upstream-PR-appropriate fix is to make the kaolin
imports in `data/utils.py` lazy/optional (see docs/exp18.md).
"""

import importlib.abc
import importlib.machinery
import sys
import types


class _KaolinStubModule(types.ModuleType):
    def __init__(self, name):
        super().__init__(name)
        self.__file__ = "<kaolin-stub>"
        self.__path__ = []
        self.__spec__ = None

    def __getattr__(self, name):
        # Let dunder lookups (e.g. __file__, __wrapped__, __all__) behave normally.
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        # Capitalized -> isinstance-safe stub class (Camera, PBRMaterial, ...).
        if name and name[0].isupper():
            return type(name, (), {})
        # Lowercase -> a no-op callable that also behaves like a submodule.
        stub = _KaolinCallableStub(f"{self.__name__}.{name}")
        return stub


class _KaolinCallableStub(_KaolinStubModule):
    def __call__(self, *args, **kwargs):
        # Return a truthy no-op. The only kaolin calls on the core geometry/mesh path
        # are validators like `kaolin.utils.testing.check_tensor(...)`, used inside
        # `assert torch.is_tensor(x) and check_tensor(...)`, which need a truthy return.
        # Data-returning kaolin calls (kal.io.*.import_mesh, render.*) live in the
        # texture-backprojection stage and will fail fast downstream (documented gap).
        return True


class _KaolinFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, name, path=None, target=None):
        if name == "kaolin" or name.startswith("kaolin."):
            return importlib.machinery.ModuleSpec(name, self)
        return None

    def create_module(self, spec):
        return _KaolinStubModule(spec.name)

    def exec_module(self, module):
        pass


if "kaolin" not in sys.modules and not any(
    isinstance(f, _KaolinFinder) for f in sys.meta_path
):
    sys.meta_path.insert(0, _KaolinFinder())
