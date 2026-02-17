from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

_include_dirs = ["src/cthreading"]

monitoring_module = Extension(
    "cthreading._monitoring",
    sources=["src/cthreading/monitoring.c"],
    include_dirs=_include_dirs,
    extra_compile_args=[],
)

sync_module = Extension(
    "cthreading._sync",
    sources=["src/cthreading/sync.c"],
    include_dirs=_include_dirs,
    extra_compile_args=[],
)

queue_module = Extension(
    "cthreading._queue",
    sources=["src/cthreading/queue.c"],
    include_dirs=_include_dirs,
    extra_compile_args=[],
)

threading_module = Extension(
    "cthreading._threading",
    sources=["src/cthreading/threading.c"],
    include_dirs=_include_dirs,
    extra_compile_args=[],
)

tasks_module = Extension(
    "cthreading._tasks",
    sources=["src/cthreading/tasks.c"],
    include_dirs=_include_dirs,
    extra_compile_args=[],
)

class BuildExt(build_ext):
    def build_extensions(self) -> None:
        if self.compiler.compiler_type == "msvc":
            for ext in self.extensions:
                ext.extra_compile_args = [
                    *ext.extra_compile_args,
                    "/std:c11",
                    "/experimental:c11atomics",
                ]
        else:
            for ext in self.extensions:
                ext.extra_compile_args = [*ext.extra_compile_args, "-std=c11"]
        super().build_extensions()

setup(
    ext_modules=[
        monitoring_module,
        sync_module,
        queue_module,
        threading_module,
        tasks_module,
    ],
    cmdclass={"build_ext": BuildExt},
)