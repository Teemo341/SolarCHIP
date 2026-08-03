# Monkey-patch: 修复 fsspec 在 NAS 上保存 checkpoint 时的跨设备 rename 失败
# 让临时文件创建在目标文件所在目录，而不是 /tmp，避免 Errno 18
import fsspec.implementations.local as fsspec_local
from fsspec.compression import compr
import tempfile as _tempfile
import os
_orig_open = fsspec_local.LocalFileOpener._open
def _patched_open(self):
    if self.f is None or self.f.closed:
        if self.autocommit or "w" not in self.mode:
            self.f = open(self.path, mode=self.mode)
            if self.compression:
                compress = compr[self.compression]
                self.f = compress(self.f, mode=self.mode)
        else:
            target_dir = os.path.dirname(self.path) or "."
            os.makedirs(target_dir, exist_ok=True)
            i, name = _tempfile.mkstemp(dir=target_dir)
            os.close(i)
            self.temp = name
            self.f = open(name, mode=self.mode)
        if "w" not in self.mode:
            self.size = self.f.seek(0, 2)
            self.f.seek(0)
            self.f.size = self.size
fsspec_local.LocalFileOpener._open = _patched_open