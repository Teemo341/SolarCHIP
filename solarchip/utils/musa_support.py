import torch

# Monkey-patch: 修复 MUSA 设备下 Lightning _InfiniteBarrier 使用 gloo backend 报错
# "No backend type associated with device type musa"
# gloo 后端不支持 musa 设备类型，且系统使用 MCCL 而非 NCCL。
# _InfiniteBarrier 只需 CPU 同步，在 MUSA 上回退使用默认 group 的 barrier()
if hasattr(torch, 'musa') and torch.musa.is_available():
    from lightning.fabric.utilities import distributed as _lf_dist
    _orig_barrier_enter = _lf_dist._InfiniteBarrier.__enter__
    _orig_barrier_exit = _lf_dist._InfiniteBarrier.__exit__

    def _patched_barrier_enter(self):
        # 跳过 Gloo new_group：Docker 容器中 Gloo TCP 通信在 C++ 层直接崩溃，
        # Python try/except 无法捕获。_InfiniteBarrier 只需 CPU 同步，
        # 直接用主进程组(MCCL)的 barrier 即可。
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
            self.group = None

    def _patched_barrier_exit(self, *args, **kwargs):
        if hasattr(self, 'group') and self.group is not None:
            torch.distributed.destroy_process_group(self.group)

    _lf_dist._InfiniteBarrier.__enter__ = _patched_barrier_enter
    _lf_dist._InfiniteBarrier.__exit__ = _patched_barrier_exit

# Monkey-patch: MUSA 设备上 Lightning DDP 使用 backend="nccl" 初始化进程组，
# 但系统只有 MCCL（Moore Threads Collective Communication Library）
if hasattr(torch, 'musa') and torch.musa.is_available():
    _orig_init_process_group = torch.distributed.init_process_group
    def _patched_init_process_group(backend=None, *args, **kwargs):
        if backend == 'nccl':
            backend = 'mccl'
        # 处理 device_id：cuda:N -> musa:N
        device_id = kwargs.get('device_id', None)
        if device_id is not None and isinstance(device_id, torch.device) and device_id.type == 'cuda':
            kwargs = dict(kwargs)
            kwargs['device_id'] = torch.device('musa', device_id.index)
        return _orig_init_process_group(backend=backend, *args, **kwargs)
    torch.distributed.init_process_group = _patched_init_process_group

    _orig_new_group = torch.distributed.new_group
    def _patched_new_group(*args, **kwargs):
        backend = kwargs.get('backend', None)
        if backend == 'nccl':
            kwargs = dict(kwargs)
            kwargs['backend'] = 'mccl'
            return _orig_new_group(*args, **kwargs)
        return _orig_new_group(*args, **kwargs)
    torch.distributed.new_group = _patched_new_group

    # Patch _new_process_group_helper 内部的 device_id 转换：
    # mccl 后端只认 musa:N，但 Lightning 创建的是 cuda:N
    import torch.distributed.distributed_c10d as _c10d
    _orig_helper = _c10d._new_process_group_helper
    def _patched_helper(*args, **kwargs):
        device_id = kwargs.get('device_id', None)
        if device_id is not None and isinstance(device_id, torch.device) and device_id.type == 'cuda':
            kwargs = dict(kwargs)
            kwargs['device_id'] = torch.device('musa', device_id.index)
        return _orig_helper(*args, **kwargs)
    _c10d._new_process_group_helper = _patched_helper

# Monkey-patch: MUSA 设备上 Lightning 内部直接调用 torch.cuda.* API，
# 但 torch.cuda.Stream 等是 dummy stub（CUDA 不可用），需重定向到 torch.musa
if hasattr(torch, 'musa') and torch.musa.is_available():
    # 包装函数：将 cuda:N 设备转为 musa:N，避免 "Expected a musa device, but got: cuda:N" 错误
    def _wrap_musa_fn(fn):
        import functools
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            new_args = []
            for a in args:
                if isinstance(a, torch.device) and a.type == 'cuda':
                    a = torch.device('musa', a.index)
                new_args.append(a)
            return fn(*new_args, **kwargs)
        return wrapper

    torch.cuda.Stream = torch.musa.Stream  # Stream 构造时 Lightning 传 device，但 musa.Stream 能处理
    torch.cuda.stream = _wrap_musa_fn(torch.musa.stream)
    torch.cuda.current_stream = _wrap_musa_fn(torch.musa.current_stream)
    torch.cuda.synchronize = _wrap_musa_fn(torch.musa.synchronize)
    torch.cuda.device_count = torch.musa.device_count
    torch.cuda.is_available = torch.musa.is_available
    torch.cuda.empty_cache = torch.musa.empty_cache
    torch.cuda.current_device = torch.musa.current_device
    torch.cuda.set_device = _wrap_musa_fn(torch.musa.set_device)
    torch.cuda.get_device_name = _wrap_musa_fn(torch.musa.get_device_name)
    torch.cuda.get_device_properties = _wrap_musa_fn(torch.musa.get_device_properties)
    torch.cuda.get_device_capability = _wrap_musa_fn(torch.musa.get_device_capability)

    # Patch Lightning CUDAAccelerator: 强制使用 musa:N 设备而非 cuda:N
    # 否则 model.to(cuda:N) 会失败，因为 MUSA 用 PrivateUse1 后端
    from lightning.pytorch.accelerators.cuda import CUDAAccelerator
    _orig_get_parallel_devices = CUDAAccelerator.get_parallel_devices
    @staticmethod
    def _patched_get_parallel_devices(devices):
        return [torch.device('musa', i) for i in devices]
    CUDAAccelerator.get_parallel_devices = _patched_get_parallel_devices

    # setup_device 也需 patch：接受 musa 设备类型
    _orig_setup_device = CUDAAccelerator.setup_device
    def _patched_setup_device(self, device):
        if device.type == 'musa':
            device = torch.device('cuda', device.index)
        return _orig_setup_device(self, device)
    CUDAAccelerator.setup_device = _patched_setup_device