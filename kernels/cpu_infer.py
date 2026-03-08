"""
CPUInfer: 异步 CPU 推理线程池的简化封装。

这是 nano-ktrans 中最底层的基础设施，负责管理一个 CPU 线程池。
通过它，我们可以把 MoE 的专家计算任务异步提交到 CPU 上执行，
同时 GPU 继续并行处理其他专家。

核心概念：
- WorkerPool: 一个分子池的 CPU 线程池（可按 NUMA 节点分配线程）
- submit_with_cuda_stream: 将任务提交到 CPU 线程池，并与 CUDA 流同步
- sync_with_cuda_stream: 等待所有 CPU 任务完成，并与 CUDA 流同步

依赖：kt_kernel_ext C++ 扩展（由 kt-kernel 包提供）
"""

from kt_kernel import kt_kernel_ext


class CPUInferEngine:
    """
    简化的 CPU 异步推理引擎单例。

    用法：
        engine = CPUInferEngine.get_instance(num_threads=32, numa_pools=2)
        engine.submit(task)    # 非阻塞
        engine.sync()          # 阻塞等待
    """

    _instance = None

    def __init__(self, num_threads: int = 32, numa_pools: int = 2):
        """
        初始化 CPU 线程池。

        Args:
            num_threads:  CPU 推理总线程数
            numa_pools:   NUMA 子池数量（通常等于 NUMA 节点数）
        """
        # 构建 WorkerPool 配置
        config = kt_kernel_ext.WorkerPoolConfig()
        config.subpool_count = numa_pools
        config.subpool_numa_map = list(range(numa_pools))

        # 将线程平均分配到各个子池
        threads_per_pool = num_threads // numa_pools
        remainder = num_threads % numa_pools
        config.subpool_thread_count = [
            threads_per_pool + (1 if i < remainder else 0)
            for i in range(numa_pools)
        ]

        # 创建底层 C++ CPUInfer 实例
        self._engine = kt_kernel_ext.CPUInfer(config)

    @classmethod
    def get_instance(cls, num_threads: int = 32, numa_pools: int = 2) -> "CPUInferEngine":
        """获取单例实例（整个进程共享一个线程池）"""
        if cls._instance is None:
            cls._instance = cls(num_threads, numa_pools)
        return cls._instance

    @property
    def backend(self):
        """返回底层 C++ WorkerPool 指针，用于传给 MOEConfig"""
        return self._engine.backend_

    def submit(self, task):
        """
        提交一个 CPU 任务（非阻塞）。

        task 是 C++ 层返回的 (func_ptr, args_ptr) 对，
        例如 moe.forward_task(...) 或 moe.load_weights_task(...)。
        """
        self._engine.submit(task)

    def sync(self, allow_pending: int = 0):
        """
        等待所有已提交的任务完成。

        Args:
            allow_pending: 允许保留多少个挂起任务不等待
        """
        self._engine.sync(allow_pending)

    def submit_with_cuda_stream(self, cuda_stream, task):
        """
        提交 CPU 任务，并与指定 CUDA 流同步。

        这是 Hybrid MoE 的核心：
        1. GPU 先把 hidden_states 拷贝到 CPU pinned memory（在 CUDA 流上）
        2. CPU 线程池等待 CUDA 拷贝完成后开始计算
        3. GPU 继续执行其他操作
        """
        self._engine.submit_with_cuda_stream(cuda_stream, task)

    def sync_with_cuda_stream(self, cuda_stream, allow_pending: int = 0):
        """
        等待 CPU 任务完成，并通知 CUDA 流可以继续。

        在调用 sync 后，GPU 可以安全地从 CPU 输出 buffer 拷贝结果。
        """
        self._engine.sync_with_cuda_stream(cuda_stream, allow_pending)
