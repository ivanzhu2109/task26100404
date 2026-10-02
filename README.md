# task26100404

GPU 整组排程库（纯 Python，无第三方依赖）。

## 用法

```python
from gpu_scheduler import GpuScheduler

scheduler = GpuScheduler()
scheduler.register("job-a", priority=1, gpu_count=2, compatible_gpus={0, 1, 2})
scheduler.register("job-b", priority=2, gpu_count=1, compatible_gpus={2})

plan = scheduler.plan(available_gpus={0, 1, 2})
print(dict(plan.assignments))  # {'job-a': frozenset({0, 1}), 'job-b': frozenset({2})}

scheduler.withdraw("job-a")  # 作业启动或取消后由上层撤回
```

- 优先级 1 最高、5 最低，同级按登记先后排序。
- 入选集合严格按队列顺序字典序最优；无法容纳的作业被跳过，不阻塞后续作业。
- 入选集合确定后，在所有可行分配中最大化保留上一份计划的作业-GPU 配对。
- 每次 `plan()` 的输出自动成为下一次排程的历史计划。

## 测试

```bash
python3 -m unittest discover -s tests -v
```
