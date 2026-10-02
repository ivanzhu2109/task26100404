# gpu_scheduler

GPU 整组排程库：一次为多个推理作业分配 GPU，严格兑现队列顺序，并尽量保留上次分配。

## 用法

```python
from gpu_scheduler import Job, Scheduler

s = Scheduler()
s.register(Job("job-a", priority=1, gpu_count=2, compatible_gpus=frozenset({"g0", "g1", "g2"})))
s.register(Job("job-b", priority=2, gpu_count=1, compatible_gpus=frozenset({"g2"})))

plan = s.plan(frozenset({"g0", "g1", "g2"}))  # 上层提供当前可用 GPU
print(plan.assignments)  # {"job-a": {"g0", "g1"}, "job-b": {"g2"}}

s.withdraw("job-a")  # 作业启动/取消后由上层撤回
```

## 规则

- 优先级 1（最高）至 5（最低），同级按登记先后排序。
- 只有获得所需数量的不同兼容 GPU 才入选，不允许部分分配；一张 GPU 一份计划内只属于一个作业。
- 入选集合严格按队列顺序字典序最优：靠前作业优先，无法容纳的跳过，继续考虑后面的。
- 入选集合确定后，在所有可行分配中保留上一份计划的 作业-GPU 配对 最多。
- 每次 `plan()` 的输出自动成为下一次排程的上一份计划；撤回作业或 GPU 不可用后旧分配不再占用资源。

## 测试

```bash
python3 -m pytest tests/ -q
```
