# ThermionicTest1

## 目的
`ThermionicTest1` 是针对修改后的 1D 热电子铯 PIC-DSMC 模型的一个短周期验证案例。
主要用于：
- 检查更新后的回调与重采样逻辑是否稳定，
- 检查粒子数增长是否受控，
- 检查 MPI/OpenMP 并行配置是否正常，
- 对“卡住”问题进行定点定位。

这不是正式的长周期物理生产算例。

## 文件
- `thermionicCesium-dsmc-Lietz-fixed`
  - 短测试输入卡。
- `thermionicCesium-dsmc-Lietz-fixed.py`
  - 短测试 Python 驱动。
- `thermionicCesium-dsmc-Lietz-fixed.pre-maskA`
  - 本轮并行退激修正前的输入卡备份。
- `thermionicCesium-dsmc-Lietz-fixed.pre-maskA.py`
  - 本轮并行退激修正前、单 rank 可直接运行的脚本备份。
- `readme.pre-maskA.md`
  - 本轮修改前的说明备份。
- `Trae-Solution.pre-maskA.md`
  - 本轮修改前的方案讨论备份。

## 当前短测试设置
- `gap = 0.5 mm`
- `nz = 12000`
- `dt = 4.98774027872218e-14 s`
- `max_step_sim = 50000`
- 场/粒子诊断间隔：`25000` 步
- checkpoint 间隔：`5000` 步
- `amr.max_grid_size = 240`
- `amr.blocking_factor = 16`
- 动态负载均衡间隔：`5000` 步

`50000` 步大约对应：
- `2.49 ns` 物理时间

这足够用于：
- 检查是否存在粒子数失控，
- 检查重采样和回调是否运行正常，
- 在较低时间成本下复现“卡住”现象。

## 推荐运行命令
### 纯 MPI 基准
```bash
OMP_NUM_THREADS=1 mpirun -np 8 python thermionicCesium-dsmc-Lietz-fixed.py
```

### 混合并行测试
```bash
OMP_NUM_THREADS=8 mpirun -np 2 python thermionicCesium-dsmc-Lietz-fixed.py
```

## 对 i7-13700K 的建议
Intel i7-13700K 具有：
- `8` 个 P 核
- `8` 个 E 核
- `24` 个硬件线程

建议优先比较两组：
1. `8 MPI x 1 OMP`
2. `2 MPI x 8 OMP`

原因：
- 第一组更适合看 MPI 负载均衡；
- 第二组更适合看 Python 回调是否把程序拖进低并行度阶段。

## 卡顿定位文件
这个目录现在会为每个 MPI rank 自动生成两个轻量级调试文件：
- `callback_status.rankN.txt`
  - 会被频繁覆盖；
  - 用于显示当前 rank 正停在哪个回调阶段。
- `callback_profile.rankN.log`
  - 每 `100` 步追加一条；
  - 记录回调耗时分解和本地宏粒子数。

如果程序再次“卡住”，优先检查：
- `callback_status.rank0.txt`
- `callback_status.rank1.txt`
- 以及对应的 `callback_profile.rankN.log`

它们会帮助判断问题更可能发生在：
- `before_collisions`
- `after_collisions`
- `after_step`

本轮又对 `before_collisions` 里的退激回调做了进一步细化：
- `callback_status.rankN.txt` 现在会额外显示：
  - `before_collisions:dexcite_species`
  - `before_collisions:dexcite_progress`
  - `before_collisions:dexcite_before_add`
  - `before_collisions:dexcite_add_particles`
- `callback_profile.rankN.log` 现在会额外记录：
  - `dexcite_species`
  - `dexcite_summary`
  - 以及累计时间分量：
    - `dexcite_scan`
    - `dexcite_buffer`
    - `dexcite_add`

这些新增信息用来判断卡顿究竟发生在：
- `age >= lifetime` 的筛选阶段，
- 退激目标分桶与原粒子失效标记阶段，
- 还是 `add_particles(...)` 阶段。

## restart 说明
由于 checkpoint 间隔改成了 `5000` 步，后续如果要从最近检查点继续跑，不需要再从头开始。
当前目录下的 checkpoint 会出现在：
- `diags/diagThermionicTest1_restartXXXXX/`

如果本轮要从 `020000` 步继续跑，请确认输入卡中的：
- `amr.restart = "diags/diagThermionicTest1_restart020000"`

## 建议的排查顺序
1. 先复现一次卡顿。
2. 不要立刻结束进程，先查看 `callback_status.rankN.txt`。
3. 再查看最近的 `callback_profile.rankN.log`。
4. 根据停住的位置，判断是：
   - 退激逻辑变重，
   - 激发态年龄递增变重，
   - 寿命初始化变重，
   - 还是某个 rank 的粒子数局部暴涨。

## 当前并行退激修正
本轮已经按照我们前面论证过的“掩码版方案 A”做了 Python-only 修改，不涉及 WarpX C++ 层和重新编译。

### 现有证据
目前最关键的观测是：
- 程序卡死在 `STEP 43836 starts ...`
- 对应 Python 回调内部是 `step=43835`
- `rank2` 停在：
  - `before_collisions:dexcite_add_particles`
  - `target=cesium_6p particles=1`
- 其他 rank 停在：
  - `after_collisions:done`

这说明这次不是：
- `dexcite_scan` 太慢
- 也不是大量退激粒子导致的纯计算超时

而是：
- 只有一个 rank 进入了 `pc_target.add_particles(...)`
- 其他 rank 没有同时进入

### 为什么这会导致 MPI 卡死
这个判断直接来自 WarpX/AMReX 源码：

1. Python 层 `add_particles(...)` 会进入 `add_n_particles(...)`
   - 见 `warpx/Python/pywarpx/extensions/WarpXParticleContainer.py`
2. C++ 的 `WarpXParticleContainer::AddNParticles(...)` 在结尾无条件调用 `Redistribute()`
   - 见 `warpx/Source/Particles/WarpXParticleContainer.cpp`
3. AMReX 源码明确写道：
   - `All the processors need to participate in Redistribute() though.`
   - 见 `amrex/Src/Particle/AMReX_ParticleInit.H`

因此，当前最核心的问题不是“只加 1 个粒子为什么这么慢”，而是：

- `add_particles(...)` 背后带着一个需要并行参与的 `Redistribute()`
- 如果只有部分 rank 调用，就可能永久等待

这也解释了一个重要现象：

- 少 MPI 不一定能彻底避免问题
- 只有单 rank 时，这类“部分 rank 进入、部分 rank 未进入”的 MPI 不一致才不会发生

也就是说，当前障碍的本质更像是：

- **MPI 集体参与条件被破坏**

而不仅仅是：

- “退激函数太慢”

### 修正依据
这个实现严格建立在下面三条源码事实之上：

1. Python 层 `add_particles(...)` 会进入 `add_n_particles(...)`
   - 见 `warpx/Python/pywarpx/extensions/WarpXParticleContainer.py`
2. C++ 的 `WarpXParticleContainer::AddNParticles(...)` 在结尾无条件调用 `Redistribute()`
   - 见 `warpx/Source/Particles/WarpXParticleContainer.cpp`
3. AMReX 源码明确写道：
   - `All the processors need to participate in Redistribute() though.`
   - 见 `amrex/Src/Particle/AMReX_ParticleInit.H`

因此，本轮修正的目标不是“让某个 rank 更快地调用 `add_particles(...)`”，而是：
- 保证所有 MPI ranks 在真正执行 `add_particles(...)` 时
- 对同一组目标态
- 按完全一致的顺序
- 调用完全一致的次数

### 已实施方案：掩码版 A
这是当前在“不改 WarpX 绑定层”的前提下，最保守且最贴近现有源码约束的 Python-only 实现。

核心思想：
- 每个 rank 仍然先本地扫描退激并构造自己的 `particle_buffer`
- 但本地 `particle_buffer` 不再直接决定谁去调用 `add_particles(...)`
- 脚本先用当前 `specie_decay_data` 和 `specie_name_list` 构造固定顺序的“可写入目标态列表”
- 每个 rank 再为这一固定列表生成一个本地活跃掩码
- 通过一次 MPI 全局规约得到“本步全局活跃目标态掩码”
- 最后，所有 rank 只对这些“全局活跃目标态”按固定顺序 collective 调用 `add_particles(...)`
- 某个 rank 对某个目标态本地没有粒子时，就传入长度为 0 的数组

这个方案的优点：
- 只依赖当前 pywarpx 已暴露的接口
- 完全基于现有源码行为推导出来
- 相比“每步对全部目标态空/非空都 collective add”，只对本步全局活跃目标态调用，空调用明显更少
- 不需要修改 WarpX C++ 层

这个方案的代价：
- 仍然会引入一次额外的 MPI 掩码规约
- 当本步全局活跃目标态较多时，`add_particles(...)` 仍然可能较重
- 这首先是正确性修复，不是最终性能优化

因此，这个方案更像是：
- **一个保守且安全的正确性修复方案**
- 不一定是最终最优性能方案

### 代码层的具体实现点
当前脚本中已经新增了下面几类逻辑：

1. `mpi4py` 检测与 communicator 初始化
   - 若有 MPI 环境，则使用 `MPI.COMM_WORLD`
   - 单 rank 情况下会自然退化为本地掩码
2. 固定顺序目标态列表构造
   - 基于 `specie_decay_data` 中出现过的目标态
   - 再与 `specie_name_list` 取交集并保留既有顺序
3. 本地活跃掩码构造
   - 只标记当前 rank 本步确实有退激产物的目标态
4. 全局活跃掩码规约
   - 使用一次 `MPI.Allreduce(..., MPI.MAX)` 得到本步全局并集
5. collective `add_particles(...)`
   - 所有 rank 只对全局活跃目标态按固定顺序调用
   - 本地空载荷 rank 传零长度数组

### Barrier 处理
本轮实现里没有主动加入 `Barrier`。

原因是：
- 只要所有 rank 已经对同一组目标态按同一顺序执行同次数的 `add_particles(...)`
- 那么每次 `add_particles(...)` 背后的 `Redistribute()` 自身就已经是并行参与点

因此：
- `Barrier` 不是这个修复的必要条件
- 当前更重要的是“调用集合、调用顺序、调用次数完全一致”

### 方案 B：更推荐的源程序级修正方向
从源码角度看，更理想的做法不是“每次 Python 加粒子都立刻 Redistribute”，而是：

1. 各 rank 先本地累积退激后产生的新粒子
2. 本地插入粒子时不立刻触发 `Redistribute()`
3. 在一个所有 rank 都参与的统一阶段，再进行一次 collective 的 `Redistribute()`

这个思路之所以有依据，是因为：
- AMReX/pyAMReX 体系里本来就存在带 `local` 语义的粒子添加思路
- 当前 pywarpx 扩展文件里甚至有注释说明：
  - 它覆盖了 pyAMReX 的一个签名
  - `add_particles(other: ParticleContainer, local: bool = False)`
  - 见 `warpx/Python/pywarpx/extensions/WarpXParticleContainer.py`
- AMReX 侧源码也能看到：
  - `if (! local) { Redistribute(); }`
  - 见 `amrex/Src/Particle/AMReX_ParticleContainerI.H`

这说明从框架设计上看：
- “本地添加”
- “之后统一 Redistribute”

本来就是合理方向。

但当前问题是：
- pywarpx 现在暴露给我们使用的这个 `add_particles(...)` 路径
- 会直接进入 `AddNParticles(...)`
- 并在里面立刻 `Redistribute()`

因此，如果要走这条更理想的修正路线，后续很可能需要：
- 给 pywarpx/pybind 层增加一个新的接口

例如下面两类之一：
1. `add_particles_local(...)`
   - 只做本地插入
   - 不立即 `Redistribute()`
2. 暴露 `MultiParticleContainer.Redistribute()` 或单物种 `Redistribute()`
   - 让 Python 先完成本地累积
   - 然后再由所有 rank 同步调用一次 collective redistribute

这个方向的优点：
- 更符合并行语义
- 更可能避免“每个目标态单独触发一次 Redistribute”带来的额外代价
- 更适合后续做性能优化

这个方向的缺点：
- 需要扩展绑定层
- 工作量和验证成本高于纯 Python 修改

### 当前建议
在不立即修改源代码的前提下，当前建议分两层理解：

1. **短期保守修复**
   - 采用“所有 rank 按固定顺序 collective 调用 `add_particles(...)`，空 rank 传空数组”的思路
   - 目标是先保证不会因为 `Redistribute()` 参与不一致而挂住

2. **中期推荐修复**
   - 研究并扩展 pywarpx/pybind 接口
   - 支持“本地添加 + 统一 collective redistribute”的两阶段方案

当前更倾向于认为：
- 真正值得长期保留的方案是第 2 类
- 第 1 类更像是验证根因和保证正确性的过渡方案
