# 二维、三维波动鳍代码整合

## 来源和范围

- 目标基线：`kkdxtd/IAMReX` 的 `main`，提交 `2cf74f43`。
- 二维来源：`kkdxtd/IAMReX-2dIBMdev` 的 `main`，提交 `78e0184b`。
- 三维来源：整合时本地 `IAMReX/Source` 及 `Tutorials/fin3d` 的工作文件。

二维仓库与目标仓库没有共同 Git 祖先，因此采用代码移植，不合入旧仓库的
删除记录、旧求解器版本或历史计算结果。目标基线的水平集质量修正、相体积
诊断、椭球和 RKPM 代码予以保留。

## 模块选择

| 配置 | 实现 | 输入前缀 | 算例 |
| --- | --- | --- | --- |
| `DIM=2`, `USE_PARTICLES=TRUE` | `DiffusedFiber.H/.cpp` | `fiber.input` | `Tutorials/Fiber` |
| `DIM=3`, `USE_PARTICLES=TRUE`, `PARTICLE_PARALLEL=FALSE` | `DiffusedIB.H/.cpp` | `particle.input`, `geometry_type=3` | `Tutorials/fin3d` |
| `DIM=3`, `PARTICLE_PARALLEL=TRUE` | 原有 `DiffusedIB_Parallel.H/.cpp` | 原有粒子配置 | 原有 RKPM 算例 |

`PARTICLE_PARALLEL` 是原有粒子/RKPM 实现的选择开关，不等于 `USE_MPI`。
二维和三维鳍算例仍可使用 `USE_MPI=TRUE`。三维鳍需要使用普通 DIBM 路径；
本次没有把 `geometry_type=3` 加入 RKPM 实现。二维、三维是分别编译的程序。

## 整合适配

- 将二维细丝代码提取为独立模块，按维度选择头文件和源码，隔离三维碰撞结构。
- 保留二维绝对 x 坐标行波、原有点距/权重和惩罚力公式，以及三维转角行波、
  边缘半权重、解析速度、力矩和 VTK 输出。
- 将预定几何和目标速度对齐到新流场的 `time+dt`。二维初始化及重启根据
  当前物理时间重建几何，力输出采用实际最细层步数，适配 AMR 子循环。
- 三维普通 DIBM 的新接口与 RKPM 的旧接口分别调用，保留原有 RKPM 路径。
- 初始化和输出挂钩仅在启用 DIBM 时访问鳍/粒子对象。
- 保留本地 `velocity_magnitude` 派生变量，二维和三维均可用于流场输出。
- 二维算例移除无效的第三维输入分量；三维保留本地有效参数，并修正文档与
  注释：当前频率为 `1.0 Hz`，`amr.plot_int=-1` 关闭周期流场/鳍 VTK 输出。

二维 CSV 保留旧代码的受力符号和密度约定，三维 CSV 为流体作用于鳍的
有量纲力/力矩。详细公式见两个算例的 README，比较结果时需先统一约定。

## 验证与文件范围

本次仅进行文本差异、维度分支、接口对应、输入和提交清单检查。
**未执行编译、求解器运行或数值测试；编译与数值验证由用户完成。**
上传提交使用 `[skip ci]` 避免触发本次提交的 GitHub Actions 编译测试。
后续自行提交时若不需要自动测试，也应注意仓库已有的 CI 触发规则。

仅新增/修改源码、Make 配置、输入参数、说明文档和已有后处理脚本。
不包含本地的 `IB_Fin_*.csv`、`IB_Fiber_*.csv`、`fin_vtk`、`force-list`、
plotfile、checkpoint、日志或可执行文件。忽略规则不会删除本地结果；
目标仓库基线中原有的参考数据也没有在本次改动中删除。
