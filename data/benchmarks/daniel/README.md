# DANIEL reference solutions

This directory retains the upstream Google OR-Tools objective and timing arrays in
`or_solution/` used by this study's public-benchmark comparison. Evaluation uses
`evaluation/exp_single_obj.py` and `src/utils/benchmark_gaps.py` in the PEACH
repository. These arrays do not contain operation schedules; this directory is not a standalone DANIEL training installation.
Only the twelve aggregate reference arrays required by the paper are included. The SD1/SD2 instances in the parent directory and these reference arrays come from the [official DANIEL repository](https://github.com/wrqccc/FJSP-DRL). Its [instance directory](https://github.com/wrqccc/FJSP-DRL/tree/main/data) and [OR-Tools arrays](https://github.com/wrqccc/FJSP-DRL/tree/main/or_solution) provide the upstream sources.

Runqing Wang, Gang Wang, Jian Sun, Fang Deng, and Jie Chen,
“Flexible Job Shop Scheduling via Dual Attention Network-Based Reinforcement
Learning,” *IEEE Transactions on Neural Networks and Learning Systems*, 2023.
[DOI: 10.1109/TNNLS.2023.3306421](https://doi.org/10.1109/TNNLS.2023.3306421).

```bibtex
@ARTICLE{10246328,
  author={Wang, Runqing and Wang, Gang and Sun, Jian and Deng, Fang and Chen, Jie},
  journal={IEEE Transactions on Neural Networks and Learning Systems},
  title={Flexible Job Shop Scheduling via Dual Attention Network-Based Reinforcement Learning},
  year={2023},
  doi={10.1109/TNNLS.2023.3306421}
}
```
