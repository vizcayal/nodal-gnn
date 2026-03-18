---
license: cc-by-sa-4.0
tags:
- energy
- optimization
- optimal_power_flow
- power_grid
pretty_name: PGLearn Optimal Power Flow (small)
size_categories:
- 1M<n<10M
task_categories:
- tabular-regression
dataset_info:
  config_name: 14_ieee
  features:
  - name: case/json
    dtype: string
  - name: input/pd
    sequence: float32
    length: 11
  - name: input/qd
    sequence: float32
    length: 11
  - name: input/gen_status
    sequence: bool
    length: 5
  - name: input/branch_status
    sequence: bool
    length: 20
  - name: input/seed
    dtype: int64
  - name: ACOPF/primal/vm
    sequence: float32
    length: 14
  - name: ACOPF/primal/va
    sequence: float32
    length: 14
  - name: ACOPF/primal/pg
    sequence: float32
    length: 5
  - name: ACOPF/primal/qg
    sequence: float32
    length: 5
  - name: ACOPF/primal/pf
    sequence: float32
    length: 20
  - name: ACOPF/primal/pt
    sequence: float32
    length: 20
  - name: ACOPF/primal/qf
    sequence: float32
    length: 20
  - name: ACOPF/primal/qt
    sequence: float32
    length: 20
  - name: ACOPF/dual/kcl_p
    sequence: float32
    length: 14
  - name: ACOPF/dual/kcl_q
    sequence: float32
    length: 14
  - name: ACOPF/dual/vm
    sequence: float32
    length: 14
  - name: ACOPF/dual/pg
    sequence: float32
    length: 5
  - name: ACOPF/dual/qg
    sequence: float32
    length: 5
  - name: ACOPF/dual/ohm_pf
    sequence: float32
    length: 20
  - name: ACOPF/dual/ohm_pt
    sequence: float32
    length: 20
  - name: ACOPF/dual/ohm_qf
    sequence: float32
    length: 20
  - name: ACOPF/dual/ohm_qt
    sequence: float32
    length: 20
  - name: ACOPF/dual/pf
    sequence: float32
    length: 20
  - name: ACOPF/dual/pt
    sequence: float32
    length: 20
  - name: ACOPF/dual/qf
    sequence: float32
    length: 20
  - name: ACOPF/dual/qt
    sequence: float32
    length: 20
  - name: ACOPF/dual/va_diff
    sequence: float32
    length: 20
  - name: ACOPF/dual/sm_fr
    sequence: float32
    length: 20
  - name: ACOPF/dual/sm_to
    sequence: float32
    length: 20
  - name: ACOPF/dual/slack_bus
    dtype: float32
  - name: ACOPF/meta/seed
    dtype: int64
  - name: ACOPF/meta/formulation
    dtype: string
  - name: ACOPF/meta/primal_objective_value
    dtype: float32
  - name: ACOPF/meta/dual_objective_value
    dtype: float32
  - name: ACOPF/meta/primal_status
    dtype: string
  - name: ACOPF/meta/dual_status
    dtype: string
  - name: ACOPF/meta/termination_status
    dtype: string
  - name: ACOPF/meta/build_time
    dtype: float32
  - name: ACOPF/meta/extract_time
    dtype: float32
  - name: ACOPF/meta/solve_time
    dtype: float32
  - name: DCOPF/primal/va
    sequence: float32
    length: 14
  - name: DCOPF/primal/pg
    sequence: float32
    length: 5
  - name: DCOPF/primal/pf
    sequence: float32
    length: 20
  - name: DCOPF/dual/kcl_p
    sequence: float32
    length: 14
  - name: DCOPF/dual/pg
    sequence: float32
    length: 5
  - name: DCOPF/dual/ohm_pf
    sequence: float32
    length: 20
  - name: DCOPF/dual/pf
    sequence: float32
    length: 20
  - name: DCOPF/dual/va_diff
    sequence: float32
    length: 20
  - name: DCOPF/dual/slack_bus
    dtype: float32
  - name: DCOPF/meta/seed
    dtype: int64
  - name: DCOPF/meta/formulation
    dtype: string
  - name: DCOPF/meta/primal_objective_value
    dtype: float32
  - name: DCOPF/meta/dual_objective_value
    dtype: float32
  - name: DCOPF/meta/primal_status
    dtype: string
  - name: DCOPF/meta/dual_status
    dtype: string
  - name: DCOPF/meta/termination_status
    dtype: string
  - name: DCOPF/meta/build_time
    dtype: float32
  - name: DCOPF/meta/extract_time
    dtype: float32
  - name: DCOPF/meta/solve_time
    dtype: float32
  - name: SOCOPF/primal/w
    sequence: float32
    length: 14
  - name: SOCOPF/primal/pg
    sequence: float32
    length: 5
  - name: SOCOPF/primal/qg
    sequence: float32
    length: 5
  - name: SOCOPF/primal/pf
    sequence: float32
    length: 20
  - name: SOCOPF/primal/pt
    sequence: float32
    length: 20
  - name: SOCOPF/primal/qf
    sequence: float32
    length: 20
  - name: SOCOPF/primal/qt
    sequence: float32
    length: 20
  - name: SOCOPF/primal/wr
    sequence: float32
    length: 20
  - name: SOCOPF/primal/wi
    sequence: float32
    length: 20
  - name: SOCOPF/dual/kcl_p
    sequence: float32
    length: 14
  - name: SOCOPF/dual/kcl_q
    sequence: float32
    length: 14
  - name: SOCOPF/dual/w
    sequence: float32
    length: 14
  - name: SOCOPF/dual/pg
    sequence: float32
    length: 5
  - name: SOCOPF/dual/qg
    sequence: float32
    length: 5
  - name: SOCOPF/dual/ohm_pf
    sequence: float32
    length: 20
  - name: SOCOPF/dual/ohm_pt
    sequence: float32
    length: 20
  - name: SOCOPF/dual/ohm_qf
    sequence: float32
    length: 20
  - name: SOCOPF/dual/ohm_qt
    sequence: float32
    length: 20
  - name: SOCOPF/dual/jabr
    dtype:
      array2_d:
        shape:
        - 20
        - 4
        dtype: float32
  - name: SOCOPF/dual/sm_fr
    dtype:
      array2_d:
        shape:
        - 20
        - 3
        dtype: float32
  - name: SOCOPF/dual/sm_to
    dtype:
      array2_d:
        shape:
        - 20
        - 3
        dtype: float32
  - name: SOCOPF/dual/va_diff
    sequence: float32
    length: 20
  - name: SOCOPF/dual/wr
    sequence: float32
    length: 20
  - name: SOCOPF/dual/wi
    sequence: float32
    length: 20
  - name: SOCOPF/dual/pf
    sequence: float32
    length: 20
  - name: SOCOPF/dual/pt
    sequence: float32
    length: 20
  - name: SOCOPF/dual/qf
    sequence: float32
    length: 20
  - name: SOCOPF/dual/qt
    sequence: float32
    length: 20
  - name: SOCOPF/meta/seed
    dtype: int64
  - name: SOCOPF/meta/formulation
    dtype: string
  - name: SOCOPF/meta/primal_objective_value
    dtype: float32
  - name: SOCOPF/meta/dual_objective_value
    dtype: float32
  - name: SOCOPF/meta/primal_status
    dtype: string
  - name: SOCOPF/meta/dual_status
    dtype: string
  - name: SOCOPF/meta/termination_status
    dtype: string
  - name: SOCOPF/meta/build_time
    dtype: float32
  - name: SOCOPF/meta/extract_time
    dtype: float32
  - name: SOCOPF/meta/solve_time
    dtype: float32
  splits:
  - name: train
    num_bytes: 29023242427
    num_examples: 756205
  - name: test
    num_bytes: 7255839392
    num_examples: 189052
  download_size: 5716674030
  dataset_size: 36279081819
configs:
- config_name: 14_ieee
  data_files:
  - split: train
    path: 14_ieee/train-*
  - split: test
    path: 14_ieee/test-*
  default: true
---
