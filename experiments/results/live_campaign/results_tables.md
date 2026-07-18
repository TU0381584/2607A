| Arm | Slice | SLA compliance % (mean±std) | Blocks/episode (mean±std) | Mean backlog margin (mean±std) | Mean inferred MOS (mean±std) | n seeds |
|---|---|---|---|---|---|---|
| baseline | embb | 73.7±18.6 | 0.0±0.0 | -257627.345±182170.839 | 1.260±0.025 | 3 |
| baseline | urllc | 73.4±18.8 | 0.0±0.0 | -223096.257±157753.407 | 1.241±0.331 | 3 |
| baseline | mmtc | 73.8±18.5 | 0.0±0.0 | -147.806±129.862 | 4.648±0.088 | 3 |
| dqn_sla | embb | 100.0±0.0 | 0.0±0.0 | 1.000±0.000 | 1.223±0.000 | 3 |
| dqn_sla | urllc | 100.0±0.0 | 0.0±0.0 | 0.733±0.002 | 1.495±0.046 | 3 |
| dqn_sla | mmtc | 100.0±0.0 | 0.0±0.0 | 0.741±0.008 | 4.778±0.005 | 3 |
| a2c_sla | embb | 100.0±0.0 | 0.0±0.0 | 1.000±0.000 | 1.223±0.000 | 3 |
| a2c_sla | urllc | 100.0±0.0 | 0.0±0.0 | 0.750±0.012 | 1.661±0.120 | 3 |
| a2c_sla | mmtc | 100.0±0.0 | 0.0±0.0 | 0.731±0.003 | 4.772±0.002 | 3 |
| dqn_qoe | embb | 100.0±0.0 | 0.0±0.0 | 1.000±0.000 | 1.223±0.000 | 3 |
| dqn_qoe | urllc | 100.0±0.0 | 0.0±0.0 | 0.738±0.006 | 1.552±0.058 | 3 |
| dqn_qoe | mmtc | 100.0±0.0 | 0.0±0.0 | 0.732±0.002 | 4.773±0.001 | 3 |
| a2c_qoe | embb | 100.0±0.0 | 42.5±1.0 | 0.734±0.001 | 1.224±0.000 | 3 |
| a2c_qoe | urllc | 100.0±0.0 | 41.2±1.9 | 0.751±0.007 | 1.660±0.064 | 3 |
| a2c_qoe | mmtc | 100.0±0.0 | 37.5±2.9 | 0.723±0.008 | 4.767±0.005 | 3 |

| Arm | Mean episodic reward (mean±std) | n seeds |
|---|---|---|
| baseline | 1.4663±2.9828 | 3 |
| dqn_sla | 5.6473±0.2005 | 3 |
| a2c_sla | 5.6473±0.2005 | 3 |
| dqn_qoe | 0.3184±0.0049 | 3 |
| a2c_qoe | 0.3877±0.0049 | 3 |

### Paired-seed win/loss (SLA compliance, vs. baseline, summed across 3 slices x n seeds)
| Arm | Wins | Losses | Ties | Cohen's d (compliance, per slice) |
|---|---|---|---|---|
| dqn_sla | 6 | 0 | 3 | embb=1.63, urllc=1.63, mmtc=1.63 |
| a2c_sla | 6 | 0 | 3 | embb=1.63, urllc=1.63, mmtc=1.63 |
| dqn_qoe | 6 | 0 | 3 | embb=1.63, urllc=1.63, mmtc=1.63 |
| a2c_qoe | 6 | 0 | 3 | embb=1.63, urllc=1.63, mmtc=1.63 |
