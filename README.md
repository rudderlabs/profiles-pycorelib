# profiles-pycorelib

A Python Native package that registers the core python models

## Models

- CommonColumnUnion
- PyPlot

## Usage Examples:

### - CommonColumnUnionModel

#### Here's an example of how a CommonColumnUnionModel can be defined in a YAML configuration file:

```yaml
- name: test_common_column_union
  model_type: common_column_union
  model_spec:
    inputs:
      - inputs/tbl_a
      - inputs/tbl_b
      - inputs/tbl_c
```

### - PyPlotModel

#### Here's an example of how a PyPlotModel can be defined in a YAML configuration file:

```yaml
- name: total_ids_vs_credits_spent
  model_type: pyplot
  model_spec:
    materialization:
      output_type: image
      run_type: discrete
    title: Total IDs vs. Credits Spent
    size: 8x8 # Optional with Defaults
    grid: true # Optional with Defaults
    x_axis:
      label: Total IDs (in thousands)
      column: total_ids
      input: models/id_stitcher_runtime
      transformation: x / 1000
    y_axis:
      label: Credits Spent
      column: run_time_in_sec
      input: models/id_stitcher_runtime
      transformation: y / 3600
```
