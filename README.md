# profiles-pycorelib

A Python Native package that registers the core python models

## Models

- Common Col Union

## This is how a GraphModel can be defined:

```yaml
- name: test_graph
  model_type: graph
  model_spec:
    num_of_graphs: 2
    fig_height:
      - 8
      - 8
    fig_width:
      - 6
      - 6
    label_x:
      - Total IDs (in thousands)
      - Total Users (in thousands)
    label_y:
      - Credits Spent
      - Credits Spent
    title:
      - Total IDs vs. Credits Spent
      - Total Users vs. Credits Spent
    output_folder:
      - reports
      - reports
    col_x:
      - total_records
      - run_time_in_sec
    col_y:
      - post_stitched_ids
      - run_time_in_sec
    tables:
      - - models/input_table_size
        - models/id_stitcher_runtime
      - - models/input_table_size
        - models/feature_table_runtime
    img_name:
      - total_ids_vs_credits_spent.png
      - total_users_vs_credits_spent.png
    materialization:
      output_type: image
      run_type: discrete
```
