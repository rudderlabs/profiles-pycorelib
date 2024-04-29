# profiles-pycorelib

A Python Native package that registers the core python models

## Models

- Common Col Union

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
    extension: .png
    x_axis:
      label: Total IDs (in thousands)
      column: total_ids
      inputs:
        - models/input_table_size
        - models/input_table_size_2
      transformation: x / 1000
    y_axis:
      label: Credits Spent
      column: run_time_in_sec
      inputs:
        - models/id_stitcher_runtime
        - models/id_stitcher_runtime_2
      transformation: y / 3600
```

- Attribution Model

A sample yaml that uses this is as follows:
```
  - name: attribution_scores_revenue
    model_type: campaign_attribution_scores
    model_spec:
      touchpoint_var: campaigns_list
      days_since_first_seen_var: days_since_first_seen
      first_seen_since: 90
      conversion_entity_var: gross_amt_spent_overall
      entity: user      
```

This yaml would be part of a standard profiles project with following mandatory entity vars:
1. `campaigns_list`: This has to be of array type, with structure similar to [campaign1, page1, event2, campaign2 etc..]. It can have duplicates, and the model treats each independently. It does not deduplicate the list. The order of the list is important, as the model treats the first element as the first touchpoint, and the last element as the conversion touchpoint.
2. `days_since_first_seen_var`: This is the number of days since the user was first seen. It is a numeric field. It is used to filter out touchpoints that are older than the specified number of days. So the output of this model would include campaign performance only for the users seen since the last N days, where N is the value of `first_seen_since` parameter.
3. `conversion_entity_var`: This is the entity that is being measured. It is a numeric/boolean field. It is used to calculate the attribution scores. For example, to attribute revenue to different campaign sources, this would be something like amount_spent_overall. To attribute conversions to different campaign sources, this would be a boolean entity var such as is_converted, is_payer, has_signed_up etc. The boolean column can have values as True/False, or 1/0. Both are handled. 1 is treated as True and 0 as False. WARNING: If this is a numeric column, markov model treats it as boolean only, with anything > 0 is counted as a conversion. This is a temporary limitation and will be fixed in future

Limitations in V0:
1. Not calculating shapley values. Only - first touch, last touch, linear, and markov. We can also include other models such as time decay, position based etc.
2. Has strict assumptions on the datatypes- mainly touch points - we should probably support comma separated string types too and convert that as arraytype in this model
3. The data gets loaded in memory. So, if the data is too large, it might crash on UI. This is typically not a problem for local runs. 