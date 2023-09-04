# rudderstack-profiles-common-column-union

## File Structure

```bash
.
├── __init__.py # Registers Model Type
└── model.py # Model Type Definition
```

## CommonColumnUnionModel

### ModelType Class

- TypeName - Model Type
- BuildSpecSchema - Schema for model_spec in yaml
- GetMaterialRecipe() - Get Material Recipe
- Validate() - Validate model
- GetEntityKey() - Model entity key (optional)
- GetContract() - Model output contract (optional)

### Recipe

- Describe() - Return content for model's compile output
- Prepare() - Add input dependencies to material
- Execute() - Run and create the material in warehouse

### Sample Usage

```yaml
# profiles.yaml
models:
  - name: common_column_union
    model_type: common_column_union
    model_spec:
      inputs: ... # list of inputs
```
