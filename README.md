# rudderstack-profiles-common-column-union

Python Native Model definition for Common Column Union

## Steps to Test Locally

- Ensure you have the latest pb binary(not v 0.8.0) installed locally

  ```bash
  # Pull latest main from wht repo and run(inside wht repo)
  make build
  ```

- Install the latest profiles-rudderstack(not v0.8.0)

  ```bash
  # This ignore installing the binary included in profiles-rudderstack
  WHT_CI=true pip install git+https://github.com/rudderlabs/pywht@wht_test_v44#subdirectory=profiles_rudderstack
  ```

- Install the python package

  ```bash
  cd ./common_col_union
  pip install .
  ```

- Edit the profiles.yaml and inputs.yaml if required, ensure all tables listed exists.
