# VirtualDB

VirtualDB provides a unified query interface across heterogeneous datasets with
different experimental condition structures and terminologies. Each dataset
defines experimental conditions in its own way, with properties stored at
different hierarchy levels (repository, dataset, or field) and using different
naming conventions. VirtualDB uses an external YAML configuration to map these
varying structures to a common schema, normalize factor level names (e.g.,
"D-glucose", "dextrose", "glu" all become "glucose"), and enable cross-dataset
queries with standardized field names and values.

## API Reference

::: tfbpapi.virtual_db.VirtualDB
    options:
      show_root_heading: true
      show_source: true

### Helper Functions

::: tfbpapi.virtual_db.get_nested_value
    options:
      show_root_heading: true

::: tfbpapi.virtual_db.normalize_value
    options:
      show_root_heading: true
