# VirtualDB

VirtualDB provides a SQL query interface across heterogeneous HuggingFace
datasets using an in-memory DuckDB database. Each dataset defines experimental
conditions in its own way, with properties stored at different hierarchy levels
(repository, dataset, or field) and using different naming conventions.
VirtualDB uses an external YAML configuration to map these varying structures
to a common schema, normalize factor level names (e.g., "D-glucose",
"dextrose", "glu" all become "glucose"), and enable cross-dataset queries with
standardized field names and values.

For primary datasets, VirtualDB creates:

- **`<db_name>_meta`** -- one row per sample with derived metadata columns
- **`<db_name>`** -- full measurement-level data joined to the metadata view

For comparative analysis datasets, VirtualDB creates:

- **`<db_name>_expanded`** -- the raw data with composite ID fields parsed
  into `<link_field>_source` (aliased to configured `db_name`) and
  `<link_field>_id` (sample_id) columns

See the [configuration guide](virtual_db_configuration.md) for setup details
and the [tutorial](tutorials/virtual_db_tutorial.ipynb) for usage examples.

## API Reference

::: tfbpapi.virtual_db.VirtualDB
    options:
      show_root_heading: true
      show_source: true
