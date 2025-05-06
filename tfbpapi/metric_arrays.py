import logging
from collections.abc import Callable

import pandas as pd

logger = logging.getLogger(__name__)


def metric_arrays(
    res_dict: dict[str, pd.DataFrame | dict[str, pd.DataFrame]],
    metrics_dict: dict[str, Callable],
    rownames: str = "target_symbol",
    colnames: str = "regulator_symbol",
    row_dedup_func: Callable | None = None,
    drop_incomplete_rows: bool = True,
) -> dict[str, pd.DataFrame]:
    """
    Extract specified metrics from an AbstractRecordsAndFilesAPI instance's
    read(retrieve_files=True) results object.

    :param res_dict: The output of an AbstractRecordsAndFiles instance.
    :param metrics_dict: A dictionary where the keys are metrics and the values are
        functions to apply to rows in the event that there are multiple rows with
        the same rownames. Set to None to raise error if duplicate rownames are found.
    :param rownames: Column name to use for row labels.
    :param colnames: Column name to use for column labels.
    :param drop_incomplete_rows: When True, drops rows and columns with all NaN values.

    :return: A dictionary where the metric is the key and the value is a DataFrame.
        The column values are metric values, and the column names correspond
        to `colnames` in the metadata DataFrame.

    :raises AttributeError: If the values in `colnames` or `rownames` are not unique
    :raises KeyError: If the res_dict does not have keys 'metadata' and 'data'
    :raises KeyError: If the data dictionary does not have the same keys as the 'id'
        column
    :raises ValueError: If the metadata does not have an 'id' column
    :raises ValueError: If either the metadata or the data dictionary values are not
        DataFrames
    :raises ValueError: If the `colnames` is not in the res_dict metadata
    :raises ValueError: If the `rownames` is not in the res_dict data
    :raises ValueError: If the metrics are not in the data dictionary

    """

    # Check required keys
    if not all(k in res_dict for k in ["metadata", "data"]):
        raise KeyError("res_dict must have keys 'metadata' and 'data'")

    metadata: pd.DataFrame = res_dict["metadata"]

    # Verify 'id' in metadata
    if "id" not in metadata.columns:
        raise ValueError("metadata must have an 'id' column")

    # Check for missing keys in 'data'
    missing_keys = [k for k in metadata["id"] if str(k) not in res_dict["data"]]
    if missing_keys:
        raise KeyError(
            f"Data dictionary must have the same keys as the 'id' "
            f"column. Missing keys: {missing_keys}"
        )

    # Ensure all data dictionary values are DataFrames
    if not all(isinstance(v, pd.DataFrame) for v in res_dict["data"].values()):
        raise ValueError("All values in the data dictionary must be DataFrames")

    # Verify rownames in data and colnames in metadata
    if colnames not in metadata.columns:
        raise ValueError(f"colnames '{colnames}' not in metadata")
    data_with_missing_rownames = [
        id for id, df in res_dict["data"].items() if rownames not in df.columns
    ]
    if data_with_missing_rownames:
        raise ValueError(
            f"rownames '{rownames}' not in data for ids: {data_with_missing_rownames}"
        )

    # Factorize unique row and column labels
    row_labels = pd.Index(
        {item for df in res_dict["data"].values() for item in df[rownames].unique()}
    )

    # Initialize output dictionary with NaN DataFrames for each metric
    output_dict = {
        m: pd.DataFrame(index=pd.Index(row_labels, name=rownames))
        for m in metrics_dict.keys()
    }

    # Populate DataFrames with metric values
    info_msgs = set()
    for _, row in metadata.iterrows():
        try:
            data = res_dict["data"][row["id"]]
        except KeyError:
            info_msgs.add("casting `id` to str to extract data from res_dict['data']")
            data = res_dict["data"][str(row["id"])]

        for metric, row_dedup_func in metrics_dict.items():
            # Filter data to include only the rownames and metric columns
            if metric not in data.columns:
                raise ValueError(
                    f"Metric '{metric}' not found in data for id '{row['id']}'"
                )

            metric_data = data[[rownames, metric]]

            # Handle deduplication if row_dedup_func is provided
            if row_dedup_func is not None:
                metric_data = (
                    metric_data.groupby(rownames)[metric]
                    .apply(row_dedup_func)
                    .reset_index()
                )
            else:
                # Ensure no duplicates exist if no deduplication function is provided
                if metric_data[rownames].duplicated().any():
                    raise ValueError(
                        f"Duplicate entries found for metric '{metric}' "
                        f"in id '{row['id']}' without dedup_func"
                    )

            # test if row[colnames] is already in output_dict[metric]. If it is, add a
            # replicate suffix and try again, Continue doing this until the column name
            # is unique
            colname = row[colnames]
            suffix = 2
            while colname in output_dict[metric].columns:
                colname = f"{row[colnames]}_rep{suffix}"
                suffix += 1
            if suffix > 2:
                info_msgs.add(
                    f"Column name '{row[colnames]}' already exists in "
                    f"output DataFrame for metric '{metric}'. "
                    f"Renaming to '{colname}'"
                )
            # Join metric data with output DataFrame for the metric
            output_dict[metric] = output_dict[metric].join(
                metric_data.set_index(rownames).rename(columns={metric: colname}),
                how="left",
            )
    logger.info("; ".join(info_msgs))

    # Drop incomplete rows and columns if drop_incomplete_rows is True
    if drop_incomplete_rows:
        for metric, df in output_dict.items():
            # Drop rows and columns where all values are NaN
            initial_shape = df.shape
            output_dict[metric] = df.dropna(axis=0)
            final_shape = output_dict[metric].shape

            dropped_rows = initial_shape[0] - final_shape[0]
            dropped_columns = initial_shape[1] - final_shape[1]

            if dropped_rows > 0 or dropped_columns > 0:
                logger.warning(
                    f"{dropped_rows} rows and {dropped_columns} "
                    f"columns with incomplete "
                    f"records were dropped for metric '{metric}'."
                )

    return output_dict
