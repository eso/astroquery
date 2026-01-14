"""
utils.py: helper functions for the astropy.eso module
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import numpy as np
from astropy.table import Column, Table

DEFAULT_LEAD_COLS_RAW = ['object', 'ra', 'dec', 'dp_id', 'date_obs', 'prog_id']
DEFAULT_LEAD_COLS_PHASE3 = ['target_name', 's_ra', 's_dec', 'dp_id', 'date_obs', 'proposal_id']


@dataclass
class _UserParams:
    """
    Parameters set by the user
    """
    table_name: str
    column_name: str = None
    allowed_values: Union[List[str], str] = None
    cone_ra: float = None
    cone_dec: float = None
    cone_radius: float = None
    columns: Union[List, str] = None
    column_filters: Dict[str, str] = None
    top: int = None
    order_by: str = ''
    order_by_desc: bool = True
    count_only: bool = False
    get_query_payload: bool = False
    print_help: bool = False
    authenticated: bool = False


def _split_str_as_list_of_str(column_str: str):
    if column_str == '':
        column_list = []
    else:
        column_list = list(map(lambda x: x.strip(), column_str.split(',')))
    return column_list


def _raise_if_has_deprecated_keys(filters: Optional[Dict[str, str]]) -> bool:
    if not filters:
        return

    if any(k in filters for k in ("box", "coord1", "coord2")):
        raise ValueError(
            "box, coord1 and coord2 are deprecated; "
            "use cone_ra, cone_dec and cone_radius instead."
        )

    if any(k in filters for k in ("etime", "stime")):
        raise ValueError(
            "'stime' and 'etime' are deprecated; "
            "use instead 'exp_start' together with '<', '>', 'between'. Examples:\n"
            "\tcolumn_filters = {'exp_start': '< 2024-01-01'}\n"
            "\tcolumn_filters = {'exp_start': '>= 2023-01-01'}\n"
            "\tcolumn_filters = {'exp_start': \"between '2023-01-01' and '2024-01-01'\"}\n"
        )


def _build_where_constraints(
        column_name: str,
        allowed_values: Union[List[str], str],
        column_filters: Dict[str, str]) -> str:
    def _format_helper(av):
        if isinstance(av, str):
            av = _split_str_as_list_of_str(av)
        quoted_values = [f"'{v.strip()}'" for v in av]
        return f"{column_name} in ({', '.join(quoted_values)})"

    column_filters = column_filters or {}
    where_constraints = []
    if allowed_values:
        where_constraints.append(_format_helper(allowed_values))

    where_constraints += [
        f"{k} {_adql_sanitize_op_val(v)}" for k, v in column_filters.items()
    ]
    return where_constraints


def _reorder_columns(table: Table,
                     leading_cols: Optional[List[str]] = None):
    """
    Reorders the columns of the pased table so that the
    colums given by the list leading_cols are first.
    If no leading cols are passed, it defaults to
    ['object', 'ra', 'dec', 'dp_id', 'date_obs']
    Returns a table with the columns reordered.
    """
    if not isinstance(table, Table):
        return table

    leading_cols = leading_cols or DEFAULT_LEAD_COLS_RAW
    first_cols = []
    last_cols = table.colnames[:]
    for x in leading_cols:
        if x in last_cols:
            last_cols.remove(x)
            first_cols.append(x)
    last_cols = first_cols + last_cols
    table = table[last_cols]
    return table


def _adql_sanitize_op_val(op_val):
    """
    Expected input:
        "= 5", "< 3.14", "like '%John Doe%'", "in ('item1', 'item2')"
        or just string values like "ESO", "ALMA", "'ALMA'", "John Doe"

    Logic:
        returns "<operator> <value>" if operator is provided.
        Defaults to "= <value>" otherwise.
    """
    supported_operators = ["<=", ">=", "!=", "=", ">", "<",
                           "not like ", "not in ", "not between ",
                           "like ", "between ", "in "]  # order matters

    if not isinstance(op_val, str):
        return f"= {op_val}"

    op_val = op_val.strip()
    for s in supported_operators:
        if op_val.lower().startswith(s):
            operator, value = s, op_val[len(s):].strip()
            return f"{operator} {value}"

    # Default case: no operator. Assign "="
    value = op_val if (op_val.startswith("'") and op_val.endswith("'")) else f"'{op_val}'"
    return f"= {value}"


def raise_if_coords_not_valid(cone_ra: Optional[float] = None,
                              cone_dec: Optional[float] = None,
                              cone_radius: Optional[float] = None) -> bool:
    """
    ra, dec, radius must be either present all three
    or absent all three. Moreover, they must be float
    """
    are_all_none = (cone_ra is None) and (cone_dec is None) and (cone_radius is None)
    are_all_float = isinstance(cone_ra, (float, int)) and \
        isinstance(cone_dec, (float, int)) and \
        isinstance(cone_radius, (float, int))
    is_a_valid_combination = are_all_none or are_all_float
    if not is_a_valid_combination:
        raise ValueError(
            "Either all three (cone_ra, cone_dec, cone_radius) must be present or none.\n"
            "Values provided:\n"
            f"\tcone_ra = {cone_ra}, cone_dec = {cone_dec}, cone_radius = {cone_radius}"
        )


def _build_adql_string(user_params: _UserParams) -> str:
    """
    Return the adql string corresponding to the parameters passed
    See adql examples at https://archive.eso.org/tap_obs/examples
    """
    query_string = None
    columns = user_params.columns or []

    # We assume the coordinates passed are valid
    where_circle = []
    if user_params.cone_radius is not None:
        where_circle += [
            'intersects(s_region, circle(\'ICRS\', '
            f'{user_params.cone_ra}, {user_params.cone_dec}, {user_params.cone_radius}))=1']

    wc = _build_where_constraints(user_params.column_name,
                                  user_params.allowed_values,
                                  user_params.column_filters) + where_circle

    if isinstance(columns, str):
        columns = _split_str_as_list_of_str(columns)
    if columns is None or len(columns) < 1:
        columns = ['*']
    if user_params.count_only:
        columns = ['count(*)']

    # Build the query
    query_string = ', '.join(columns) + ' from ' + user_params.table_name
    if len(wc) > 0:
        where_string = ' where ' + ' and '.join(wc)
        query_string += where_string

    if len(user_params.order_by) > 0 and not user_params.count_only:
        order_string = ' order by ' + user_params.order_by + (' desc ' if user_params.order_by_desc else ' asc ')
        query_string += order_string

    if user_params.top is not None:
        query_string = f"select top {user_params.top} " + query_string
    else:
        query_string = "select " + query_string

    return query_string.strip()


def _normalize_catalogue_list(values: Union[str, List[str], None],
                              label: str) -> Optional[List[str]]:
    """
    Normalize a catalogue filter to a list of strings.

    Parameters
    ----------
    values : str, list of str, or None
        Filter values to normalize.
    label : str
        Label used in error messages.

    Returns
    -------
    list of str or None
        Normalized list of strings. Returns ``None`` for empty inputs.
    """
    if values is None:
        return None
    if isinstance(values, str):
        cleaned = values.strip()
        return [cleaned] if cleaned else None
    if hasattr(values, "tolist"):
        values = values.tolist()
    if not isinstance(values, (list, tuple, set)):
        raise TypeError(f"`{label}` must be a string or list of strings.")
    normalized = []
    for value in values:
        if not isinstance(value, str):
            raise TypeError(f"All `{label}` entries must be strings.")
        cleaned = value.strip()
        if cleaned:
            normalized.append(cleaned)
    return normalized or None


def _catalogue_like_conditions(column_name: str,
                               values: Optional[List[str]]) -> Optional[str]:
    """
    Build a LIKE clause joined by OR for catalogue filters.
    """
    if not values:
        return None
    return " OR ".join(f"{column_name} LIKE '{value}'" for value in values)


def _build_catalogue_metadata_query(all_versions: bool,
                                    collections: Optional[List[str]],
                                    tables: Optional[List[str]]) -> str:
    """
    Build the ADQL query for catalogue metadata.
    """
    query = """
        SELECT
            collection, title, version, table_name, filter, instrument, telescope, publication_date,
            ref.description AS description, number_rows, number_columns, rel_descr_url, acknowledgment,
            cat_id, mjd_obs, mjd_end, skysqdeg, bibliography, document_id, kc.from_column AS from_column,
            k.target_table AS target_table, kc.target_column AS target_column, schema_name
        FROM TAP_SCHEMA.tables AS ref
        LEFT OUTER JOIN TAP_SCHEMA.keys AS k ON ref.table_name = k.from_table
        LEFT OUTER JOIN TAP_SCHEMA.key_columns AS kc ON k.key_id = kc.key_id
        WHERE schema_name = 'safcat'
    """
    if not all_versions:
        query += """
        AND cat_id IN (
            SELECT t1.cat_id
            FROM TAP_SCHEMA.tables t1
            LEFT JOIN TAP_SCHEMA.tables t2 ON (t1.title = t2.title AND t1.version < t2.version)
            WHERE t2.title IS NULL
        )
        """
    collections_clause = _catalogue_like_conditions("collection", collections)
    if collections_clause:
        query += f" AND ({collections_clause})"
    tables_clause = _catalogue_like_conditions("table_name", tables)
    if tables_clause:
        query += f" AND ({tables_clause})"
    return query


def _build_catalogue_columns_query(collections: Optional[List[str]],
                                   tables: Optional[List[str]]) -> str:
    """
    Build the ADQL query for catalogue column metadata.
    """
    collections = collections or ["%"]
    tables = tables or ["%"]
    collections_clause = _catalogue_like_conditions("collection", collections)
    tables_clause = _catalogue_like_conditions("table_name", tables)
    return f"""
        SELECT table_name, column_name, ucd, datatype, description, unit
        FROM TAP_SCHEMA.columns
        WHERE table_name IN (
            SELECT table_name FROM TAP_SCHEMA.tables WHERE {collections_clause}
        )
        AND ({tables_clause})
    """


def _build_catalogue_table_query(table_name: str,
                                 columns: Union[List[str], str, None],
                                 conditions_dict: Optional[Dict[str, str]],
                                 order_by: Optional[str],
                                 order: str,
                                 top: Optional[int]) -> str:
    """
    Build the ADQL query to retrieve rows from a catalogue table.
    """
    if columns is None:
        column_list = []
    elif isinstance(columns, str):
        column_list = _split_str_as_list_of_str(columns)
    else:
        if hasattr(columns, "tolist"):
            columns = columns.tolist()
        if not isinstance(columns, (list, tuple, set)):
            raise TypeError("`columns` must be a string or list of strings.")
        column_list = []
        for col in columns:
            if not isinstance(col, str):
                raise TypeError("All `columns` entries must be strings.")
            column_list.append(col.strip())
    select_columns = ", ".join(column_list) if column_list else "*"
    select_clause = f"SELECT {'TOP ' + str(top) + ' ' if top else ''}{select_columns}"

    conditions_clause = ""
    if conditions_dict:
        if not isinstance(conditions_dict, dict):
            raise TypeError("`conditions_dict` must be a dictionary.")
        conditions = []
        for key, value in conditions_dict.items():
            if value is None:
                continue
            conditions.append(f"{key} {_adql_sanitize_op_val(value)}")
        if conditions:
            conditions_clause = " WHERE " + " AND ".join(conditions)

    order_clause = ""
    if order_by:
        order_token = (order or "ascending").strip().lower()
        if order_token in ("ascending", "asc", ""):
            direction = "asc"
        elif order_token in ("descending", "desc"):
            direction = "desc"
        else:
            raise ValueError("`order` must be 'ascending' or 'descending'.")
        order_clause = f" ORDER BY {order_by} {direction}"

    return f"{select_clause} FROM {table_name}{conditions_clause}{order_clause}"


def _set_last_version(table: Table, update: bool = True) -> None:
    """
    Append or update a ``last_version`` column based on title/version metadata.
    """
    required_cols = ("title", "version")
    if any(col not in table.colnames for col in required_cols):
        return
    if "last_version" in table.colnames and not update:
        return
    if "last_version" in table.colnames:
        table.remove_column("last_version")

    titles = table["title"]
    versions = table["version"]
    try:
        versions_numeric = versions.astype(float)
        use_numeric = True
    except Exception:
        versions_numeric = versions
        use_numeric = False

    last_version_flags = np.zeros(len(table), dtype=bool)
    for title in np.unique(titles):
        mask = titles == title
        version_values = versions_numeric[mask]
        latest = np.nanmax(version_values) if use_numeric else max(version_values)
        last_version_flags[mask] = (version_values == latest)

    table.add_column(
        Column(
            data=last_version_flags,
            name="last_version",
            dtype=bool,
            description="True if this is the latest version of the catalogue",
        )
    )
