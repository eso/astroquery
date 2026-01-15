# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
===========================
ESO Astroquery Catalogue tests
===========================

European Southern Observatory (ESO)

"""
from astropy.table import Table

from ...eso import Eso

CATALOGUE_TABLE_NAME = "KiDS_DR4_1_ugriZYJHKs_cat_fits"


def _catalogue_metadata_table():
    return Table({
        "collection": ["KIDS"],
        "title": ["KiDS DR4"],
        "version": ["1.0"],
        "table_name": [CATALOGUE_TABLE_NAME],
    })


def _catalogue_columns_table():
    return Table({
        "table_name": [CATALOGUE_TABLE_NAME, CATALOGUE_TABLE_NAME, CATALOGUE_TABLE_NAME],
        "column_name": ["ra", "dec", "source_id"],
        "ucd": ["pos.eq.ra;meta.main", "pos.eq.dec;meta.main", "meta.id;meta.main"],
        "datatype": ["double", "double", "char"],
        "description": ["", "", ""],
        "unit": ["deg", "deg", ""],
    })


def monkey_catalogue_tap(query, **kwargs):
    _ = kwargs
    if "FROM TAP_SCHEMA.tables AS ref" in query:
        return _catalogue_metadata_table()
    if "FROM TAP_SCHEMA.columns" in query:
        return _catalogue_columns_table()
    raise AssertionError(f"Unexpected catalogue query: {query}")


def test_list_catalogues_returns_table(monkeypatch):
    # monkeypatch instructions from https://pytest.org/latest/monkeypatch.html
    eso = Eso()
    monkeypatch.setattr(eso, "_run_catalogue_tap_query", monkey_catalogue_tap)
    result = eso.list_catalogues()

    assert isinstance(result, Table)
    assert len(result) > 0
    assert "table_name" in result.colnames
    assert "table_RA" in result.colnames
    assert result["table_name"][0] == CATALOGUE_TABLE_NAME


def test_list_catalogues_info_returns_table(monkeypatch):
    # monkeypatch instructions from https://pytest.org/latest/monkeypatch.html
    eso = Eso()
    monkeypatch.setattr(eso, "_run_catalogue_tap_query", monkey_catalogue_tap)
    result = eso.list_catalogues_info(collections="KIDS")

    assert isinstance(result, Table)
    assert len(result) > 0
    assert {"table_name", "column_name", "ucd", "datatype", "description", "unit"}.issubset(
        result.colnames
    )


def test_query_catalogues_builds_expected_adql_with_columns_and_filters(monkeypatch):
    # monkeypatch instructions from https://pytest.org/latest/monkeypatch.html
    eso = Eso()
    captured = {}

    def mock_list_catalogues(*args, **kwargs):
        _ = args, kwargs
        return Table({
            "table_name": [CATALOGUE_TABLE_NAME],
            "number_rows": [10],
        })

    def mock_list_catalogues_info(*args, **kwargs):
        _ = args, kwargs
        return Table({
            "table_name": [CATALOGUE_TABLE_NAME, CATALOGUE_TABLE_NAME],
            "column_name": ["colA", "colB"],
            "ucd": ["", ""],
        })

    def mock_run_catalogue_query(query, maxrec=None, type_of_query="sync", authenticated=False):
        captured["query"] = query
        captured["maxrec"] = maxrec
        captured["type_of_query"] = type_of_query
        captured["authenticated"] = authenticated
        return Table({"colA": [1], "colB": [2]})

    monkeypatch.setattr(eso, "list_catalogues", mock_list_catalogues)
    monkeypatch.setattr(eso, "list_catalogues_info", mock_list_catalogues_info)
    monkeypatch.setattr(eso, "_run_catalogue_tap_query", mock_run_catalogue_query)

    result = eso.query_catalogues(
        tables=CATALOGUE_TABLE_NAME,
        columns=["colA", "colB"],
        conditions_dict={"colC": "> 0", "colD": "foo"},
        maxrec=5,
    )

    assert isinstance(result, Table)
    assert f"FROM {CATALOGUE_TABLE_NAME}" in captured["query"]
    assert "SELECT colA, colB" in captured["query"]
    assert "colC > 0" in captured["query"]
    assert "colD = 'foo'" in captured["query"]
    assert captured["maxrec"] == 5


def test_query_catalogues_unknown_collection_returns_none(monkeypatch):
    # monkeypatch instructions from https://pytest.org/latest/monkeypatch.html
    eso = Eso()

    def mock_list_catalogues(*args, **kwargs):
        _ = args, kwargs
        return Table()

    monkeypatch.setattr(eso, "list_catalogues", mock_list_catalogues)
    result = eso.query_catalogues(collections="NO_SUCH_COLLECTION")

    assert result is None
