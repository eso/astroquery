# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
===========================
ESO Astroquery Catalogue tests
===========================

European Southern Observatory (ESO)

"""
import pytest
from astropy.table import Table

from ...eso import Eso

CATALOGUE_TABLE_NAME = "KiDS_DR4_1_ugriZYJHKs_cat_fits"
CATALOGUE_TABLE_FULL = f"safcat.{CATALOGUE_TABLE_NAME}"


def _catalogue_metadata_table():
    return Table({
        "table_name": [CATALOGUE_TABLE_FULL],
    })


def monkey_catalogue_tap(query, **kwargs):
    _ = kwargs
    if "tap_schema.tables" in query.lower():
        return _catalogue_metadata_table()
    raise AssertionError(f"Unexpected catalogue query: {query}")


def test_list_catalogues_returns_table(monkeypatch):
    # monkeypatch instructions from https://pytest.org/latest/monkeypatch.html
    eso = Eso()
    monkeypatch.setattr(eso, "query_tap", monkey_catalogue_tap)
    result = eso.list_catalogues()

    assert isinstance(result, list)
    assert len(result) > 0
    assert all(isinstance(value, str) for value in result)
    assert CATALOGUE_TABLE_NAME in result


def test_query_catalogues_builds_expected_adql_with_columns_and_filters(monkeypatch):
    # monkeypatch instructions from https://pytest.org/latest/monkeypatch.html
    eso = Eso()
    monkeypatch.setattr(eso, "list_catalogues", lambda *args, **kwargs: [CATALOGUE_TABLE_NAME])

    query = eso.query_catalogue(
        catalogue=CATALOGUE_TABLE_NAME,
        columns=["colA", "colB"],
        column_filters={"colC": "> 0", "colD": "foo"},
        get_query_payload=True,
    )

    assert f"from {CATALOGUE_TABLE_FULL}" in query
    assert "select colA, colB" in query
    assert "colC > 0" in query
    assert "colD = 'foo'" in query


def test_query_catalogues_unknown_collection_raises(monkeypatch):
    # monkeypatch instructions from https://pytest.org/latest/monkeypatch.html
    eso = Eso()

    monkeypatch.setattr(eso, "list_catalogues", lambda *args, **kwargs: [CATALOGUE_TABLE_NAME])

    with pytest.raises(ValueError):
        eso.query_catalogue(catalogue="NO_SUCH_COLLECTION")


def test_query_catalogues_help_returns_table(monkeypatch):
    eso = Eso()
    monkeypatch.setattr(eso, "list_catalogues", lambda *args, **kwargs: [CATALOGUE_TABLE_NAME])

    columns = Table({
        "column_name": ["col_ra", "col_dec"],
        "datatype": ["double", "double"],
        "unit": ["deg", "deg"],
        "ucd": ["pos.eq.ra;meta.main", "pos.eq.dec;meta.main"],
    })
    monkeypatch.setattr(eso, "_columns_table", lambda *args, **kwargs: columns)
    monkeypatch.setattr(eso, "query_tap", lambda *args, **kwargs: Table({"count": [1]}))

    result = eso.query_catalogue(catalogue=CATALOGUE_TABLE_NAME, help=True)

    assert isinstance(result, Table)
    assert "column_name" in result.colnames


def test_query_catalogues_cone_uses_ucd_columns(monkeypatch):
    eso = Eso()
    monkeypatch.setattr(eso, "list_catalogues", lambda *args, **kwargs: [CATALOGUE_TABLE_NAME])
    monkeypatch.setattr(eso, "_catalogue_radec_columns", lambda *args, **kwargs: ("col_ra", "col_dec"))

    query = eso.query_catalogue(
        catalogue=CATALOGUE_TABLE_NAME,
        cone_ra=41.2863,
        cone_dec=-55.7406,
        cone_radius=0.04,
        get_query_payload=True,
    )

    assert "CONTAINS(point('', col_ra, col_dec), circle('', 41.2863, -55.7406, 0.04)) = 1" in query
