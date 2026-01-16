
***************
Troubleshooting
***************

Clearing Cache
==============

If you encounter repeated query failures or see outdated or inconsistent results, you may be dealing with a stale or corrupted local cache. You can clear the entire Astropy cache for ESO with the following command:

.. doctest-skip::

    >>> from astroquery.eso import Eso
    >>> Eso.clear_cache()

This will remove all cached ESO queries and downloaded metadata files. Data products already downloaded will remain unless manually deleted.

.. _column-filters-fix:

Using Correct ``column_filters``
================================

If your query fails or silently returns no results, it might be because you're using column names that are **accepted in the ESO web interface (WDB)** but **not supported by the TAP/ADQL interface** that now is used within ``astroquery.eso``. A common case involves using ``stime`` and ``etime``, which are not valid TAP fields. Instead, use ``exp_start``, the TAP-compliant column representing the observation start time. This field supports SQL-style date filtering.

Below are examples of invalid filter usage and their corrected TAP-compatible versions.

Filtering between two dates
---------------------------

.. doctest-skip::

    # ❌ Invalid (WDB-specific fields, not recognized by TAP)
    column_filters = {
        "stime": "2024-01-01",
        "etime": "2024-12-31"
    }

.. doctest-skip::

    # ✅ Correct (TAP-compliant syntax using 'exp_start')
    column_filters = {
        "exp_start": "between '2024-01-01' and '2024-12-31'"
    }

Filtering with only a start date
--------------------------------

.. doctest-skip::

    # ❌ Invalid
    column_filters = {
        "stime": "2024-01-01"
    }

.. doctest-skip::

    # ✅ Correct
    column_filters = {
        "exp_start": "> '2024-01-01'"
    }

Filtering with only an end date
-------------------------------

.. doctest-skip::

    # ❌ Invalid
    column_filters = {
        "etime": "2024-12-31"
    }

.. doctest-skip::

    # ✅ Correct
    column_filters = {
        "exp_start": "< '2024-12-31'"
    }