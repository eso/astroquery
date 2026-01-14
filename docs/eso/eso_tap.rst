
***********************************************
Query Raw or Reduced Data using the TAP Service
***********************************************

The ESO `TAP service <https://archive.eso.org/programmatic/#TAP>`_ allows you to issue custom `ADQL <https://www.ivoa.net/documents/ADQL/>`_ (Astronomical Data Query Language) queries against the archive metadata, offering fine-grained control over your search. TAP queries can be issued against different tables, depending on the type of data you're interested in:

- The ``ivoa.ObsCore`` table provides standardized metadata for **fully calibrated (Phase 3) data products**.
- The ``dbo.raw`` table provides access to **raw observational data** across all ESO instruments.
- The ``ist.<instrument_name>`` tables (e.g. ``ist.midi``, ``ist.muse``) allow more detailed queries tailored to **instrument-specific raw metadata**.

These various query options have also been demonstrated earlier in this documentation using high-level `astroquery.eso` interfaces such as :meth:`~astroquery.eso.EsoClass.query_instrument`, :meth:`~astroquery.eso.EsoClass.query_main`, and :meth:`~astroquery.eso.EsoClass.list_surveys`. Using ADQL directly through TAP enables greater flexibility when building complex queries that combine constraints across multiple metadata fields.

Query for Raw Data (Generic)
============================

The following example demonstrates how to query the ``dbo.raw`` table for raw data products from the ``MUSE`` instrument, specifically looking for observations of the object ``NGC300``:

.. doctest-remote-data::

    >>> query = """
    ...         SELECT *
    ...         FROM dbo.raw
    ...         WHERE object = 'NGC300'
    ...             AND instrument = 'muse'
    ...          """
    >>> result = eso.query_tap_service(query)
    >>> result

Query for Raw Data (Instrument-Specific)
========================================

The following example demonstrates how to query the ``ist.muse`` table for raw data products from the ``MUSE`` instrument, specifically looking for observations taken during the night with a moon illumination below 30% and an average seeing (FWHM) below 1.0 arcsec:

.. doctest-remote-data::

    >>> query = """
    ...         SELECT *
    ...         FROM ist.muse
    ...         WHERE night_flag = 'NIGHT'
    ...             AND moon_illu < 30
    ...             AND dimm_fwhm_avg < 1.0
    ...             AND exptime > 100
    ...             AND lst between 0 and 6   
    ...          """
    >>> result = eso.query_tap_service(query)
    >>> result

Query for Reduced Data Products
===============================

The following example queries the ``ivoa.ObsCore`` table to find fully calibrated (``calib_level=3``) multi-extension observations (``multi_ob='M'``) from the ``SPHERE`` survey, with spatial pixel scales smaller than 0.2 arcsec:

.. doctest-remote-data::

    >>> query = """
    ...          SELECT obs_collection, calib_level, multi_ob, filter, s_pixel_scale, instrument_name 
    ...          FROM ivoa.ObsCore 
    ...          WHERE obs_collection IN ('sphere') 
    ...              AND calib_level = 3 
    ...              AND multi_ob = 'M' 
    ...              AND s_pixel_scale < 0.2
    ...          """
    >>> result = eso.query_tap_service(query)
    >>> result
    <Table length=15>
    obs_collection calib_level multi_ob filter s_pixel_scale instrument_name
                                                arcsec                   
        object        int32     object  object    float64         object    
    -------------- ----------- -------- ------ ------------- ---------------
            SPHERE           3        M    K12        0.0122          SPHERE
            SPHERE           3        M    K12        0.0122          SPHERE
            ...
            SPHERE           3        M      H        0.0122          SPHERE

.. tip:: 

    For more information about the TAP and how to write ADQL queries, refer to the following resources:

    - `ESO TAP+ documentation <https://archive.eso.org/programmatic/>`_: Describes ESO's implementation of TAP and the available services.
    - `IVOA TAP standard <https://www.ivoa.net/documents/TAP/>`_: The official specification from the International Virtual Observatory Alliance.
    - `ADQL specification <https://www.ivoa.net/documents/ADQL/>`_: Defines the query language used to interact with TAP services.

Download Data
=============

To download the data returned by the query, you can use the :meth:`~astroquery.eso.EsoClass.retrieve_data` method. This method takes a list of data product IDs (``dp_id``) and downloads the corresponding files from the ESO archive.

.. doctest-remote-data::
    >>> eso.retrieve_data(table["dp_id"])

The ``data_files`` points to the decompressed dataset filenames that have been locally downloaded. The default location of the decompressed datasets can be adjusted by providing a ``destination`` keyword in the call to :meth:`~astroquery.eso.EsoClass.retrieve_data`.

.. doctest-skip::
    >>> data_files = eso.retrieve_data(table["dp_id"], destination="./eso_data/")