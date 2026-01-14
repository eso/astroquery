
**********************
Query for Reduced Data
**********************

In addition to raw observational files, the ESO Science Archive provides access to a wide range of **processed (reduced) data products**, also known as **Phase 3** data. These include science-ready images, spectra, and datacubes that have been calibrated and validated by ESO or by contributing survey teams.

This section demonstrates how to search for and retrieve these reduced products using ``astroquery.eso``. The examples focus on **Phase 3 survey data**, which are organized by instrument, observing program, and survey tile.

Available Surveys
=================

The list of available surveys can be obtained with :meth:`~astroquery.eso.EsoClass.list_surveys` as follows:

.. doctest-remote-data::

    >>> surveys = eso.list_surveys()
    >>> surveys
    ['081.C-0827', '092.A-0472', '096.B-0054', '1100.A-0528', '1101.A-0127', '193.D-0232',
    '195.B-0283', '196.B-0578', '196.D-0214', '197.A-0384', '198.A-0708', '60.A-9284H',
    '60.A-9493', 'ADHOC', 'ALCOHOLS', 'ALLSMOG', 'ALMA', 'AMAZE', 'AMBRE', 'APEX-SciOps',
    'ARP_VST', 'ATLASGAL', 'CAFFEINE', 'ENTROPY', 'ePESSTOplus', 'ERIS-NIX',
    'ERIS-SPIFFIER', 'ESPRESSO', 'ESSENCE', 'FDS', 'FEROS', 'Fornax3D', 'FORS2-SPEC',
    'GAIAESO', 'GCAV', 'GIRAFFE', 'GOODS_FORS2', 'GOODS_ISAAC', 'GOODS_VIMOS_IMAG',
    'GOODS_VIMOS_SPEC', 'GW170817', 'HARPS', 'HAWKI', 'HUGS', 'INSPIRE', 'KIDS', 'KMOS',
    'LEGA-C', 'LESS', 'MAGIC', 'MUSE', 'MUSE-DEEP', 'MUSE-STD', 'MW-BULGE-PSFPHOT',
    'NGTS', 'NIRPS', 'OMEGACAM_INAF', 'PENELLOPE', 'PESSTO', 'PHANGS', 'PIONIER',
    'SHARKS', 'SPHERE', 'SUPER', 'UltraVISTA', 'UVES', 'UVES_SQUAD', 'VANDELS', 'VEGAS',
    'VEILS', 'VEXAS', 'VHS', 'VIDEO', 'VIKING', 'VIMOS', 'VINROUGE', 'VIPERS', 'VISIONS',
    'VMC', 'VPHASplus', 'VST-ATLAS', 'VVV', 'VVVX', 'XQ-100', 'XSGRB', 'XSHOOTER',
    'XShootU', 'XSL', 'ZCOSMOS']

Available Query Constraints
===========================

As before, list the possible columns in :meth:`~astroquery.eso.EsoClass.query_surveys` that can be queried with: 

.. doctest-remote-data::

    >>> eso.query_surveys(help=True) # get help on the ESO query
    INFO: 
    Columns present in the table ivoa.ObsCore:
        column_name     datatype    xtype     unit 
    ------------------- -------- ----------- ------
               abmaglim   double                mag
         access_estsize     long              kbyte
          access_format     char                   
             access_url     char                   
          bib_reference     char                   
            calib_level      int                                 
                    ...
            target_name     char                   

    Number of records present in the table ivoa.ObsCore:
    4559928
    [astroquery.eso.core]

Query with Constraints (Specific Survey)
========================================

Let's assume that we work with the ``HARPS`` survey, and that we are interested in
target ``HD203608``. The archive can be queried as follows:

.. doctest-remote-data::

    >>> table = eso.query_surveys(surveys="HARPS", 
    ...                           column_filters= {"target_name": "HD203608"})
    >>> table
    <Table length=1000>
    target_name    s_ra     s_dec              dp_id             proposal_id  abmaglim access_estsize ...   snr    strehl t_exptime     t_max          t_min      t_resolution t_xel
                deg       deg                                                mag        kbyte      ...                     s           d              d             s            
    object    float64   float64             object               object    float64      int64      ... float64 float64  float64     float64        float64       float64    int64
    ----------- --------- --------- --------------------------- ------------- -------- -------------- ... ------- ------- --------- -------------- -------------- ------------ -----
    HD203608 321.61455 -65.36429 ADP.2014-09-16T11:03:30.940 077.D-0720(A)       --           5261 ...    60.9      --      33.0 53956.24265204 53956.24227009     33.00048    --
    HD203608 321.61761 -65.36485 ADP.2014-09-16T11:03:31.020 077.D-0720(A)       --           5261 ...    87.0      --    32.999 53953.36835125 53953.36796931    32.999616    --
    HD203608 321.60594 -65.36528 ADP.2014-09-16T11:03:31.067 077.D-0720(A)       --           5261 ...    73.9      --      33.0 53956.15534682 53956.15496487     33.00048    --
    ...
    HD203608 321.61113 -65.37211 ADP.2014-09-16T11:05:14.863 077.D-0720(A)       --           5261 ...    95.2      --    32.999 53954.99642615 53954.99604421    32.999616    --

The returned table has a ``dp_id`` column, which can be used to retrieve the datasets with
:meth:`~astroquery.eso.EsoClass.retrieve_data`: ``eso.retrieve_data(table["dp_id"][0])``.
More details about this method in the following section.

Query with Constraints (Specific Instrument)
============================================

You can also query a specific instrument using the same method (e.g., ``HARPS``). For example, to retrieve **all** available HARPS data products regardless of the associated survey towards ``HD203608`` is given the following query:

.. doctest-remote-data::

    >>> table = eso.query_surveys(column_filters={"instrument_name": "HARPS", 
    ...                                            "target_name": "HD203608"})
    >>> table
    <Table length=1000>
    target_name    s_ra     s_dec              dp_id             proposal_id  abmaglim access_estsize               access_format                ... s_xel2   snr    strehl t_exptime     t_max          t_min      t_resolution t_xel
                deg       deg                                                mag        kbyte                                                 ...                            s           d              d             s            
    object    float64   float64             object               object    float64      int64                        object                   ... int64  float64 float64  float64     float64        float64       float64    int64
    ----------- --------- --------- --------------------------- ------------- -------- -------------- ------------------------------------------ ... ------ ------- ------- --------- -------------- -------------- ------------ -----
    HD203608 321.61455 -65.36429 ADP.2014-09-16T11:03:30.940 077.D-0720(A)       --           5261 application/x-votable+xml;content=datalink ...     --    60.9      --      33.0 53956.24265204 53956.24227009     33.00048    --
    HD203608 321.61761 -65.36485 ADP.2014-09-16T11:03:31.020 077.D-0720(A)       --           5261 application/x-votable+xml;content=datalink ...     --    87.0      --    32.999 53953.36835125 53953.36796931    32.999616    --
    HD203608 321.60594 -65.36528 ADP.2014-09-16T11:03:31.067 077.D-0720(A)       --           5261 application/x-votable+xml;content=datalink ...     --    73.9      --      33.0 53956.15534682 53956.15496487     33.00048    --
    ...
    HD203608 321.61113 -65.37211 ADP.2014-09-16T11:05:14.863 077.D-0720(A)       --           5261 application/x-votable+xml;content=datalink ...     --    95.2      --    32.999 53954.99642615 53954.99604421    32.999616    --

.. tip:: 

    Keep in mind that the definition of a ``survey`` (also referred to as a **collection** in the ESO Science Archive) is not the same as the definition of an **instrument**. The ``instrument_name`` refers to the actual hardware that acquired the data (e.g., ``HARPS``, ``MUSE``), whereas the ``obs_collection`` identifies the scientific program, survey, or processing pipeline associated with the data product. 

    In many cases, survey names match the instrument name (e.g., ``HARPS``, ``MUSE``, ``XSHOOTER``), which typically indicates **Phase 3 products processed and curated by ESO**. However, when the collection name differs (e.g., ``AMBRE``, ``GAIAESO``, ``PHANGS``), it usually denotes **community-contributed data** from large collaborations or specific science teams.

    So, for example, querying for ``eso.query_surveys(column_filters={"instrument_name": "HARPS"})`` will return all products taken with the HARPS instrument, across all programs and collections. In contrast, filtering on ``eso.query_surveys(surveys="HARPS"}`` will return only the `HARPS data reduced by ESO <https://doi.eso.org/10.18727/archive/33>`_.

    You can inspect the collection for each result via the ``obs_collection`` column in your results table.

Download Data
=============

To download the data returned by the query, you can use the :meth:`~astroquery.eso.EsoClass.retrieve_data` method. This method takes a list of data product IDs (``dp_id``) and downloads the corresponding files from the ESO archive.

.. doctest-remote-data::
    >>> eso.retrieve_data(table["dp_id"])

The ``data_files`` points to the decompressed dataset filenames that have been locally downloaded. The default location of the decompressed datasets can be adjusted by providing a ``destination`` keyword in the call to :meth:`~astroquery.eso.EsoClass.retrieve_data`.

.. doctest-skip::
    >>> data_files = eso.retrieve_data(table["dp_id"], destination="./eso_data/")