************************
Catalogues - Cone Search
************************

In the simplest case, catalogue queries can be constrained by sky
position (cone search). For catalogue tables, this is currently done by
building an ADQL cone-search expression and passing it through
``column_filters`` in ``eso.query_catalogue``.

.. note::
    The ``cone_ra``, ``cone_dec``, and ``cone_radius`` arguments are not
    currently implemented in ``eso.query_catalogue`` (as they are with e.g. ``eso.query_main``).
    For catalogue cone searches, use a custom ADQL spatial filter as shown
    below.


Identify Coordinate Columns
===========================

Before building a cone search, inspect the catalogue schema with
``help=True`` and identify the right ascension and declination columns from
their UCD values:

- ``pos.eq.ra;meta.main`` for right ascension
- ``pos.eq.dec;meta.main`` for declination

For example:

.. doctest-skip::

    >>> eso.query_catalogue(catalogue='KiDS_DR4_1_ugriZYJHKs_cat_fits', help=True)
    ...
                RAJ2000   DOUBLE              deg                   pos.eq.ra;meta.main
               DECJ2000   DOUBLE              deg                  pos.eq.dec;meta.main
    ...

In this case the column names are ``RAJ2000`` and ``DECJ2000``.
However, different catalogues may use different coordinate column names (e.g.
``RA2000``/``DEC2000`` or ``RA_Spec``/``Dec_Spec``), so checking the UCD is the
most robust way to identify the correct columns.

Query with Cone Filter
===========================

Catalogue cone searches are implemented using the ADQL spatial function
``CONTAINS(POINT, CIRCLE)``. Since ``eso.query_catalogue`` builds its
``WHERE`` clause from the ``column_filters`` dictionary, the spatial
constraint must be provided as a key with value ``1`` (corresponding to
``CONTAINS(...) = 1`` in ADQL). 

Under the hood, ``eso.query_catalogue`` submits an ADQL query via TAP.
For example, a cone search on the KiDS DR4 catalogue corresponds to an
ADQL statement of the form:

.. code-block:: sql

    SELECT *
    FROM KiDS_DR4_1_ugriZYJHKs_cat_fits
    WHERE CONTAINS(POINT('', RAJ2000, DECJ2000), CIRCLE('', <ra_deg>, <dec_deg>, <radius_deg>))=1

Here:

- ``POINT('', RAJ2000, DECJ2000)`` defines the source position
  using the catalogue RA and Dec columns.
- ``CIRCLE('', <ra_deg>, <dec_deg>, <radius_deg>)`` defines the
  search region in degrees.
- ``CONTAINS`` evaluates to ``1`` when the source lies within
  the specified region.

When using ``column_filters``, the dictionary *key* is inserted as the
left-hand side of the ``WHERE`` clause, and the dictionary *value*
becomes the right-hand side. Therefore, providing a key like
``CONTAINS(...)`` with value ``1`` yields ``CONTAINS(...) = 1`` in ADQL.

The helper function below constructs this automatically.

Helper Function
---------------

Define target of interest and search radius:

.. doctest-skip::

    >>> from astroquery.eso import Eso
    >>> from astropy.coordinates import SkyCoord
    >>> import astropy.units as u
    >>> eso = Eso()

    >>> coords = SkyCoord.from_name("NGC1097")
    >>> radius = 3 * u.arcmin

Helper function to construct ADQL cone search predicate:

.. doctest-skip::

    >>>  def make_cone_filter(
    ...        ra,
    ...        dec,
    ...        radius,
    ...        ra_name="RAJ2000",
    ...        dec_name="DECJ2000",
    ...        ):
    ...        """
    ...        Construct an ADQL cone-search predicate for catalogue queries.
    ...
    ...        Parameters
    ...        ----------
    ...        ra : astropy.units.Quantity
    ...            Right ascension of cone centre.
    ...        dec : astropy.units.Quantity
    ...            Declination of cone centre.
    ...        radius : astropy.units.Quantity
    ...            Search radius.
    ...        ra_name : str, optional
    ...            Name of the RA column in the catalogue
    ...            (default: 'RAJ2000').
    ...        dec_name : str, optional
    ...            Name of the Dec column in the catalogue
    ...            (default: 'DECJ2000').
    ...
    ...        Returns
    ...        -------
    ...        dict
    ...            Dictionary suitable for ``column_filters`` of the form
    ...            ``{predicate: 1}``, corresponding to
    ...            ``CONTAINS(...) = 1`` in ADQL.
    ...        """
    ...
    ...        ra_deg = ra.to_value(u.deg)
    ...        dec_deg = dec.to_value(u.deg)
    ...        rad_deg = radius.to_value(u.deg)
    ...
    ...        predicate = (
    ...            f"CONTAINS("
    ...            f"POINT('', {ra_name}, {dec_name}), "
    ...            f"CIRCLE('', {ra_deg}, {dec_deg}, {rad_deg})"
    ...            f")"
    ...        )
    ...
    ...        return {predicate: 1}

Run search with cone filter:

.. doctest-skip::

    >>> column_filters = make_cone_filter(
    ...     ra=coords.ra,
    ...     dec=coords.dec,
    ...     radius=radius,
    ... )

    >>> table = eso.query_catalogue(
    ...     catalogue="KiDS_DR4_1_ugriZYJHKs_cat_fits",
    ...     column_filters=column_filters,
    ... )
    >>> table
    <Table length=55>
    Level    ALPHA_J2000      A_IMAGE         A_WORLD    ...      Z_B_MAX             Z_B_MIN               Z_ML       
    count        deg           pixel            deg      ...                                                           
    float32     float64        float64         float32    ...      float64             float64             float64      
    ---------- ----------- ------------------ ------------ ... ------------------ ------------------- -------------------
    0.02452336   41.624103           3.264482 0.0001940674 ...               0.76                0.69                0.73
    0.02450416   41.632288           1.199719 7.132157e-05 ...               1.35                0.52                1.87
    0.02449745   41.634118           1.714497 0.0001019072 ...               1.03 0.41000000000000003                 7.0
    0.02455906    41.62684 1.2852190000000001 7.640447e-05 ...               1.62                0.98                1.65
        ...         ...                ...          ... ...                ...                 ...                 ...
    0.02458552    41.62298           1.037333  6.16569e-05 ... 1.6400000000000001                0.63                3.66
    0.02453066   41.625326           1.191271 7.081894e-05 ...               1.38                0.53 0.24000000000000002
    0.02451551   41.624227           1.582422 9.406044e-05 ...               0.63 0.43000000000000005                0.55
    0.02454042   41.618984            2.15449 0.0001280547 ... 0.6799999999999999                0.17                0.11

Combining with Additional Column Filters
----------------------------------------

The cone filter returns a standard ``column_filters`` dictionary.
Additional constraints (e.g. magnitude limits) can be added using
normal dictionary updates.

.. doctest::

    >>> # Add a magnitude constraint
    >>> column_filters.update({"MAG_AUTO": "<22"})

    >>> table = eso.query_catalogue(
    ...     catalogue="KiDS_DR4_1_ugriZYJHKs_cat_fits",
    ...     column_filters=column_filters,
    ... )
    <Table length=6>
    Level    ALPHA_J2000      A_IMAGE         A_WORLD     Agaper ...   Ypos     Z_B   Z_B_MAX       Z_B_MIN              Z_ML       
    count        deg           pixel            deg       arcsec ...  pixel                                                         
    float32     float64        float64         float32    float32 ... float32  float64 float64       float64            float64      
    ---------- ----------- ------------------ ------------ ------- ... -------- ------- ------- ------------------- ------------------
    0.02452336   41.624103           3.264482 0.0001940674    0.99 ... 9077.715    0.73    0.76                0.69               0.73
    0.02450728   41.622675           3.695437 0.0002196531    1.06 ... 9153.268    0.51    0.54 0.48000000000000004               0.51
    0.02450339    41.63336           2.607809  0.000155017     0.9 ... 9207.265    0.96    1.35  0.8200000000000001 1.6400000000000001
    0.02452449    41.62697 2.9889900000000003 0.0001776759    0.95 ... 9217.702    0.72    0.75  0.6699999999999999               0.72
    0.02452196   41.626775           2.787715 0.0001656992    0.92 ...   9311.3    0.56    0.59                0.53               0.56
    0.0245021   41.629953           5.531547 0.0003288784    1.38 ... 9338.087     0.7    0.74  0.6599999999999999                0.7