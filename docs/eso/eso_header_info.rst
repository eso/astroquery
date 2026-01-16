
***************************
Extended Header Information
***************************

Only a small subset of the keywords present in the data products can be obtained with :meth:`~astroquery.eso.EsoClass.query_instrument`, :meth:`~astroquery.eso.EsoClass.query_main`, :meth:`~astroquery.eso.EsoClass.query_surveys`, or :meth:`~astroquery.eso.EsoClass.query_tap_service`.

There is however a way to get the full primary header of the FITS data products, using :meth:`~astroquery.eso.EsoClass.get_headers`.

.. doctest-remote-data::

    >>> table = eso.query_instrument('midi',
    ...                     column_filters={
    ...                         'object': 'NGC4151',
    ...                         'date_obs': "<='2008-01-01'"
    ...                     },
    ...                     columns=['object', 'date_obs', 'dp_id'])
    >>> table_headers = eso.get_headers(table["dp_id"])
    >>> len(table_headers.columns)
    336
    >>> table_headers
               DP.ID             SIMPLE BITPIX ...   HIERARCH ESO OCS EXPO7 FNAME2     HIERARCH ESO OCS EXPO8 FNAME1     HIERARCH ESO OCS EXPO8 FNAME2
    ---------------------------- ------ ------ ... --------------------------------- --------------------------------- ---------------------------------
    MIDI.2007-02-07T07:01:51.000   True     16 ...
    MIDI.2007-02-07T07:02:49.000   True     16 ...
    MIDI.2007-02-07T07:03:30.695   True     16 ...
                             ...
    MIDI.2007-02-07T07:20:06.695   True     16 ... MIDI.2007-02-07T07:20:06.695.fits
    MIDI.2007-02-07T07:22:57.000   True     16 ... MIDI.2007-02-07T07:20:06.695.fits MIDI.2007-02-07T07:22:57.000.fits
    MIDI.2007-02-07T07:23:38.695   True     16 ... MIDI.2007-02-07T07:20:06.695.fits MIDI.2007-02-07T07:22:57.000.fits MIDI.2007-02-07T07:23:38.695.fits


As shown above, for each data product ID (``DP.ID``; note that this is equivalent to "dp_id" in ``table``), the full header (336 columns in our case) of the archive FITS file is collected. In the above table ``table_headers``, there are as many rows as in the column ``table['dp_id']``.