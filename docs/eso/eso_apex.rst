
*******************
Query for APEX Data
*******************

This notebook provides an example of how to query the ESO Science Archive for **APEX data**. 

APEX data are available within the archive and can be queried using any of the standard astroquery methods:  
- :meth:`~astroquery.eso.EsoClass.query_survey` for **reduced data**  
- :meth:`~astroquery.eso.EsoClass.query_main` for **raw data**  
- :meth:`~astroquery.eso.EsoClass.query_instrument` for **raw data (instrument specific)**

In addition, there is the option to use :meth:`~astroquery.eso.EsoClass.query_apex_quicklooks`, which is a dedicated query for retrieving **APEX Quick Look products**.

The workflow for APEX data is somewhat different from that of other ESO-operated instruments and telescopes. The **APEX Quick Looks** aim to address some of these differences by providing users with ``.tar`` files that include additional data outputs — such as diagnostic plots, observing logs, and (in some cases) a reduced version of the dataset. These ``.tar`` files also include ``class`` (``.apex``) files that allow users to **re-run the data reduction**, if desired. This is often more useful than working directly with the ``.fits`` files available in the archive and can be **particularly helpful for heterodyne datasets**. For heterodyne observations, the calibrated ``.class`` file is typically included, while the corresponding uncalibrated ``.fits`` files are usually **not needed** and cannot be processed with standard reduction software.

The APEX Science Archive provides access to raw observations and associated data products for observations performed with APEX since **July 11, 2005**. The Quick Look products are designed to help users preview and assess the data quality, providing useful summaries that complement the raw observations.

This notebook will guide you through a step-by-step workflow to **identify, query, and download** APEX Quick Look data products from the ESO Science Archive.

Here, we focus on APEX observations from the [**ALCOHOLS survey**](https://www.eso.org/rm/api/v1/public/releaseDescriptions/199) (12CO(3–2) line emission in the Milky Way) available in the ESO Archive. We will first search for the **reduced ALCOHOLS data products**, then identify the corresponding raw datasets using the **instrument-specific query**. Once the raw data is identified, we can use it to locate and retrieve the **associated APEX Quick Look products**. We follow this workflow because APEX Quick Looks typically require knowledge of the **APEX proposal ID**, which may not always be known in advance — this is **not** the same as the ESO programme ID. If you know your APEX proposal ID (for example, if you are querying your own data), you can search for APEX Quick Look files directly.

Note that this workflow is **not specific to the ALCOHOLS survey** and can be used to query APEX Quick Look products for any project.

Query Reduced APEX data
=======================

We first query for the reduced data from the **ALCOHOLS** survey, and retrieve the (ESO) proposal ID. 

.. doctest-remote-data::

    >>> table_reduced = eso.query_surveys("ALCOHOLS") # query the ESO archive for the ALCOHOLS survey
    >>> proposal_id = list(set(table_reduced['proposal_id'])) # extract unique proposal IDs from the query result
    
    >>> # Check if we have a single proposal ID or multiple
    >>> if len(proposal_id) == 1:
    >>>    proposal_id = proposal_id[0]
    >>> else:
    >>>    print("Warning: Multiple proposal IDs found...")
    
    >>> proposal_id = proposal_id.split('(')[0] # extract the first part of the proposal ID before any parentheses (i.e. the run ID)
    >>> print(f"Proposal ID: {proposal_id}")
    Proposal ID: 094.C-0935

Note that multiple proposal IDs may be returned, which would require minor changes to the above example to loop through the projects.

Available Query Constraints
===========================

As always, it is good practice to check the available columns to search in the instrument-specific query.

.. doctest-remote-data::

    >>> eso.query_instrument("APEX", help=True) 
    INFO: 
    Columns present in the table ist.APEX:
       column_name    datatype    xtype     unit
    ----------------- -------- ----------- -----
       access_estsize     long             kbyte
           access_url     char                  
                 bwid    float               GHz
             channels      int                  
         datalink_url     char                  
             date_obs     char                  
                  dec   double               deg
               dp_cat     char                  
                dp_id     char                  
              dp_tech     char                  
              dp_type     char                  
              ecl_lat   double               deg
              ecl_lon   double               deg
            exp_start     char   timestamp      
             exposure    float                 s
              exptime    float                 s
                 febe     char                  
                 freq    float               GHz
              freqres    float                  
              gal_lat   double               deg
              gal_lon   double               deg
           instrument     char                  
           lambda_max   double                nm
           lambda_min   double                nm
        last_mod_date     char   timestamp      
                 line     char                  
                  lst    float                 s
              mjd_obs   double                 d
                npols      int                  
                nsubs    short                  
                ob_id      int                  
               object     char                  
    observer_initials     char                  
       observing_mode     char                  
             origfile     char                  
               period      int                  
               pi_coi     char                  
              prog_id     char                  
           prog_title     char                  
            prog_type     char                  
           project_id     char                  
                   ra   double               deg
         release_date     char   timestamp      
             restfreq    float                Hz
             s_region     char adql:REGION      
             scangeom     char                  
             scanmode     char                  
              scannum      int                  
             scantype     char                  
              skyfreq    float                Hz
         tel_airm_end    float                  
       tel_airm_start    float                  
              tel_alt    float               deg
               tel_az    float               deg
            telescope     char                  
             wobcycle    float                 s
             wobthrow    float               deg
              wobused     char                  
    
    Number of records present in the table ist.APEX:
    913029
     [astroquery.eso.core]

Query Raw APEX data
===================

We now query for raw data from the APEX instrument, using the proposal ID we retrieved from the previous query.

.. doctest-remote-data::

    >>> table_raw = eso.query_instrument("APEX", column_filters={"prog_id": f"like '{proposal_id}%'"}) # query the APEX instrument for data related to the proposal ID
    >>> project_id = list(set(table_raw["project_id"])) # extract unique project IDs from the raw data query
    >>> project_id = project_id[0] # Assuming we only have one project ID
    >>> print(f"Project ID: {project_id}")
    Project ID: E-094.C-0935A-2014

In this case, we know there is only **one** APEX proposal ID, but if there were multiple IDs, we would need to loop through them.

.. tip::
    In the :meth:`~astroquery.eso.EsoClass.query_surveys` query, the ``"proposal_id"`` column refers to the **ESO programme ID**. In contrast, in an APEX-specific query using :meth:`~astroquery.eso.EsoClass.query_instrument`, the ``"prog_id"`` column also refers to the **ESO programme ID**, **not** the **APEX proposal ID**. The APEX proposal ID is instead found in the ``"project_id"`` column in the :meth:`~astroquery.eso.EsoClass.query_instrument` query—this is the value used to identify APEX Quick Look products.

Query APEX Quick Look products
==============================

We can check the available columns to search in the query.

.. doctest-remote-data::

    >>> eso.query_apex_quicklooks(help=True)
    INFO: 
    Columns present in the table ist.apex_quicklooks:
      column_name   datatype   xtype    unit
    --------------- -------- --------- -----
     access_estsize     long           kbyte
         access_url     char                
         instrument     char                
    instrument_type     char                
            partner     char                
             pi_coi     char                
            prog_id     char                
         prog_title     char                
          prog_type     char                
         project_id     char                
       quicklook_id     char                
       release_date     char timestamp      
    
    Number of records present in the table ist.apex_quicklooks:
    282296
     [astroquery.eso.core]

And now, query for the APEX Quick Look products using the APEX proposal ID (``project_id``) we retrieved from the previous query.

.. doctest-remote-data::

    >>> table_quicklooks = eso.query_apex_quicklooks(project_id) 
    >>> table_quicklooks  
    <Table length=15>
    access_estsize                               access_url                               instrument instrument_type partner ...                                    prog_title                                   prog_type     project_id             quicklook_id              release_date      
        kbyte                                                                                                                ...                                                                                                                                                                  
        int64                                      object                                   object        object      object ...                                      object                                       object        object                  object                    object         
    -------------- ---------------------------------------------------------------------- ---------- --------------- ------- ... ------------------------------------------------------------------------------- --------- ------------------ --------------------------- ------------------------
            846755 https://dataportal.eso.org/dataPortal/file/E-094.C-0935A.2014DEC10.TAR    APEXHET      Heterodyne     ESO ... The APEX Large CO Heterodyne Outflow Legacy Supercam survey of Orion (ALCOHOLS)    Normal E-094.C-0935A-2014 E-094.C-0935A.2014DEC10.TAR 2014-12-10T07:05:44.397Z
               ...                                                                    ...        ...             ...     ... ...                                                                             ...       ...                ...                         ...                      ...
          40963041 https://dataportal.eso.org/dataPortal/file/E-094.C-0935A.2015AUG07.TAR    APEXHET      Heterodyne     ESO ... The APEX Large CO Heterodyne Outflow Legacy Supercam survey of Orion (ALCOHOLS)    Normal E-094.C-0935A-2014 E-094.C-0935A.2015AUG07.TAR 2015-04-25T18:41:53.900Z
              6389 https://dataportal.eso.org/dataPortal/file/E-094.C-0935A.2015AUG22.TAR    APEXHET      Heterodyne     ESO ... The APEX Large CO Heterodyne Outflow Legacy Supercam survey of Orion (ALCOHOLS)    Normal E-094.C-0935A-2014 E-094.C-0935A.2015AUG22.TAR 2015-04-25T18:41:53.900Z


As can be seen from the output above, there is one APEX Quick Look product available per UT date, per APEX proposal ID. 

Also note that the APEX Quick Look products are available in `.tar` (`.TAR`) format, which can be downloaded and extracted (see below). 

Download APEX Quick Look products
=================================

Finally, we can download the APEX Quick Look products using the `eso.retrieve_data` function.

.. doctest-remote-data::

    >>> eso.retrieve_data(table_quicklooks[0]["quicklook_id"])  