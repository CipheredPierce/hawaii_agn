"""
PyQSOFit_cat.py
------------------------
Catalog for storing and recreating QSOFit results 
Uses hierarchical HDF5 file.


Hierarchy
------------------------
    /YYYYMMDD/                         <- date group (fit run date)
        HHMMSS[_label]/                 <- bulk_fit group (one per run)
            <objid>/                    <- Identifier for specific object
                init_params/            <- All parameters set before a fit
                    user_input          <- The parameter valies of QSO.Fit()
                    line_priors         <- The parameters established for emission lines
                    conti_priors        <- The parameters established for the continuum
                result_params           <- Result parameters from QSO.Fit() results
                spectra                 <- Arrays needed to replot the spectra
                user                    <- Additional information that was included by the user
                                            (Note: Any additional quality/parameter will appear in the entire bulk run)



"""

import os
import datetime
import warnings
import numpy as np
import pandas as pd
import h5py as h5
from astropy.table import Table
from pyqsofit.PyQSOFit import QSOFit

#--------------------------------
# Helpers
#--------------------------------


# --------------------------------
# Main Class
# -------------------------------
class QSOCat:
    
    def __init__(self, file_name: str, path: str | None = "."):
        self.path = os.path.join(path, file_name) # Path for the file to be stored in and filename
        self.file_name = file_name

        # Set by _make_run_name; naming for hierarchy
        self._date_key: str
        self._run_name: str

    def _make_run_name(self, label: str | None = None):
        """
        Build self.date_key, self.run_name strings from current system time

        """

        current_time = datetime.datetime.now()
        self._date_key = current_time.strftime("%Y_%m_%d")
        self._run_name = current_time.strftime("%H_%M_%S")

        if (label):
            formatted = label.strip().replace(" ", "_").replace("/", "_")
            self._run_name = f"{self._run_name}_{formatted}"

        print(self._date_key)
        print(self._run_name)

        return

    def start_run(self, 
                  label: str | None = None,
                  verbose: bool | None = True
                  ):
        """
        Initializing a run

        params:
        label: str, optional
            Readable lable appended to run name
        verbose: bool, optional
            If true, program will provide feedback, useful for debugging

        Returns: 
            Full run key, e.g. "2025_06_25/12_00_00_night1"


        """
        self.verbose = verbose

        # Creating date_key and run_name
        self._make_run_name(label)


        return
    
    def end_run(self):
        """
        Ends the current bulk run and closes file

        """

        return
    
    def _require_run(self):
        """
        Raise error if start_run() has not been called yet
        """
        if(self._date_key == None):
            raise RuntimeError(
                "No active run. Call start_run before append()"
            )
    
    def append(
            self,
            qso: QSOFit,
            objid: str,
            nsmooth: int,
            and_mask: bool,
            or_mask: bool,
            reject_badpix: bool,
            deredden: bool,
            extra_attrs: dict | None = None):
        
        """
        Append one fit result to the current file. Will create file if it does not exist.

        params:
            qso: QSOFit
                QSOFit object after calling qso.Fit(save_result = False, ...)
            
            objid: 
                Unique identifier for this object
                examples: 
                    "JWST12345"
                    "COSMOS202012345"
                    "JWST12345_COSMOS20206789"

            nsmooth, and_mask, or_mask, reject_badpix, and deredden are all parameters for QSO.Fit()
            that are not saved internally by PyQSOFit
            
            extra_attrs: dict, optional 
                Keys become column names: values become data (Supports int, float, str values)
                example:
                    extra_attrs = {
                        "instrument": "PFS",
                        "logLBol": 45.65
                        }

                All keys must be valid column names (no spaces or special chars).
                A warning is given if the key would shadow an existing column and key is ignored
                Each key (column) is required for the entire run
        """
        self._require_run

        # Opening the file and creating appropriate groups
        try:
            with h5.File(self.path, "r") as f:
                if self.verbose:
                    print("File opened successfully")
        except:
            # Create new file and build structure
            print("File did not exist, creating new one")
            with h5.File(self.file_name, "a") as f:
                # Check for first run of local day
                if(self.date_key not in list(f.keys())):
                    f.create_group("self.date_key")


        # Create new file
        
        # Set up group hierarchy
       
        # Saving data to the file

        # Building init_params

        # Getting data from qso object
        input_params_list = ["wave_range", "wave_mask", "decompose_host", "host_prior", "host_prior_scale", 
                        "host_line_mask", "decomp_na_mask", "qso_type", "npca_qso", "host_type", 
                        "npca_gal", "Fe_uv_op", "poly", "BC", "rej_abs_conti", "n_pix_min_conti", 
                        "linefit", "rej_abs_line", "MC", "MCMC", "nsamp", "nburn", "nthin", "epsilon_jitter"]

        input_params = {}
        for field in input_params_list:
            input_params[field] = getattr(qso, field)
        input_params["nsmooth"] = nsmooth
        input_params["and_mask"] = and_mask
        input_params["or_mask"] = or_mask
        input_params["deredden"] = deredden

        # Converting to df then hd5 dataset
        input_df = pd.DataFrame([input_params])

            
        