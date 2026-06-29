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
from astropy.io import fits
from pyqsofit.PyQSOFit import QSOFit
import matplotlib

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
        self._date_key = None
        self._run_name = None

        self.q = None
        self.objid = None # Unique identifier for an object, see append()

# --------------------------
# Adding new spectra
# --------------------------
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

        print(f"Inside make_run_name: Date {self._date_key}, Time {self._run_name}")

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
    
    def _require_run(self):
        """
        Raise error if start_run() has not been called yet
        """
        if(self._date_key == None):
            raise RuntimeError(
                "No active run. Call start_run before append()"
            )
        return
    
    def _build_file(self, objid: str):
        """
        Building the hierarchy file system
        """
        # Opening the file and creating appropriate groups
        try:
            with h5.File(self.path, "r") as f:
                if self.verbose:
                    print("File opened successfully")           
        except:
            # Create new file and build structure
            print("File did not exist, creating new one")
            with h5.File(self.file_name, "a") as f:
                # Creating folder for local day
                f.create_group(self._date_key)

        with h5.File(self.file_name, "a") as f:
              # Check for first run of local day
            if(self._date_key not in list(f.keys())):
                date_grp = f.create_group(self._date_key)
            else:
                date_grp = f[self._date_key]
            # Check for first obj in run
            if(self._run_name not in list(date_grp.keys())):
                run_grp = date_grp.create_group(self._run_name)
            else:
                run_grp = date_grp[self._run_name]
            # Check for duplicate object names
            if(objid in list(run_grp.keys())):
                raise RuntimeError(f"Object ID {objid} already exists in this group. IDs must be unique")
            # Build obj structure
            obj_grp = run_grp.create_group(objid)

            init_grp = obj_grp.create_group("init_params")
            init_grp.create_group("user_input")
            init_grp.create_group("line_priors")
            init_grp.create_group("conti_priors")

            scalar_grp = obj_grp.create_group("scalar_results")
            scalar_grp.create_group("line_results")
            scalar_grp.create_group("conti_result")

            obj_grp.create_group("spectra")
            obj_grp.create_group("extra_attrs")


        return
    
    def _get_init_params(self,
                        nsmooth: int,
                        and_mask: bool,
                        or_mask: bool,
                        reject_badpix: bool,
                        deredden: bool,):
        input_params_list = ["z", "wave_range", "wave_mask", "decompose_host", "host_prior", "host_prior_scale", 
                        "host_line_mask", "decomp_na_mask", "qso_type", "npca_qso", "host_type", 
                        "npca_gal", "Fe_uv_op", "poly", "BC", "rej_abs_conti", "n_pix_min_conti", 
                        "linefit", "rej_abs_line", "MC", "MCMC", "nsamp", "nburn", "nthin", "epsilon_jitter"]

        input_params = {}
        for field in input_params_list:
            input_params[field] = getattr(self.q, field)
        input_params["nsmooth"] = nsmooth
        input_params["and_mask"] = and_mask
        input_params["or_mask"] = or_mask
        input_params["reject_badpix"] = reject_badpix
        input_params["deredden"] = deredden

        # Converting to df then hd5 dataset
        input_df = pd.DataFrame([input_params])
        input_df.to_hdf(self.file_name, key=f"{self._date_key}/{self._run_name}/{self.objid}/init_params/user_input")
        
        # Line priors are stored as astropy.io.fits.fitsrec.FITS_rec, can be converted directly to pd DataFrame
        line_priors_df = pd.DataFrame(self.q.linelist)
        line_priors_df.to_hdf(self.file_name, key= f"{self._date_key}/{self._run_name}/{self.objid}/init_params/line_priors")

        # conti_priors are stored without labels
        f = fits.open(self.q.param_file_name)
        conti_table = f["CONTI_PRIORS"].data
        conti_cols = ["parname","initial", "min", "max", "vary"]

        conti_priors_df = pd.DataFrame(conti_table, columns = conti_cols)
        conti_priors_df.to_hdf(self.file_name, key = f"{self._date_key}/{self._run_name}/{self.objid}/init_params/conti_priors")

        if(self.verbose == True):
            print("init_params built")

        return

    def _save_spectra(self):
        spectra_list = ["fur_result", "gauss_result", "line_flux", "f_line_model", 
                        "uniq_linecomp_sort", "wave_mask", "wave", "wave_prereduced", 
                        "flux_prereduced", "decomposed", "host", "tmp_all"]
        
        if(self.q.decomposed == True):
            spectra_list.append("qso")
        
        spectra = {}
        for field in spectra_list:
            spectra[field] = getattr(self.q, field)

        spectra_df = pd.DataFrame([spectra])
        spectra_df.to_hdf(self.file_name, key = f"{self._date_key}/{self._run_name}/{self.objid}/spectra")
        return


    def append(
            self,
            objid: str,
            q: QSOFit,
            nsmooth: int,
            and_mask: bool,
            or_mask: bool,
            reject_badpix: bool,
            deredden: bool,
            extra_attrs: dict | None = None):
        
        """
        Append one fit result to the current file. Will create file if it does not exist.

        params:
            q: QSOFit
                QSOFit object after calling q.Fit(save_result = False, ...)
            
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
        self._require_run()
        self._build_file(objid)
        
        # Getting init_params from qso object
        self.objid = objid
        self.q = q
        self._get_init_params(nsmooth, and_mask, or_mask, reject_badpix, deredden)

        # Getting scalar result params from qso object
        line_result_df = pd.DataFrame([q.line_result], columns=q.line_result_name)
        line_result_df.to_hdf(self.file_name, key=f"{self._date_key}/{self._run_name}/{self.objid}/scalar_results/line_results")
        conti_result_df = pd.DataFrame([q.conti_result], columns=q.conti_result_name)
        conti_result_df.to_hdf(self.file_name, key=f"{self._date_key}/{self._run_name}/{self.objid}/scalar_results/conti_results")
    

        # Getting spectral results params from qso object
        self._save_spectra()

        # Saving extra_attr
        if(len(extra_attrs) > 0):
            extra_attrs_df = pd.DataFrame([extra_attrs])
            extra_attrs_df.to_hdf(self.file_name, key=f"{self._date_key}/{self._run_name}/{self.objid}/extra_attrs")

        if(self.verbose == True):
            print(f"{objid} appended successfully")
            
        return
    
#------------------------------------
# Navigation
#------------------------------------
    def list_runs(self) -> list[str]:
        """
        Return all run keys in the file, sorted chronologically.

        Each entry is a full path like ``"/20250601/20250601_143000_night1"``.
        """
        if not os.path.exists(self.path):
            return []
        runs = []
        with h5.File(self.path, "r") as f:
            for date_grp in sorted(f.keys()):
                for run_grp in sorted(f[date_grp].keys()):
                    runs.append(f"/{date_grp}/{run_grp}")
        return runs

    def list_dates(self) -> list[str]:
            """Return all date groups (``YYYYMMDD``) present in the file."""
            if not os.path.exists(self.path):
                return []
            with h5.File(self.path, "r") as f:
                return sorted(f.keys())
# These functions have been taken from the PyQSOFit source code to re-create the plots. Update as needed


            
        