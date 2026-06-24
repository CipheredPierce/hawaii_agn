"""
pyqsofit_catalog_2
------------------------------
A class for storing PyQSOFit results using pandas HDFStore 

 Properties:
        -----------
        .wave: array
            the rest wavelength, some pixels have been removed.
            
        .flux: array
            the rest flux. Dereddened and *(1+z) flux.  
            
        .err: array
            the error.
        
        .wave_prereduced: array
            the wavelength after removing bad pixels, masking, deredden, spectral trim, and smoothing.
            
        .flux_prereduced: array
            the flux after removing bad pixels, masking, deredden, spectral trim, and smoothing.
            
        .err_prereduced: array
            the error after removing bad pixels, masking, deredden, spectral trim, and smoothing.
            
        .host: array
            the model of host galaxy from PCA method
               
        .qso: array
            the model of a quasar from PCA method.
            
        .SN_ratio_conti: float
            the mean S/N ratio of 1350, 3000 and 5100A.
            
        .conti_fit.: structure 
            all the continuum fitting results, including best-fit parameters and Chisquare, etc. For details,
            see https://lmfit.github.io/lmfit-py/fitting.html
            
        .f_conti_model: array
            the continuum model including power-law, polynomial, optical/UV FeII, Balmer continuum.
            
        .f_bc_model: array
            the Balmer continuum model.
            
        .f_fe_uv: array
            the UV FeII model.
            
        .f_fe_op: array
            the optical FeII model.
            
        .f_pl_model: array
            the power-law model.
            
        .f_poly_model: array
            the polynomial model.
            
        .PL_poly_BC: array
            The combination of Powerlaw, polynomial and Balmer continuum model.
            
        .line_flux: array
            the emission line flux after subtracting the .f_conti_model.
        
        .line_fit: structrue
            Line fitting results for last complexes (From Lya to Ha) , including best-fit parameters, errors (lmfit derived) and Chisquare, etc. For details,
            see https://lmfit.github.io/lmfit-py/fitting.html
        
        .gauss_result: array
            3*n Gaussian parameters for all lines in the format of [scale, centerwave, sigma ], n is number of Gaussians for all complexes.
            ADD UNITS
            
        gauss_result_all: array
            [nsamp, 3*n] Gaussian parameters for all lines in the format of [scale, centerwave, sigma ], n is number of Gaussians for all complexes.
            ADD UNITS
            
        .conti_result: array
            continuum parameters, including widely used continuum parameters and monochromatic flux at 1350, 3000
            and 5100 Angstrom, etc. The corresponding names are listed in .conti_result_name. For all continuum fitting results,
            go to .conti_fit.params. 
            
        .conti_result_name: array
            the names for .conti_result.
            
        .fur_result: array
            emission line parameters, including FWHM, sigma, EW, measured from whole model of each main broad emission line covered.
            The corresponding names are listed in .line_result_name.
            
        .fur_result_name: array
            the names for .fur_result.
            
        .line_result: array
            emission line parameters, including FWHM, sigma, EW, measured from whole model of each main broad emission line covered,
            and fitting parameters of each Gaussian component. The corresponding names are listed in .line_result_name.
            
        .line_result_name: array
            the names for .line_result.
            
        .uniq_linecomp_sort: array
            the sorted complex names.
            
        .all_comp_range: array
            the start and end wavelength for each complex. e.g., Hb is [4640.  5100.] AA.
            
        .linelist: array
            the information listed in the param_file_name (qsopar.fits).

"""

import os
import warnings
import numpy as np
import pandas as pd
from PyQSOFit import QSO
import h5py
from astropy.table import Table
import datetime

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _infer_min_itemsize(result: dict) -> dict:
    """Return min_itemsize dict for string columns so pandas doesn't truncate."""
    sizes = {}
    for k, v in result.items():
        if isinstance(v, str):
            sizes[k] = max(len(v), 32)
    return sizes


def _sanitize_result(result: dict) -> dict:
    """
    Convert any numpy scalars / 0-d arrays to plain Python types so pandas
    doesn't complain, and drop None values with a NaN replacement.
    """
    clean = {}
    for k, v in result.items():
        if v is None:
            clean[k] = np.nan
        elif isinstance(v, np.ndarray) and v.ndim == 0:
            clean[k] = v.item()
        elif isinstance(v, (np.integer,)):
            clean[k] = int(v)
        elif isinstance(v, (np.floating,)):
            clean[k] = float(v)
        else:
            clean[k] = v
    return clean

# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class QSOCatalog:
    """
    Append-friendly HDF5 catalog for PyQSOFit scalar results.

    Parameters
    ----------
    scalar_path : str
        Path to the scalar-only .h5 file.  Created on first append if it doesn't exist.
    full_path : str
        Path to the full storage file (inclues spectrum arrays for plotting). Created on first append if
        it doesn't exist
    param_key : str
        HDF5 key for the scalar parameter table.  Default ``"results"``.
    spectra_group : str
        HDF5 group path under which per-object spectra are stored.
        Default "spectra". (Only utilized for scalar+spectra file)
    """

    def __init__(self, path: str, param_key: str = "results",
                 spectra_group: str = "spectra"):
        self.path = path
        self.param_key = param_key
        self.spectra_group = spectra_group
        self._min_itemsize: dict = {}   # grows as string columns are seen

    # ------------------------------------------------------------------
    # Writing
    # ------------------------------------------------------------------

    def append(
        self,
        result: QSO | None = None,
        fit_time: datetime.datetime | None = None
    ) -> None:
        """
        Append one fit result to the catalog.

        Parameters
        ----------
        result : QSO
            The QSO object from PyQSOFit that stores the results. This object
            contains both scalars and spectrum results that will be stored
            in different hd5 files
        fit_time: datetime.datetime
            
        """
        result = _sanitize_result(result)

        if objid is not None:
            result["objid"] = objid

        # ---- scalar params -------------------------------------------
        df_new = pd.DataFrame([result])
        itemsize = _infer_min_itemsize(result)
        self._min_itemsize.update(itemsize)

        mode = "a"   # append / create
        try:
            df_new.to_hdf(
                self.path,
                key=self.param_key,
                mode=mode,
                append=True,
                format="table",
                data_columns=True,          # every column is queryable
                min_itemsize=self._min_itemsize,
                complevel=5,                # light compression
                complib="blosc",
            )
        except Exception as exc:
            # If the store exists but min_itemsize grew (longer string),
            # we need to rebuild — rare but handle gracefully.
            if "Cannot serialize" in str(exc) or "itemsize" in str(exc).lower():
                self._rebuild_with_new_row(df_new)
            else:
                raise

        # ---- spectra / arrays ----------------------------------------
        if objid is not None and any(
            a is not None for a in [wave, flux, err, model, extra_arrays]
        ):
            self._store_spectrum(
                objid, wave=wave, flux=flux, err=err,
                conti=conti, line=line, extra_arrays=extra_arrays or {},
                attrs={k: result[k] for k in ("ra", "dec", "redshift")
                       if k in result},
            )