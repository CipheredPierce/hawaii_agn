"""
pyqsofit_catalog.py
-------------------
Catalog for accumulating PyQSOFit QSOFit results into a hierarchical HDF5 file.

Dependencies

Hierarchy
---------
    /YYYYMMDD/                              ← date group  (fit run date)
        YYYYMMDD_HHMMSS[_label]/            ← bulk_fit group (one per run)
            params                          ← scalar table  (one row per object)
            spectra/                        ← (full catalog only)
                <objid>/
                    wave, flux, err,
                    f_conti_model, f_line_model,
                    f_pl_model, f_fe_mgii_model, f_fe_balmer_model,
                    f_bc_model, f_poly_model, line_flux,
                    host, qso                (present only if decomposed)
                    attrs: ra, dec, redshift
 
Usage
-----
    from pyqsofit_catalog import QSOCatalog
 
    # --- scalar-only catalog ---
    cat = QSOCatalog("params_only.h5")
 
    # --- full catalog (scalars + spectra) ---
    cat_full = QSOCatalog("full_catalog.h5")
 
    # Call once before the fitting loop.
    # extra_scalars are constants stamped onto every row in the run.
    cat.start_run(
        label="night1",
        extra_scalars={"instrument": "SDSS", "plate": 1234, "seeing": 1.2},
    )
    cat_full.start_run(
        label="night1",
        extra_scalars={"instrument": "SDSS", "plate": 1234, "seeing": 1.2},
    )
 
    # After each qso.Fit(...) call:
    cat.append(qso, objid="J1234+5678")
    cat_full.append(qso, objid="J1234+5678", store_spectra=True)
 
    # Query / load (scoped to the active run):
    df = cat.load()
    df_highz = cat.query("redshift > 2.0")
 
    # Load from a specific past run:
    df = cat.load(run_key="/20250601/20250601_143000_night1")
 
    # Export the active run to FITS:
    cat.to_fits("run_results.fits")
 
    # Overview:
    print(cat)
    cat.list_runs()
"""

import os
import datetime
import warnings
import numpy as np
import pandas as pd
import h5py
from astropy.table import Table


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_run_name(label: str | None = None) -> tuple[str, str]:
    """
    Build (date_key, run_name) strings from the current system time.

    Returns
    -------
    date_key : str  e.g. ``"20250601"``
    run_name : str  e.g. ``"20250601_143000"`` or ``"20250601_143000_night1"``
    """
    now = datetime.datetime.now()
    date_key = now.strftime("%Y%m%d")
    run_name = now.strftime("%Y%m%d_%H%M%S")
    if label:
        safe = label.strip().replace(" ", "_").replace("/", "_")
        run_name = f"{run_name}_{safe}"
    return date_key, run_name
 

# The full set of host result column names produced by PyQSOFit when
# decompose_host=True succeeds.  When decomposition is skipped or fails,
# these columns are absent from conti_result — we fill them with NaN so
# every row has an identical schema and HDF5 append never complains.
_HOST_RESULT_NAMES = [
    "SN_host", "rchi2_decomp", "frac_host_4200", "frac_host_5100",
    "Dn4000", "sigma", "sigma_err", "v_off", "v_off_err", "rchi2_ppxf",
]
 
 
def _qso_to_scalar_dict(qso) -> dict:
    """
    Extract all scalar fit results from a QSOFit object into a flat dict.
 
    PyQSOFit stores results in two paired arrays:
      - qso.conti_result / qso.conti_result_name  (continuum parameters)
      - qso.line_result  / qso.line_result_name   (emission line parameters)
 
    Host decomposition columns (SN_host, rchi2_decomp, etc.) are only
    present in conti_result when decompose_host=True succeeded.  When they
    are absent we fill them with NaN so every row has the same schema and
    HDF5 append never raises a column-mismatch error.
 
    Values are stored as strings in the numpy arrays, so we cast each one
    to float where possible and leave the rest as str (e.g. complex names).
    """
    result = {}
 
    # Continuum results
    if hasattr(qso, 'conti_result') and len(qso.conti_result) > 0:
        for name, val in zip(qso.conti_result_name, qso.conti_result):
            try:
                result[name] = float(val)
            except (ValueError, TypeError):
                result[name] = str(val)
 
    # Emission line results
    if hasattr(qso, 'line_result') and len(qso.line_result) > 0:
        for name, val in zip(qso.line_result_name, qso.line_result):
            try:
                result[name] = float(val)
            except (ValueError, TypeError):
                result[name] = str(val)
 
    # Fill any missing host columns with NaN so the schema is consistent
    # regardless of whether decompose_host succeeded for this object.
    for name in _HOST_RESULT_NAMES:
        if name not in result:
            result[name] = np.nan
 
    # Also pad gal_par / qso_par columns if present on other rows but not
    # this one.  We detect the max index seen on the qso object itself so
    # we don't need to hard-code the PCA component count.
    if hasattr(qso, 'host_result_name'):
        for name in qso.host_result_name:
            if name not in result:
                result[name] = np.nan
 
    return result


def _infer_min_itemsize(result: dict) -> dict:
    """Return min_itemsize dict for string columns so pandas doesn't truncate."""
    return {k: max(len(v), 64) for k, v in result.items() if isinstance(v, str)}


def _sanitize_result(result: dict) -> dict:
    """Convert numpy scalars / 0-d arrays to plain Python types."""
    clean = {}
    for k, v in result.items():
        if v is None:
            clean[k] = np.nan
        elif isinstance(v, np.ndarray) and v.ndim == 0:
            clean[k] = v.item()
        elif isinstance(v, np.integer):
            clean[k] = int(v)
        elif isinstance(v, np.floating):
            clean[k] = float(v)
        else:
            clean[k] = v
    return clean


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class QSOCatalog:
    """
    Append-friendly HDF5 catalog for PyQSOFit QSOFit objects.

    The internal hierarchy is::

        /YYYYMMDD/
            YYYYMMDD_HHMMSS[_label]/
                params          ← pandas scalar table (one row per object)
                spectra/        ← per-object spectral arrays (full catalog only)
                    <objid>/
                        wave, flux, err, f_conti_model, f_line_model,
                        f_pl_model, f_fe_mgii_model, f_fe_balmer_model,
                        f_bc_model, f_poly_model, line_flux,
                        host, qso  (only if decompose_host was True)

    Parameters
    ----------
    path : str
        Path to the .h5 file.  Created on first append if it doesn't exist.
    """

    def __init__(self, path: str):
        self.path = path
        self._date_key: str | None = None
        self._run_name: str | None = None
        self._param_key: str | None = None
        self._spectra_prefix: str | None = None
        self._min_itemsize: dict = {}

        # Run-wide constants stamped onto every row; set by start_run()
        self._extra_scalars: dict = {}

        # Per-object scalar keys declared on the first append() call.
        # All subsequent appends must supply exactly these keys.
        # None means no first append has occurred yet this run.
        self._per_object_keys: set | None = None

    # ------------------------------------------------------------------
    # Run management
    # ------------------------------------------------------------------

    def start_run(
            self,
            label: str | None = None,
            extra_scalars: dict | None = None,
        ) -> str:
            """
            Initialise a new bulk-fit run. Call once before the fitting loop.

            Parameters
            ----------
            label : str, optional
                Short human-readable label appended to the run name,
                e.g. ``"night1"`` → ``"20250601_143000_night1"``.
            extra_scalars : dict, optional
                Run-wide constants to stamp onto every row in this run.
                Keys become column names; values set both the data and the
                default for that column.  Supports int, float, and str values.

                Example::

                    extra_scalars={
                        "instrument": "SDSS",   # str column, value "SDSS" on every row
                        "plate":      1234,     # int column, value 1234 on every row
                        "seeing":     1.2,      # float column, value 1.2 on every row
                    }

                All keys must be valid pandas column names (no spaces or
                special characters).  A warning is raised if a key would
                shadow an existing PyQSOFit result column.

            Returns
            -------
            str
                Full run key, e.g. ``"/20250601/20250601_143000_night1"``.
            """
            self._date_key, self._run_name = _make_run_name(label)
            self._param_key      = f"{self._date_key}/{self._run_name}/params"
            self._spectra_prefix = f"{self._date_key}/{self._run_name}/spectra"
            self._min_itemsize    = {}
            self._extra_scalars   = _sanitize_result(extra_scalars or {})
            self._per_object_keys = None   # reset; first append will declare keys

            run_key = f"/{self._date_key}/{self._run_name}"
            if self._extra_scalars:
                print(f"Run started → {run_key}  |  extra scalars: {list(self._extra_scalars.keys())}")
            else:
                print(f"Run started → {run_key}")
            return run_key
   
    def _require_run(self) -> None:
        """Raise if start_run() has not been called yet."""
        if self._param_key is None:
            raise RuntimeError(
                "No active run. Call start_run() before append()."
            )

    # ------------------------------------------------------------------
    # Writing
    # ------------------------------------------------------------------
    def append(
        self,
        qso,
        objid: str,
        store_spectra: bool = False,
        extra_scalars: dict | None = None,
    ) -> None:
        """
        Append one fit result to the current run.
 
        Parameters
        ----------
        qso : QSOFit
            A QSOFit object after calling qso.Fit(save_result=False, ...).
        objid : str, optional
            Unique identifier for this object, e.g. ``"J1234+5678"``.
            Required if store_spectra=True. Falls back to qso.sdss_name
            if available and objid is not given.
        store_spectra : bool, optional
            If True, store all spectral arrays under spectra/<objid>/.
            Default False (scalar-only catalog).
        extra_scalars : dict, optional
            Per-object scalar values to store alongside the PyQSOFit
            results, e.g. ``{"logLbol": 46.2, "flag_broad": 1}``.
 
            The first call that provides extra_scalars declares the
            canonical set of keys for this run.  Every subsequent call
            must supply exactly the same keys — missing or extra keys
            raise a ValueError to keep column widths consistent across
            all rows.
 
            These are merged after the run-wide extra_scalars from
            start_run(), so per-object values always win over run-wide
            defaults when keys overlap.
        """
        self._require_run()
 
        # Build scalar dict from the QSOFit result arrays
        result = _qso_to_scalar_dict(qso)
        result = _sanitize_result(result)

           # ---- merge run-wide extra scalars (from start_run) ----------
        # PyQSOFit result columns always win over run-wide constants.
        if self._extra_scalars:
            shadowed = [k for k in self._extra_scalars if k in result]
            if shadowed:
                warnings.warn(
                    f"extra_scalars key(s) {shadowed} shadow existing "
                    f"PyQSOFit result columns and will be ignored.",
                    stacklevel=2,
                )
            for k, v in self._extra_scalars.items():
                if k not in result:
                    result[k] = v

        # ---- merge per-object extra scalars (from this append call) --
        per_obj = _sanitize_result(extra_scalars or {})

        if per_obj:
            if self._per_object_keys is None:
                # First append that supplies per-object scalars:
                # declare the canonical key set for this run.
                self._per_object_keys = set(per_obj.keys())
            else:
                missing = self._per_object_keys - set(per_obj.keys())
                extra   = set(per_obj.keys()) - self._per_object_keys
                if missing or extra:
                    raise ValueError(
                        f"extra_scalars keys do not match the schema declared "
                        f"on the first append of this run.\n"
                        f"  Expected : {sorted(self._per_object_keys)}\n"
                        f"  Got      : {sorted(per_obj.keys())}\n"
                        f"  Missing  : {sorted(missing)}\n"
                        f"  Unexpected: {sorted(extra)}"
                    )
            # Per-object values overwrite run-wide values on key clash.
            result.update(per_obj)
        elif self._per_object_keys is not None:
            # A previous append declared per-object keys but this call
            # didn't supply any — enforce consistency.
            raise ValueError(
                f"This append() call is missing per-object extra_scalars. "
                f"Expected keys: {sorted(self._per_object_keys)}"
            )

        # ---- scalar params -------------------------------------------
        df_new = pd.DataFrame([result])
        self._min_itemsize.update(_infer_min_itemsize(result))

        try:
            df_new.to_hdf(
                self.path,
                key=self._param_key,
                mode="a",
                append=True,
                format="table",
                data_columns=True,
                min_itemsize=self._min_itemsize,
                complevel=5,
                complib="blosc",
            )
        except Exception as exc:
            if "Cannot serialize" in str(exc) or "itemsize" in str(exc).lower():
                self._rebuild_with_new_row(df_new)
            else:
                raise

        # ---- spectral arrays -----------------------------------------
        if store_spectra:
            if objid is None:
                self._store_spectrum(qso, objid)

    def _store_spectrum(self, qso, objid: str) -> None:
        """
        Write all spectral arrays from a QSOFit object into
        spectra/<objid>/ within the active run group.

        Stored arrays
        -------------
        Always (if present on qso):
            wave, flux, err           rest-frame spectrum
            f_conti_model             full continuum model
            f_line_model              full emission line model
            f_pl_model                power-law component
            f_fe_mgii_model           UV FeII component
            f_fe_balmer_model         optical FeII component
            f_bc_model                Balmer continuum component
            f_poly_model              polynomial component
            line_flux                 continuum-subtracted flux

        Only when decompose_host=True was used:
            host                      host galaxy template
            qso  (stored as qso_temp) QSO PCA template
        """
        self._require_run()
        grp_path = f"{self._spectra_prefix}/{objid}"

        # All possible spectral arrays and the attribute name on the qso object
        candidate_arrays = [
            "wave",
            "flux",
            "err",
            "f_conti_model",
            "f_line_model",
            "f_pl_model",
            "f_fe_mgii_model",
            "f_fe_balmer_model",
            "f_bc_model",
            "f_poly_model",
            "line_flux",
            "host",
        ]
        # qso.qso clashes with the class name so we rename it on storage
        qso_template_attr = "qso"
        qso_template_key  = "qso_template"

        with h5py.File(self.path, "a") as f:
            if grp_path in f:
                warnings.warn(
                    f"Spectrum for '{objid}' already exists in this run — overwriting.",
                    stacklevel=3,
                )
                del f[grp_path]

            grp = f.require_group(grp_path)

            # Store each array that exists and is a non-empty ndarray
            for attr in candidate_arrays:
                arr = getattr(qso, attr, None)
                if arr is not None and isinstance(arr, np.ndarray) and arr.size > 0:
                    grp.create_dataset(
                        attr,
                        data=np.asarray(arr, dtype="f8"),
                        compression="gzip",
                        compression_opts=4,
                    )

            # Store qso PCA template under a safe key name
            arr = getattr(qso, qso_template_attr, None)
            if arr is not None and isinstance(arr, np.ndarray) and arr.size > 0:
                grp.create_dataset(
                    qso_template_key,
                    data=np.asarray(arr, dtype="f8"),
                    compression="gzip",
                    compression_opts=4,
                )

            # Store scalar metadata as HDF5 group attributes for
            # quick access without loading the full params table
            for attr in ("ra", "dec", "z"):
                val = getattr(qso, attr, None)
                if val is not None:
                    try:
                        grp.attrs[attr] = float(val)
                    except (TypeError, ValueError):
                        pass

    def _rebuild_with_new_row(self, df_new: pd.DataFrame) -> None:
        """
        Fallback when a string column needs more space than originally
        allocated: load existing table, concatenate, and overwrite.
        """
        warnings.warn(
            "String column width increased — rebuilding param table. "
            "This is a one-time cost.",
            stacklevel=3,
        )
        try:
            df_old = pd.read_hdf(self.path, key=self._param_key)
            df_combined = pd.concat([df_old, df_new], ignore_index=True)
        except KeyError:
            df_combined = df_new

        df_combined.to_hdf(
            self.path,
            key=self._param_key,
            mode="a",
            format="table",
            data_columns=True,
            min_itemsize=self._min_itemsize,
            complevel=5,
            complib="blosc",
        )

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------

    def _resolve_param_key(self, run_key: str | None) -> str:
        """
        Return the HDF5 key for the scalar table of a given run.
        If run_key is None, uses the active run.
        """
        if run_key is not None:
            return run_key.lstrip("/") + "/params"
        self._require_run()
        return self._param_key

    def load(
        self,
        columns: list[str] | None = None,
        run_key: str | None = None,
    ) -> pd.DataFrame:
        """
        Load the full scalar parameter table for a run.

        Parameters
        ----------
        columns : list of str, optional
            Subset of columns to load. Loads all if omitted.
        run_key : str, optional
            Override the active run, e.g. ``"/20250601/20250601_143000_night1"``.
        """
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"Catalog not found: {self.path}")
        key = self._resolve_param_key(run_key)
        return pd.read_hdf(self.path, key=key, columns=columns)

    def query(
        self,
        condition: str,
        columns: list[str] | None = None,
        run_key: str | None = None,
    ) -> pd.DataFrame:
        """
        Load only rows matching a pandas query string — without reading
        the whole table into memory.

        Examples
        --------
        >>> cat.query("redshift > 2.0")
        >>> cat.query("MgII_whole_br_fwhm > 3000 & redshift < 2.5")
        >>> cat.query("SN_ratio_conti > 10", columns=["ra", "dec", "redshift"])
        >>> cat.query("redshift > 2", run_key="/20250601/20250601_120000_test")
        """
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"Catalog not found: {self.path}")
        key = self._resolve_param_key(run_key)
        return pd.read_hdf(self.path, key=key, where=condition, columns=columns)

    def get_spectrum(
        self,
        objid: str,
        run_key: str | None = None,
    ) -> dict:
        """
        Retrieve all stored spectral arrays for one object.

        Returns
        -------
        dict
            Keys are array names (``wave``, ``flux``, ``err``,
            ``f_conti_model``, etc.). Scalar metadata (ra, dec, z)
            are under the ``"attrs"`` key.
        """
        if run_key is not None:
            prefix = run_key.lstrip("/") + "/spectra"
        else:
            self._require_run()
            prefix = self._spectra_prefix

        grp_path = f"{prefix}/{objid}"
        with h5py.File(self.path, "r") as f:
            if grp_path not in f:
                raise KeyError(
                    f"No spectrum stored for '{objid}' under '{prefix}'"
                )
            grp = f[grp_path]
            result = {k: grp[k][:] for k in grp.keys()}
            result["attrs"] = dict(grp.attrs)
        return result

    def list_spectra(self, run_key: str | None = None) -> list[str]:
        """Return object IDs that have stored spectra in a given run."""
        if not os.path.exists(self.path):
            return []
        if run_key is not None:
            prefix = run_key.lstrip("/") + "/spectra"
        else:
            if self._spectra_prefix is None:
                return []
            prefix = self._spectra_prefix
        with h5py.File(self.path, "r") as f:
            if prefix not in f:
                return []
            return list(f[prefix].keys())

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------

    def list_runs(self) -> list[str]:
        """
        Return all run keys in the file, sorted chronologically.

        Each entry is a full path like ``"/20250601/20250601_143000_night1"``.
        """
        if not os.path.exists(self.path):
            return []
        runs = []
        with h5py.File(self.path, "r") as f:
            for date_grp in sorted(f.keys()):
                for run_grp in sorted(f[date_grp].keys()):
                    runs.append(f"/{date_grp}/{run_grp}")
        return runs

    def list_dates(self) -> list[str]:
        """Return all date groups (``YYYYMMDD``) present in the file."""
        if not os.path.exists(self.path):
            return []
        with h5py.File(self.path, "r") as f:
            return sorted(f.keys())

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def to_fits(
        self,
        output_path: str,
        overwrite: bool = True,
        run_key: str | None = None,
    ) -> None:
        """
        Export the scalar parameter table of a run to a FITS BINTABLE.

        Parameters
        ----------
        output_path : str
            Destination .fits path.
        overwrite : bool
            Overwrite if the file already exists.
        run_key : str, optional
            Run to export. Defaults to the active run.
        """
        df = self.load(run_key=run_key)
        for col in df.select_dtypes(include="object").columns:
            df[col] = df[col].astype(str)
        Table.from_pandas(df).write(output_path, overwrite=overwrite)
        print(f"Exported {len(df)} rows → {output_path}")

    def to_csv(
        self,
        output_path: str,
        run_key: str | None = None,
        **kwargs,
    ) -> None:
        """Export the scalar parameter table of a run to CSV."""
        self.load(run_key=run_key).to_csv(output_path, index=False, **kwargs)
        print(f"Exported → {output_path}")

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Number of objects in the active run's scalar table."""
        try:
            return len(self.load())
        except (FileNotFoundError, RuntimeError):
            return 0

    def __repr__(self) -> str:
        size_mb = (
            os.path.getsize(self.path) / 1e6
            if os.path.exists(self.path) else 0.0
        )
        runs   = self.list_runs()
        active = (
            f"/{self._date_key}/{self._run_name}"
            if self._run_name else "None"
        )
        try:
            n_obj  = len(self.load())
            n_spec = len(self.list_spectra())
        except (FileNotFoundError, RuntimeError):
            n_obj = n_spec = 0

        return (
            f"QSOCatalog('{self.path}')\n"
            f"  Active run       : {active}\n"
            f"  Objects (params) : {n_obj}\n"
            f"  Objects (spectra): {n_spec}\n"
            f"  Total runs       : {len(runs)}\n"
            f"  File size        : {size_mb:.2f} MB"
        )

    def column_names(self, run_key: str | None = None) -> list[str]:
        """Return the list of scalar column names for a run."""
        return list(self.load(run_key=run_key).columns)

    def summary(self, run_key: str | None = None) -> pd.DataFrame:
        """Return basic statistics for all numeric columns in a run."""
        return self.load(run_key=run_key).describe()