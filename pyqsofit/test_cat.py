from PyQSOFit_cat import QSOCat
import pandas as pd
import h5py
import sys
import numpy as np
from pyqsofit.PyQSOFit import QSOFit
from astropy.io import fits
import matplotlib.pyplot as plt
import csv
import warnings
from scipy.ndimage import gaussian_filter1d
QSOFit.set_mpl_style()

def print_h5_tree(g, indent=0):
    for key in g.keys():
        obj = g[key]

        if isinstance(obj, h5py.Group):
            print("  " * indent + f"📁 {key}")
            print_h5_tree(obj, indent + 1)
        else:
            print("  " * indent + f"📄 {key} {obj.shape}")
    return

if __name__ == "__main__":

    selector = 3
    # SED
    match selector:
        case 1:
            data_path = "/home/mprdi/Projects/GitHub/hawaii_agn/data/initial_spectra/all_agn_spectra_flux_conv_rebinned_rescale.fits"
            csv_path = "/home/mprdi/Projects/GitHub/hawaii_agn/obj_info/all_agn_more_info.csv"

        case 2:
            data_path = "/home/mprdi/Projects/GitHub/hawaii_agn/data/initial_spectra/sed_agn_spectra_flux_conv_rebinned_rescale.fits"
            csv_path = "/home/mprdi/Projects/GitHub/hawaii_agn/obj_info/sed_agn_more_info.csv"

        case 3:
            data_path = "/home/mprdi/Projects/GitHub/hawaii_agn/data/initial_spectra/all_agn_spectra_flux_conv_rebinned_rescale.fits"
            csv_path = "/home/mprdi/Projects/GitHub/hawaii_agn/obj_info/JWST_PFS_subsample.csv"

    data = fits.open(data_path)

    # Each one of these is a 2D np array
    lam = data["WAVELENGTH"].data
    flux = data["FLUX"].data
    err = data["ERROR"].data
    z = data["INFO"].data["z"]
    obCode = data["INFO"].data["id"]
    
    # Link data_path id of object with correct row in csv_path
    df = pd.read_csv(csv_path)
    df = df.set_index("obCode") 

    # Fitting object

    # CSV index's of 4 chosen spectra (high z & br, high z & na, low z & br, low z & na)
    final = [123, 180, 391, 349]
    interesting = [193, 365, 492, 176, 500]
    final = [391, 180]

    file_name = "test_cat4.h5"
    cat = QSOCat(file_name)
    cat.start_run("label1", True)

    # Loop through values
    for i in final:
        # Find the current variable at a given index
        # Pulled from data_path
        flux_i = flux[i, :] * 1e17
        err_i = err[i, :] * 1e17
        lam_i = lam[i, :]
        id_i = int(obCode[i])

        # Get redshift
        z_i = float(z[i]) # Redshift from fits file (matches zsk in csv file)
        if(z_i < 0):
            z_i = df.loc[id_i, "zpfs"]
            z_err = df.loc[id_i, "zpfs_err"]
            warnings.warn(f"zsk not found for obcode {id_i}, using zpfs. Error: {z_err}", UserWarning)

        # Pulled from csv_path
        # Only compatible with 
        if(selector == 3):
            ra = df.loc[id_i, "matched_web_ra"]
            dec = df.loc[id_i, "matched_web_dec"]
            obj_id = df.loc[id_i, "matched_web_id_int"]
            lbol = df.loc[id_i, "logLbol"]
            survey = "COSMOSWb"
        else:
            ra = df.loc[id_i, "ra"]
            dec = df.loc[id_i, "dec"]
            obj_id = id_i
            lbol = df.loc[id_i, "logLbol"]
            survey = "COSMOS2020"

        # Save additional info in results
        extra_params = {"logLbol" : lbol}

        # Preform the fitting
        q = QSOFit(lam_i, flux_i, err_i, z_i, ra, dec, plateid = obj_id, path='/home/mprdi/Projects/GitHub/hawaii_agn/data/initial_spectra')
        q.Fit(name=f"{survey}{obj_id}",  # customize the name of given targets. Default: plate-mjd-fiber
            # prepocessing parameters
            nsmooth=1,  # do n-pixel smoothing to the raw input flux and err spectra
            and_mask=False,  # delete the and masked pixels
            or_mask=False,  # delete the or masked pixels
            reject_badpix=False,  # reject 10 most possible outliers by the test of pointDistGESD
            deredden=True,  # correct the Galactic extinction
            wave_range = None,  # trim input wavelength
            wave_mask=None,  # 2-D array, mask the given range(s)

            # host decomposition parameters
            decompose_host=True,  # If True, the host galaxy-QSO decomposition will be applied
            host_prior=False, # If True, the code will adopt prior-informed method to assist decomposition. Currently, only 'CZBIN1' and 'DZBIN1' model for QSO PCA are available. And the model for galaxy must be PCA too.
            host_prior_scale = 0.2,
            #host_prior_scale=0.2, # scale of prior panelty. Usually, 0.2 works fine for SDSS spectra. Adjust it smaller if you find the prior affect the fitting results too much.

            host_line_mask=True, # If True, the line region of galaxy will be masked when subtracted from original spectra.
            decomp_na_mask=True, # If True, the narrow line region will be masked when perform decomposition
            qso_type='CZBIN1', # PCA template name for quasar
            npca_qso=10, # numebr of quasar templates
            host_type='PCA', # template name for galaxy
            npca_gal=5, # number of galaxy templates
            
            # continuum model fit parameters
            Fe_uv_op=True,  # If True, fit continuum with UV and optical FeII template
            poly=False,  # If True, fit continuum with the polynomial component to account for the dust reddening
            BC=False,  # If True, fit continuum with Balmer continua from 1000 to 3646A
            initial_guess=None,  # Initial parameters for continuum model, read the annotation of this function for detail
            rej_abs_conti=False,  # If True, it will iterately reject 3 sigma outlier absorption pixels in the continuum
            n_pix_min_conti=100,  # Minimum number of negative pixels for host continuuum fit to be rejected.

            # emission line fit parameters
            linefit=True,  # If True, the emission line will be fitted
            rej_abs_line=False,
            # If True, it will iterately reject 3 sigma outlier absorption pixels in the emission lines
            
            # fitting method selection
            MC=False,
            # If True, do Monte Carlo resampling of the spectrum based on the input error array to produce the MC error array
            MCMC=False,
            # If True, do Markov Chain Monte Carlo sampling of the posterior probability densities to produce the error array
            nsamp=200,
            # The number of trials of the MC process (if MC=True) or number samples to run MCMC chain (if MCMC=True)

            # advanced fitting parameters
            param_file_name='testqsopar.fits',  # Name of the qso fitting parameter FITS file.
            nburn=20,  # The number of burn-in samples to run MCMC chain
            nthin=10,  # To set the MCMC chain returns every n samples
            epsilon_jitter=0.,
            # Initial jitter for every initial guass to avoid local minimum. (Under test, not recommanded to change)

            # customize the results
            save_result=True,  # If True, all the fitting results will be saved to a fits file
            save_fits_name= "fit_" + str(id_i) + "_" + str(i) + "_1",  # The output name of the result fits
            save_fits_path='/home/mprdi/Projects/GitHub/hawaii_agn/data/results_spectra',  # The output path of the result fits
            plot_fig=True,  # If True, the fitting results will be plotted
            save_fig=False,  # If True, the figure will be saved
            plot_corner=False,  # Whether or not to plot the corner plot results if MCMC=True

            # debugging mode
            verbose=False,  # turn on (True) or off (False) debugging output

            # sublevel parameters for figure plot and emcee
            kwargs_plot={
                'save_fig_path': '/home/mprdi/Projects/GitHub/hawaii_agn/data/results_spectra',  # The output path of the figure
                'broad_fwhm'   : 1200  # km/s, lower limit that code decide if a line component belongs to broad component
            },
            kwargs_conti_emcee={},
            kwargs_line_emcee={})
            
        # Save the fit
        obj_name = survey + str(obj_id).zfill(5)
        cat.append(obj_name, q, 1, False, False, False, True, extra_params)

  

    # with h5py.File(file_name, "r") as f:
    #     print_h5_tree(f)

    #df = pd.read_hdf(file_name, 