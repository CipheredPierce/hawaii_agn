#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import libraries
get_ipython().run_line_magic('matplotlib', 'inline')
import sys
import numpy as np
from pyqsofit.PyQSOFit import QSOFit
from astropy.io import fits
import matplotlib.pyplot as plt
import csv
import pandas as pd
from scipy.ndimage import gaussian_filter1d
QSOFit.set_mpl_style()

print(sys.executable)


# In[2]:


# Set variables
startInt, endInt = 13, 14

selector = 2
# SED
match selector:
    case 1:
        data_path = "/home/mprdi/Projects/GitHub/hawaii_agn/data/initial_spectra/all_agn_spectra_flux_conv_rebinned_rescale.fits"
        csvPath = "/home/mprdi/Projects/GitHub/hawaii_agn/obj_info/all_agn_more_info.csv"

    case 2:
        data_path = "/home/mprdi/Projects/GitHub/hawaii_agn/data/initial_spectra/sed_agn_spectra_flux_conv_rebinned_rescale.fits"
        csvPath = "/home/mprdi/Projects/GitHub/hawaii_agn/obj_info/sed_agn_more_info.csv"

    case 3:
        data_path = "/home/mprdi/Projects/GitHub/hawaii_agn/data/initial_spectra/all_agn_spectra_flux_conv_rebinned_rescale.fits"
        csv_path = ""



data = fits.open(data_path)
data.info()
print(type(data[1]))

# Each one of these is a 2D np array
lam = data["WAVELENGTH"].data
flux = data["FLUX"].data
err = data["ERROR"].data
z = data["INFO"].data["z"]
obCode = data["INFO"].data["id"]
#print(obCode)



# In[11]:


# Find data in INFO column
print(data["INFO"].columns)

df = pd.read_csv(csvPath)
df = df.set_index("obCode")
df.info()
print(df)


# In[21]:


# Preform the fitting routine

startInt = 55
endInt = 15

# Define empty lists
qList = []
FWHM_list = []
nameList = []
fwhm_indices = []
fwhm_values = []
sigma_values = []

# Restricting wavelengths
lam_min = 2000
lam_max = 5500

# Loop through values
# for i in range(startInt, endInt):
# #for i in range(flux.shape[0]):

#         # Find the current variable at a given index
#         flux_i = flux[i, :] * 1e17
#         err_i = err[i, :] * 1e17
#         lam_i = lam[i, :]
#         z_i = float(z[i])
        
#         id_i = int(obCode[i])
#         print(id_i)

#         ra = df.loc[id_i, "ra"]
#         dec = df.loc[id_i, "dec"]
#         plateid = df.loc[id_i, "objId"]

#         # Convert rest-frame to observed-frame so PyQSOFit can correct it back
#         #lam_obs = lam_i * (1 + z_i)
#         lam_obs = lam_i

        # save_fits_name = "result_test", 
        #         save_result= False,

        # # Preform the fitting
        # q = QSOFit(lam_obs, flux_i, err_i, z_i, path='/home/mprdi/Projects/GitHub/hawaii_agn/data/initial_spectra', ra=ra, dec=dec, plateid=plateid)
        # q.Fit(      
        #         deredden = True,
        #         decompose_host = True,
        #         host_line_mask = False,
        #         decomp_na_mask = False,
        #         wave_range = [lam_min, lam_max],
        #         Fe_uv_op = True,
        #         linefit = True,
        #         plot_fig = True,
        #         param_file_name = "/home/mprdi/Projects/GitHub/hawaii_agn/PyQSOFit/example/qsopar.fits", 
        #         save_fits_path="/home/mprdi/Projects/GitHub/hawaii_agn/data/results_spectra",
        #         verbose = False,
                
        #         # sublevel parameters for figure plot and emcee
        #          kwargs_plot={
        #                 'broad_fwhm'   : 1200  # km/s, lower limit that code decide if a line component belongs to broad component
        #         },
        #         kwargs_conti_emcee={},
        #         kwargs_line_emcee={}) 
        # qList.append(q)

for i in range(14, 15):
    # Find the current variable at a given index
    flux_i = flux[i, :] * 1e17
    err_i = err[i, :] * 1e17
    lam_i = lam[i, :]
    z_i = float(z[i])

    # Separting the spectra based on redshift
    if(z_i >= 0.9):
        id_i = int(obCode[i])

        ra = df.loc[id_i, "ra"]
        dec = df.loc[id_i, "dec"]
        plateid = df.loc[id_i, "objId"]

        # Convert rest-frame to observed-frame so PyQSOFit can correct it back
        #lam_obs = lam_i * (1 + z_i)
        lam_obs = lam_i

        # Preform the fitting
        q = QSOFit(lam_obs, flux_i, err_i, z_i, path='/home/mprdi/Projects/GitHub/hawaii_agn/data/initial_spectra', ra=ra, dec=dec, plateid=plateid)
        q.Fit( 
              plot_fig = True, 
              save_fig=False, 
              param_file_name = "/home/mprdi/Projects/GitHub/hawaii_agn/PyQSOFit/example/qsopar.fits", 
        ) 
        qList.append(q)

        # Record the values from the fitting routine
        # lineResult = q.line_result
        # lineResultName = q.line_result_name
        # emission_num = len(lineResultName)/4


        # # Find the list of names and indexs that have "fwhm"
        # for idx, name in enumerate(lineResultName):
        #     if "fwhm" in name.lower():
        #         first = name.split("_")[0]
        #         nameList.append(first)
        #         fwhm_indices.append(idx)

        # nameList = list(dict.fromkeys(nameList))

        # print("Names:", nameList)
        # print("Indices:", fwhm_indices)

        # # Find the respective fwhm and sigma values for the broad emission lines
        # for num in fwhm_indices:
        #     fwhm_values.append(lineResult[num])
        #     sigma_values.append(lineResult[num+1])

        # print(fwhm_values)
        # print(sigma_values)



        # Saving the results
        
        # Record the values from the fitting routine
        # lineResult = q.line_result
        # lineResultName = q.line_result_name
        # emission_num = len(lineResultName)/4


        # # Find the list of names and indexs that have "fwhm"
        # for idx, name in enumerate(lineResultName):
        #     if "fwhm" in name.lower():
        #         first = name.split("_")[0]
        #         nameList.append(first)
        #         fwhm_indices.append(idx)

        # nameList = list(dict.fromkeys(nameList))

        # print("Names:", nameList)
        # print("Indices:", fwhm_indices)

        # # Find the respective fwhm and sigma values for the broad emission lines
        # for num in fwhm_indices:
        #     fwhm_values.append(lineResult[num])
        #     sigma_values.append(lineResult[num+1])

        # print(fwhm_values)
        # print(sigma_values)


# In[5]:


# Line fitting results
print(type(q.err))
print(q.err)

print('\n')
print(q.line_result)


# In[6]:


# Reading the results

df = pd.DataFrame({
    'Parameter': q.line_result_name,
    'Value': q.line_result
})

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

err_rows = df[df['Parameter'].str.contains('scale', na=False)]
display(err_rows)
display(df)


# In[7]:


for q in qList:
    q.plot_fig()
    display(q.fig)


# In[8]:


# plot wavelength vs flux and mark prominent emission lines 
# NIRCam 30mmas pixel scale, sci, err

startInt = 0
endInt = 0
 

# Loop through values
for i in range(startInt, endInt):
    flux_i = flux[i, :]
    lam_i = lam[i, :]
    z_i = float(z[i])
    id_i = int(obCode[i])

    flux_smooth = gaussian_filter1d(flux_i, sigma=5)


    #Plot flux vs lam
    plt.figure(figsize=(30,10))
    plt.plot(lam_i[75:4500], flux_i[75:4500], alpha=0.3, label='Data', linestyle="--", color="black", linewidth=.5)
    plt.plot(lam_i[80:4500], flux_smooth[80:4500], linewidth=2, label='Smoothed', color="tab:blue")
    plt.xlabel("Rest Wavelength (Å)")
    plt.ylabel("Flux")

    # Plot emissions
    plt.axvline(x=6564.61 * (1 + z_i), color='tab:red', linestyle=':', label=r"H$\alpha$", linewidth=5)
    plt.axvline(x=4862.68 * (1 + z_i), color='tab:orange', linestyle=':', label=r"H$\beta$", linewidth=5)
    plt.axvline(x=2799.117 * (1 + z_i), color='tab:pink', linestyle=':', label="Mg II", linewidth=5)
    plt.axvline(x=1908.734 * (1 + z_i), color='tab:purple', linestyle=':', label="C III", linewidth=5)


    # Apply limits to the x and y axes
    plt.xlim(3750,9000) #(3750, 9000)
    flux_slice = flux_i[75:4500]
    y_low = np.percentile(flux_slice, 5)
    y_high = np.percentile(flux_slice, 95)
    margin = (y_high - y_low) * 0.2  # 20% padding
    plt.ylim(y_low - margin, y_high + margin) 

    plt.title(f"ID: {id_i}, csv: {i}, z = {z_i}")
    plt.legend()
    plt.show()

    print(z_i)
    print(lam_i.min(), lam_i.max())
    print(id_i)

    

# switch for unceranties, input pareneter, mc sampling

