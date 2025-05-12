'''
Functions used to convert the data cube of the fire dataset (which is not in your typical WCS format) to
WCS format to allow for accurate extractions of the moment maps.
'''

from astropy.io import fits

def dataCube(file_name):
    #----------------------------------------------------
    # 1) Open the original file
    #----------------------------------------------------
    hdul = fits.open(file_name)
    data = hdul[0].data   # shape likely (300, 300, 200) in Python
    old_header = hdul[0].header
    hdul.close()

    print("Original data shape (Python) =", data.shape)
    # Should be (300, 300, 200) if NAXIS1=200, NAXIS2=300, NAXIS3=300 in header.

    #----------------------------------------------------
    # 2) Create a new header for a "typical" spectral cube
    #----------------------------------------------------
    new_header = fits.Header()

    # Primary FITS keywords
    new_header["SIMPLE"]  = True
    new_header["BITPIX"]  = -64
    new_header["NAXIS"]   = 3
    new_header["NAXIS1"]  = 300   # X dimension
    new_header["NAXIS2"]  = 300   # Y dimension
    new_header["NAXIS3"]  = 200   # Velocity dimension

    new_header["EXTEND"]  = True
    new_header["BUNIT"]   = "arbitrary"  # e.g. flux units or "arbitrary" for sim

    # Copy over any info you'd like from old_header, if desired:
    # For example:
    new_header["DATE"]     = old_header.get("DATE", "Unknown")
    new_header["UNIVERSE"] = old_header.get("UNIVERSE", "")
    new_header["REDSHIFT"] = old_header.get("REDSHIFT", 0.0)

    # (Optionally keep your custom keys like FOVSIZE, PIXSIZE, etc.)
    new_header["FOVSIZE"] = old_header.get("FOVSIZE", 30.0)
    new_header["PIXSIZE"] = old_header.get("PIXSIZE", 0.1)
    new_header["VLIM"]    = old_header.get("VLIM", 500.0)
    new_header["DELTAV"]  = old_header.get("DELTAV", 5.0)
    new_header["INCL"]    = old_header.get("INCL", 70.0)

    #----------------------------------------------------
    # 3) Define WCS for each axis
    #     Axis 1,2 = "linear" spatial in kpc
    #     Axis 3   = velocity in km/s
    #----------------------------------------------------

    #Values for reference pixel, and delta values are all guesses due to them being omitted from
    #header file

    # -- Axis 1 (X) --
    new_header["CTYPE1"] = "RA---TAN"
    new_header["CUNIT1"] = "deg"
    new_header["CRVAL1"] = 0.0       # 'RA' at the reference pixel
    new_header["CRPIX1"] = 150.0     # center pixel if NAXIS1=300
    new_header["CDELT1"] = -0.00027778  # e.g. ~1 arcsec per pixel

    # -- Axis 2 (Y) --
    new_header["CTYPE2"] = "DEC--TAN"
    new_header["CUNIT2"] = "deg"
    new_header["CRVAL2"] = 0.0
    new_header["CRPIX2"] = 150.0
    new_header["CDELT2"] = 0.00027778   # sign can be + or -

    # -- Axis 3 (Velocity) --
    # We'll place v=0 at channel 101 so that channel 1 ~ -500 km/s
    # given DELTAV=5 and 200 channels total
    new_header["CTYPE3"] = "VELO-LSR"   # recognized as velocity
    new_header["CUNIT3"] = "km/s"
    new_header["CRVAL3"] = 0.0         # velocity at ref pixel
    new_header["CRPIX3"] = 101.0       # ref pixel is middle
    new_header["CDELT3"] = 5.0         # channel width in km/s

    # You may also set SPECSYS or RESTFRQ if relevant:
    new_header["SPECSYS"] = "LSRK"     # or "TOPOCENT", etc.

    #----------------------------------------------------
    # 4) If data is already shape (300,300,200), perfect!
    #    Otherwise, reorder/transpose the array so that
    #    data.shape is indeed (NAXIS2=300, NAXIS1=300, NAXIS3=200).
    #    But typically from astropy you'll already have (300,300,200).
    #----------------------------------------------------
    print("Final data shape (Python) =", data.shape)

    #----------------------------------------------------
    # 5) Write out a new FITS file with this updated header
    #----------------------------------------------------
    hdu = fits.PrimaryHDU(data=data, header=new_header)

    return hdu