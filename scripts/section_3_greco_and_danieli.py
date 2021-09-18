# Standard library
import os

# Third-party
import matplotlib.pyplot as plt
from astropy.table import Table

# Project
from artpop import show_image

# Path to data
data_path = os.path.join(os.pardir, 'data')


###############################################################################
# Section 3.2 assumes that we have created the necessary isochrone parameters
# Here, we will generate them using a MIST isochrone in the data directory.
###############################################################################

iso_fn = os.path.join(data_path, 'feh_m1.00_vvcrit0.4_LSST_10gyr_test_iso')
iso_table = Table.read(iso_fn, format='ascii')
mini = iso_table['initial_mass']
mact = iso_table['star_mass']
mags = iso_table[['LSST_u', 'LSST_g', 'LSST_r', 'LSST_i', 'LSST_z', 'LSST_y']]


###############################################################################
# Code from Section 3.2
###############################################################################

# part 1: Isochrone
from artpop.stars import Isochrone

iso = Isochrone(mini, mact, mags)

g_i =  iso.ssp_color(
    blue="LSST_g", # blue filter
    red="LSST_i",  # red filter
    imf="salpeter" # initial mass function
)

m_survive = iso.ssp_surviving_mass("salpeter")

print('artpop.Isochrone example:')
print(f'g-i = {g_i:.2f}')
print(f'surviving mass [normalized to 1 M_sun]= {m_survive:.2f} M_sun\n')

# part 2: SSP
from artpop.stars import SSP

ssp = SSP(
    isochrone=iso,  # Isochrone object
    num_stars=1e5,  # number of stars
    imf="salpeter", # initial mass function
)

i = ssp.total_mag("LSST_i")
g_i = ssp.integrated_color("LSST_g", "LSST_i")

print('artpop.SSP example:')
print(f'SSP M_i = {i:.2f}')
print(f'SSP g-i = {g_i:.2f}\n')


###############################################################################
# Code from Section 3.3
###############################################################################

from astropy import units as u
from artpop import Source
from artpop.space import plummer_xy

xy_dim = (501, 501) # image dimensions
pixel_scale = 0.2 * u.arcsec / u.pixel

# returns a 2D numpy array
xy = plummer_xy(
    num_stars=ssp.num_stars,
    scale_radius=500*u.pc,
    distance=8*u.Mpc,
    xy_dim=xy_dim,
    pixel_scale=pixel_scale
)

# ssp magnitudes stored in astropy table
src = Source(
    xy=xy,              # image coordinates
    mags=ssp.mag_table, # magnitude table
    xy_dim=xy_dim       # image dimensions
)


from artpop.image import IdealImager
imager = IdealImager()


from artpop.image import moffat_psf

# returns a 2D numpy array
psf = moffat_psf(
    fwhm=0.6*u.arcsec,
    pixel_scale=pixel_scale
)

obs = imager.observe(src, "LSST_i", psf)

# ArtPop's show_image function is useful for visualization
print('\nDisplaying artificial image...')
show_image(obs.image)
plt.show()
