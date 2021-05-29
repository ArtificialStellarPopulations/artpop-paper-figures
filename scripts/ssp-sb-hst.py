# Third-party
import numpy as np
import matplotlib.pyplot as plt
from astropy.visualization import make_lupton_rgb
from astropy import units as u
from astropy.io import fits
# from astropy.visualization import *

# Project
import artpop
from artpop import constant_sb_stars_per_pix

# Load matplotlib style
plt.style.use('jpg.mplstyle')


###############################################################################
# Instrumental and observational parameters
###############################################################################
xy_dim = 2001
phot_system ='HST_ACSWF'
bands = ['F814W','F555W','F435W']
pixel_scale = 0.05
exptime = 50 * u.min

# read in Tiny Tim-modeled PSFs
psf = {bands[i]: fits.getdata('../data/'+str(bands[i])+'_psf_20as.fits') 
for i in range(3)}

# Initialize art imager
imager = artpop.ArtImager(phot_system)
###############################################################################


###############################################################################
# Stellar population parameters
###############################################################################
log_age = 10
feh = -1
sbs = [26.0, 24.0, 20.0]
sb_band = 'ACS_WFC_F814W'

distance = 1 * u.Mpc


###############################################################################
# Make constant distance SSP, varying surface brightness figure
###############################################################################

fig, axes = plt.subplots(1, len(sbs), figsize=(15, 5),
                         subplot_kw=dict(xticks=[], yticks=[]))
fig.subplots_adjust(wspace=0.03, hspace=0.03)

q = 0.5
stretch = 0.01
m = 0.

# create MIST uniform spatial distribution SSP sources at various sb
for i, sb in enumerate(sbs):
    src = artpop.MISTUniformSSP(
        log_age, feh, phot_system,
        distance, xy_dim, pixel_scale,
        sb, sb_band
    )

    images = []

    # mock observe in F435W, F555W, and F814W
    for num, band in enumerate(bands):
        obs = imager.observe(src, f'ACS_WFC_{band}', 
            exptime, psf=psf[band]
        )
        images.append(obs.image)

    # create RGB image
    images = [0.4*images[0], 0.7*images[1], 0.6*images[2]]
    rgb = make_lupton_rgb(*images, Q=q, stretch=stretch, minimum=m)

    # plot image
    _, _ax = artpop.show_image(rgb, subplots=(fig, axes[i]))

    # add labels
    axes[i].text(0.04, 0.96, '$\mu_I \sim'+str(int(sb))+'\,\mathrm{mag\,arcsec}^{-2}$', 
        c='black', fontsize=25, 
        bbox=dict(facecolor='white', edgecolor='white',linewidth=2, 
            boxstyle='round,pad=0.25', alpha=0.8),
         horizontalalignment='left', verticalalignment='top', 
         transform=axes[i].transAxes)

fig.savefig('../figures/artpop_sb.pdf', dpi=250)
fig.savefig('../figures/artpop_sb.png', dpi=250)


