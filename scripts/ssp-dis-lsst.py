# Third-party
import numpy as np
import matplotlib.pyplot as plt
from astropy.visualization import make_lupton_rgb
from astropy import units as u

# Project
import artpop

# Load matplotlib style
plt.style.use('jpg.mplstyle')


###############################################################################
# Instrumental and observational parameters
###############################################################################
xy_dim = 251
phot_system ='LSST'
pixel_scale = 0.2
seeing = [0.7] * 3 

psf = {'irg'[i]: artpop.moffat_psf(seeing[i], pixel_scale) for i in range(3)}

# Initialize ideal imager
imager = artpop.IdealImager()
###############################################################################


###############################################################################
# Stellar population parameters
###############################################################################
log_age = 9
feh = -1
sb = 24.0
sb_band = 'LSST_i' 
###############################################################################


###############################################################################
# Make constant surface brightness SSP, varying distance figure
###############################################################################
distances = [1.0, 4.0, 8.0, 16.0] * u.Mpc

fig, axes = plt.subplots(1, len(distances), figsize=(20, 5),
                         subplot_kw=dict(xticks=[], yticks=[]))

fig.subplots_adjust(wspace=0.03, hspace=0.03)

stretch = 0.5

# create MIST uniform spatial distribution SSP sources at various distnaces
for i, distance in enumerate(distances):
    src = artpop.MISTUniformSSP(
        log_age, feh, phot_system,
        distance, xy_dim, pixel_scale,
        sb, sb_band
    )

    images = []

    # mock observe in gri
    for num, band in enumerate('irg'):
        obs = imager.observe(
            src, f'LSST_{band}', psf=psf[band]
        )
        images.append(obs.image)

    # create RGB image
    rgb = make_lupton_rgb(*images, stretch=stretch)

     # plot image
    _, _ax = artpop.show_image(rgb, subplots=(fig, axes[i]))

    # add labels
    axes[i].text(0.06, 0.94, str(int(distance.value))+' Mpc', c='black', 
        fontsize=30, bbox=dict(facecolor='white', edgecolor='white',
            linewidth=2, boxstyle='round,pad=0.25', alpha=0.8),
         horizontalalignment='left', verticalalignment='top', 
         transform=axes[i].transAxes)

fig.savefig('../figures/artpop_dis.pdf')
