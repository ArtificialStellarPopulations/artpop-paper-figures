# Standard library
import os

# Third-party
import numpy as np
import matplotlib.pyplot as plt
from astropy.visualization import make_lupton_rgb
from astropy import units as u
from astropy.io import fits

# Project
import artpop

# load matplotlib style
plt.style.use(artpop.jpg_style)

# path to figures
fig_path = os.path.join(os.pardir, 'figures')


###############################################################################
# Instrumental and observational parameters
###############################################################################
xy_dim = [701, 501]
phot_system ='HST_ACSWF'
bands = [f'ACS_WFC_{b}' for b in ['F814W','F606W','F475W']]
pixel_scale = 0.05
exptime = 90 * u.min

# read in Tiny Tim-modeled PSFs
z = 120
psf = {b: fits.getdata(f'../data/{b}.fits')[z:-z, z:-z] for b in bands}

# Initialize art imager
imager = artpop.ArtImager(phot_system, diameter=2.4, read_noise=3)


###############################################################################
# Stellar population parameters
###############################################################################
log_age = 10.0
feh = -1.6
sbs = [26.0, 23.0, 20.0]
sb_band = 'ACS_WFC_F814W'
d_vals = np.array([8, 2, 0.5]) # Mpc


###############################################################################
# Make distance - SB grid figure
###############################################################################

fig, axes = plt.subplots(
    len(d_vals), len(sbs),
    figsize=(15, int(15.5 * xy_dim[1]/xy_dim[0])),
    subplot_kw=dict(xticks=[], yticks=[])
)

fig.subplots_adjust(wspace=0.03, hspace=0.03)

# label surface brightnesses
y = 1.015
title_fs = 25
unit_lab = '\,\mathrm{mag\,arcsec}^{-2}$'
kw = dict(fontsize=title_fs, y=y)
axes[0, 0].set_title('$\mu_I = ' + str(int(sbs[0])) + unit_lab, **kw)
axes[0, 1].set_title('$' + str(int(sbs[1])) + unit_lab, **kw)
axes[0, 2].set_title('$' + str(int(sbs[2])) + unit_lab, **kw)

# label distances
l = r'\ Mpc}$'
axes[2, 0].set_ylabel(r'$\mathrm{D = ' + str(d_vals[2]) + l, fontsize=title_fs)
axes[1, 0].set_ylabel(r'$\mathrm{' + str(d_vals[1]) + l, fontsize=title_fs)
axes[0, 0].set_ylabel(r'$\mathrm{' + str(d_vals[0]) + l, fontsize=title_fs)

Q = 8
stretch = 0.5
count = 0
axes = axes.flatten()

# some scaling to make the image pretty
scale = dict(ACS_WFC_F814W=1, ACS_WFC_F606W=1.2, ACS_WFC_F475W=1.42)

for d in d_vals:

    # create MIST uniform spatial distribution SSP sources at various sb
    for sb in sbs:
        mag_lim_kw = dict(mag_limit=None, mag_limit_band=None)
        dist_mod = 5 * np.log10(d * 1e6) - 5

        # setting mag limit for discrete sources when sb = 20
        # if we don't do this, we will have memory issues
        if sb == 20:
            mag_lim_kw = dict(mag_limit=dist_mod + 4.5, mag_limit_band=sb_band)

        src = artpop.MISTUniformSSP(
            log_age, feh, phot_system, d, xy_dim, pixel_scale,
            sb, sb_band, **mag_lim_kw
        )

        print(f'D = {d}, mu = {sb}, N_star = {len(src.mags):.3e}')

        images = []

        # mock observe in F474W, F606W, and F814W
        for num, band in enumerate(bands):
            zpt = 28
            sky_sb = [25, 26, 27][num] # add faint sky noise
            obs = imager.observe(src, band, exptime, psf=psf[band],
                                 sky_sb=sky_sb, zpt=zpt)
            images.append(obs.image * scale[band])

        # create RGB image
        rgb = make_lupton_rgb(*images, Q=Q, stretch=stretch)

        # plot image
        _, _ax = artpop.show_image(rgb, subplots=(fig, axes[count]),
                                   rasterized=True)

        count += 1

fig.savefig(os.path.join(fig_path, 'artpop_sb.pdf'), dpi=170)
