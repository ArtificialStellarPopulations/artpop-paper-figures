# Third-party
import numpy as np
import matplotlib.pyplot as plt
from astropy.visualization import make_lupton_rgb
from astropy import units as u
from astropy.io import fits

# Project
import artpop
from artpop.util import embed_slices

# Load matplotlib style
plt.style.use('jpg.mplstyle')


###############################################################################
# Instrumental and observational parameters
###############################################################################
xy_dim = 1001
phot_system ='DECam'
pixel_scale = 0.263 
seeing = [0.97, 1.07, 1.25] #irg

psf = {'irg'[i]: artpop.moffat_psf(seeing[i], pixel_scale, shape=61) 
for i in range(3)}
zpt = 30

# Initialize ideal imager
imager = artpop.IdealImager()
###############################################################################


###############################################################################
# Dwarf galaxy parameters
###############################################################################
log_age = 10.1
feh = -2.0

total_mass = 5e5 * u.Msun
r_eff = 150 * u.pc
distance = 1000 * u.kpc 

n = 0.6
theta = 30
ellip = 0.25
###############################################################################


###############################################################################
# Create MIST Sersic SSP source
###############################################################################
src = artpop.MISTSersicSSP(
	log_age, feh, phot_system,
	r_eff, n, theta, ellip, distance, xy_dim,
	pixel_scale, total_mass=total_mass, label_type='phases'
    )
###############################################################################


###############################################################################
# Left panel: mock observe in gri  
###############################################################################
fig, axes = plt.subplots(1, 3, figsize=(15,5))

# fig.subplots_adjust(wspace=0.1, hspace=0.06)

images = []
for num, band in enumerate('irg'):
    obs = imager.observe(
        src, f'DECam_{band}', psf=psf[band], zpt=zpt
            )
    images.append(obs.image)

# create RGB image
stretch = 15
mock_rgb = make_lupton_rgb(*images, stretch=stretch)

axes[0].imshow(mock_rgb)
axes[0].text(0.5, -0.05, 'model, ' +
    '$\mathrm{M}_\star = 5\cdot10^5\,\mathrm{M}_{\odot}$,' +
    ' 1 Mpc', c='black', fontsize=17, 
         bbox=dict(facecolor='white', edgecolor='white',
            linewidth=2, boxstyle='round,pad=0.4', alpha=0.8),
         horizontalalignment='center',
         verticalalignment='top', transform=axes[0].transAxes)
axes[0].invert_xaxis()
axes[0].set_axis_off()
axes[0].set_box_aspect(1)
###############################################################################


###############################################################################
# Middle panel: inject model to DES image
###############################################################################
datapath = '/Users/shanydanieli/projects/artpop/data/des/'
image_g, hdr_g = fits.getdata(datapath+'DES0128-4249_r2624p01_g.fits.fz', 
    header=True)
image_r, hdr_r = fits.getdata(datapath+'DES0128-4249_r2624p01_r.fits.fz', 
    header=True)
image_i, hdr_i = fits.getdata(datapath+'DES0128-4249_r2624p01_i.fits.fz', 
    header=True)

img_slice, arr_slice = embed_slices((4000, 5500), 
    images[0].shape, image_g.shape)
image_i[img_slice] += images[0][arr_slice]
image_r[img_slice] += images[1][arr_slice]
image_g[img_slice] += images[2][arr_slice]

mock_rgb_inject = make_lupton_rgb(image_i, image_r, image_g, stretch=15, Q=8)
axes[1].imshow(mock_rgb_inject)
axes[1].text(0.5, -0.05, 'model injected to DES data', c='black', fontsize=17, 
         bbox=dict(facecolor='white', edgecolor='white', 
            linewidth=2, boxstyle='round,pad=0.4', alpha=0.8),
         horizontalalignment='center',
         verticalalignment='top', transform=axes[1].transAxes)
axes[1].set_xlim(5000,6000)
axes[1].set_ylim(3500,4500)
axes[1].set_axis_off()
axes[1].set_box_aspect(1)
###############################################################################


###############################################################################
# Right panel: make phase Color-Magnitude Diagram
###############################################################################
# get g and r-band mag tables
mag_g = src.sp.mag_table['DECam_g']
mag_r = src.sp.mag_table['DECam_r']

lim_mag_g = 24.33

# phases = ['MS', 'RGB', 'CHeB', 'EAGB', 'TPAGB', 'postAGB']
phases = ['RGB', 'CHeB', 'EAGB', 'TPAGB']
colors = ['r', 'purple', 'green', 'blue']

for num, phase in enumerate(phases):
    mask = src.sp.select_phase(phase=phase)
    mag_g_mask = mag_g[mask]
    mag_r_mask = mag_r[mask]

    axes[2].scatter(mag_g_mask-mag_r_mask, mag_g_mask, s=3, 
        color=colors[num], label=phase)

axes[2].axhline(lim_mag_g,0,0.55,c='k',ls='--')
axes[2].axhline(lim_mag_g,0.65,1,c='k',ls='--')
axes[2].text(0.76, 0.5, 'DES DR1 \n $g$-band \n magnitude limit', 
    horizontalalignment='center', verticalalignment='center', 
    transform=axes[2].transAxes,size=12)

axes[2].set_xlabel(r'$g-r$')
axes[2].set_ylabel(r'$g$')
axes[2].set_xlim(-0.5,1.5)
axes[2].set_ylim(22.5,27)
axes[2].set_box_aspect(1)
axes[2].invert_yaxis()
lgnd = axes[2].legend(loc='best', bbox_to_anchor=(0.035, 0.62), 
    prop={'size': 12})
lgnd.legendHandles[0]._sizes = [30]
lgnd.legendHandles[1]._sizes = [30]
lgnd.legendHandles[2]._sizes = [30]
lgnd.legendHandles[3]._sizes = [30]

fig.savefig('../figures/dwarf_des.png', dpi=400, bbox_inches='tight')
