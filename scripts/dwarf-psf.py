# Third-party
import numpy as np
import matplotlib.pyplot as plt
from astropy.visualization import make_lupton_rgb
from astropy import units as u
from astropy.modeling.models import Gaussian1D

# Project
import artpop

# Load matplotlib style
plt.style.use('jpg.mplstyle')

def round_up_to_odd(f):
    return np.ceil(f) // 2 * 2 + 1

###############################################################################
# Instrumental and observational parameters
###############################################################################
phot_system = ['SDSSugriz','HSC','HST_ACSWF']
pixel_scale = {phot_system[i]: [0.4, 0.17, 0.05][i] for i in range(3)}
xy_dim_asec = 80
xy_dim = {phot_system[i]: round_up_to_odd(xy_dim_asec/
	pixel_scale[phot_system[i]]) for i in range(3)}
fwhm = {phot_system[i]: [0.7, 0.4, 0.1][i] for i in range(3)}

# Initialize ideal imager
imager = artpop.IdealImager()
###############################################################################


###############################################################################
# Dwarf galaxy parameters
###############################################################################
log_age = 10.0
feh = -1.0

total_mass = 2e6 * u.Msun
r_eff = 0.35 * u.kpc
distance = 8.0 * u.Mpc

n=0.6
theta=30
ellip=0.3
###############################################################################


###############################################################################
# Make a dwarf galaxy sources
# Mock observe in SDSS, HSC, and HST
###############################################################################
# create MIST Sersic SSP source
# observe in SDSS
src_sdss = artpop.MISTSersicSSP(
	log_age, feh, phot_system,
	r_eff, n, theta, ellip, distance, xy_dim['SDSSugriz'],
	pixel_scale['SDSSugriz'], total_mass=total_mass
    )
psf_sdss = artpop.gaussian_psf(fwhm['SDSSugriz'], pixel_scale['SDSSugriz'])
obs_sdss = imager.observe(src_sdss, 'SDSS_i', psf_sdss)

# observe in HSC
src_hsc = artpop.MISTSersicSSP(
	log_age, feh, phot_system,
	r_eff, n, theta, ellip, distance, xy_dim['HSC'],
	pixel_scale['HSC'], total_mass=total_mass
    )
psf_hsc = artpop.gaussian_psf(fwhm['HSC'], pixel_scale['HSC'])
obs_hsc = imager.observe(src_hsc, 'hsc_i', psf_hsc)

# observe in HST
src_hst = artpop.MISTSersicSSP(
	log_age, feh, phot_system,
	r_eff, n, theta, ellip, distance, xy_dim['HST_ACSWF'],
	pixel_scale['HST_ACSWF'], total_mass=total_mass
    )
psf_hst = artpop.gaussian_psf(fwhm['HST_ACSWF'], pixel_scale['HST_ACSWF'])
obs_hst = imager.observe(src_hst, 'ACS_WFC_F814W', psf_hst)
###############################################################################


###############################################################################
# Make Gaussian functions for plotting
###############################################################################
r = np.arange(7,13,0.01)
s1 = Gaussian1D(mean=10,stddev=fwhm['SDSSugriz'])
s2 = Gaussian1D(mean=10,stddev=fwhm['HSC'])
s3 = Gaussian1D(mean=10,stddev=fwhm['HST_ACSWF'])
###############################################################################


###############################################################################
# Make PSF figure
###############################################################################
fig, axes = plt.subplots(1, 3, figsize=(15, 6),
                         subplot_kw=dict(xticks=[], yticks=[]))                         

fig.subplots_adjust(wspace=0.03, hspace=0.03)

percentile=[0.1, 99.9]

vmin, vmax = np.nanpercentile(obs_hst.image, percentile)
axes[0].imshow(obs_hst.image, cmap='gray_r', 
	rasterized=True, origin='lower', vmin=vmin, vmax=vmax)

vmin, vmax = np.nanpercentile(obs_hsc.image, percentile)
axes[1].imshow(obs_hsc.image, cmap='gray_r', 
	rasterized=True, origin='lower', vmin=vmin, vmax=vmax)

vmin, vmax = np.nanpercentile(obs_sdss.image, percentile)
axes[2].imshow(obs_sdss.image, cmap='gray_r', 
	rasterized=True, origin='lower', vmin=vmin, vmax=vmax)

axes[1].hlines(25, 300, 417, color='#cc4c02')
axes[1].text(300,35,'20 arcsec', color='#cc4c02', fontsize=19)

# plot Gaussians
axins1 = axes[0].inset_axes([0.05, 0.05, 0.25, 0.25]) 
axins1.plot(r, s3(r)+3, lw=2,c='#2c7fb8')
axins1.set_axis_off()

axins2 = axes[1].inset_axes([0.05, 0.05, 0.25, 0.25]) 
axins2.plot(r, s2(r)+3, lw=2,c='#2c7fb8')
axins2.set_axis_off()

axins3 = axes[2].inset_axes([0.05, 0.05, 0.25, 0.25]) 
axins3.plot(r, s1(r)+3, lw=2,c='#2c7fb8')
axins3.set_axis_off()


for i in range(3):
	axes[i].set_box_aspect(1)

fig.suptitle(r'$\mathrm{M}_\star=2\cdot 10^6\,\mathrm{M}_{\odot} \  ,' +
	'  \ \mathrm{D}=8\,\mathrm{Mpc}$', fontsize=25)

axes[0].set_title('$\mathrm{PSF}_{\mathrm{FWHM}} = 0.1\,\mathrm{arcsec}$',
	fontsize=25)
axes[1].set_title('$0.4\,\mathrm{arcsec}$',fontsize=25)
axes[2].set_title('$0.7\,\mathrm{arcsec}$',fontsize=25)

plt.savefig('../figures/artpop_psf.pdf')
plt.savefig('../figures/artpop_psf.png', dpi=300)
