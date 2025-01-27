# Standard library
import os

# Third-party
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from astropy.io import fits
from astropy import units as u
from astropy.visualization import make_lupton_rgb

# Project
import artpop

# load matplotlib style
plt.style.use(artpop.jpg_style)

# path to figures
fig_path = os.path.join(os.pardir, 'figures')


###############################################################################
# GC parameters and MIST isochrone object
###############################################################################
phot_system = 'HST_ACSWF'
pixel_scale = 0.05

log_age = 10.1
feh = -2
r_s = 0.8 * u.pc
total_mass = 2e5 * u.Msun
distance = 5 * u.kpc

mist = artpop.MISTIsochrone(log_age, feh, phot_system)
###############################################################################


###############################################################################
# Make IMF-weighted CMD figure
###############################################################################
imf = 'kroupa'

g = mist.isochrone_full['ACS_WFC_F475W']
I = mist.isochrone_full['ACS_WFC_F814W']
g_I = g - I

# calculate IMF weights
wghts = artpop.imf_dict[imf](mist.mini)
wghts /= wghts.max()
log_wghts = np.log10(wghts)

fig, ax = plt.subplots(figsize=(9, 8))
kw = dict(s=50, marker='o',cmap='bone')

# distance modulus
dist_mod = 5 * np.log10(distance.to('pc').value) - 5

# scatter plot with color bar
sax = ax.scatter(g_I, I + dist_mod, c=log_wghts, vmin=-1.5, vmax=0.1, **kw)

# make color bar
cbaxes = fig.add_axes([0.47, 0.53, 0.38, 0.03])
cbar = plt.colorbar(sax, orientation='horizontal', cax=cbaxes)
cbar.ax.set_xlabel(r'log$_{10}$(dN/dM)', fontsize=30, labelpad=-74)
cbar.ax.tick_params(length=6, labelsize=20)
cbar.ax.xaxis.set_ticks_position('bottom')

ax.invert_yaxis()
ax.set_xlabel(r'$g_{475} - I_{814}$', fontsize=38)
ax.set_ylabel(r'$I_{814}$', fontsize=38, labelpad=15)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')

# SSP annotations
fs = 31
y = 0.32
dy = 0.1
x = 0.13
ha = 'left'
ax.text(x, y, f'D = {int(distance.value)} kpc', transform=ax.transAxes,
        ha=ha, va='center', fontsize=fs)
ax.text(x, y - dy, f'[Fe/H] = {feh}', transform=ax.transAxes,
        ha=ha, va='center', fontsize=fs)
ax.text(x, y - 2 * dy, f'Age = {round(10**10.1 /1e9, 1)} Gyr',
        transform=ax.transAxes, ha=ha, va='center', fontsize=fs)

ax.tick_params('both', labelsize=23)
fig.savefig(os.path.join(fig_path, 'main-diagram', 'cmd.png'), dpi=300)
###############################################################################


###############################################################################
# Make xy Plummer positions figure
###############################################################################
xy_dim = np.array([2501, 2501])
x_0, y_0 = xy_dim / 2

r_pix = u.radian.to('arcsec') * (r_s / distance).decompose().value
r_pix /= pixel_scale

fig, ax = plt.subplots(figsize=(8, 8))

fs = 36
ax.set_xlabel('$x$', fontsize=fs)
ax.set_ylabel('$y$', fontsize=fs, labelpad=11)

# sample 5e3 star positions from Plummer profile
xy = artpop.plummer_xy(
    num_stars=5e3,
    distance=distance,
    xy_dim=xy_dim,
    pixel_scale=pixel_scale,
    scale_radius=r_s,
    drop_outside=True
)

star_c = 'gray'
ax.plot(xy[:, 0], xy[:, 1], 'o', ms=0.5, c=star_c)
ax.set(aspect='equal', xticks=[], yticks=[])

circ_c = 'k'
ax.add_patch(Circle((x_0, y_0), r_pix, fc='none', ec=circ_c, lw=2))
ax.axvline(x=x_0, ls='--', c='k', alpha=1, zorder=-10)
ax.axhline(y=y_0, ls='--', c='k', alpha=1, zorder=-10)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)

y = y_0 + 1.3 * r_pix
ax.plot([x_0, x_0 + r_pix], [y, y], '-', c=circ_c, lw=2)
ax.text(x_0 + 0.5 * r_pix, y + 40, r'$r_s$', fontsize=35, ha='center')
ax.set(xlim=[0, xy_dim[0] - 1], ylim=[0, xy_dim[1] - 1])

fig.savefig(os.path.join(fig_path, 'main-diagram', 'plummer.png'), dpi=300)
###############################################################################


###############################################################################
# Make RGB PSF images
###############################################################################
cmaps = ['Reds', 'Greens', 'Blues']
bands = ['ACS_WFC_F814W', 'ACS_WFC_F606W', 'ACS_WFC_F475W']
psf = {b: fits.getdata(f"../data/{b}.fits") for b in bands}

zoom = 120
for b, c in zip(bands, cmaps):
    fig, ax = artpop.show_image(psf[b][zoom:-zoom, zoom:-zoom],
                                [0, 99.5], cmap=c, figsize=(8, 8))
    fn = f'psf_{c.lower()}.png'
    fig.savefig(os.path.join(fig_path, 'main-diagram', fn), dpi=300)
###############################################################################


###############################################################################
# Make RGB GC image
###############################################################################

# random state for reproducibility
rng = np.random.RandomState(123)

phot_system = 'HST_ACSWF'
src = artpop.MISTPlummerSSP(log_age, feh, phot_system, r_s,
                            distance, xy_dim, pixel_scale,
                            total_mass=total_mass, random_state=rng)

# HST-like artificial imager
imager = artpop.ArtImager(phot_system, diameter=2.4, read_noise=3)

# some scaling to make the image pretty
scale = dict(ACS_WFC_F814W=1, ACS_WFC_F606W=1.3, ACS_WFC_F475W=1.7)

exptime = 90 * u.min
images =[]

# mock observe
for b in bands:
    zpt = 22
    _psf = fits.getdata(f'../data/{b}.fits')
    obs = imager.observe(src, b, exptime=exptime, psf=_psf, zpt=zpt)
    images.append(obs.image * scale[b])

rgb = make_lupton_rgb(*images, stretch=0.4, Q=8)
fig, ax = artpop.show_image(rgb)
fig.savefig(os.path.join(fig_path, 'main-diagram', 'gc_rgb.png'), dpi=300)
###############################################################################


###############################################################################
# Make noiseless and noisy grid images
###############################################################################
xy_dim = 51
cmaps = ['Reds', 'Greens', 'Blues']

for c in cmaps:

    image = np.random.uniform(size=(10, 10))

    plt.figure(figsize=(10, 10))
    plt.imshow(image/image, cmap=c)
    plt.xlim(0, 9)
    plt.ylim(0, 9)
    plt.gca().grid(True, c='k', lw=1.5)
    plt.gca().tick_params(length=0)
    plt.gca().set_xticklabels([])
    plt.gca().set_yticklabels([])
    fn = f'inject_{c.lower()}_noiseless.png'
    plt.savefig(os.path.join(fig_path, 'main-diagram', fn), dpi=300)

    plt.figure(figsize=(10, 10))
    plt.imshow(image, cmap=c)
    plt.gca().tick_params(length=0)
    plt.gca().set_xticklabels([])
    plt.gca().set_yticklabels([])
    fn = f'inject_{c.lower()}.png'
    plt.savefig(os.path.join(fig_path, 'main-diagram', fn), dpi=300)
