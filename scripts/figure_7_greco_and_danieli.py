# Standard library
import os

# Third-party
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy import units as u
from astropy.table import Table
from astropy.visualization import make_lupton_rgb

# Project
import artpop
plt.style.use(artpop.jpg_style)

# path to figures
fig_path = os.path.join(os.pardir, 'figures')

# path to data
data_path = os.path.join(os.pardir, 'data')


###############################################################################
# Load MIST and PARSEC isochrones, which we pre-calculated using FSPS
###############################################################################
mist_fn = os.path.join(data_path, 'mist_iso_logage_8.55_feh_m1.5.csv')
parsec_fn = os.path.join(data_path, 'parsec_iso_logage_8.55_feh_m1.5.csv')
mist_iso = Table.read(mist_fn)
parsec_iso = Table.read(parsec_fn)
mask = parsec_iso['mini'] >= 0.1

bands = ['ACS_WFC_F814W', 'ACS_WFC_F606W', 'ACS_WFC_F475W']
parsec = artpop.Isochrone(parsec_iso['mini'][mask], parsec_iso['mact'][mask],
                          parsec_iso[mask][bands])
mist = artpop.Isochrone(mist_iso['mini'], mist_iso['mact'], mist_iso[bands])


###############################################################################
# Sample a single set of xy positions and stellar masses
###############################################################################
num_stars = 5e6
distance = 1.5 * u.Mpc

xy_dim = 2061
pixel_scale = 0.05

xy = artpop.sersic_xy(
    num_stars,
    n = 0.9,
    ellip = 0.35,
    r_eff = 100 * u.pc,
    theta = 0 * u.deg,
    distance = distance,
    xy_dim = xy_dim,
    pixel_scale = pixel_scale,
    drop_outside=True
)

# force mass range to be the same
print(f"MIST m_max = {mist_iso['mini'].max():.4f}")
print(f"MIST m_min = {mist_iso['mini'].min():.4f}\n")
print(f"PARSEC m_max = {parsec_iso['mini'].max():.4f}")
print(f"PARSEC m_min = {parsec_iso['mini'].min():.4f}")
m_max = min(parsec_iso['mini'].max(), mist_iso['mini'].max())
m_min = mist_iso['mini'].min()

print(f'm_min = {m_min:.4f}, m_max = {m_max:.4f}')
masses = artpop.sample_imf(len(xy), m_min=m_min, m_max=m_max)


###############################################################################
# Create source objects
###############################################################################
dmod = 5 * np.log10(distance.to('pc').value) - 5
parsec_mags = Table({b: parsec.interpolate(b, masses) + dmod for b in bands})
mist_mags = Table({b: mist.interpolate(b, masses) + dmod for b in bands})

parsec_src = artpop.Source(xy, parsec_mags, xy_dim, pixel_scale)
mist_src = artpop.Source(xy, mist_mags, xy_dim, pixel_scale)


mist_image = []
parsec_image = []


###############################################################################
# Mock observe sources
###############################################################################
zpt = 29
stretch = 0.4
sky_sb = [21, 22, 23]
exptime = 2 * 90 * u.min

imager = artpop.ArtImager(phot_system='HST_ACSWF', diameter=2.4, read_noise=3)

mist_image = []
parsec_image = []
for i, b in enumerate(bands):
    psf = fits.getdata(f'../data/{b}.fits')
    kw = dict(bandpass=b, exptime=exptime, sky_sb=sky_sb[i], psf=psf, zpt=zpt)
    mist_obs = imager.observe(mist_src, **kw)
    mist_image.append(mist_obs.image)
    parsec_obs = imager.observe(parsec_src, **kw)
    parsec_image.append(parsec_obs.image)
mist_rgb = make_lupton_rgb(*mist_image, stretch=stretch)
parsec_rgb = make_lupton_rgb(*parsec_image, stretch=stretch)


###############################################################################
# Make figure
###############################################################################
g_parsec = parsec_mags['ACS_WFC_F475W']
I_parsec = parsec_mags['ACS_WFC_F814W']
g_mist = mist_mags['ACS_WFC_F475W']
I_mist = mist_mags['ACS_WFC_F814W']

parsec_gI = -2.5 * np.log10(np.sum(10**(-0.4*g_parsec))) +\
             2.5 * np.log10(np.sum(10**(-0.4*I_parsec)))
mist_gI = -2.5 * np.log10(np.sum(10**(-0.4*g_mist))) +\
           2.5 * np.log10(np.sum(10**(-0.4*I_mist)))

scale = 1.1
fig, ax = plt.subplots(1, 3, figsize=(20*scale, 6*scale))
fig.subplots_adjust(wspace=0.25)

artpop.show_image(mist_rgb, subplots=(fig, ax[0]), rasterized=True)
ax[0].set(xticks=[], yticks=[])

title_fs = 30
ax[0].set_title('MIST Dwarf Galaxy', fontsize=title_fs, pad=11)
ax[0].set_xlabel('$g_{475} - I_{814} = ' + str(round(mist_gI, 1)) + '$',
                 fontsize=title_fs, labelpad=10)

ax[1].plot(g_parsec - I_parsec, I_parsec, '.', c='tab:red', label='PARSEC',
           rasterized=True)
ax[1].plot(g_mist - I_mist, I_mist, '.', c='tab:blue', label='MIST',
           rasterized=True)

x = 0.92
y = 0.21
dy = 0.1
ha = 'right'
ax[1].text(x, y, 'MIST', c='tab:blue', transform=ax[1].transAxes,
           fontsize=31, ha=ha, va='center')
ax[1].text(x, y - dy, 'PARSEC', c='tab:red', transform=ax[1].transAxes,
           fontsize=31, ha=ha, va='center')

ax[1].set_ylabel('$I_{814}$', fontsize=30, labelpad=10)
ax[1].set_xlabel('$g_{475} - I_{814}$', fontsize=30)
ax[1].tick_params(labelsize=18)
ax[1].set_ylim(19.4, 38.4)
ax[1].minorticks_on()
ax[1].invert_yaxis()

x = 0.92
y = 0.72
dy = 0.1
dx = 0.3
fs = 22
ha = 'right'
ax[1].text(x, y, 'log(Age/yr) = 8.55',  transform=ax[1].transAxes,
           fontsize=fs, ha=ha, va='center')

ax[1].text(x, y - dy, '[Fe/H] = $-1.5$', transform=ax[1].transAxes,
           fontsize=fs, ha=ha, va='center')

ax[1].text(x, y - 2 * dy, f'D = {distance.value} Mpc',
           transform=ax[1].transAxes, fontsize=fs, ha=ha, va='center')

ax[1].text(x, y - 3 * dy, r'N$_\star = 5 \times 10^6$',
           transform=ax[1].transAxes, fontsize=fs, ha=ha, va='center')


annotate_kw = dict(
    xy=(-0.04, -0.008),
    xytext=(-0.04, 1.004),
    xycoords='axes fraction',
    arrowprops=dict(facecolor='black',  arrowstyle='|-|,widthA=0.6,widthB=0.6')
)

ax[0].set_ylabel('1 kpc', fontsize=30, labelpad=23)
ax[0].annotate('', **annotate_kw)

artpop.show_image(parsec_rgb, subplots=(fig, ax[2]), rasterized=True)
ax[2].set_title('PARSEC Dwarf Galaxy', fontsize=title_fs, pad=11)
ax[2].set_ylabel('1 kpc', fontsize=30, labelpad=23)

ax[2].set(xticks=[], yticks=[])
ax[2].set_xlabel('$g_{475} - I_{814} = $ ' + f'{parsec_gI:.1f}',
                 fontsize=title_fs, labelpad=10)
ax[2].annotate('', **annotate_kw)

fig.savefig(os.path.join(fig_path, 'mist_vs_parsec_dwarf.pdf'), dpi=170);
