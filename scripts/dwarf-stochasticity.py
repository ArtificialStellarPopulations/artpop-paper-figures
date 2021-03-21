# Third-party
# from math import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.visualization import make_lupton_rgb


# Project
import artpop
from artpop.util import embed_slices

# Load matplotlib style
plt.style.use('jpg.mplstyle')


###############################################################################
# Instrumental and observational parameters
###############################################################################
xy_dim = 1001
phot_system = 'DECam'
pixel_scale = 0.263 

psf_i = artpop.moffat_psf(0.97, pixel_scale, 61)
psf_r = artpop.moffat_psf(1.07, pixel_scale, 61)
psf_g = artpop.moffat_psf(1.25, pixel_scale, 61)

# Initialize artificial imager
imager = artpop.IdealImager()
zpt = 32
###############################################################################


###############################################################################
# Dwarf galaxy parameters
###############################################################################
log_age = 10.1
feh = -2.0

total_mass =1e5
r_eff = 150 * u.pc
distance = 6000 * u.kpc 

n = 0.6
theta = 30
ellip = 0.25
###############################################################################


###############################################################################
# Make 1000 dwarfs with identical input parameters
###############################################################################
N_dwarfs = 1000
index = np.arange(N_dwarfs)
df = pd.DataFrame(index=index, columns = ['g_mag', 'r_mag', 'int_color', 'num_rgb', 'num_cheb', 'num_agb','num_eagb', 'num_tpagb'])

lowest_color = 1
lowest_idx = 0
highest_color = 0
highest_idx = 0

for i in range(N_dwarfs):
    print('making dwarf # '+str(i))

    # create MIST Sersic SSP object    
    ssp = artpop.MISTSSP(log_age=log_age, feh=feh, phot_system=phot_system, 
                         total_mass=total_mass, distance=distance, imf='kroupa')
        
    # create Sersic source with the SSP just created
    src = artpop.SersicSP(ssp, r_eff=r_eff, n=n, theta=theta, 
                          ellip=ellip, xy_dim=xy_dim, pixel_scale=pixel_scale,
                          num_r_eff=10, dx=0, dy=0)

	# populate the data frame with useful information on the SSP 
    df.loc[df.index[i], 'g_mag'] = ssp.total_mag(bandpass='DECam_g')
    df.loc[df.index[i], 'r_mag'] = ssp.total_mag(bandpass='DECam_r')
    df.loc[df.index[i], 'int_color'] = ssp.integrated_color('DECam_g','DECam_r')
    
    mask_rgb = ssp.select_phase(phase='RGB')
    df.loc[df.index[i], 'num_rgb'] = sum(mask_rgb)
    mask_cheb = ssp.select_phase(phase='CHeB')
    df.loc[df.index[i], 'num_cheb'] = sum(mask_cheb)
    mask_agb = ssp.select_phase(phase='AGB')
    df.loc[df.index[i], 'num_agb'] = sum(mask_agb)
    mask_eagb = ssp.select_phase(phase='EAGB')
    df.loc[df.index[i], 'num_eagb'] = sum(mask_eagb)
    mask_tpagb = ssp.select_phase(phase='TPAGB')
    df.loc[df.index[i], 'num_tpagb'] = sum(mask_tpagb)

	# keep the lowest and highest color sources out of the sample        
    if ssp.integrated_color('DECam_g','DECam_r') < lowest_color:
        lowest_color = ssp.integrated_color('DECam_g','DECam_r')
        lowest_idx = i
        src_low = src
        ssp_low = ssp
    elif ssp.integrated_color('DECam_g','DECam_r') > highest_color:
        highest_color = ssp.integrated_color('DECam_g','DECam_r')
        highest_idx = i
        src_high = src
        ssp_high = ssp
###############################################################################
   

###############################################################################
# Get statistics for the various stellar pops from the simulated sources
###############################################################################
dfn = df.convert_dtypes()

# red giant branch stars
num_rgb = df.num_rgb.to_numpy()
mean_rgb = np.mean(num_rgb)

# core helium burning stars
num_cheb = df.num_cheb.to_numpy()
mean_cheb = np.mean(num_cheb)

# early asymptotic giant branch
num_eagb = df.num_eagb.to_numpy()
mean_eagb = np.mean(num_eagb)
eagb_stars = dfn.groupby(['num_eagb'])
mean_eagb_bin = eagb_stars.mean()
num_eagb_bin = mean_eagb_bin.index.values
describe_eagb = eagb_stars.describe()
eagb_g_mean = describe_eagb['g_mag']['mean'].to_numpy()
eagb_g_std = describe_eagb['g_mag']['std'].to_numpy()
eagb_color_mean = describe_eagb['int_color']['mean'].to_numpy()
eagb_color_std = describe_eagb['int_color']['std'].to_numpy()

# thermally pulsating asymptotic giant branch stars
num_tpagb = df.num_tpagb.to_numpy()
mean_tpagb = np.mean(num_tpagb)
tpagb_stars = dfn.groupby(['num_tpagb'])
mean_tpagb_bin = tpagb_stars.mean()
num_tpagb_bin = mean_tpagb_bin.index.values
describe_tpagb = tpagb_stars.describe()
tpagb_g_mean = describe_tpagb['g_mag']['mean'].to_numpy()
tpagb_g_std = describe_tpagb['g_mag']['std'].to_numpy()
tpagb_color_mean = describe_tpagb['int_color']['mean'].to_numpy()
tpagb_color_std = describe_tpagb['int_color']['std'].to_numpy()
###############################################################################


###############################################################################
# mock observe in gri two galaxies from the sample
###############################################################################
obs_i = imager.observe(src_low, 'DECam_i', psf_i, zpt=zpt)
obs_r = imager.observe(src_low, 'DECam_r', psf_r, zpt=zpt)
obs_g = imager.observe(src_low, 'DECam_g', psf_g, zpt=zpt)
mock_rgb_low = make_lupton_rgb(obs_i.image, obs_r.image, obs_g.image, stretch=15)

obs_i = imager.observe(src_high, 'DECam_i', psf_i, zpt=zpt)
obs_r = imager.observe(src_high, 'DECam_r', psf_r, zpt=zpt)
obs_g = imager.observe(src_high, 'DECam_g', psf_g, zpt=zpt)
mock_rgb_high = make_lupton_rgb(obs_i.image, obs_r.image, obs_g.image, stretch=15)
###############################################################################


###############################################################################
# Make dwarf galaxies stochastically sampled IMF figure
###############################################################################
fig, ((ax1,ax2),(ax3,ax4),(ax5,ax6)) = plt.subplots(3,2, figsize=(15/1.5,15))

# distributions of the # of stars in each stellar population 
binwidth = 1
data_hist_rgb = num_rgb-mean_rgb
bins_rgb = range(min(data_hist_rgb), max(data_hist_rgb) + binwidth, binwidth)
ax1.hist(num_rgb-mean_rgb, bins=bins_rgb, label=r'RGB', histtype='step', facecolor=None, edgecolor='r', lw=2)

data_hist_cheb = num_cheb-mean_cheb
bins_cheb = range(min(data_hist_cheb), max(data_hist_cheb) + binwidth, binwidth)
ax1.hist(num_cheb-mean_cheb, bins=bins_cheb, label=r'CHeB', histtype='step', facecolor=None, edgecolor='b', lw=2)

data_hist_eagb = num_eagb-mean_eagb
bins_eagb = range(min(data_hist_eagb), max(data_hist_eagb) + binwidth, binwidth)
ax1.hist(num_eagb-mean_eagb, bins=bins_eagb, label=r'EAGB', histtype='step', facecolor=None, edgecolor='g', lw=2)

data_hist_tpagb = num_tpagb-mean_tpagb
bins_tpagb = range(min(data_hist_tpagb), max(data_hist_tpagb) + binwidth, binwidth)
ax1.hist(num_tpagb-mean_tpagb, bins=bins_tpagb, label=r'TPAGB', histtype='step', facecolor=None, edgecolor='orange', lw=2)

ax1.set_xlabel(r'N$_{\mathrm{stars}}- \bar \mathrm{N}_{\mathrm{stars}}$')
ax1.set_ylabel(r'N$_{\mathrm{gal}}$')
ax1.text(0.05,0.4,'N$_{\mathrm{RGB, mean}}=$'+str(mean_rgb)+
         '\n' + 'N$_{\mathrm{CHeB, mean}}=$'+str(mean_cheb)+
         '\n' + 'N$_{\mathrm{EAGB, mean}}=$'+str(mean_eagb)+
         '\n' + 'N$_{\mathrm{TPAGB, mean}}=$'+str(mean_tpagb)
         , fontsize=15, transform=ax1.transAxes)
ax1.legend(loc='upper left', fontsize=12)

# integrated color distribution
ax2.hist(df.int_color, bins=20, color='#54278f', alpha=0.3, histtype='stepfilled', edgecolor='#54278f', lw=4)
ax2.set_xlabel(r'$g-r$')
ax2.set_ylabel(r'N$_\mathrm{gal}$')

# g-band magnitude as a function of N_stars for EAGB and TPAGB stars
ax3.scatter(num_eagb_bin/np.max(num_eagb_bin) ,eagb_g_mean, color='g', label='EAGB')
ax3.errorbar(num_eagb_bin/np.max(num_eagb_bin) ,eagb_g_mean, yerr=eagb_g_std, color='g', fmt='.')
ax3.scatter(num_tpagb_bin/np.max(num_tpagb_bin) ,tpagb_g_mean, color='orange', label='TPAGB')
ax3.errorbar(num_tpagb_bin/np.max(num_tpagb_bin) ,tpagb_g_mean, yerr=tpagb_g_std, color='orange', fmt='.')
ax3.set_xlabel(r'N$_\mathrm{stars}/\mathrm{N}_\mathrm{stars,max}$')
ax3.set_ylabel(r'$g$')
ax3.legend(fontsize=15)

# g-r color as a function of N_stars for EAGB and TPAGB stars
ax4.scatter(num_eagb_bin/np.max(num_eagb_bin) ,eagb_color_mean, color='g', label='EAGB')
ax4.errorbar(num_eagb_bin/np.max(num_eagb_bin) ,eagb_color_mean, yerr=eagb_color_std, color='g', fmt='.')
ax4.scatter(num_tpagb_bin/np.max(num_tpagb_bin) ,tpagb_color_mean, color='orange', label='TPAGB')
ax4.errorbar(num_tpagb_bin/np.max(num_tpagb_bin) ,tpagb_color_mean, yerr=tpagb_color_std, color='orange', fmt='.')
ax4.set_xlabel(r'N$_\mathrm{stars}/\mathrm{N}_\mathrm{stars,max}$')
ax4.set_ylabel(r'$g-r$')

# artificial image for the dwarf with the lowest g-r metallicity
ax5.imshow(mock_rgb_low[400:600,400:600])
low_color = round(df.int_color[lowest_idx],2)
low_mag =  round(df.g_mag[lowest_idx],2)
ax5.text(8,20,'dwarf \#1',color='w', fontsize=20)
ax5.text(8,175,r'$m_g$ = '+str(low_mag)+' mag',color='w', fontsize=20)
ax5.text(8,190,r'$g-r$ = '+str(low_color)+' mag',color='w', fontsize=20)
ax5.axis('off')


# artificial image for the dwarf with the highest g-r metallicity
ax6.imshow(mock_rgb_high[400:600,400:600])
high_color = round(df.int_color[highest_idx],2)
high_mag =  round(df.g_mag[highest_idx],2)
ax6.text(8,20,'dwarf \#2',color='w', fontsize=20)
ax6.text(8,175,r'$m_g$ = '+str(high_mag)+' mag',color='w', fontsize=20)
ax6.text(8,190,r'$g-r$ = '+str(high_color)+' mag',color='w', fontsize=20)
ax6.axis('off')

plt.savefig('../figures/dwarf_stoc.jpeg', dpi=400, bbox_inches='tight')



















