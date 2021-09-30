# Standard library
import os

# Third-party
import numpy as np
import matplotlib.pyplot as plt

# Project
import artpop

# load matplotlib style
plt.style.use(artpop.jpg_style)

# path to figures
fig_path = os.path.join(os.pardir, 'figures')


###############################################################################
# Sun AB and Vega magnitudes for converting to Solar luminosities
###############################################################################
V_sun_ab = 4.81
V_sun_vega = 4.81

I_sun_ab = 4.52
I_sun_vega = 4.10

sun_mag = dict(
    V=dict(vega=V_sun_vega, ab=V_sun_ab),
    I=dict(vega=I_sun_vega, ab=I_sun_ab)
)

# Solar luminosity as function of filter, absolute mag, and magnitude system
L_sun = lambda m, band, system: 10**(0.4*(sun_mag[band][system] - m))
###############################################################################


###############################################################################
# Calculate SSP mass, mags, and SBF mags using ArtPop (takes ~6 minutes)
###############################################################################
imf = 'kroupa'
phot_system = 'UBVRIplus'
log_ages = artpop.MISTIsochrone._log_age_grid[40:-3]

# SSP mass
mass = {}

# SSP magnitudes
V = {}
I = {}

# SSP SBF magnitudes
Vbar = {}
Ibar = {}

for feh in [-1.5, 0.0]:
    print(f'[Fe/H] = {feh}')

    mass[feh] = []

    V[feh] = []
    I[feh] = []

    Vbar[feh] = []
    Ibar[feh] = []

    for log_age in log_ages:

        mist = artpop.MISTIsochrone(log_age, feh, phot_system)

        mass[feh].append(mist.ssp_surviving_mass('kroupa'))

        V[feh].append(mist.ssp_mag('Bessell_V', imf))
        I[feh].append(mist.ssp_mag('Bessell_I', imf))

        Vbar[feh].append(mist.ssp_sbf_mag('Bessell_V', imf))
        Ibar[feh].append(mist.ssp_sbf_mag('Bessell_I', imf))

    mass[feh] = np.array(mass[feh])

    V[feh] = np.array(V[feh])
    I[feh] = np.array(I[feh])

    Vbar[feh] = np.array(Vbar[feh])
    Ibar[feh] = np.array(Ibar[feh])
###############################################################################


###############################################################################
# Make SPS demo figure for paper
###############################################################################
fig, ax = plt.subplots(2, 2, figsize=(12, 10), sharex=True)
fig.subplots_adjust(hspace=0.05, wspace=0.23)

lw = 2.5
ax = ax.flatten()

for feh in [0.0, -1.5]:

    ls = '-' if feh == -1.5 else '--'
    kw = dict(lw=lw, ls=ls)

    label = lambda b: {0.0: '', -1.5: '$M / L_' + b + '$'}[feh]

    m_l = mass[feh] / L_sun(I[feh], 'I', 'ab')
    ax[0].plot(10**log_ages, m_l, c='tab:red', label=label('I'), **kw)

    m_l = mass[feh] / L_sun(V[feh], 'V', 'ab')
    ax[0].plot(10**log_ages, m_l, c='tab:blue', label=label('V'), **kw)

    sign = '' if feh >= 0 else '$-$'
    label = lambda b: {0.0: '', -1.5: '$\overline{' + b + '}$'}[feh]
    ax[1].plot(10**log_ages, V[feh] - I[feh], c='k',  **kw)
    ax[2].plot(10**log_ages, Ibar[feh], c='tab:red', label=label('I'),**kw)
    ax[2].plot(10**log_ages, Vbar[feh], c='tab:blue', label=label('V'), **kw)
    ax[3].plot(10**log_ages, Vbar[feh] - Ibar[feh], c='k',
               label=f'{sign}{abs(feh)}', **kw)

ax[0].set(yscale='log')
ax[0].set_ylim(ymax=10)
ax[0].set_ylabel('$M / L$ [$M_\odot/L_\odot$]', fontsize=25, labelpad=4)
ax[0].set_yticks([0.1, 1, 10])
ax[0].set_yticklabels([0.1, 1, 10])
ax[0].legend(loc='upper left', fontsize=22, frameon=False, borderaxespad=0.9)

ax[1].set_ylabel('$V - I$', fontsize=25, labelpad=-12)
ax[2].set_ylabel('SBF Magnitude', fontsize=25)
ax[3].set_ylabel('$\overline{V} - \overline{I}$', fontsize=25, labelpad=10)

ax[3].legend(loc='lower right', frameon=True, title='[Fe/H]', handlelength=1.7,
             handletextpad=0.5, fontsize=21, title_fontsize=20,
             borderaxespad=0.8)

ax[2].legend(loc='lower left', handlelength=1.7,
             handletextpad=0.5, fontsize=21, title_fontsize=22,
             borderaxespad=0.8)

ax[2].invert_yaxis()

for i in range(4):
    ax[i].minorticks_on()
    ax[i].set_xscale('log')
    ax[i].tick_params('y', pad=6, labelsize=20)
    ax[i].tick_params('x', pad=8, labelsize=22, length=7)

for i in [2, 3]:
    ax[i].set_xticks([1e7, 1e8, 1e9, 1e10])
    ax[i].set_xlabel('Age [yr]', fontsize=28)

fig.savefig(os.path.join(fig_path, 'sps_demo.pdf'))
