# Third-party
import numpy as np
import matplotlib.pyplot as plt

# Project
import artpop

# Load matplotlib style
plt.style.use('jpg.mplstyle')

# Zero point converter based on MIST calculations
zpt_convert = artpop.load_zero_point_converter()


###############################################################################
# Sun AB and Vega magnitudes for converting to Solar luminosities
###############################################################################
U_sun_ab = 6.35
U_sun_vega = 5.55

B_sun_ab = 5.35
B_sun_vega = 5.46

R_sun_ab = 4.60
R_sun_vega = 4.41

sun_mag = dict(
    U=dict(vega=U_sun_vega, ab=U_sun_ab),
    B=dict(vega=B_sun_vega, ab=B_sun_ab),
    R=dict(vega=R_sun_vega, ab=R_sun_ab),
)

# Solar luminosity as function of filter, absolute mag, and magnitude system
L_sun = lambda m, band, system: 10**(0.4*(sun_mag[band][system] - m))
###############################################################################


###############################################################################
# Calculate SSP mass, mags, and SBF mags using ArtPop (takes ~8 minutes)
###############################################################################
imf = 'kroupa'
phot_system = 'UBVRIplus'
log_ages = artpop.MISTIsochrone._log_age_grid[40:-3]

# SSP mass
mass = {}

# SSP magnitudes
U = {}
B = {}
R = {}

# SSP SBF magnitudes
Ubar = {}
Bbar = {}
Rbar = {}

for feh in [-1.5, 0.0]:
    print(f'[Fe/H] = {feh}')

    mass[feh] = []

    U[feh] = []
    B[feh] = []
    R[feh] = []

    Ubar[feh] = []
    Bbar[feh] = []
    Rbar[feh] = []

    for log_age in log_ages:

        mist = artpop.MISTIsochrone(log_age, feh, phot_system)

        mass[feh].append(mist.ssp_surviving_mass('kroupa'))

        U[feh].append(mist.ssp_mag('Bessell_U', imf))
        B[feh].append(mist.ssp_mag('Bessell_B', imf))
        R[feh].append(mist.ssp_mag('Bessell_R', imf))

        Ubar[feh].append(mist.ssp_sbf_mag('Bessell_U', imf))
        Bbar[feh].append(mist.ssp_sbf_mag('Bessell_B', imf))
        Rbar[feh].append(mist.ssp_sbf_mag('Bessell_R', imf))

    mass[feh] = np.array(mass[feh])

    U[feh] = np.array(U[feh]) + zpt_convert.to_ab('Bessell_U')
    B[feh] = np.array(B[feh]) + zpt_convert.to_ab('Bessell_B')
    R[feh] = np.array(R[feh]) + zpt_convert.to_ab('Bessell_R')

    Ubar[feh] = np.array(Ubar[feh]) + zpt_convert.to_ab('Bessell_U')
    Bbar[feh] = np.array(Bbar[feh]) + zpt_convert.to_ab('Bessell_B')
    Rbar[feh] = np.array(Rbar[feh]) + zpt_convert.to_ab('Bessell_R')
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

    m_l = mass[feh] / L_sun(R[feh], 'R', 'ab')
    ax[0].plot(10**log_ages, m_l, c='tab:red', label=label('R'), **kw)

    m_l = mass[feh] / L_sun(B[feh], 'B', 'ab')
    ax[0].plot(10**log_ages, m_l, c='tab:blue', label=label('B'), **kw)

    m_l = mass[feh] / L_sun(B[feh], 'U', 'ab')
    ax[0].plot(10**log_ages, m_l, c='blueviolet', label=label('U'), **kw)

    sign = '' if feh >= 0 else '$-$'
    ax[1].plot(10**log_ages, U[feh] - R[feh], c='k',  **kw)
    ax[2].plot(10**log_ages, B[feh] - R[feh], c='k', **kw)
    ax[3].plot(10**log_ages, Bbar[feh] - Rbar[feh], c='k',
               label=f'{sign}{abs(feh)}', **kw)


ax[0].set(yscale='log')
ax[0].set_ylim(ymax=10)
ax[0].set_ylabel('$M / L$ [$M_\odot/L_\odot$]', fontsize=25, labelpad=4)
ax[0].set_yticks([0.1, 1, 10])
ax[0].set_yticklabels([0.1, 1, 10])
ax[0].legend(loc='upper left', fontsize=20, frameon=False, borderaxespad=0.9)

ax[1].set_ylim(-0.8, 3)
ax[2].set_ylim(-0.7, 1.5)

ax[1].set_ylabel('$U - R$', fontsize=25, labelpad=8)
ax[2].set_ylabel('$B - R$', fontsize=25, labelpad=-4)
ax[3].set_ylabel('$\overline{B} - \overline{R}$', fontsize=25, labelpad=8)

ax[3].legend(loc='lower right', frameon=True, title='[Fe/H]', handlelength=1.7,
             handletextpad=0.5, fontsize=21, title_fontsize=20,
             borderaxespad=0.8)

for i in range(4):
    ax[i].minorticks_on()
    ax[i].set_xscale('log')
    ax[i].tick_params('y', pad=6, labelsize=20)
    ax[i].tick_params('x', pad=8, labelsize=22, length=7)

for i in [2, 3]:
    ax[i].set_xticks([1e7, 1e8, 1e9, 1e10])
    ax[i].set_xlabel('Age [yr]', fontsize=28)

fig.savefig('../figures/sps_demo.pdf')
