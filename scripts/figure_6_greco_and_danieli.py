# Standard library
import os
import pickle
from argparse import ArgumentParser

# Third-party
import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.visualization import make_lupton_rgb

# Project
import artpop

# load matplotlib style
plt.style.use(artpop.jpg_style)

# path to figures
fig_path = os.path.join(os.pardir, 'figures')


###############################################################################
# Parse command-line arguments
###############################################################################
parser = ArgumentParser()
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument('--make-pops', action='store_true',
                   help='Generate SSPs and save as pickle files in pop_path. '
                        'Note this takes ~30 min on a 2018 Macbook Pro and '
                        '~20 GB of disk space.')
group.add_argument('--make-plot', action='store_true')
parser.add_argument('--pop-path', default='.', help='path to SSP pkl files')
parser.add_argument('--num-pops', default=1000, type=int,
                    help='number of SSPs to generate and save')
args = parser.parse_args()


###############################################################################
# Model parameters
###############################################################################
phot_system = 'DECam'
pixel_scale = 0.263
log_age = 10.1
feh = -2.0
total_mass = 1e5 * u.M_sun
r_eff = 150 * u.pc
distance = 2.5 * u.Mpc
num_pops = args.num_pops

struct_kw = dict(
    n = 1.0,
    r_eff = r_eff,
    ellip = 0.3,
    theta = 135,
    xy_dim = 351,
    pixel_scale = pixel_scale,
)

mu_sky_vals = [19.9, 21.2, 22.0] # sky brightness
exptimes = np.array([2, 1, 1]) * u.hr # exposure times


###############################################################################
# If make_pops, then generate num_pops SSPs and save as pickle files
###############################################################################
if args.make_pops:
    gi_colors = []
    print('WARNING: SSP files will take up ~20 GB of disk space when N=1000.')
    for n in range(num_pops):
        if n % 100 == 0:
            print(f'generating ssp number {n}')
        ssp = artpop.MISTSSP(
            log_age=log_age,
            feh=feh,
            phot_system=phot_system,
            total_mass=total_mass
        )
        ssp.to_pickle(os.path.join(args.pop_path, f'ssp-{n}.pkl'))
        gi_colors.append([n, ssp.integrated_color('DECam_g', 'DECam_i')])
    output = np.vstack(gi_colors)
    out_fn = os.path.join(args.pop_path, 'g_i_colors.txt')
    np.savetxt(out_fn, output, fmt='%.0f %.6f')


###############################################################################
# If make_plot, generate dwarf stochasticity figure
###############################################################################

if args.make_plot:
    pkl_fn = lambda i: os.path.join(args.pop_path, f'ssp-{i}.pkl')
    gi_colors = np.loadtxt(os.path.join(args.pop_path, 'g_i_colors.txt'))[:, 1]

    idx_red = gi_colors.argmax()
    idx_blue = gi_colors.argmin()
    idx_med = np.abs(gi_colors - np.median(gi_colors)).argmin()

    for i in idx_blue, idx_red, idx_med:
        assert os.path.isfile(pkl_fn(i)), pkl_fn(i) + ' does not exist'

    ssp_red = artpop.MISTSSP.from_pickle(pkl_fn(idx_red))
    ssp_med = artpop.MISTSSP.from_pickle(pkl_fn(idx_med))
    ssp_blue = artpop.MISTSSP.from_pickle(pkl_fn(idx_blue))

    ssp_red.set_distance(distance)
    ssp_med.set_distance(distance)
    ssp_blue.set_distance(distance)

    src_red = artpop.SersicSP(ssp_red, **struct_kw)
    src_med = artpop.SersicSP(ssp_med, **struct_kw)
    src_blue = artpop.SersicSP(ssp_blue, **struct_kw)

   ############################################################################
   # generate RGB images of min, median, and max g-i dwarfs
   ############################################################################

    Q = 8
    stretch = 0.18
    fwhm = dict(i=0.97, r=1.07, g=1.25)

    imager = artpop.ArtImager('DECam', diameter=4*u.m)

    colors = ['blue', 'med', 'red']
    src_dict = dict(blue=src_blue, med=src_med, red=src_red)
    psf = {b: artpop.moffat_psf(fwhm[b], pixel_scale, 61) for b in 'gri'}

    rgb = dict()
    for c in colors:
        images = []
        # mock observe in gri
        for num, band in enumerate('irg'):
            obs = imager.observe(
                src_dict[c], f'DECam_{band}', exptimes[num], psf=psf[band],
                sky_sb=mu_sky_vals[num]
            )
            images.append(obs.image)
        # create RGB image
        rgb[c] = make_lupton_rgb(*images, stretch=stretch, Q=Q)

   ############################################################################
   # make the figure
   ############################################################################

    fs = 22
    axes = plt.figure(figsize=(9.5, 7), tight_layout=True).subplot_mosaic(
        """
        ABC
        DDD
        """
    )

    for a, c in zip('ABC', colors):
        axes[a].set(xticks=[], yticks=[], aspect='equal')
        artpop.show_image(rgb[c], subplots=(None, axes[a]), rasterized=True)
        gi = src_dict[c].sp.integrated_color('DECam_g', 'DECam_i')
        axes[a].set_title(f'$g-i$ = {gi:.2f}', fontsize=fs, y=1.015)

    arrowstyle = '|-|,widthA=0.6,widthB=0.6'
    axes['A'].set_ylabel('800 pc', fontsize=fs, labelpad=23)
    axes['A'].annotate('',
        xy=(-0.06, -0.009), xytext=(-0.06, 1.005), xycoords='axes fraction',
        arrowprops=dict(facecolor='black',  arrowstyle=arrowstyle)
    )

    kw = dict(bins='auto', range=[0.535, 0.795])
    axes['D'].hist(gi_colors, color='lightgrey', **kw)
    axes['D'].hist(gi_colors, color='k', histtype='step', lw=3., **kw)
    axes['D'].set_ylabel('Number of Realizations', fontsize=fs-1)
    axes['D'].set_xlabel('$g-i$', fontsize=fs+5)
    axes['D'].tick_params(labelsize=fs-5)

    print(f'median(g-i) = {np.median(gi_colors):.2f}')
    print(f'stddev(g-i) = {np.std(gi_colors):.2f}')
    print(f' delta(g-i) = {gi_colors.ptp():.2f}')

    fs = 20
    y = 0.9
    dy = 0.17
    dx = 0.3
    ha = 'left'
    kw = dict(transform=axes['D'].transAxes, fontsize=fs, ha=ha, va='center')

    x = 0.105
    axes['D'].text(x, y, 'log(Age/yr) = 10.1', **kw)
    axes['D'].text(x, y - dy, '[Fe/H] = $-2$', **kw)
    axes['D'].text(x, y - 2 * dy, r'M$_\star = 10^5$ M$_\odot$', **kw);

    x = 0.65
    axes['D'].text(x, y, '$r_\mathrm{eff}$ = 150 pc', **kw)
    axes['D'].text(x, y - dy, f'D = 2.5 Mpc', **kw)

    axes['D'].spines['right'].set_visible(False)
    axes['D'].spines['top'].set_visible(False)
    axes['D'].yaxis.set_ticks_position('left')
    axes['D'].xaxis.set_ticks_position('bottom')

    arrowprops = dict(
        lw=3, color='tab:red',
        connectionstyle="arc3",
        arrowstyle="->,head_length=0.8,head_width=0.5",
    )
    kw = dict(
        xycoords='figure fraction',
        textcoords='figure fraction',
        arrowprops=arrowprops
    )

    axes['D'].annotate('', xy=(0.935, 0.55), xytext=(0.935, 0.12), **kw)

    arrowprops['color'] = 'tab:blue'
    axes['D'].annotate('', xy=(0.14, 0.55), xytext=(0.14, 0.12), **kw)

    arrowprops['color'] = 'tab:green'
    axes['D'].annotate('', xy=(0.538, 0.55), xytext=(0.538, 0.45),  **kw)

    for _ax, c in zip('ABC', ['blue', 'green', 'red']):
        for s in ['left', 'right', 'bottom', 'top']:
            axes[_ax].spines[s].set_color(f'tab:{c}')
            axes[_ax].spines[s].set_linewidth(3)

    plt.savefig(os.path.join(fig_path, 'dwarf_stoc.pdf'), dpi=170)
