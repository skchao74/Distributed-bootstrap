import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, ScalarFormatter
import numpy as np
import pandas as pd
import glob


class ScalarFormatterForceFormat(ScalarFormatter):
    def _set_format(self, vmin, vmax):
        self.format = "%1.2f"


def gen_plt_hd_rel(cov, rad, ks, km, dis, des, dim='', supp='a'):
    d_str = [r'$d=2^3$', r'$d=2^5$', r'$d=2^7$']
    if dim == 'hd':
        d_str = [r'$s_0=2^2$', r'$s_0=2^4$']
    if dim == 'hd' and dis == 'glm':
        d_str = [r'$s_0=2^1$', r'$s_0=2^3$']
    if dim == 'hd':
        dim += '_'

    for i2 in np.arange(2):
        for i3 in np.arange(4):
            if i2 != 0 or (i3 != 0 and i3 != 3):
                continue
            if i2 == 0:
                s2 = 'p95'
                p = 0.95
            else:
                s2 = 'p9'
                p = 0.9
            if i3 == 0:
                s3 = 'inf'
            elif i3 == 1:
                s3 = '2'
            elif i3 == 2:
                s3 = '1'
            else:
                s3 = 'pt'

            clr = plt.rcParams['axes.prop_cycle'].by_key()['color']

            max_wid = np.ma.masked_invalid(np.array(
                [[[np.nanmax(np.mean(rad[i, i2, i3, l, j, :, :], 1)[:km[j]]) for i in np.arange(2)] for j in
                  np.arange(2)] for l in np.arange(4)]
            )).max()
            ym = max(max_wid * 1.05, 1.05) * 2

            fig, ([[ax1, ax2], [ax11, ax22]]) = plt.subplots(2, 2, figsize=(6.5, 4.5), constrained_layout=True)

            yfmt = ScalarFormatterForceFormat()
            yfmt.set_powerlimits((0, 0))

            ax1.plot(np.log2(ks[:km[0]]), np.mean(cov[0, i2, i3, 0, 0, :, :], 1)[:km[0]], alpha=0.8,
                     label=r'$\tau=1$',
                     linewidth=0.8, marker='v', markerfacecolor='None', markeredgewidth=0.8, color=clr[0])
            ax1.plot(np.log2(ks[:km[0]]), np.mean(cov[0, i2, i3, 1, 0, :, :], 1)[:km[0]], alpha=0.8,
                     label=r'$\tau=2$',
                     linewidth=0.8, marker='o', markerfacecolor='None', markeredgewidth=0.8, color=clr[1])
            ax1.plot(np.log2(ks[:km[0]]), np.mean(cov[0, i2, i3, 2, 0, :, :], 1)[:km[0]], alpha=0.8,
                     label=r'$\tau=3$', linewidth=0.8, marker='s', markerfacecolor='None', markeredgewidth=0.8,
                     color=clr[4])
            ax1.plot(np.log2(ks[:km[0]]), np.mean(cov[0, i2, i3, 3, 0, :, :], 1)[:km[0]], alpha=0.8,
                     label=r'$\tau=4$',
                     linewidth=0.8, marker='D', markerfacecolor='None', markeredgewidth=0.8, color=clr[5])
            ax1.axhline(y=p, color='k', linestyle='-', linewidth=0.8, alpha=0.8)
            ax1.set_ylim(-0.05, 1.05)
            ax1.set_xlabel(r'$\log_2 k$')
            ax1.set_ylabel('Coverage', rotation=0, position=(0, 1.05), fontsize=10)
            ax1.set_title('k-grad, ' + d_str[0], fontsize=12)
            ax1.grid(True)

            ax2.plot(np.log2(ks)[:km[1]], np.mean(cov[0, i2, i3, 0, 1, :, :], 1)[:km[1]], alpha=0.8,
                     label=r'$\tau=1$',
                     linewidth=0.8, marker='v', markerfacecolor='None', markeredgewidth=0.8, color=clr[0])
            ax2.plot(np.log2(ks)[:km[1]], np.mean(cov[0, i2, i3, 1, 1, :, :], 1)[:km[1]], alpha=0.8,
                     label=r'$\tau=2$',
                     linewidth=0.8, marker='o', markerfacecolor='None', markeredgewidth=0.8, color=clr[1])
            ax2.plot(np.log2(ks)[:km[1]], np.mean(cov[0, i2, i3, 2, 1, :, :], 1)[:km[1]], alpha=0.8,
                     label=r'$\tau=3$', linewidth=0.8, marker='s', markerfacecolor='None', markeredgewidth=0.8,
                     color=clr[4])
            ax2.plot(np.log2(ks)[:km[1]], np.mean(cov[0, i2, i3, 3, 1, :, :], 1)[:km[1]], alpha=0.8,
                     label=r'$\tau=4$',
                     linewidth=0.8, marker='D', markerfacecolor='None', markeredgewidth=0.8, color=clr[5])
            ax2.axhline(y=p, color='k', linestyle='-', linewidth=0.8, alpha=0.8)
            ax2.set_ylim(-0.05, 1.05)
            ax2.set_xlabel(r'$\log_2 k$')
            ax2.set_title('k-grad, ' + d_str[1], fontsize=12)
            ax2.set_yticklabels([])
            ax2.yaxis.set_tick_params(length=0)
            ax2.grid(True)

            ax4 = ax1.twinx()
            ax4.plot(np.log2(ks[:km[0]]), np.mean(rad[0, i2, i3, 0, 0, :, :], 1)[:km[0]], '--', alpha=0.8,
                     label=r'$\tau=1$', linewidth=0.8, marker='v', markerfacecolor='None', markeredgewidth=0.8,
                     color=clr[0])
            ax4.plot(np.log2(ks[:km[0]]), np.mean(rad[0, i2, i3, 1, 0, :, :], 1)[:km[0]], '--', alpha=0.8,
                     label=r'$\tau=2$', linewidth=0.8, marker='o', markerfacecolor='None', markeredgewidth=0.8,
                     color=clr[1])
            ax4.plot(np.log2(ks[:km[0]]), np.mean(rad[0, i2, i3, 2, 0, :, :], 1)[:km[0]], '--', alpha=0.8,
                     label=r'$\tau=3$', linewidth=0.8, marker='s', markerfacecolor='None', markeredgewidth=0.8,
                     color=clr[4])
            ax4.plot(np.log2(ks[:km[0]]), np.mean(rad[0, i2, i3, 3, 0, :, :], 1)[:km[0]], '--', alpha=0.8,
                     label=r'$\tau=4$', linewidth=0.8, marker='D', markerfacecolor='None', markeredgewidth=0.8,
                     color=clr[5])
            ax4.axhline(y=1, color='k', linestyle='--', linewidth=0.8, alpha=0.8)
            ax4.set_ylim(-ym / 1.05 * 0.05, ym)
            ax4.set_yticklabels([])
            ax4.yaxis.set_tick_params(length=0)
            ax4.grid(False)
            ax1.legend(fontsize=6, markerscale=0.8, loc=(0, 0.5))

            ax5 = ax2.twinx()
            ax5.plot(np.log2(ks[:km[1]]), np.mean(rad[0, i2, i3, 0, 1, :, :], 1)[:km[1]], '--', alpha=0.8,
                     label=r'$\tau=1$',
                     linewidth=0.8, marker='v', markerfacecolor='None', markeredgewidth=0.8, color=clr[0])
            ax5.plot(np.log2(ks[:km[1]]), np.mean(rad[0, i2, i3, 1, 1, :, :], 1)[:km[1]], '--', alpha=0.8,
                     label=r'$\tau=2$',
                     linewidth=0.8, marker='o', markerfacecolor='None', markeredgewidth=0.8, color=clr[1])
            ax5.plot(np.log2(ks[:km[1]]), np.mean(rad[0, i2, i3, 2, 1, :, :], 1)[:km[1]], '--', alpha=0.8,
                     label=r'$\tau=3$',
                     linewidth=0.8, marker='s', markerfacecolor='None', markeredgewidth=0.8, color=clr[4])
            ax5.plot(np.log2(ks[:km[1]]), np.mean(rad[0, i2, i3, 3, 1, :, :], 1)[:km[1]], '--', alpha=0.8,
                     label=r'$\tau=4$',
                     linewidth=0.8, marker='D', markerfacecolor='None', markeredgewidth=0.8, color=clr[5])
            ax5.axhline(y=1, color='k', linestyle='--', linewidth=0.8, alpha=0.8)
            ax5.set_ylim(-ym / 1.05 * 0.05, ym)
            ax5.set_ylabel(r'$\mathrm{\frac{Width}{OracleWidth}}$', rotation=0, position=(1, 1.2))
            l = ax2.get_ylim()
            l2 = ax5.get_ylim()
            f = lambda x: l2[0] + (x - l[0]) / (l[1] - l[0]) * (l2[1] - l2[0])
            ticks = f(ax2.get_yticks())
            ax5.yaxis.set_major_locator(FixedLocator(ticks))
            ax5.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
            ax5.yaxis.set_major_formatter(yfmt)
            ax5.grid(False)
            dummy_lines = [ax2.plot([], [], c="black", ls='-', linewidth=0.8)[0],
                           ax2.plot([], [], c="black", ls='--', linewidth=0.8)[0]]
            ax2.legend([dummy_lines[i] for i in [0, 1]], ["coverage", "width"], fontsize=6)

            ax11.plot(np.log2(ks[:km[0]]), np.mean(cov[1, i2, i3, 0, 0, :, :], 1)[:km[0]], alpha=0.8,
                      label=r'$\tau=1$',
                      linewidth=0.8, marker='v', markerfacecolor='None', markeredgewidth=0.8, color=clr[0])
            ax11.plot(np.log2(ks[:km[0]]), np.mean(cov[1, i2, i3, 1, 0, :, :], 1)[:km[0]], alpha=0.8,
                      label=r'$\tau=2$',
                      linewidth=0.8, marker='o', markerfacecolor='None', markeredgewidth=0.8, color=clr[1])
            ax11.plot(np.log2(ks[:km[0]]), np.mean(cov[1, i2, i3, 2, 0, :, :], 1)[:km[0]], alpha=0.8,
                      label=r'$\tau=3$', linewidth=0.8, marker='s', markerfacecolor='None', markeredgewidth=0.8,
                      color=clr[4])
            ax11.plot(np.log2(ks[:km[0]]), np.mean(cov[1, i2, i3, 3, 0, :, :], 1)[:km[0]], alpha=0.8,
                      label=r'$\tau=4$',
                      linewidth=0.8, marker='D', markerfacecolor='None', markeredgewidth=0.8, color=clr[5])
            ax11.axhline(y=p, color='k', linestyle='-', linewidth=0.8, alpha=0.8)
            ax11.set_ylim(-0.05, 1.05)
            ax11.set_xlabel(r'$\log_2 k$')
            ax11.set_ylabel('Coverage', rotation=0, position=(0, 1.05), fontsize=10)
            ax11.set_title('n+k\u22121-grad, ' + d_str[0], fontsize=12)
            ax11.grid(True)

            ax22.plot(np.log2(ks)[:km[1]], np.mean(cov[1, i2, i3, 0, 1, :, :], 1)[:km[1]], alpha=0.8,
                      label=r'$\tau=1$',
                      linewidth=0.8, marker='v', markerfacecolor='None', markeredgewidth=0.8, color=clr[0])
            ax22.plot(np.log2(ks)[:km[1]], np.mean(cov[1, i2, i3, 1, 1, :, :], 1)[:km[1]], alpha=0.8,
                      label=r'$\tau=2$',
                      linewidth=0.8, marker='o', markerfacecolor='None', markeredgewidth=0.8, color=clr[1])
            ax22.plot(np.log2(ks)[:km[1]], np.mean(cov[1, i2, i3, 2, 1, :, :], 1)[:km[1]], alpha=0.8,
                      label=r'$\tau=3$', linewidth=0.8, marker='s', markerfacecolor='None', markeredgewidth=0.8,
                      color=clr[4])
            ax22.plot(np.log2(ks)[:km[1]], np.mean(cov[1, i2, i3, 3, 1, :, :], 1)[:km[1]], alpha=0.8,
                      label=r'$\tau=4$',
                      linewidth=0.8, marker='D', markerfacecolor='None', markeredgewidth=0.8, color=clr[5])
            ax22.axhline(y=p, color='k', linestyle='-', linewidth=0.8, alpha=0.8)
            ax22.set_ylim(-0.05, 1.05)
            ax22.set_xlabel(r'$\log_2 k$')
            ax22.set_title('n+k\u22121-grad, ' + d_str[1], fontsize=12)
            ax22.set_yticklabels([])
            ax22.yaxis.set_tick_params(length=0)
            ax22.grid(True)

            ax44 = ax11.twinx()
            ax44.plot(np.log2(ks[:km[0]]), np.mean(rad[1, i2, i3, 0, 0, :, :], 1)[:km[0]], '--', alpha=0.8,
                      label=r'$\tau=1$', linewidth=0.8, marker='v', markerfacecolor='None', markeredgewidth=0.8,
                      color=clr[0])
            ax44.plot(np.log2(ks[:km[0]]), np.mean(rad[1, i2, i3, 1, 0, :, :], 1)[:km[0]], '--', alpha=0.8,
                      label=r'$\tau=2$', linewidth=0.8, marker='o', markerfacecolor='None', markeredgewidth=0.8,
                      color=clr[1])
            ax44.plot(np.log2(ks[:km[0]]), np.mean(rad[1, i2, i3, 2, 0, :, :], 1)[:km[0]], '--', alpha=0.8,
                      label=r'$\tau=3$', linewidth=0.8, marker='s', markerfacecolor='None', markeredgewidth=0.8,
                      color=clr[4])
            ax44.plot(np.log2(ks[:km[0]]), np.mean(rad[1, i2, i3, 3, 0, :, :], 1)[:km[0]], '--', alpha=0.8,
                      label=r'$\tau=4$', linewidth=0.8, marker='D', markerfacecolor='None', markeredgewidth=0.8,
                      color=clr[5])
            ax44.axhline(y=1, color='k', linestyle='--', linewidth=0.8, alpha=0.8)
            ax44.set_ylim(-ym / 1.05 * 0.05, ym)
            ax44.set_yticklabels([])
            ax44.yaxis.set_tick_params(length=0)
            ax44.grid(False)

            ax55 = ax22.twinx()
            ax55.plot(np.log2(ks[:km[1]]), np.mean(rad[1, i2, i3, 0, 1, :, :], 1)[:km[1]], '--', alpha=0.8,
                      label=r'$\tau=1$',
                      linewidth=0.8, marker='v', markerfacecolor='None', markeredgewidth=0.8, color=clr[0])
            ax55.plot(np.log2(ks[:km[1]]), np.mean(rad[1, i2, i3, 1, 1, :, :], 1)[:km[1]], '--', alpha=0.8,
                      label=r'$\tau=2$',
                      linewidth=0.8, marker='o', markerfacecolor='None', markeredgewidth=0.8, color=clr[1])
            ax55.plot(np.log2(ks[:km[1]]), np.mean(rad[1, i2, i3, 2, 1, :, :], 1)[:km[1]], '--', alpha=0.8,
                      label=r'$\tau=3$',
                      linewidth=0.8, marker='s', markerfacecolor='None', markeredgewidth=0.8, color=clr[4])
            ax55.plot(np.log2(ks[:km[1]]), np.mean(rad[1, i2, i3, 3, 1, :, :], 1)[:km[1]], '--', alpha=0.8,
                      label=r'$\tau=4$',
                      linewidth=0.8, marker='D', markerfacecolor='None', markeredgewidth=0.8, color=clr[5])
            ax55.axhline(y=1, color='k', linestyle='--', linewidth=0.8, alpha=0.8)
            ax55.set_ylim(-ym / 1.05 * 0.05, ym)
            ax55.set_ylabel(r'$\mathrm{\frac{Width}{OracleWidth}}$', rotation=0, position=(1, 1.2))
            l = ax22.get_ylim()
            l2 = ax55.get_ylim()
            f = lambda x: l2[0] + (x - l[0]) / (l[1] - l[0]) * (l2[1] - l2[0])
            ticks = f(ax22.get_yticks())
            ax55.yaxis.set_major_locator(FixedLocator(ticks))
            ax55.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
            ax55.yaxis.set_major_formatter(yfmt)
            ax55.grid(False)

            fig.savefig(
                dim + dis + '_' + des + '_' + s2 + '_' + s3 + '_' + supp + '_jasa.pdf',
                bbox_inches='tight')


plt.style.use('ggplot')

num_k = 5
num_s = 2

ks = 2 ** np.arange(2, 7)
km = [5, 5]

cov = np.full((2, 2, 4, 4, 3, num_s, num_k, 1000), np.nan)
rad = np.full((2, 2, 4, 4, 3, num_s, num_k, 1000), np.nan)
truerad = np.full((2, 4, 3, num_s), np.nan)

csv_list = glob.glob("*.csv")
for f in csv_list:
    f0 = f[3:-4].split("_")
    if f0[4] != 'cov':
        continue
    if f0[2] == 'k':
        i1 = 0
    else:
        i1 = 1
    if f0[3] == 'p95':
        i2 = 0
    else:
        i2 = 1
    if f0[5] == 'inf':
        i3 = 0
    elif f0[5] == '2':
        i3 = 1
    elif f0[5] == '1':
        i3 = 2
    else:
        i3 = 3
    i4 = int(f0[6][1]) - 1
    df = pd.read_csv(f, header=None)
    df = np.array(df)
    df = df.reshape(int(df.shape[0] / 3 / num_s), 3, num_s, -1).transpose((1, 2, 3, 0))
    cov[i1, i2, i3, i4, :, :, :, :df.shape[3]] = df[:, :, 1:, :]

csv_list = glob.glob("*.csv")
for f in csv_list:
    f0 = f[3:-4].split("_")
    if f0[4] != 'rad':
        continue
    if f0[2] == 'k':
        i1 = 0
    else:
        i1 = 1
    if f0[3] == 'p95':
        i2 = 0
    else:
        i2 = 1
    if f0[5] == 'inf':
        i3 = 0
    elif f0[5] == '2':
        i3 = 1
    elif f0[5] == '1':
        i3 = 2
    else:
        i3 = 3
    i4 = int(f0[6][1]) - 1
    df = pd.read_csv(f, header=None)
    df = np.array(df)
    df = df.reshape(int(df.shape[0] / 3 / num_s), 3, num_s, -1).transpose((1, 2, 3, 0))
    rad[i1, i2, i3, i4, :, :, :, :df.shape[3]] = df[:, :, 1:, :]

df = pd.read_csv("hd_lm_tp_truerad_supp.csv", header=None)
df = np.array(df)
df = df.reshape(int(df.shape[0] / 9), 3, num_s, -1).transpose((3, 1, 2, 0))
truerad = np.percentile(df[1:, :, :, :], (95, 90), 3)

rel_rad = rad * 1
for i1 in range(rad.shape[1]):
    for i2 in range(rad.shape[2]):
        for i4 in range(rad.shape[4]):
            for i5 in range(rad.shape[5]):
                rel_rad[:, i1, i2, :, i4, i5] /= truerad[i1, i2, i4, i5]

gen_plt_hd_rel(cov[:, :, :, :, 0, :, :, :], rel_rad[:, :, :, :, 0, :, :, :], ks, km, 'lm', 'tp', 'hd', 'a')
gen_plt_hd_rel(cov[:, :, :, :, 1, :, :, :], rel_rad[:, :, :, :, 1, :, :, :], ks, km, 'lm', 'tp', 'hd', 's')
gen_plt_hd_rel(cov[:, :, :, :, 2, :, :, :], rel_rad[:, :, :, :, 2, :, :, :], ks, km, 'lm', 'tp', 'hd', 'n')
