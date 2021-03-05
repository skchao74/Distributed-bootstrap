import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, ScalarFormatter, MaxNLocator, IndexLocator
import numpy as np
import pandas as pd
import glob


class ScalarFormatterForceFormat(ScalarFormatter):
    def _set_format(self, vmin, vmax):  # Override function that finds format to use.
        self.format = "%1.2f"  # Give format here


beta_t_d200 = np.array(pd.read_csv("split/beta_t_d200.csv", header=None))
beta_t_d500 = np.array(pd.read_csv("split/beta_t_d500.csv", header=None))
beta_t_d1000 = np.array(pd.read_csv("split/beta_t_d1000.csv", header=None))

cd_d200 = np.array(pd.read_csv("split/cd_d200.csv", header=None))[0]
cd_d500 = np.array(pd.read_csv("split/cd_d500.csv", header=None))[0]
cd_d1000 = np.array(pd.read_csv("split/cd_d1000.csv", header=None))[0]

columns_d200 = np.array(pd.read_csv("split/columns_d200.csv", header=None)).ravel()
columns_d500 = np.array(pd.read_csv("split/columns_d500.csv", header=None)).ravel()
columns_d1000 = np.array(pd.read_csv("split/columns_d1000.csv", header=None)).ravel()

num_sig_d200 = np.vstack((np.sum((np.abs(beta_t_d200) > cd_d200)[:4], 0), np.sum((np.abs(beta_t_d200) > cd_d200)[5:], 0)))
num_sig_d500 = np.vstack((np.sum((np.abs(beta_t_d500) > cd_d500)[:4], 0), np.sum((np.abs(beta_t_d500) > cd_d500)[5:], 0)))
num_sig_d1000 = np.vstack((np.sum((np.abs(beta_t_d1000) > cd_d1000)[:4], 0), np.sum((np.abs(beta_t_d1000) > cd_d1000)[5:], 0)))

plt.style.use('ggplot')

fig, ax1 = plt.subplots(1, 1, figsize=(4, 3), constrained_layout=True)

clr = plt.rcParams['axes.prop_cycle'].by_key()['color']

ax1.plot(np.arange(1, 7), num_sig_d200[0], alpha=0.7, label=r'$d=200$, relevant', marker='o', markerfacecolor='None', color=clr[0])
ax1.plot(np.arange(1, 7), num_sig_d500[0], alpha=0.7, label=r'$d=500$, relevant', marker='v', markerfacecolor='None', color=clr[1])
ax1.plot(np.arange(1, 7), num_sig_d1000[0], alpha=0.7, label=r'$d=1000$, relevant', marker='s', markerfacecolor='None', color=clr[5])
ax1.plot(np.arange(1, 7), num_sig_d200[1], '--', alpha=0.7, label=r'$d=200$, spurious', marker='o', markerfacecolor='None', color=clr[0])
ax1.plot(np.arange(1, 7), num_sig_d500[1], '--', alpha=0.7, label=r'$d=500$, spurious', marker='v', markerfacecolor='None', color=clr[1])
ax1.plot(np.arange(1, 7), num_sig_d1000[1], '--', alpha=0.7, label=r'$d=1000$, spurious', marker='s', markerfacecolor='None', color=clr[5])
ax1.axhline(y=4, color=clr[3], linestyle='--', alpha=0.4, linewidth=0.7)
ax1.set_ylim(-0.5, 4.5)
ax1.set_xlabel(r'$\tau$')
ax1.set_ylabel('Number of sig. variables')
ax1.xaxis.set_major_locator(IndexLocator(base=1, offset=0))
ax1.legend()
ax1.grid(True)

fig.savefig('num_split.pdf')

plt.style.use('ggplot')

fig, ax2 = plt.subplots(1, 1, figsize=(5, 3), constrained_layout=True)

clr = plt.rcParams['axes.prop_cycle'].by_key()['color']

for ct, me, y in zip(beta_t_d1000[:4, 1], [cd_d1000[1]] * len(beta_t_d1000[:4, 1]), range(len(beta_t_d1000[:4, 1]))[::-1]):
    ax2.plot((ct - me, ct + me), (y, y), '_-', color=clr[4], markersize=8, mew=1.5)
    ax2.plot(ct, y, 'o', color=clr[5], markersize=5)
    ax2.axvline(x=0, color=clr[3], linestyle='--', alpha=0.4, linewidth=0.7)
ax2.grid(True)
ax2.set_yticks(range(len(beta_t_d1000[:4, 1])))
labels = [item.get_text() for item in ax2.get_yticklabels()]
labels = columns_d1000[:4][::-1]
ax2.set_yticklabels(labels)

fig.savefig('ci_d1000_split.pdf')
