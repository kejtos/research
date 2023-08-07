import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

N_RANKS = 200
N_RANKS_PLOT = 35
PERCENTS = np.array([1, 1, 0.97826086, 0.94501278, 0.90281329, 0.86061381, 0.81969309, 0.77877237, 0.73401534, 0.68670076, 0.65217391, 0.61764705, 0.60230179, 0.56521739, 0.53580562, 0.48209718, 0.45907928, 0.43606138,
                     0.42071611, 0.39386189, 0.36700767, 0.34015345, 0.30946291, 0.28132992, 0.26982097, 0.24424552, 0.23145780, 0.22122762, 0.20588235, 0.19693094, 0.18925831, 0.17774936, 0.16240409, 0.15217391, 0.14322250, 0.13682864,
                     0.12404092, 0.11892583, 0.11125319, 0.10485933, 0.10230179, 0.09846547, 0.09335038, 0.08951406, 0.08951406, 0.08823529, 0.08567774, 0.08312020, 0.07928388, 0.07672634, 0.07544757])


ranks = np.arange(1, N_RANKS)
log_ranks = np.log(ranks)

log_diffs_1 = np.empty((N_RANKS-1))

for i in range(1, N_RANKS-1):
    log_diffs_1[i] = log_ranks[i] - log_ranks[0]

diffs_in_log_diffs = np.diff(log_diffs_1[1:])

table1 = []
table2 = []
x = []
for i in range(1,50):
    table1.append((log_diffs_1[i+1:]-log_diffs_1[1:-i])*0.508)
    table2.append(np.exp(table1[i-1]))
    x.append(np.arange(2+i, table1[i-1].shape[0]+2+i))

maxes = [seq[0] for seq in table1]
table1[0]

flat1 = ranks*0.020
flat2 = np.exp(ranks)

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

plt.figure(figsize=(10, 8))

text_coos = []
ises = []
for i in range(0, 12, 3):
    ises.append(i)
    y = table1[i][:N_RANKS_PLOT]
    cross_rank_mask = y > (i+1)*0.02
    y2 = y[cross_rank_mask]
    y2[0], y2[-1] = 5, 0
    # cross_rank = np.where(cross_rank_mask)[0].shape[0] + 3 + i
    mask = table1[i] > (i+1)*0.02
    anti_mast = ~mask
    cross_rank = x[i][mask]
    left_line_x = np.where(mask)[0].shape[0]*[3+i]
    right_line_x = np.where(mask)[0].shape[0]*[np.where(anti_mast)[0][0]+2+i]

    plt.subplot(2, 2, int(i/3+1))
    plt.fill_between(x[i][mask], 0, 5, color='#007FFF', alpha=0.1)
    plt.step(x[i][:N_RANKS_PLOT], y, where='post', color=colors[0], label='Change using LOGDIF')
    plt.fill_between(x[i][mask], table1[i][mask], (i+1)*0.02, step="post", color=colors[0], alpha=0.35)
    plt.plot(x[i][:N_RANKS_PLOT], (N_RANKS_PLOT)*[(i+1)*0.02+0.0005], color=colors[1], label='Change using RANKDIF')
    plt.plot(left_line_x, y[cross_rank_mask], color=colors[0])

    plt.plot(right_line_x, y2, markersize=8, linestyle='--', dashes=(5, 3), color='lightskyblue', label='Turning point')
    # plt.text(cross_rank[-1] + 0.3, (i+1)*0.02 + 0.005, f'Rank: {cross_rank[-1]}', fontsize=10)
    # plt.text(15, 0.7, f'{(1-PERCENTS[cross_rank[-1]])*100:.1f}% of games', fontsize=10, color='#FFC107')
    text_coos.append(cross_rank[-1])
    plt.xlabel('Rank of the weaker team after change')
    plt.ylabel('Change in log(odds) of favorite winning')
    plt.title(f'Change in RANKDIF by {i+1}')
    plt.legend(loc='best', prop={'size': 8}, handlelength=0.5, markerscale=0.5)

    add_ticks = []
    if left_line_x[0] not in [10,20,30,40]:
        for _ in range(5):
            add_ticks.append(left_line_x[0])

    if right_line_x[0] not in [10,20,30]:
        add_ticks.append(right_line_x[0])

    tick_locations = np.concatenate(([10,20,30,40], [*add_ticks]))
    plt.gca().xaxis.set_major_locator(ticker.FixedLocator(tick_locations))

for i, ax in enumerate(plt.gcf().get_axes()):
    ax.set_xlim(ises[i]+2, ises[i]+34)
    ax.set_ylim(0, np.ceil(y[0]*1.05))
    ax.text(0.3, 0.65, f'{(1-PERCENTS[text_coos[i]])*100:.1f}% of games', fontsize=10, color='#1f77b4', transform=ax.transAxes)
    ax.text(text_coos[i] + 0.3, (i+1)*0.02 + 0.005, f'Rank: {text_coos[i]}', fontsize=10, transform=ax.transAxes)

plt.subplots_adjust(hspace=0.35)
# plt.tight_layout()
plt.show()
