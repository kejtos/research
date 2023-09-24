import numpy as np
import research.functions as functions
from research.functions import *
import importlib
import numexpr as ne
from scipy.stats import percentileofscore as perc_sc

importlib.reload(functions)
from research.functions import *

## --- SETUP --- ###
percentiles = np.array([95, 99]) # percentiles for plots
reps = 1_000 # number of permutations
k = 3 # length of strings

directory = 'D:\\Plocha\\Doktorát\\Hot hand\\Data\\'
rng = np.random.default_rng(42)
tile_frenzies = ['Tile Frenzy', 'Tile Frenzy Mini']
scen_types = 'flicks' # flicks, tiles, all
#------------------------------------------------------------------------------------------------------------------------#

## --- RETRIEVING THE DATA --- ###
list_zipfiles = get_zipfile_names(directory=directory)

list_sets_scenario_names = []
players = []
n_scen = 0
for player, zipname in enumerate(list_zipfiles, 1):
    list_of_names = get_scenario_names(zipname=zipname)
    scenario_names = sorted({re.search('(.*)(?= - Challenge)', name)[0] for name in list_of_names})
    players.append(player)
    list_sets_scenario_names.append(scenario_names)
    n_scen += len(list_of_names)
dict_scenario_names = {player: name for player, name in zip(players, list_sets_scenario_names)}


if scen_types in ('flicks', 'all'):
    list_of_regs = ['(?i)reflex', '(?i)flick', '(?i)microshot','(?i)micro shot','(?i)microshots','(?i)micro shots', '(?i)speed', '(?i)timing', '(?i)popcorn']
    flick_scens = set()
    for key, names in dict_scenario_names.items():
        for name in names:
            for regex in list_of_regs:
                if re.search(regex, name):
                    flick_scens.add(name)

    flick_delete = [r'Reflex Poke-Micro++ Regenerating Goated', r'VT Multiclick 120 Intermediate Speed Focus LG56', r'Close flick v2', r"KovaaK's League S6 - beanClick 250% Speed", r'Flicker Plaza No Strafes Dash Master',
                    r'Floating Heads Timing Easy', r'Fireworks Flick [x-small]', r'Target Acquisition Flick Small 400ms', r'Houston Outlaws - Speed Switching', r'psalmflick TPS Strafing',
                    r'VT Pasu Rasp Intermediate Speed Focus LG56', r'Target Acquisition Flick Easy', r'Valorant Small Flicks', r'Flicker Plaza Hard', r'Flicker XYZ', r'Flip off Flick Random',
                    r'Jumbo Tile Frenzy Flick 180', r'Flicker Plaza', r'Tamspeed 2bpes', r'e1se Braking Reflex Flick EASY', r'Target Acquisition Flick', r'VSS GP9 +50% speed', 'Floating Heads Timing 400% no mag',
                    r'Reflex Tracking a+', r'Target Acquisition Flick 350ms', r'Flicker XYZ Easy', r'FuglaaXYZ Voltaic No Blinks but the bot has taken a speed potion', r'KovaaK_s League S6 - beanClick 250_ Speed',
                    r'Tamspeed 2bp Bronze', r'Floating Heads Timing 400_', r'Ground Plaza Voltaic 1 Invincible Always Dash Speed No Dash', r'Flicker Plaza Grandmaster', r'1wall2targets_smallflicks 60s',
                    r'PlasmaFlick 180', r'Reactive Flick to Track', r'Target Acquisition Flick Small', r'Floating Heads Timing 400%', r'Eclipse Flick Warmup', r'Popcorn Goated TI easy', r'Popcorn Gauntlet Raspberry Master']

    flick_scens = [scen for scen in flick_scens if not re.search('(?i)tracking', scen)]
    flick_scens = [scen for scen in flick_scens if not re.search('(?i)smooth', scen)]
    flick_scens = [scen for scen in flick_scens if scen not in flick_delete]
    if scen_types == 'all':
        all_scens = flick_scens + tile_frenzies
        df = create_df_from_zipcsvs(list_zipfiles=list_zipfiles, names_of_scenarios=all_scens)
    else:
        df = create_df_from_zipcsvs(list_zipfiles=list_zipfiles, names_of_scenarios=flick_scens)
elif scen_types == 'tiles':
    df = create_df_from_zipcsvs(list_zipfiles=list_zipfiles, names_of_scenarios=tile_frenzies)

np_shots = df.loc[(df['Accuracy'] <= 0.9) & (df['Accuracy'] >= 0.1), 'String_of_shots'].to_numpy()
n_runs = np_shots.shape[0]
#------------------------------------------------------------------------------------------------------------------------#

### --- CALCULATIONS --- ###
faulty_shots = []
p_vals_dep = np.zeros((n_runs, 3))
D_hat = np.zeros(n_runs)
P_hat_p = np.zeros(n_runs)
entropies = np.zeros(n_runs)
mask = np.zeros((n_runs), dtype='bool')
p_vals_ent = np.zeros(n_runs)
t_stats_Pp_dist = np.zeros((n_runs,reps))
t_stats_D_dist = np.zeros((n_runs,reps))
entropy_dists = np.zeros((n_runs,reps))

for i in range(n_runs):
    arr = np_shots[i,].copy()
    # DEPENDENCE SIMULTANEOUS
    try:
        D_had_p_value, P_hat_p_p_value = get_p_values(arr, reps=reps, k=k)
        p_vals_dep[i,0] = D_had_p_value
        p_vals_dep[i,1] = P_hat_p_p_value
        mask[i] = True
    except ValueError:
        p_vals_dep[i,0] = np.nan
        p_vals_dep[i,1] = np.nan
        faulty_shots.append(np_shots[i,])
    p_vals_dep[i,2] = i

    if mask[i]:
        # DEPENDENCE JOINT
        repos = 3
        D_hat_dist, P_hat_p_dist, _ = get_D_hat_P_hat_p_dists(shots=arr, k=k, reps=reps*repos, seed=42)
        while D_hat_dist.size < reps or P_hat_p_dist.size < reps:
            D_hat_dist, P_hat_p_dist, _ = get_D_hat_P_hat_p_dists(shots=arr, k=k, reps=reps*repos, seed=42)
            repos += 1
        t_stats_D_dist[i,] = D_hat_dist[:reps]
        t_stats_Pp_dist[i,] = P_hat_p_dist[:reps]
        D_hat[i], P_hat_p[i], _ = get_D_hat_P_hat_p(shots=arr, reps=1, k=k)
        # ENTROPY SIMULTANEOUS
        H_s = entropy(arr)
        H_s_dist = entropy_dist(shots=arr, reps=reps, seed=42)
        p_vals_ent[i] = entropy_pval(stat=H_s, dist=H_s_dist)
        # ENTROPY JOINT
        entropy_dists[i,] = H_s_dist
        entropies[i,] = H_s

np_shots, t_stats_D_dist, t_stats_Pp_dist, D_hat, P_hat_p, p_vals_ent, entropy_dists, entropies = np_shots[mask], t_stats_D_dist[mask], t_stats_Pp_dist[mask], D_hat[mask], P_hat_p[mask], p_vals_ent[mask], entropy_dists[mask], entropies[mask]

n_runs = np_shots.shape[0]
p_vals_notnan = p_vals_dep[~np.isnan(p_vals_dep).any(axis=1)]

D_had_p_values = np.sort(p_vals_notnan[:,0])
P_hat_p_p_value = np.sort(p_vals_notnan[:,1])

mean_D_dist = np.mean(t_stats_D_dist, axis=0)
mean_Pp_dist = np.mean(t_stats_Pp_dist, axis=0)
mean_D = np.mean(D_hat, axis=0)
mean_Pp = np.mean(P_hat_p, axis=0)

D_hat_p_val = perc_sc(mean_D_dist, mean_D, kind='rank')
P_hat_p_val = perc_sc(P_hat_p_dist, mean_Pp, kind='rank')
D_hat_p_val_j = ne.evaluate('(100-D_hat_p_val)/100')
P_hat_p_p_val_j = ne.evaluate('(100-P_hat_p_val)/100')

mean_entropy_dist = np.mean(entropy_dists, axis=0)
mean_entropy = np.mean(entropies, axis=0)
entropy_p_val = perc_sc(mean_entropy_dist, mean_entropy, kind='rank')
entropy_p_val_j = ne.evaluate('(100-entropy_p_val)/100')

hot_hands_D = get_n_rejets(p_values=D_had_p_values, alpha=0.05, sidak=False)
hot_hands_P_p = get_n_rejets(p_values=P_hat_p_p_value, alpha=0.05, sidak=False)
hot_hands_D_sidak = get_n_rejets(p_values=D_had_p_values, alpha=0.05, sidak=True)
hot_hands_P_p_sidak = get_n_rejets(p_values=P_hat_p_p_value, alpha=0.05, sidak=True)
hot_hands_D_FDR = get_FDR(p_values=D_had_p_values, alpha=0.05)
hot_hands_P_p_FDR = get_FDR(p_values=P_hat_p_p_value, alpha=0.05)
hot_hands_ent = get_n_rejets(p_values=p_vals_ent, alpha=0.05, sidak=False)
hot_hands_ent_sidak = get_n_rejets(p_values=p_vals_ent, alpha=0.05, sidak=True)
hot_hands_ent_FDR = get_FDR(p_values=p_vals_ent, alpha=0.05)
#------------------------------------------------------------------------------------------------------------------------#

### --- PRINTS --- ###
print(f'''\n
Tested runs:    {p_vals_notnan.shape[0]}\n

<<< --- DEPENDENCE --- >>>\n

SIMULTANEOUS
 Individual   <D, P>:   <{hot_hands_D}, {hot_hands_P_p}>
       FWER   <D, P>:   <{hot_hands_D_sidak}, {hot_hands_P_p_sidak}>
        FDR   <D, P>:   <{hot_hands_D_FDR}, {hot_hands_P_p_FDR}>\n

JOINT
Joint P <t, p-value>: <{mean_Pp:.4f}, {D_hat_p_val_j:.4f}>
Joint D <t, p-value>: <{mean_D:.4f}, {P_hat_p_p_val_j:.4f}>\n

<<< --- ENTROPY --- >>>\n

SIMULTANEOUS
          Individual: {hot_hands_ent}
                FWER: {hot_hands_ent_sidak}
                 FDR: {hot_hands_ent_FDR}\n

JOINT
  Joint <t, p-value>: {mean_entropy:.4f}, {entropy_p_val_j:.4f}
''')
#------------------------------------------------------------------------------------------------------------------------#

### --- SERIAL DEPENDENCE PLOT --- ###
arr = np_shots[324,]
arr

D_hat, P_hat_p, p = get_D_hat_P_hat_p(shots=arr, reps=1, k=k)
D_hat_dist, P_hat_p_dist, _ = get_D_hat_P_hat_p_dists(shots=arr, k=k, reps=reps, seed=42)

plt_D_hat, ax_D_hat = plot_stat_dist(stat_distribution=D_hat_dist, stat=D_hat, percentiles=percentiles, bins=50)
ax_D_hat.axis([-0.0, 1.2, 0, 17000])
ax_D_hat.set_title(r'Distribution of a plug-in estimator $\widehat{D}_{73,2}(X_{20})$ for a string of a scenario Reflex Flick - Fair', fontsize=15)
ax_D_hat.set_xlabel(r'$\widehat{D}_{73,2}(X_{20})$', fontsize=12)
ax_D_hat.set_ylabel('Frequency', fontsize=12)

ax_D_hat.text(0.86, 8000, 'Value of the estimator', fontsize=14, color='#599750', weight='bold')
ax_D_hat.text(0.875, 2100, r'$\mathbf{\alpha = 0.05}$', fontsize=14, color='#FFCF9C')
ax_D_hat.text(0.92, 700, r'$\mathbf{\alpha = 0.01}$', fontsize=14, color='#FF7878')
plt_D_hat.show()

plt_P_hat_p, ax_P_hat_p = plot_stat_dist(stat_distribution=P_hat_p_dist, stat=P_hat_p, percentiles=percentiles, bins=50)
ax_P_hat_p.set_title(r'Distribution of a plug-in estimator $\widehat{P}_{137,2}(X_{324})-\widehat{p}_{137,324}$ for a string of a scenario Tile Frenzy', fontsize=15)
ax_P_hat_p.set_xlabel(r'$\widehat{P}_{137,2}(X_{324})-\widehat{p}_{137,324}$', fontsize=12)
ax_P_hat_p.set_ylabel('Frequency', fontsize=12)

ax_P_hat_p.text(0.02, 8000, 'Value of the estimator', fontsize=14, color='#599750', weight='bold')
ax_P_hat_p.text(0.065, 2100, r'$\mathbf{\alpha = 0.05}$', fontsize=14, color='#FFCF9C')
ax_P_hat_p.text(0.092, 700, r'$\mathbf{\alpha = 0.01}$', fontsize=14, color='#FF7878')
plt_P_hat_p.show()
#------------------------------------------------------------------------------------------------------------------------#

### --- ENTROPY PLOT --- ###
arr = np_shots[1,]

H_s = entropy(arr)
H_s_dist = entropy_dist(shots=arr, reps=reps, seed=42)
p_val = entropy_pval(stat=H_s, dist=H_s_dist)

plt_H_s, ax_H_s = plot_ent_dist(stat_distribution=H_s_dist, stat=H_s, percentiles=100-percentiles, bins=50)
ax_H_s.spines[['right', 'top']].set_visible(False)
ax_H_s.set_xlabel('Entropy', fontsize=28)
ax_H_s.set_ylabel('Frequency', fontsize=28)

ax_H_s.text(H_s+1, 65000, r'Entropy of $\mathbf{X_i}$', fontsize=28, color='#599750', weight='bold')
ax_H_s.text(H_s+1, 25000, r'$\mathbf{\alpha = 0.05}$', fontsize=28, color='#FFCF9C')
ax_H_s.text(H_s-5, 15000, r'$\mathbf{\alpha = 0.01}$', fontsize=28, color='#FF7878')

current_values = plt.gca().get_yticks()
plt.gca().set_yticklabels(['{:,.0f}'.format(x) for x in current_values])
plt.yticks(fontsize=24)
plt.xticks(fontsize=24)
ax_H_s.set_xlim(-115, -80)

plt_H_s.show()
#------------------------------------------------------------------------------------------------------------------------#

### --- SUMMARY --- ###
df.loc[(df['Accuracy'] <= 0.9) & (df['Accuracy'] >= 0.1), 'Accuracy'].describe()
df.loc[(df['Accuracy'] <= 0.9) & (df['Accuracy'] >= 0.1), 'Number_of_shots'].describe()

df['Number_of_shots'].describe()