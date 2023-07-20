import numpy as np
import functions
from functions import *
import importlib
import numexpr as ne

importlib.reload(functions)
from functions import *

from scipy.stats import percentileofscore as perc_sc

directory = 'D:\\Plocha\\Doktorát\\Hot hand\\Data\\'
rng = np.random.default_rng(42)
tile_frenzies = ['Tile Frenzy', 'Tile Frenzy Mini']

list_zipfiles = get_zipfile_names(directory = directory)

list_sets_scenario_names = []
players = []
n_scen = 0
for player, zipname in enumerate(list_zipfiles, 1):
    list_of_names = get_scenario_names(zipname = zipname)
    scenario_names = sorted({re.search('(.*)(?= - Challenge)', name)[0] for name in list_of_names})
    players.append(player)
    list_sets_scenario_names.append(scenario_names)
    n_scen += len(list_of_names)
dict_scenario_names = {player: name for player, name in zip(players, list_sets_scenario_names)}

list_of_regs = ['(?i)reflex', '(?i)flick', '(?i)microshot','(?i)micro shot','(?i)microshots','(?i)micro shots', '(?i)speed', '(?i)timing']
flick_scens = set()
for key, names in dict_scenario_names.items():
    for name in names:
        for regex in list_of_regs:
            if re.search(regex, name):
                flick_scens.add(name)

flick_delete = ['Reflex Poke-Micro++ Regenerating Goated', 'VT Multiclick 120 Intermediate Speed Focus LG56', 'Close flick v2', "KovaaK's League S6 - beanClick 250% Speed", 'Flicker Plaza No Strafes Dash Master', 'Floating Heads Timing Easy','Fireworks Flick [x-small]',
               'Target Acquisition Flick Small 400ms', 'Houston Outlaws - Speed Switching', 'psalmflick TPS Strafing', 'VT Pasu Rasp Intermediate Speed Focus LG56', 'Target Acquisition Flick Easy', 'Valorant Small Flicks', 'Flicker Plaza Hard',
               'Flicker XYZ', 'Flip off Flick Random', 'Jumbo Tile Frenzy Flick 180', 'Flicker Plaza', 'Tamspeed 2bpes', 'e1se Braking Reflex Flick EASY', 'Target Acquisition Flick', 'VSS GP9 +50% speed', 'Floating Heads Timing 400% no mag',
               'Reflex Tracking a+', 'Target Acquisition Flick 350ms', 'Flicker XYZ Easy','FuglaaXYZ Voltaic No Blinks but the bot has taken a speed potion', 'KovaaK_s League S6 - beanClick 250_ Speed', 'Tamspeed 2bp Bronze', 'Floating Heads Timing 400_',
               'Ground Plaza Voltaic 1 Invincible Always Dash Speed No Dash', 'Flicker Plaza Grandmaster', '1wall2targets_smallflicks 60s', 'PlasmaFlick 180', 'Reactive Flick to Track', 'Target Acquisition Flick Small', 'Floating Heads Timing 400%', 'Eclipse Flick Warmup']

flick_scens = [scen for scen in flick_scens if scen not in flick_delete]
all_scens = flick_scens + tile_frenzies
df = create_df_from_zipcsvs(list_zipfiles = list_zipfiles, names_of_scenarios = flick_scens)
df = create_df_from_zipcsvs(list_zipfiles = list_zipfiles, names_of_scenarios = tile_frenzies)
df = create_df_from_zipcsvs(list_zipfiles = list_zipfiles, names_of_scenarios = all_scens)

np_shots = df.loc[df['Accuracy'] <= 0.9, 'String_of_shots'].to_numpy()
df.loc[df['Accuracy'] <= 0.9, 'Accuracy'].describe()
df.loc[df['Accuracy'] <= 0.9, 'Number_of_shots'].describe()

df['Number_of_shots'].describe()

################################################################################################################################################################################################################################################
################################################################################################################################################################################################################################################
### --- SERIAL DEPENDENCY --- ###
shots = np_shots[324,]
shots
percentiles = np.array([95, 99])
reps = 10_000

k = 3

D_hat, P_hat_p, p = get_D_hat_P_hat_p(shots=shots, reps=1, k=k)
D_hat_dist, P_hat_p_dist, _ = get_D_hat_P_hat_p_dists(shots=shots, k=k, reps=reps, seed=42)

### --- EXAMPLE GRAPH --- ###
plt_D_hat, ax_D_hat = plot_stat_dist(stat_distribution=D_hat_dist, stat=D_hat, percentiles=percentiles, bins=50)
ax_D_hat.axis([-0.0, 1.2, 0, 17000])
ax_D_hat.set_title(r'Distribution of a plug-in estimator $\widehat{D}_{73,2}(X_{20})$ for a string of a scenario Reflex Flick - Fair', fontsize=15)
ax_D_hat.set_xlabel(r'$\widehat{D}_{73,2}(X_{20})$', fontsize=12)
ax_D_hat.set_ylabel('Frequency', fontsize=12)

ax_D_hat.text(0.86, 8000, 'Value of the estimator', fontsize=14, color='#599750', weight='bold')
ax_D_hat.text(0.875, 2100, r'$\mathbf{\alpha = 0.05}$', fontsize=14, color='#FFCF9C')
ax_D_hat.text(0.92, 700, r'$\mathbf{\alpha = 0.01}$', fontsize=14, color='#FF7878')
plt_D_hat.show()

plt_P_hat_p, ax_P_hat_p = plot_stat_dist(stat_distribution = P_hat_p_dist, stat = P_hat_p, percentiles = percentiles, bins = 50)
ax_P_hat_p.set_title(r'Distribution of a plug-in estimator $\widehat{P}_{137,2}(X_{324})-\widehat{p}_{137,324}$ for a string of a scenario Tile Frenzy', fontsize=15)
ax_P_hat_p.set_xlabel(r'$\widehat{P}_{137,2}(X_{324})-\widehat{p}_{137,324}$', fontsize=12)
ax_P_hat_p.set_ylabel('Frequency', fontsize=12)

ax_P_hat_p.text(0.02, 8000, 'Value of the estimator', fontsize=14, color='#599750', weight='bold')
ax_P_hat_p.text(0.065, 2100, r'$\mathbf{\alpha = 0.05}$', fontsize=14, color='#FFCF9C')
ax_P_hat_p.text(0.092, 700, r'$\mathbf{\alpha = 0.01}$', fontsize=14, color='#FF7878')
plt_P_hat_p.show()

faulty_shots = []
lenos = df.shape[0]
p_vals = np.zeros((lenos, 3))

for i in range(lenos):
    shots = np_shots[i,]
    try:
        D_had_p_value, P_hat_p_p_value = get_p_values(shots, reps = reps, k = k)
        p_vals[i,0] = D_had_p_value
        p_vals[i,1] = P_hat_p_p_value
    except ValueError:
        p_vals[i,0] = np.nan
        p_vals[i,1] = np.nan
        faulty_shots.append(np_shots[i,])
    p_vals[i,2] = i

p_vals_notnan = p_vals[~np.isnan(p_vals).any(axis=1)]

D_had_p_values = np.sort(p_vals_notnan[:,0])
P_hat_p_p_value = np.sort(p_vals_notnan[:,1])

np.min(P_hat_p_p_value)

hot_hands_D = get_n_rejets(p_values = D_had_p_values, alpha = 0.05, sidak = False)
hot_hands_P_p = get_n_rejets(p_values = P_hat_p_p_value, alpha = 0.05, sidak = False)
hot_hands_D_sidak = get_n_rejets(p_values = D_had_p_values, alpha = 0.05, sidak = True)
hot_hands_P_p_sidak = get_n_rejets(p_values = P_hat_p_p_value, alpha = 0.05, sidak = True)
hot_hands_D_FDR = get_FDR(p_values = D_had_p_values, alpha = 0.05)
hot_hands_P_p_FDR = get_FDR(p_values = P_hat_p_p_value, alpha = 0.05)

print(p_vals_notnan.shape[0])
print(hot_hands_D)
print(hot_hands_P_p)
print(hot_hands_D_sidak)
print(hot_hands_P_p_sidak)
print(hot_hands_D_FDR)
print(hot_hands_P_p_FDR)

#------------------------------------------------------------------------------------------------------------------------#
# Tohle je asi pokus o joint inferenci
D_p = {}
Pp_p = {}
for k in range(2, 4):
    lenos = df.shape[0]
    t_stats_Pp_dist = np.zeros((lenos,reps))
    t_stats_D_dist = np.zeros((lenos,reps))
    D_hat = np.zeros((lenos))
    P_hat_p = np.zeros((lenos))
    for i in range(lenos):
        shots = np_shots[i,]

        D_hat_dist, P_hat_p_dist, _ = get_D_hat_P_hat_p_dists(shots=shots, k=k, reps=reps*2, seed=42)

        repos = 3
        while D_hat_dist.size < reps or P_hat_p_dist.size < reps:
            D_hat_dist, P_hat_p_dist, _ = get_D_hat_P_hat_p_dists(shots=shots, k=k, reps=reps*repos, seed=42)
            repos += 1

        t_stats_D_dist[i,] = D_hat_dist[:reps]
        t_stats_Pp_dist[i,] = P_hat_p_dist[:reps]

        try:
            D_hat[i], P_hat_p[i], _ = get_D_hat_P_hat_p(shots = shots, reps = 1, k = k)
        except ValueError:
            try:
                D_hat[i] = get_D_hat_P_hat_p(shots = shots, reps = 1, k = k)[0]
            except ValueError:
                D_hat[i] = np.nan
            try:
                P_hat_p[i] = get_D_hat_P_hat_p(shots = shots, reps = 1, k = k)[1]
            except ValueError:
                P_hat_p[i] = np.nan

    mean_D_dist = np.mean(t_stats_D_dist, axis=0)
    mean_Pp_dist = np.mean(t_stats_Pp_dist, axis=0)
    mean_D = np.mean(D_hat, axis=0)
    mena_Pp = np.mean(P_hat_p, axis=0)


    D_had_al = perc_sc(mean_D_dist, mean_D, kind = 'rank')
    P_hat_p_al = perc_sc(P_hat_p_dist, mena_Pp, kind = 'rank')
    D_had_p_value = ne.evaluate('(100-D_had_al)/100')
    P_hat_p_p_value = ne.evaluate('(100-P_hat_p_al)/100')
    D_p[f'k = {k}'] = D_had_p_value.item()
    Pp_p[f'k = {k}'] = P_hat_p_p_value.item()
#------------------------------------------------------------------------------------------------------------------------#


################################################################################################################################################################################################################################################
################################################################################################################################################################################################################################################
### --- ENTROPY --- ### Zkusit se podívat na joint inferenci
percentiles = np.array([95, 99])
reps = 1_000_000
arr = np_shots[1,]
# arr = np.array([1,1,1,1,1,0,0,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,1,1,1,0,1,1,1,1,1,1,0,1,1,0,1,1,1,1,1,0,0,1,0,1,0,1,1,1,0,0,1,0,1,0,1,0,1,1,1,0,1,1,1,1,0,0,0,0])

H_s = entropy(arr)
H_s_dist = entropy_dist(shots = arr, reps = reps, seed = 42)
p_val = entropy_pval(stat = H_s, dist = H_s_dist)
p_val

plt_H_s, ax_H_s = plot_ent_dist(stat_distribution = H_s_dist, stat = H_s, percentiles = 100-percentiles, bins = 50)
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


faulty_shots = []
lenos = np_shots.shape[0]
p_vals = np.zeros((lenos))

for i in range(lenos):
    arr = np_shots[i,]
    H_s = entropy(arr)
    H_s_dist = entropy_dist(shots = arr, reps = reps, seed = 42)
    p_vals[i] = entropy_pval(stat = H_s, dist = H_s_dist)

np.min(p_vals)
p_vals[np.argsort(p_vals)[:7]]

hot_hands_ent = get_n_rejets(p_values = p_vals, alpha = 0.05, sidak = False)
hot_hands_ent_sidak = get_n_rejets(p_values = p_vals, alpha = 0.05, sidak = True)
hot_hands_ent_FDR = get_FDR(p_values = p_vals, alpha = 0.3)


print(f'''ENTROPY REJECTIONS
Individual: {hot_hands_ent}
      FWER: {hot_hands_ent_sidak}
       FDR: {hot_hands_ent_FDR}''')


