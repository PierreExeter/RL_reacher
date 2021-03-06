#!/bin/bash


nsteps=3000     # each episode last 150 timesteps, so evaluating for 3000 timeteps = 20 episodes
nb_seeds=2
opti_dir="logs/opti_2t_1K_Reacher2Dof-v0/"
log_dir="logs/train_0.1M_Reacher2Dof-v0/"
# log_dir2="logs/train_10K_Reacher2Dof-v0/"
save_dir="experiment_reports/train_0.1M_Reacher2Dof-v0/"
# save_dir2="experiment_reports/comp_0.2M_widowx_reacher-v5-v7_SONIC/"
env="Reacher2Dof-v0"
# env_her="NOT IMPLEMENTED"
log_rand="logs/train_0.1M_Reacher2Dof-v0_random/"
echo "ENV: ${env}"

# STEP 1
# for each seed experiment, evaluate and calculate mean reward (+std), train walltime, success ratio and average reach time
# + plot


for ((i=1;i<${nb_seeds}+1;i+=1))
do
    echo "A2C $i"
    python 3_enjoy.py --algo a2c --env ${env} -f ${log_dir} --exp-id $i --no-render -n ${nsteps}
    python my_lib/plot_1seed.py -f ${log_dir}a2c/${env}_$i/

    echo "ACKTR $i"
    python 3_enjoy.py --algo acktr --env ${env} -f ${log_dir} --exp-id $i --no-render -n ${nsteps}
    python my_lib/plot_1seed.py -f ${log_dir}acktr/${env}_$i/

    echo "DDPG $i"
    python 3_enjoy.py --algo ddpg --env ${env} -f ${log_dir} --exp-id $i --no-render -n ${nsteps}
    python my_lib/plot_1seed.py -f ${log_dir}ddpg/${env}_$i/

    echo "PPO2 $i"
    python 3_enjoy.py --algo ppo2 --env ${env} -f ${log_dir} --exp-id $i --no-render -n ${nsteps}
    python my_lib/plot_1seed.py -f ${log_dir}ppo2/${env}_$i/

    echo "SAC $i"
    python 3_enjoy.py --algo sac --env ${env} -f ${log_dir} --exp-id $i --no-render -n ${nsteps}
    python my_lib/plot_1seed.py -f ${log_dir}sac/${env}_$i/

    echo "TD3 $i"
    python 3_enjoy.py --algo td3 --env ${env} -f ${log_dir} --exp-id $i --no-render -n ${nsteps}
    python my_lib/plot_1seed.py -f ${log_dir}/td3/${env}_$i/

    echo "TRPO $i"
    python 3_enjoy.py --algo trpo --env ${env} -f ${log_dir} --exp-id $i --no-render -n ${nsteps}
    python my_lib/plot_1seed.py -f ${log_dir}trpo/${env}_$i/

    # echo "HER $i"
    # python 3_enjoy.py --algo her --env ${env_her} -f ${log_dir} --exp-id $i --no-render -n ${nsteps}
    # python my_lib/plot_1seed.py -f ${log_dir}her/${env_her}_$i/

done



# evaluate random policy
# python 3_enjoy.py --random-pol True --env ${env} -f ${log_dir} --exp-id -1 --no-render -n ${nsteps} --log-dir-random ${log_rand} # if random-pol = True, it doesn't matter to specify -f ${log_dir}
# python my_lib/clean_random_training.py -f ${log_rand}

# record video
# python -m utils.record_video --algo td3 --env ${env} -n 400 -f ${log_dir}td3/${env}_1/



# STEP 2: Get the mean of the reward and wall train time of all the seed runs in the experiment

python my_lib/plot_experiment.py -f ${log_dir}a2c/ --env ${env} --nb-seeds ${nb_seeds} -n ${nsteps}
python my_lib/plot_experiment.py -f ${log_dir}acktr/ --env ${env} --nb-seeds ${nb_seeds} -n ${nsteps}
python my_lib/plot_experiment.py -f ${log_dir}ddpg/ --env ${env} --nb-seeds ${nb_seeds} -n ${nsteps}
python my_lib/plot_experiment.py -f ${log_dir}ppo2/ --env ${env} --nb-seeds ${nb_seeds} -n ${nsteps}
python my_lib/plot_experiment.py -f ${log_dir}sac/ --env ${env} --nb-seeds ${nb_seeds} -n ${nsteps}
python my_lib/plot_experiment.py -f ${log_dir}td3/ --env ${env} --nb-seeds ${nb_seeds} -n ${nsteps}
python my_lib/plot_experiment.py -f ${log_dir}trpo/ --env ${env} --nb-seeds ${nb_seeds} -n ${nsteps}
# python my_lib/plot_experiment.py -f ${log_dir}her_sac/ --env ${env_her} --nb-seeds ${nb_seeds} -n ${nsteps}
# python my_lib/plot_experiment.py -f ${log_dir}her_td3/ --env ${env_her} --nb-seeds ${nb_seeds} -n ${nsteps}


# # STEP 3: Plot learning curves and training stats for writing report
# python my_lib/plot_experiment_comparison.py -f ${log_dir} -s ${save_dir} -r ${log_rand}
 
## STEP 4: compare learning curves between 2 envs
# python my_lib/plot_comp_envs_learning_curves.py -f1 ${log_dir} -f2 ${log_dir2} -s ${save_dir2}

# # IF OPTIMISATION
# # python my_lib/plot_opti_report.py
# python my_lib/plot_opti_best.py
# python my_lib/clean_opti_params.py -f ${opti_dir}a2c/
# python my_lib/clean_opti_params.py -f ${opti_dir}acktr/
# python my_lib/clean_opti_params.py -f ${opti_dir}ddpg/
# python my_lib/clean_opti_params.py -f ${opti_dir}ppo2/
# python my_lib/clean_opti_params.py -f ${opti_dir}sac/
# python my_lib/clean_opti_params.py -f ${opti_dir}td3/
# python my_lib/clean_opti_params.py -f ${opti_dir}trpo/
# python my_lib/clean_opti_params.py -f ${opti_dir}her/


# STEP 4: view trained agent

# python 3_enjoy.py --algo a2c --env ${env} -f ${log_dir} --exp-id 1 -n ${nsteps} --render-pybullet True
# python 3_enjoy.py --algo acktr --env ${env} -f ${log_dir} --exp-id 7 -n ${nsteps} --render-pybullet True
# python 3_enjoy.py --algo ddpg --env ${env} -f ${log_dir} --exp-id 9 -n ${nsteps} --render-pybullet True
# python 3_enjoy.py --algo ppo2 --env ${env} -f ${log_dir} --exp-id 9 -n ${nsteps} --render-pybullet True
# python 3_enjoy.py --algo sac --env ${env} -f ${log_dir} --exp-id 10 -n ${nsteps} --render-pybullet True
# python 3_enjoy.py --algo td3 --env ${env} -f ${log_dir} --exp-id 3 -n ${nsteps} --render-pybullet True
# python 3_enjoy.py --algo trpo --env ${env} -f ${log_dir} --exp-id 6 -n ${nsteps} --render-pybullet True
# python 3_enjoy.py --algo her --env ${env_her} -f ${log_dir} --exp-id 9 -n ${nsteps} --render-pybullet True   # her + sac
# python 3_enjoy.py --algo her --env ${env_her} -f ${log_dir} --exp-id 2 -n ${nsteps} --render-pybullet True  # her + td3

# python 3_enjoy.py --random-pol True --env ${env} -f ${log_dir} --exp-id -1 --no-render -n ${nsteps} --render-pybullet True
