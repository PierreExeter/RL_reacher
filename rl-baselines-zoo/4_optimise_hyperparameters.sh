#!/bin/bash

ntrials=2                            
nsteps=1000           
njobs=-1                 
sampler="tpe"
pruner="median"
log_dir="logs/opti_2t_1K_Reacher2Dof-v0/"
env="Reacher2Dof-v0"
# env_her="NOT IMPLEMENTED"


echo "A2C OPTI"
python 2_train.py --algo a2c --env ${env} -n ${nsteps} --log-folder ${log_dir} -optimize --n-trials ${ntrials} --n-jobs ${njobs} --sampler ${sampler} --pruner ${pruner}  &> submission_logs/log_a2c_opti.run

echo "ACKTR OPTI"
python 2_train.py --algo acktr --env ${env} -n ${nsteps} --log-folder ${log_dir} -optimize --n-trials ${ntrials} --n-jobs ${njobs} --sampler ${sampler} --pruner ${pruner}  &> submission_logs/log_acktr_opti.run

echo "DDPG OPTI"
python 2_train.py --algo ddpg --env ${env} -n ${nsteps} --log-folder ${log_dir} -optimize --n-trials ${ntrials} --n-jobs ${njobs} --sampler ${sampler} --pruner ${pruner}  &> submission_logs/log_ddpg_opti.run

echo "PPO2 OPTI"
python 2_train.py --algo ppo2 --env ${env} -n ${nsteps} --log-folder ${log_dir} -optimize --n-trials ${ntrials} --n-jobs ${njobs} --sampler ${sampler} --pruner ${pruner}  &> submission_logs/log_ppo2_opti.run

echo "SAC OPTI"
python 2_train.py --algo sac --env ${env} -n ${nsteps} --log-folder ${log_dir} -optimize --n-trials ${ntrials} --n-jobs ${njobs} --sampler ${sampler} --pruner ${pruner}  &> submission_logs/log_sac_opti.run

echo "TD3 OPTI"
python 2_train.py --algo td3 --env ${env} -n ${nsteps} --log-folder ${log_dir} -optimize --n-trials ${ntrials} --n-jobs ${njobs} --sampler ${sampler} --pruner ${pruner}  &> submission_logs/log_td3_opti.run

echo "TRPO OPTI"
python 2_train.py --algo trpo --env ${env} -n ${nsteps} --log-folder ${log_dir} -optimize --n-trials ${ntrials} --n-jobs 1 --sampler ${sampler} --pruner ${pruner}  &> submission_logs/log_trpo_opti.run

echo "HER OPTI"
python3 2_train.py --algo her --env ${env_her} -n ${nsteps} --log-folder ${log_dir} -optimize --n-trials ${ntrials} --n-jobs ${njobs} --sampler ${sampler} --pruner ${pruner}  &> submission_logs/log_her_td3_opti.run


# python my_lib/clean_opti_params.py --folder ${log_dir}
# python my_lib/plot_opti_best.py
# python my_lib/plot_opti_report.py
