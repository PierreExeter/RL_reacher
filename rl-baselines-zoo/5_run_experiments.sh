#!/bin/bash

# experiments over 10 initialisation seeds

nsteps=100000    
log_dir="logs/train_0.1M_Reacher2Dof-v0"
log_dir_rand="logs/train_0.1M_Reacher2Dof-v0_random"
env="Reacher2Dof-v0"
nseeds=2
# env_her="NOT IMPLEMENTED"


for ((i=0;i<${nseeds};i+=1))
do
    echo "A2C $i"
    python 2_train.py --algo a2c --env ${env} -n ${nsteps} --seed $i --log-folder ${log_dir} &> submission_logs/log_a2c_0$i.run

    echo "ACKTR $i"
    python 2_train.py --algo acktr --env ${env} -n ${nsteps} --seed $i --log-folder ${log_dir} &> submission_logs/log_acktr_0$i.run

    echo "DDPG $i"
    python 2_train.py --algo ddpg --env ${env} -n ${nsteps} --seed $i --log-folder ${log_dir} &> submission_logs/log_ddpg_0$i.run

    echo "PPO2 $i"
    python 2_train.py --algo ppo2 --env ${env} -n ${nsteps} --seed $i --log-folder ${log_dir} &> submission_logs/log_ppo2_0$i.run

    echo "SAC $i"
    python 2_train.py --algo sac --env ${env} -n ${nsteps} --seed $i --log-folder ${log_dir} &> submission_logs/log_sac_0$i.run

    echo "TD3 $i"
    python 2_train.py --algo td3 --env ${env} -n ${nsteps} --seed $i --log-folder ${log_dir} &> submission_logs/log_td3_0$i.run

    echo "TRPO $i"
    python 2_train.py --algo trpo --env ${env} -n ${nsteps} --seed $i --log-folder ${log_dir} &> submission_logs/log_trpo_0$i.run

    echo "HER $i"
    python 2_train.py --algo her --env ${env_her} -n ${nsteps} --seed $i --log-folder ${log_dir} &> submission_logs/log_her_sac_0$i.run
done


python 7_run_random_policy.py --env ${env} -n ${nsteps} --folder ${log_dir_rand} --nb-seeds ${nseeds}