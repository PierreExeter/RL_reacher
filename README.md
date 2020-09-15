# Reinforcement Learning for the Reacher Link environment
A benchmark of RL algorithms applied to the Reacher Link environment with 2 to 6 joints.


| | | |
|-|-|-|
| <img src="imgs/1joint.gif"/> 1 joint | <img src="imgs/2joint.gif"/> 2 joints |<img src="imgs/3joint.gif"/> 3 joints |
| <img src="imgs/4joint.gif"/> 4 joints | <img src="imgs/5joint.gif"/> 5 joints |<img src="imgs/6joint.gif"/> 6 joints |


## Installation

```bash
git clone https://github.com/PierreExeter/RL_reacher.git
cd RL_reacher/
conda env create -f environment.yml
conda activate reacher_link
pip install -e .   # Install the Pybullet environments locally
```

## Test installation

```bash
cd rl-baselines-zoo/
python 1_test_reacher2D.py
python 2_train.py --algo ppo2 --env Reacher6Dof-v0
```

## Optimise hyperparameters

```bash
./4_optimise_hyperparameters.sh
```
## Run experiments

```bash
./5_run_experiments.sh
```

## Evaluate trained policies

```bash
./6_get_results_exp.sh
```


TO DO NEXT: add evluation metrics (mean reward, success ratio and reach time)


## Tested on

- python 3.7
- conda 4.8.3
