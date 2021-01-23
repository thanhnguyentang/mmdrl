#!/bin/bash 
echo "Welcome to MMDQN script!"
env_name="'Breakout'"
env=Breakout
policy="'eps_greedy'"
N=200
agent=mmd_dqn_1


python main.py --env $env --agent_id mmd --agent_name $agent --gin_files ./configs/mmd_atari.gin \
    --gin_bindings "
    create_atari_environment.game_name = $env_name
    MMDAgent.bandwidth_selection_type = 'mixture' #[med/annealing/mixture/const]
    MMDAgent.policy = $policy
    MMDAgent.target_estimator = 'mean' 
    MMDAgent.num_atoms = $N
    " --run_id 0
