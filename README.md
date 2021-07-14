# MMDRL 

This is the *official* code base for our AAAI'21 paper, "**Distributional Reinforcement Learning via Moment Matching**", [arXiv](https://arxiv.org/abs/2007.12354), [AAAI Proceeding](https://ojs.aaai.org/index.php/AAAI/article/view/17104). 

![Alt Text](https://github.com/thanhnguyentang/mmdrl/blob/master/raw_result_data/My%20Movie.gif)


## Dependencies 
* tensorflow==1.15
* tensorflow-probability==0.8.0
* atari-py 
* gym==0.12.1
* gin-config 
* cv2 
* [Dopamine](https://github.com/google/dopamine) framework (already integrated into this code base)

## Instruction 
* To train and evaluate `MMDQN` in an Atari game with the default settings, use the following example command (from within the main directory `mmdrl`):
    ```
    python main.py --env Breakout --agent_id mmd \
    --agent_name mmd_dqn_1 --gin_files ./configs/mmd_atari.gin
    ```
where `env` is an Atari game name, `agent_id` is a registered agent id ('mmd' for MMDQN), `agent_name` for the directory name to save the agent training and evaluation results, and `gin_files` is a path to the hyperparameter configuration (in `gin` format). 

* For convenience, we can directly modify the bash script `run_mmdqn.sh` for various hyperparameter settings and run the bash via 
    ```
    chmod +x ./run_mmdqn.sh; ./run_mmdqn.sh 
    ```

### Main variables in MMDQN code:
* `env`: One of the 57 Atari games  
* `agent_id`: ['mmd', 'quantile', 'iqn'], agent id (for `MMDQN`, `QR-DQN` and `IQN`)
* `agent_name`: str, the experiment log saved to `./results/<env>/<agent_name>`
* `policy`: ['eps_greedy', 'ucb', 'ps'], policy used by the agent (epsilon-greey, UCB or Thompson sampling)
* `num_atoms`: int, the number of particles N 
* `bandwidth_selection_type`: 'mixture', the method for kernel bandwidth selection  
* `gin_files`: str, the path to a gin file containing the hyperparameters of the agent 
* `gin_bindings`: str, overwrite hyperparameters in a gin file 

### An overview of the MMDRL codebase: 
* `mmd_agent.py`: An implementation of `MMDQN` agent 
* `quantile_agent.py`: An implementation of `QR-DQN`  
* `main.py`: Main file to train and evaluate an agent  
* `run_mmdqn.sh`: A bash script to train and evaluate `MMDQN` agent 
* `configs/mmd_atari_gin`: Hyperparameters of `MMDQN` agent 
* `dopamine/`: The code base of Dopamine framework  


## Raw Result Data 

For the ease of re-presenting our experimental result, I have uploaded the raw result data of our algorithm `MMDQN` (and `QR-DQN`) to `/raw_result_data`

* `/raw_result_data/mmdqn_train_episode_return.csv`: The raw scores of `MMDQN` during training for the Atari games. 
* `/raw_result_data/mmdqn_eval_episode_return.csv`: The raw scores of `MMDQN` during evaluation for the Atari games. 
* `/raw_result_data/qr_train_episode_return.csv`: The raw scores of `QR-DQN` during training for the Atari games. 
* `/raw_result_data/qr_train_episode_return.csv`: The raw scores of `QR-DQN` during evaluation for the Atari games. 

`MMDQN`is trained in each of the 55 Atari games for three independent times (three random seeds). Each line of each of the `csv` files above contains the name of the game and a series of `200` numbers that represent the score that `MMDQN` obtains after each iteration. I have also uploaded the raw result data of `QR-DQN` in `/raw_result_data/qr_train_episode_return.csv`and `/raw_result_data/qr_eval_episode_return.csv`.

## Bibliography  

```
@article{Nguyen-Tang_Gupta_Venkatesh_2021,
title={Distributional Reinforcement Learning via Moment Matching}, 
volume={35}, 
url={https://ojs.aaai.org/index.php/AAAI/article/view/17104},
number={10}, 
journal={Proceedings of the AAAI Conference on Artificial Intelligence}, 
author={Nguyen-Tang, Thanh and Gupta, Sunil and Venkatesh, Svetha}, 
year={2021}, 
month={May}, 
pages={9144-9152} }
```


