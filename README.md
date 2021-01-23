# MMDRL 

This repo is the *official* code base for our AAAI'21 ["Distributional Reinforcement Learning via Moment Matching"](https://arxiv.org/abs/2007.12354). 


## Dependencies 
* tensorflow==1.15
* tensorflow-probability==0.8.0
* atari-py 
* gym==0.12.1
* gin-config 
* cv2 
* [Dopamine](https://github.com/google/dopamine) framework (already integrated into this code base)

## Instruction 
* To train and evaluate `MMDQN` in an Atari game with the default settings, use the following example command (from the main directory `sourcecode` after unzipping the supplemetary):
    ```
    python main.py --env Breakout --agent_id mmd --agent_name mmd_dqn_1 --gin_files ./configs/mmd_atari.gin
    ```
where `env` is an Atari game name, `agent_id` is a registered agent id ('mmd' for MMDQN), `agent_name` for the directory name to save the agent training and evaluation results, and `gin_files` is a path the the hyperparameter configuration. 

* For convenience, we can directly modify the bash script `run_mmdqn.sh` for various hyperparameter settings and run the bash via 
    ```
    chmod +x ./run_mmd_dqn.sh;./run_mmd_dqn.sh 
    ```

### Main variables in MMD-DQN code:   

    * `env`: One of the 57 Atari games  
    * `agent_id`: ['mmd', 'quantile', 'iqn'], agent id (for MMDQN, QR-DQN and IQN)|
    * `agent_name`: str, the experiment log saved to `./results/<env>/<agent_name>` 
    * `policy`: ['eps_greedy', 'ucb', 'ps'], policy used by the agent (epsilon-greey, UCB or Thompson sampling) 
    * `num_atoms`: int, the number of particles N 
    * `bandwidth_selection_type`: 'mixture', the method for kernel bandwidth selection  
    * `gin_files`: str, the path to a gin file containing the hyperparameters of the agent 
    * `gin_bindings`: str, overwrite hyperparameters in a gin file 

### An overview of the MMDRL codebase: 
    * `mmd_agent.py`: An implementation of MMDQN agent 
    * `quantile_agent.py`: An implementation of QR-DQN  
    * `main.py`: Main file to train and evaluate an agent  
    * `run_mmd_dqn.sh`: A bash script to train and evaluate MMDQN agent 
    * `configs/mmd_atari_gin`: Hyperparameters of MMDQN agent 
    * `dopamine/`: The code base of Dopamine framework  


## Bibliography  

```
@misc{nguyen2020distributional,
        title={Distributional Reinforcement Learning via Moment Matching},
        author={Thanh Tang Nguyen and Sunil Gupta and Svetha Venkatesh},
        year={2020},
        eprint={2007.12354},
        archivePrefix={arXiv},
        primaryClass={cs.LG}
    }

```
