## Inverse Dynamics Model (IDM)
The Inverse Dynamics Model (IDM) is a learned model that aims to predict the action given two consecutive states. In this context, we train the IDM using a transformer architecture, which is known for its ability to learn and generalize long-range dependencies in the sequences.

## Training of IDM:

Causal prediction: The transformer is trained to causally predict the next action given the current state.
Masked prediction: The transformer is trained to predict actions in a masked setting, where we randomly sample a position mask_pos and mask everything from the mask_pos*2'th state to the end, except for unmasking the very last state s_k. s.t. that masking will look like: s0,a0,s1,a1,....,s_k  

## Goal Setter (GS)
The Goal Setter (GS) is a separate neural network that generates a desired goal state. The GS's state space is the same as the underlying state space, while the action space is continuous and represents the generated goal state s_k'. The purpose of the GS is to guide the IDM by providing it with goal states to achieve.

## Training of GS:
- Define the GS as a neural network that maps from the current state to the continuous goal state (s'k) in the simplified state representation.
- Modify the PPO algorithm for the continuous action space, where the policy network of the PPO would be the GS network.
- Integrate the IDM and GS during training. At each time step, the GS provides the desired goal state, and the IDM decodes the actions to reach that goal state.
- Use the rewards and trajectory data from the IDM's interactions with the environment to update the GS's policy using PPO.

## Additional Technical Points
- Goal State Distribution: Each entry in the goal state vector is modeled as a Gaussian distribution, allowing the GS to explore a variety of goal states and search the state space more efficiently.
Reward Function Design: It is essential to carefully design the reward function for the GS to encourage goal generation that leads to desired behaviors and outcomes.
- State Representation: The choice of state representation for the goal states and how the IDM decodes the actions from them will significantly impact the overall performance of the system.
- Coordination Between IDM and GS: Training both IDM and GS can be challenging due to the requirement of coordinating between their objectives and potential difficulties in credit assignment.


# Car Racer
Developing the approach in the Car Racer env. with pretrained vision-encoder + idm 
0. DONE : train by PPO policy with 64 dim state vector, (previous was 4096)
    - 37968
1. DONE : Generate ~ 300 episodes, and store the s0,a0,r0,s1,a1,r1,...., store the latent emb of state instead of the images as vision enc. is frozen 
    - did 100 eps with mid-level agent, for better use v10 at 500 steps
2. Use this data to train the TransformerIDM and track the loss during training 
    - TransformerIDM is running, evaluate it if it learns bc as per evaluation rewards
    - Train simpler gcsl like IDM, with given s0, and randomly sampled sk, use same data loop, just make new model and trainstep
      then fix this simpler IDM for the ppo
3. Now with fixed vision encoder, fixed IDM, try train the PPO
4. setup the ppo alg. so that every entry in the output goal emb. output is gaussian
   , should also have value function trained in started way by mse loss: r_t + V(s_t) - V(s_t+1)


## car racer to dt
    - DONE : check data collet loop
    - RUNNING : 38856(100 medium), 38877(500 medium-expert) collect 500 eps with v10 500
    - RUNNING : 38880 set up to run in dt code, with reward conditioning
        -> loss did came down on mini dataset (38871), but not evaluation
        -> loss + evaluation works well, (note normalized rewards is used so REF_MAX is set to high on 38880)
    - try with no reward conditioning

# set up gcsl like idm training in dt code
    - DONE : model class 
    - DONE : trainer class 
    - R2R : 39005 try train on hopper-expert, log train loss, and also just evaluation runs even though not meaningful
        -> training loss coming down, evaluation comes to 0.31 when giving same state stacked in evaluation
    - R2R : 39145 try on car-racer

# PPO Train Goal Setter for Hopper-V2: 
    - set up ppo training file, with gcsl_idm + goal setter continous ppo 
    - set gcsl_idm into ppo_continous and see if it can match performance as per dt training
        -> does not perform similarly, perhaps normalizing is needed, normalizing not helping,
- just try train with random fixed gcsl_idm
    -> moves quickly up to 259 at 33k steps (matches standard), but then stagnates fully there
- try giving gcsl_idm parameters to the optimizer and see effect 
    -> same as when fixed, Naturally can not have effect it is only in eval. and hene torch.no_grad
- try incorporating gcsl loop for training the gcsl_idm simultaneously
    - incorporate buffer
    - incorporate training model
    - training loop, sample, loss, backprop 
    -> gcsl loss goes down, but ppo does not succeed in goal setting
        - perhaps vicreg for gcsl, or lower K
    - k 5 moves up a bit higher but then drifts down again, perhaps idm gets fixed  

- try generate real_acts from ppo too, 
    1. use ppo action, in idm train loop force actions closer by mse loss, 
    -> working but below ppo only, gcsl loss stagnates shows that it will engage in behaviour cloning of the ppo instead
    2. V0: use ppo action, in idm train loop force actions closer by mse loss, let ppo train to generate goal, and then have it minimize
       the idm closeness loss by goal generation instead
    -> on par with ppo, stable in the end, 

- try go back and use idm for act generation but regularize with vicreg
    -> flatlines

- train idm with ppo data, then save it and fix it, then train ppo goal setter with fixed "good" idm 
  to see if goal setting is learnable
  -> not seeming to work
- try not forcing ppo and idm to be close in pretraining of idm
 -> flatlining again

- cut idm goal setting idea
- vicreg temporal training for ppo state repr.
- vicret temporal training for dt

- plan 
    - pull dt data and plot it 
    - make a writing over next goals

