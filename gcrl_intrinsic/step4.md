# Goal Conditioned Reinforcement Learning
- gcsl no reward function, only from state,action data online
- dt, with reward, offline data,
- dt, with multistep action prediction + gate mixing of reward info seems better, more stable
- ppo + idm for goal setting, did not work well, so cutting that idea

- now idea is to improve offline rl method like dt,
    - with vicreg on temporal axis, ensuring representational consistency between vectors at t and t+1
        -> didn't have much positive impact as compared to previous, however it seems on par with dt
    - also tried mixing vicreg loss with act mse loss 0.5 ratio
        -> was worse than just incorporating act loss in vicreg closeness loss 
    - simple archictecture matches dt code performance, so can use that for below experiments: 53124


- car racer offline, train vicreg based temporal consistency with state+reward conditioning data only 
  - then see if it can collect its own online action data and label with different reward token, and see if these actions gets better
  - can use state token, action token and online vs video (offline) token or maybe not token differentiate between offline and online
  - then train to ouput emb on state position like: [rep,act_dim], and only force temporal vicre consistency on rep, and leave act_dim unused except in online where it used as action and collected into online buffer

  - EXP1: 53736, stream training 100 steps, 10 act steps, train both by vicreg temporal consistency [:-self.act_dim] consistency on expander
    -> not working, online loss comes below, train loss perhaps the fixed target return token gives indication 

  - EXP2: 54112 adjustments:
            - use achieved return in replay buffer rather than target return :added:
            - add predict head for t_i to predict vicreg t_i+1 :added:
            - use action head, embed_action : act_dim --> embed_dim || pred_action: embed_dim --> act_dim || embed_action :added:
                - in a sense: decoder + encoder + decoder 
                - then in inference we can use decoder + encoder
            - use vicreg on full expansion 128+128 :added:
    -> not working... 

- EXP3: like EXP2, ajustment:
            - no encoder + decoder + encoder setup
            - vicreg only on sub part

- EXP4: 54235 small loss to autoregressive predict action in online training?
        adjustments:
        - like EXP2: but with small autoregressive action prediction
    -> not working...

- EXP5: add representation tokens like: S1,A1,R1,S2,A2,R3,.... which is just random sampled from uniform, then apply the 
        the vicreg expander on here

- EXP6: change to much simpler model, 
        - a gcsl net which can take s_k, return_condition_token (rtc), s_g, and trains in the following way, where k < g
        - when streaming: sample s_k, rtc, s_g, noise_1, noise_2, and feed to net like: 
            rep1 = gcsl(s_k,rtc,noise_1), rep2 = gcsl(noise_2,rtc,s_g)
          then feed both of these to expanders: exp1 = expander(rep1), exp2 = expander(rep) 
          and push closer with vicreg loss 
        - when bootstrapping: sample s_k, rtc, s_g, a_k
          here a is the action, which in inference is generated at the i'th timestep by: 
            rep_i = gcsl(s_i,rtc,noise), and then feed to act_pred(rep_i)
          in bootstrap/online training, we predict the a_k'th action 
          and minimize (a_k - act_pred(gcsl(s_k,rtc,s_g)))^2
    Idea:
        - we learn to correlate a state s_k with a state s_g in the streaming training by conditioning on the rtc, which
        should describe the general behaviour of the trajectory and thus push the representation close together, such that 
        it represents a vector between the states 
        - in online training, we want to generate the actions, which are optimal for reaching s_g from s_k with the rtc, 
          in inference we condition on high rtc, and we don't have access to s_g so we want to emulate a high reward trajectory
          behaviour, 
          but when we train we use the idea of hindsight experience replay and we have access to the achieved s_g and rtc, 
          so we condition on the achived rtc and s_g to generate the action which matches the information in the data, 
        - the hope: is that hopefully there is a general connection in the spaces between s_k, rtc, s_g, such that 
          it will be possible to generalize or compute in the space what should be the connecting actions, hence allowing to 
          improve the actions
    (bc training seems to work well, reward reaches ~500 within 25 iterations)
    (gcsl like training alone does not reach ~500, starts high by random, then drops lower in 100 iterations)

    Runs:
    - Default:
        hopper 54348: 
            -> not working
        carracer 54354:

    - simple concat on state and goal rather than embd (does not seem to matter on bc training):
        hopper
        carracer 

    - ideas for adjustments: 
        - remove rtc on online, as it could relate action prediction to low return tokens (not effectfull alone)
        - add state predictor net for t_k to predict t_g on top of expansion (not effectfull alone)
        - make action gaussian, so output means and then sample 
        - make actions discrete like gcsl
        - small amount of discrete tokens, compute energies by multiplying into these and then force cross entropy similarity
          towards those, and let all communication bottleneck through those
        - change goal representation to be feed only s_g and not rtc
        - discrete information bottle necks:
            - set k d-dimensional tokens to be the informational bottle (ib), such that gcsl(s_k,rtc,noise) is to map
              to an attention over the k-dimensional tokens, and similarly map gcsl(noise,noise,s_g) to an attention over the k ib tokens.
            - can also be tried with the ppo planner, by swithching to discrete ppo and having the planner come up with a discrete 
              ib token to be feed to the idm, and then idm could be trained minimize mse(a_k, idm(s_k,s_g)) and mse(idm(s_k,s_g), idm(s_k,ppo(s_k))), where the ppo(s_k) will have mapped to a discrete action that is to an ib token
              -> did not work ...

Cut experimentation and switch to writing

    
