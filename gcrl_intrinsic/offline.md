# Offline training for IDM
- try IDM on 2 of the gym environments from the DT paper

1. get overview of dt code :check:
2. benchmark dt on medium replay and an expert env :check:
3. set up idm in dt set up and run for comparison :check:


# Experiment 1
- try on Hopper medium, medium-expert and medium-replay env
- try idm without ssl and vicreg
    - ssl is not readily applicable in offline
      as we have exact targets available already
    - vicreg would have been readily applicable if the action space was discrete, or can be done on an earlier representation embedding 

## Hopper-medium
- dt: 31450
- idm: 32054
## Hopper-medium-expert
- dt: 31453
- idm: 32053
## Hopper-medium-replay
- dt: 31454
- idm: 32052

for medium it works slightly for idm
but for the others simply the loss never gets compatible with existing, also it does 50 times less predictions, 

# Experiment 2
- add actions as token emb, and pred all actions, 
- keep encoding with states+ goal gating as is
- eval on MedReplay which was not as good as before 
  and Med to see if it still decent there
- use causal tgt_mask + use the pad_mask
- had to remove causal tgt_mask as it would force pad_mask to be of type bool, which would give weird error
medium-replay: 32138
medium: 32141
-> not working, 
!!!mistake in action taking, was not last decoded act in inference
Fixed versions
medium: 32216
medium-replay: 32220

# Experiment 3, 
simply add action tokens to encoder to allow that information to also be there, decode single action from here
medium-replay: 32197
medium: 32202
!!! key_pad_mask fix
medium: 32270
medium-replay: 32269
medium-expert: 32371
> decent on medium but rather unstable, not good on medium or medium-expert
> perhaps goal mixing should not mix last state into it

# Experiment 4, enc(gpt) like decoder with idm masking
- test that the causal only masking performs similar to dt
- medium-replay (causal only): 32407
- medium-replay: 32403
> not working well
stacking normal returns_to_go as dt, with causal mask only: medium-replay: 32431, not working well
with act causal masking too: 32454
- error with action reshaping after changing forward to dt code, then:
 1. should not need reshaping of course, in seq trainer
 2. should reshape back to batch first before feeding to transformer or check the shapes from dt original too
fixed shaping: medium-replay: 32741  (matches dt)
try medium: 32783 and medium-expert: 32784
(finally works, shows that masking was much error)

# Experiment 5, like Exp 4, but now with idm masking added
- medium-replay: 32786
- medium: 32790
- medium-expert: 32791
- on par with dt
- more like traj masking as we don't acces s_t+k
- slight error given idm masking is applied in inference, which it should not
    - medium-replay: 32960

# Experiment 6, interleave S+A, mix delayed R in + traj masking
- use dt forward, remove returns mixing, and use gating net to mix states, with normal returns
- use traj masking: 33006
- then try with delayed returns: 33008,33452
- medium-expert: 33455
- medium: 33453
- try medium-replay dt: 33965, with delay, not good

# Experiment 7, singleActDec with new masking 
- medium-replay: 

# All experiments are wrong! (1,2 and 3)
 - forgot that the gpt model inverts masking interpretation as compared to the torch transformer model, so have to flip it! also why it would nan crash occasionnally, when the mask was all True which would become False for all entries

# plan 
- run S+A goal mixing delayed on medium and   medium-expert, (after 16, maintainence on claaudia)
- pull data for visualization
    -> looks much more stable in medium-replay
- try dec single act
    - check if masking, pos is correct
    - mr: 33891, 33946, very swinging
- try vicreg with projector net

- try goal mixing with gate network gcsl 1dec
- 33915, not working, just cut it, and focus on training idm with goal setter
- try run Door gcsl on gym==10.0.05
- multi dec?

# Plan
- for offline gate fusion seems better more stable than default, needs multi decoding for performance with delay.
done here, can not change to single dec, dt is worse with delay. 
- for gcsl, in gym==10.0.5, door is good on org, but not good for idm_traj_dec with ssl+vicreg, 
Focus a bit on gcsl with gym==10.0.5
## Door
- gcsl_org: 34003, matches paper so env must be good now
- traj_dec_idmE: 33941, not good
- traj_dec_idmnoE: 33937, even worse
- traj_dec_idm no ssl + no vicreg: 34102

# testing why greedy is bad on ssl+vicreg traj_dec_idm
- try set teacher weights = student at beginning
- lunar traj_dec_im greedy: 34185, okay
- door traj_dec_im greedy: 34376, not working
- door traj_dec_im greedy vicreg only: 34352,
    -> was actually no vicreg, no ssl, was okay, but pusher was bad
- door traj_dec_im greedy ssl only: 34707, not working
- door traj_dec_im greedy ssl+vicreg expander: 34730, 
  working okay, can try with greedy, check pusher vicreg only?

--- ssl is the culprit, try vicreg with expander only
- door vicreg+expander explore: 34734
- door vicreg+expander greedy: 34736
- pusher vicreg+expander explore: 34739
- pusher vicreg+expander greedy: 34749

- pusher traj_dec_im greedy vicreg only: 34374, not workin
- pusher traj_dec_im greedy ssl only: 34722

the only door that worked well was when ssl was wrong in act taking, so vicreg only, when mixing ssl into it started tanking 

# gcsl expander stats
[All Expanders]

explore with ssl+vicreg
- door: 34730, 34772(not full run)
- pusher: 34770
--> Works very well in both cases, beating previous best, SELECT 

greedy with explore+vicreg
- door: 34773
--> collapse as before, cut it

greedy with vicreg
- door: 34736
- pusher: 34749
--> good early for door, but then drops, pusher mixed, perhaps needs
    increase vicreg relative weighting, but cut it

explore with vicreg
- door: 34734
- pusher: 34739
--> mixed for door, very good for pusher, cut

# Final model
gcsl_8_32f_2l_ssl_vrE..._Yes_explore
needs 2 door runs
needs 2 pusher runs
needs 3 point room runs
needs 3 point mass runs
needs 3 point lunar runs
- first try 1 of room, mass and lunar to see, then push through
eElunar: 34812, 34824, 34825, 34857
v1eElunar: 34866, 34867
eEempty: 34814, 34822, 34823
eErooms: 34815, 34816, 34821
eEDoor: 34861, 34862 (not run fully), 34921, 34922
eEPusher: 34858, 34859

gcsl org pusher:
greedy: 36198
explore: 36201, 36214, 36217

-> results look good, (lunar gym version needs be higher 0.24.0 for example)

- :check: pull and view new data, looks good 
- :check: run gcsl_org on door: 35058, 35059, 35086E,35087E 
- :check: try continous actions for CLAW: 35069, not working

- try expander vicreg for offline idm (dt)
    - 35092(wrong), no vicreg seems better
    - no vicreg (for test) 35093
- offline idm causal+pre delayed adj setup: 35274
-> idm is important it seems
- try fixing inference timesteps to 0-50: 36238
-> 
- fixing timesteps to seq length during training: 36240

## Claw
- new: 35155 
- can also try claw now to see if fixed
- gcsl_org: 34105
- traj_dec_idmE: 34115
- seems like before, just cut claw env

# dt benchmarking 
- run 3 runs of best model that is delayed+gating on
hopper, walker, cheetah for (Medium-replay,medium-expert,medium)
## delayed+gating
Hopper
    - medium: 2 left: 37639
    - medium-replay: DONE
    - medium-expert: 2 left: 37640
Cheetha
    - medium: 3 l: 37641(2runs), 37847(12k)
    - medium-replay: 3 l: 37643(2runs), 37848(12k)
    - medium-expert: 3 l: 37645, 37850(12k)
Walker
    - medium: 3 l: 37646
    - medium-replay: 3 l: 37651
    - medium-expert: 3 l: 37652
## dt
Hopper
    - medium: 2 left: 37669
    - medium-replay: 2 left: 37670
    - medium-expert: 2 left: 37671
Cheetha
    - medium: 3 l: 37647(delayed), 37653, 37844(12k)
    - medium-replay: 3 l: 37661, 37845(12k)
    - medium-expert: 3 l: 37666, 37846(12k)
Walker
    - medium: 3 l: 37650(delayed)
    - medium-replay: 3 l: 37667
    - medium-expert: 3 l: 37668

# visualize results
- ...

# more runs of the ablations on Hopper medium-replay    