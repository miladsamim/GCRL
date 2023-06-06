# Models to run:
- base: gcsl original with no exploration
- transformer: transformer d_model=8, n_layers=2, dim_feedforward=32, with (exploration)

# Experiments (3 Runs)
## EXP 1 | ENV = door
    Base 
        - slurm: 30541, 30542, 30543 | 30651, 30770, 30771
    Transformer exploration
        - slurm: 30557, 30558, 30559 | 30650
        - T slurm: 30960, 31007, 31008
    Transformer no exploration
        - slurm: 30566 (job name is wrong), 30794, 30795
        - T slurm: 31017, 31018, 31019
    - Notes: the explore versions seems quite bad
             so modify to only train explore models
## EXP 2 | ENV = pusher
    Base 
        - slurm: 30563, 30564, 30565
    Transformer exploration
        - slurm: 30560, 30561, 30562
        - T slurm: 30986, 31005, 31006
    Transformer no exploration
        - slurm: 30796, 30797, 30810
        - T slurm: 30961, 31020, 31020
## EXP 3 | ENV = pointmass_empty
    Base 
        - slurm: 30863x3, 30937 (wrong name), 30938
    Transformer exploration
        - slurm: 
        - T slurm: 31009, 31010, 31011
    Transformer no exploration
        - slurm: 30880, 30892, 30895
        - T slurm: 31025, 31027, 31028
## EXP 4 | ENV = pointmass_rooms
    Base 
        - slurm: 30864x3, 30935, 30936
    Transformer exploration
        - slurm: 30932, 30933, 30934
        - T slurm: 31012, 31013, 31014
    Transformer no exploration
        - slurm: 30866, 30870, 30872
        - T slurm: 31022, 31023, 31024
## CLAW ???
    Base 
        - slurm: 30502, 30504, 30505
    Transformer exploration
        - T slurm: 30953, 30954, 30955
    Transformer no exploration
        - T slurm: 30956, 30957, 30958

1. pull data :check:
2. analyze data: mixed results :check:
2. - re run base for pe, pr 
2. - run exploratory tra for pe, pr 
3. run CLAW
    - claw is not working well, odd action space, just scrap it 
fixed transformer ssl to take action from teacher
, run door with exploration to see effect of it, 
  run pusher with no exploration
    - determine whether to rerun transformer based exps
    > no influence on results yet, just running pusher + explore to confirm
    > explore versions look better so try rerun them first

## gcsl Expander stats

First GCSL part is over...
Now offline test to compare with decision transformer
then online plan generator + idm

5. dec. transformer plan   

