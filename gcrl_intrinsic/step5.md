Streaming
Dynamics net D: learns to predict future k states ahead in longer sequences, could be a transformer
loss is to minimize predictions to match future states, from expert behaviour to ensure that meaningfull
representations are learned

online: Action predictor:
A different net is trained to propose actions, such that actual observed frames will match the ones predicted from D
so the loss is for it to match the predicted frames of D. could be used directly or modeled as a reward function


# sequential training
- use noise input with noise labels, and see if this helps in maintaining previous learning (sleep)