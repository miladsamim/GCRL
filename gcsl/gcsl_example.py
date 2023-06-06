import sys, os 
DIR_PATH = os.path.dirname(os.path.realpath(__file__))
dir_paths = ['dependencies']
sys.path += [os.path.join(DIR_PATH, dir_path) for dir_path in dir_paths]
mujoco_path = '/home/student.aau.dk/msamim18/.mujoco/mujoco210/bin'
os.environ['LD_LIBRARY_PATH'] = os.environ.get('LD_LIBRARY_PATH', '') + ':' + mujoco_path

from gcsl.tools import dotdict
from gcsl.algo.transformer_idm import DiscreteIDMPolicy
from gcsl.algo.dec_idm import DiscreteDEC_IDMPolicy
from gcsl.algo.idm import S_DiscreteIDMPolicy
from gcsl.algo.idm_ssl import S_DiscreteIDMPolicy_SSL
from gcsl.algo.idm_online import DiscreteOnlineIDMPolicy
from gcsl.algo.pact import DiscretePACTPolicy
from gcsl.algo.pact_single import DiscretePACTSinglePolicy
# from gcsl.algo.dt_gpt import DT_GPT_Policy
from gcsl.algo.traj_idm import DiscreteTrajIDMPolicy    
from gcsl.algo.traj_idm_dec import DiscreteTrajDEC_IDMPolicy
from gcsl.algo.traj_idm_dec_simple import DiscreteTrajDEC_IDMSinglePolicy
from gcsl.algo.traj_idm_dec_claw import DiscreteTrajDEC_IDMPolicy_CLAW
from gcsl.algo.traj_idm_dec_claw_cont import DiscreteTrajDEC_IDMPolicyClawContinous

def run(output_dir='/tmp', env_name='pointmass_empty', gpu=True,
        seed=0, idm_args=None, model_args=None, **kwargs):

    import gym
    import numpy as np
    from rlutil.logging import log_utils, logger

    import rlutil.torch as torch
    import rlutil.torch.pytorch_util as ptu

    # Envs

    from gcsl import envs
    from gcsl.envs.env_utils import DiscretizedActionEnv

    # Algo
    from gcsl.algo import buffer, gcsl, variants, networks

    ptu.set_gpu(gpu)
    if not gpu:
        print('Not using GPU. Will be slow.')

    torch.manual_seed(seed)
    np.random.seed(seed)

    env = envs.create_env(env_name)
    env_params = envs.get_env_params(env_name)
    # env_params['max_timesteps'] = 2500
    print(env_params)

    env, policy, replay_buffer, gcsl_kwargs = variants.get_params(env, env_params, 
                                                idm_args=idm_args,model_args=model_args)
    gcsl_kwargs['log_tensorboard'] = True
    # gcsl_kwargs['eval_freq'] = 1000
    if idm_args.policy_class:
        policy = idm_args.policy_class(env, model_args)
    print(policy)
    algo = gcsl.GCSL(
        env,
        policy,
        replay_buffer,
        idm_args=idm_args,
        **gcsl_kwargs
    )

    exp_prefix = '%s/gcsl_org_yes_explore/' % (env_name,)
    # gcsl_8_2l_32f_ssl_vrE_bsz256_traj_tra_SingleDec_state_traj_modulation_gate_noPos_Yes_explore

    with log_utils.setup_logger(exp_prefix=exp_prefix, log_base_dir=output_dir):
        algo.train()


if __name__ == "__main__":
    params = {
        'seed': [0],
        'env_name': ['pointmass_empty'], #['lunar', 'pointmass_empty','pointmass_rooms', 'pusher', 'claw', 'door'],
        'gpu': [True],
    }
    use_gpu = True
    # remember setting dim_f
    model_args = dotdict(   
            d_model=8,
            nhead=1,
            layers=2,
            dim_f = 32,
            max_len=50,
            dropout=0.1,
            norm_first=True,
            device='cuda' if use_gpu else 'cpu',
            use_traj=False,
            zerofy=False,
            mask_type = 'all',
            centering_momentum = 0.9, # m for C
            student_sharpening = 0.1, # tau_s
            teacher_sharpening = 0.04, # tau_t
            polyak_momentum = 0.99, # l 
            out_dim=4, #4
        )   
    idm_args = dotdict(
        use_transformer = False,
        policy=None,
        # policy_class = DiscreteIDMPolicy,
        # policy_class = DiscreteDEC_IDMPolicy,
        # policy_class = S_DiscreteIDMPolicy,
        # policy_class = S_DiscreteIDMPolicy_SSL,
        # policy_class = DiscreteOnlineIDMPolicy,
        # policy_class = DiscretePACTPolicy,
        # policy_class = DT_GPT_Policy,
        # policy_class = DiscretePACTSinglePolicy,
        # policy_class = DiscreteTrajIDMPolicy,
        # policy_class = DiscreteTrajDEC_IDMPolicy,
        # policy_class = DiscreteTrajDEC_IDMPolicy_CLAW,
        # policy_class = DiscreteTrajDEC_IDMPolicyClawContinous,
        use_claw=False,
        # policy_class = DiscreteTrajDEC_IDMSinglePolicy,
        restrictive_masking = True,
        masking = None,#'traj',
        dino_loss = False,
        model_args = model_args,
        ssl = False,
        vicreg = False,  
        lambda_ = 10, # vicreg 
        mu = 10, # vicreg 
        nu = 1, # vicreg 
        expander = False, # expander in vicreg
        online_loss = False,
        pact = False
    )
    # ssl disabled in gcsl
    #['lunar', 'pointmass_empty','pointmass_rooms', 'pusher', 'claw', 'door'],
    run(output_dir=os.path.join(DIR_PATH, 'tmp'), env_name='pusher', 
        gpu=use_gpu, seed=0, idm_args=idm_args, model_args=model_args)
    