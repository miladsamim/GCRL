# reference scores from https://github.com/Farama-Foundation/D4RL/blob/master/d4rl/infos.py
REF_MIN_SCORES = {
    'hopper': -20.272305, 
    'halfcheetah': -280.178953,
    'walker2d': 1.629008,
    'carracing': -1.0,
}
REF_MAX_SCORES = {
    'hopper': 3234.3,
    'halfcheetah': 12135.0,
    'walker2d': 4592.3,
    'carracing': 240.0, # 900.0, 260.0 when normalized
}

def get_normalized_score(env_name, score):
    min_score = REF_MIN_SCORES[env_name]
    max_score = REF_MAX_SCORES[env_name]
    return (score - min_score) / (max_score - min_score)