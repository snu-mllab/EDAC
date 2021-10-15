import argparse
"""
Temporary argparser to store common arguments
"""


def parser():
    parser = argparse.ArgumentParser()

    # Variant
    parser.add_argument('-e',
                        '--env_name',
                        default='halfcheetah-random-v2',
                        type=str)
    parser.add_argument('-x',
                        "--obs_with_x",
                        type=str,
                        default='none',
                        choices=['none', 'x', 'tc', 'noise'])
    parser.add_argument('--seed', default=0, type=int)
    # Misc arguments
    parser.add_argument('--exp_postfix', default='', type=str, help='')
    parser.add_argument('--use_cpu', action='store_true')
    parser.add_argument('--log_to_tensorboard', action='store_true')
    parser.add_argument(
        '--debug',
        action='store_true',
        help='debug run flag. fixes output dir to ./results/DEBUG')
    parser.add_argument("--epoch", default=3000, type=int)
    # SAC
    parser.add_argument("--plr",
                        default=3e-4,
                        type=float,
                        help='policy learning rate')
    parser.add_argument("--qlr",
                        default=3e-4,
                        type=float,
                        help='Q learning rate')
    parser.add_argument("--num_qs",
                        default=10,
                        type=int,
                        help='number of Q-functions to be used')
    parser.add_argument(
        "--num_minqs",
        default=10,
        type=int,
        help=
        'number of Q-functions to chosen for min Q calculation (Currently obselete)'
    )
    parser.add_argument("--policy_eval_start",
                        default=0,
                        type=int,
                        help='number of behavior cloning steps')
    parser.add_argument("--tau",
                        default=5e-3,
                        type=float,
                        help='soft target tau for SAC')
    parser.add_argument("--target_update_period",
                        default=1,
                        type=int,
                        help='target update period')
    parser.add_argument("--num_q_layers",
                        default=3,
                        type=int,
                        help='number of Q-network layers')
    parser.add_argument("--num_p_layers",
                        default=3,
                        type=int,
                        help='number of policy network layers')
    parser.add_argument('--max_q_backup',
                        action='store_true',
                        help='use max q backup')
    parser.add_argument('--deterministic_backup',
                        action='store_true',
                        help='use deterministic backup')
    parser.add_argument('--sigma',
                        default=-1.0,
                        type=float,
                        help='sigma for diversifying Q-ensemble')
    
    # reward preprocessing
    parser.add_argument("--reward_mean",
                        action='store_true',
                        help='normalize rewards by its mean')
    parser.add_argument("--reward_std",
                        default=-1.0,
                        type=float,
                        help='rescale std of the rewards (when > 0)')
    parser.add_argument("--reward_add",
                        default=-1e5,
                        type=float,
                        help='add some constant to the rewards (when > -1e3)')
    parser.add_argument("--reward_scale",
                        default=-1.0,
                        type=float,
                        help='scale the rewards by some constant (when > 0)')
    parser.add_argument("--reward_pen",
                        action='store_true',
                        help='remove bonus rewards from pen task')

    args = parser.parse_args()

    assert not (args.reward_mean and (args.reward_add > -1e3))
    assert not ((args.reward_std > 0) and (args.reward_scale > 0))

    return args


def print_args(args):
    for key, val in vars(args).items():
        print('{:<20} : {}'.format(key, val))


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
