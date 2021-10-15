from run_scripts.arguments import parser

from experiment_utils.launch_experiment import launch_experiment
from experiment_configs.configs.q_learning.sac_config import get_config
from experiment_configs.algorithms.online import get_algorithm
from experiment_configs.algorithms.offline import get_offline_algorithm

if __name__ == "__main__":
    variant = dict(
        algorithm='SAC',
        collector_type='step',
        env_name='Hopper',
        env_kwargs=dict(),
        do_offline_training=True,
        do_online_training=False,
        replay_buffer_size=int(2e6),
        reward_mean=False,  # added for easy config checking
        reward_std=-1.0,  # added for easy config checking
        reward_add=-1e5,  # added for easy config checking
        reward_scale=-1.0,  # added for easy config checking
        policy_kwargs=dict(
            layer_size=256,
            num_q_layers=3,
            num_p_layers=3,
            q_activation='relu',
        ),
        trainer_kwargs=dict(
            discount=0.99,
            soft_target_tau=5e-3,
            policy_lr=3e-4,
            qf_lr=3e-4,
            use_automatic_entropy_tuning=True,
            policy_eval_start=0,
            num_qs=10,
            num_minqs=2,
            target_update_period=1,
            max_q_backup=False,
            deterministic_backup=False,
            sigma=-1.0,
        ),
        algorithm_kwargs=dict(
            num_epochs=1000,
            num_eval_steps_per_epoch=5000,
            num_trains_per_train_loop=1000,
            num_expl_steps_per_train_loop=1000,
            min_num_steps_before_training=1000,
            max_path_length=1000,
            batch_size=256,
            save_snapshot_freq=1000,
        ),
        offline_kwargs=dict(
            num_epochs=3000,
            num_eval_steps_per_epoch=3000,
            num_trains_per_train_loop=1000,
            max_path_length=1000,
            batch_size=256,
            save_snapshot_freq=3000,
        ),
    )

    experiment_kwargs = dict(
        exp_postfix='',
        use_gpu=True,
        log_to_tensorboard=False,
    )

    # Additional arguments
    args = parser()

    # Variant
    variant['env_name'] = args.env_name
    variant['seed'] = args.seed

    variant['offline_kwargs']['num_epochs'] = args.epoch
    # Save only last epoch
    variant['offline_kwargs']['save_snapshot_freq'] = args.epoch

    # Misc arguments
    variant['trainer_kwargs']['policy_lr'] = args.plr
    variant['trainer_kwargs']['qf_lr'] = args.qlr
    experiment_kwargs['exp_postfix'] = ''

    variant['policy_kwargs']['num_p_layers'] = args.num_p_layers
    variant['policy_kwargs']['num_q_layers'] = args.num_q_layers
    # variant['policy_kwargs']['num_q_layers'] = args.num_p_layers

    variant['trainer_kwargs']['num_qs'] = args.num_qs
    #variant['trainer_kwargs']['num_minqs'] = args.num_minqs
    variant['trainer_kwargs']['num_minqs'] = args.num_qs
    variant['trainer_kwargs']['soft_target_tau'] = args.tau
    variant['trainer_kwargs'][
        'target_update_period'] = args.target_update_period

    variant['trainer_kwargs']['max_q_backup'] = args.max_q_backup
    variant['trainer_kwargs'][
        'deterministic_backup'] = args.deterministic_backup
    variant['trainer_kwargs']['sigma'] = args.sigma

    variant['reward_mean'] = args.reward_mean
    variant['reward_add'] = args.reward_add
    variant['reward_std'] = args.reward_std
    variant['reward_scale'] = args.reward_scale

    variant['trainer_kwargs']['policy_eval_start'] = args.policy_eval_start

    exp_postfix = args.exp_postfix
    exp_postfix += '_{}'.format(args.num_qs)
    
    exp_postfix += '_plr{:.5f}_qlr{:.5f}'.format(args.plr, args.qlr)
    if variant['policy_kwargs']['num_p_layers'] != 3:
        exp_postfix += '_npl{}'.format(variant['policy_kwargs']['num_p_layers'])
    if variant['policy_kwargs']['num_q_layers'] != 3:
        exp_postfix += '_nql{}'.format(variant['policy_kwargs']['num_q_layers'])
    if args.policy_eval_start > 0:
        exp_postfix += '_bc{}'.format(args.policy_eval_start)
    if variant['trainer_kwargs']['max_q_backup']:
        exp_postfix += '_maxq'
    if variant['trainer_kwargs']['deterministic_backup']:
        exp_postfix += '_detq'
    if args.sigma > 0:
        exp_postfix += '_sigma{:.2f}'.format(args.sigma)

    if args.reward_mean:
        exp_postfix += '_mean'
    if args.reward_std > 0:
        exp_postfix += '_std{:.2f}'.format(args.reward_std)
    if args.reward_add > -1e3:
        exp_postfix += '_add{:.2f}'.format(args.reward_add)
    if args.reward_scale > 0:
        exp_postfix += '_scale{:.2f}'.format(args.reward_scale)
    if args.reward_pen:
        exp_postfix += '_rpen'


    experiment_kwargs['exp_postfix'] = exp_postfix
    print(exp_postfix)

    experiment_kwargs['data_args'] = {
        'reward_mean': args.reward_mean,
        'reward_std': args.reward_std,
        'reward_add': args.reward_add,
        'reward_scale': args.reward_scale,
        'reward_pen': args.reward_pen,
    }

    experiment_kwargs['debug'] = args.debug

    # Launch experiment
    launch_experiment(variant=variant,
                      get_config=get_config,
                      get_offline_algorithm=get_offline_algorithm,
                      get_algorithm=get_algorithm,
                      args=args,
                      **experiment_kwargs)
