from experiment_configs.base_experiment import experiment as run_experiment


def launch_experiment(
        # Variant
        variant,

        # Experiment config
        get_config=None,
        get_offline_algorithm=None,

        # Misc arguments
        exp_postfix='',
        use_gpu=True,
        log_to_tensorboard=False,

        # Missing data
        data_args=None,
):
    # Load experiment config
    experiment_config = dict()

    if get_config is not None:
        experiment_config['get_config'] = get_config
    if get_offline_algorithm is not None:
        experiment_config['get_offline_algorithm'] = get_offline_algorithm

    # Run experiment
    run_experiment(
        variant=variant,
        experiment_config=experiment_config,
        exp_postfix=exp_postfix,
        use_gpu=use_gpu,
        log_to_tensorboard=log_to_tensorboard,
        data_args=data_args,
    )
