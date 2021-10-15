import abc

import gtimer as gt
from lifelong_rl.core.rl_algorithms.rl_algorithm import BaseRLAlgorithm


class BatchRLAlgorithm(BaseRLAlgorithm, metaclass=abc.ABCMeta):
    # CHANGE: add sample mode (steps or paths)
    def __init__(
            self,
            trainer,
            exploration_policy,
            exploration_env,
            evaluation_env,
            exploration_data_collector,
            evaluation_data_collector,
            replay_buffer,
            max_path_length,
            num_epochs,
            num_eval_samples_per_epoch,
            num_expl_samples_per_train_loop,
            sample_mode='steps',
            num_trains_per_train_loop=1,
            num_train_loops_per_epoch=1,
            min_num_samples_before_training=0,
            save_snapshot_freq=100,
            post_epoch_funcs=None,
    ):
        super().__init__(
            trainer,
            exploration_policy,
            exploration_env,
            evaluation_env,
            exploration_data_collector,
            evaluation_data_collector,
            replay_buffer,
            save_snapshot_freq=save_snapshot_freq,
            post_epoch_funcs=post_epoch_funcs,
        )

        self.max_path_length = max_path_length
        self.num_epochs = num_epochs
        self.num_trains_per_train_loop = num_trains_per_train_loop
        self.num_train_loops_per_epoch = num_train_loops_per_epoch
        self.num_eval_samples_per_epoch = num_eval_samples_per_epoch
        self.num_expl_samples_per_train_loop = num_expl_samples_per_train_loop
        self.min_num_samples_before_training = min_num_samples_before_training
        self.sample_mode = sample_mode

    def _train(self):
        if self.min_num_samples_before_training > 0:
            init_expl_paths = self.expl_data_collector.collect_new_paths(
                self.max_path_length,
                self.min_num_samples_before_training,
                sample_mode=self.sample_mode,
                discard_incomplete_paths=True,
            )
            self.replay_buffer.add_paths(init_expl_paths)
            self.expl_data_collector.end_epoch(-1)

            self._fit_input_stats()

        for epoch in gt.timed_for(
                range(self._start_epoch, self.num_epochs),
                save_itrs=True,
        ):
            self.eval_data_collector.collect_new_paths(
                self.max_path_length,
                self.num_eval_samples_per_epoch,
                sample_mode=self.sample_mode,
                discard_incomplete_paths=True,
            )
            gt.stamp('evaluation sampling')

            for _ in range(self.num_train_loops_per_epoch):
                new_expl_paths = self.expl_data_collector.collect_new_paths(
                    self.max_path_length,
                    self.num_expl_samples_per_train_loop,
                    sample_mode=self.sample_mode,
                    discard_incomplete_paths=True,
                )
                gt.stamp('exploration sampling', unique=False)

                self.replay_buffer.add_paths(new_expl_paths)
                gt.stamp('data storing', unique=False)

                self.training_mode(True)
                for _ in range(self.num_trains_per_train_loop):
                    self.trainer.train_from_paths(new_expl_paths)
                gt.stamp('training', unique=False)
                self.training_mode(False)

            self._fit_input_stats()

            self._end_epoch(epoch)
