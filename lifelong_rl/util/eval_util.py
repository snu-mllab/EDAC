"""
Common evaluation utilities.
"""

from collections import OrderedDict
from numbers import Number
from copy import deepcopy

import numpy as np

import lifelong_rl.util.pythonplusplus as ppp


def get_generic_path_information(paths, stat_prefix=''):
    """
    Get an OrderedDict with a bunch of statistic names and values.
    """
    statistics = OrderedDict()
    returns = [sum(path["rewards"]) for path in paths]

    rewards = np.vstack([path["rewards"] for path in paths])
    if 'q_preds' in paths[0]:
        q_preds, q_trues, q_pred_true_gaps = get_q_pred_true_gaps(paths)
        statistics.update(create_stats_ordered_dict('Q pred', q_preds,
                                                    stat_prefix=stat_prefix))
        statistics.update(create_stats_ordered_dict('Q true', q_trues,
                                                    stat_prefix=stat_prefix))
        statistics.update(create_stats_ordered_dict('Q pred-true gap', q_pred_true_gaps,
                                                    stat_prefix=stat_prefix))
    statistics.update(create_stats_ordered_dict('Rewards', rewards,
                                                stat_prefix=stat_prefix))
    statistics.update(create_stats_ordered_dict('Returns', returns,
                                                stat_prefix=stat_prefix))
    actions = [path["actions"] for path in paths]
    if len(actions[0].shape) == 1:
        actions = np.hstack([path["actions"] for path in paths])
    else:
        actions = np.vstack([path["actions"] for path in paths])
    statistics.update(create_stats_ordered_dict(
        'Actions', actions, stat_prefix=stat_prefix
    ))
    statistics['Num Paths'] = len(paths)
    statistics[stat_prefix + 'Average Returns'] = get_average_returns(paths)

    '''
    for info_key in ['env_infos', 'agent_infos']:
        if info_key in paths[0]:
            all_env_infos = [
                ppp.list_of_dicts__to__dict_of_lists(p[info_key])
                for p in paths
            ]
            for k in all_env_infos[0].keys():
                final_ks = np.array([info[k][-1] for info in all_env_infos])
                first_ks = np.array([info[k][0] for info in all_env_infos])
                all_ks = np.concatenate([info[k] for info in all_env_infos])
                statistics.update(create_stats_ordered_dict(
                    stat_prefix + k,
                    final_ks,
                    stat_prefix='{}/final/'.format(info_key),
                ))
                statistics.update(create_stats_ordered_dict(
                    stat_prefix + k,
                    first_ks,
                    stat_prefix='{}/initial/'.format(info_key),
                ))
                statistics.update(create_stats_ordered_dict(
                    stat_prefix + k,
                    all_ks,
                    stat_prefix='{}/'.format(info_key),
                ))
    '''

    return statistics


def get_q_pred_true_gaps(paths, discount=0.99):
    # get Q-value gaps (predicted Q-value - true Q-value)
    q_predss = []
    q_truess = []
    q_gaps = []
    for path in paths:
        q_preds = deepcopy(path['q_preds'])
        rewards = deepcopy(path['rewards'])
        terminals = deepcopy(path['terminals'])
        entropies = deepcopy(path.get('entropies', []))
        entropies_flag = len(entropies) > 0
        q_trues = []

        last_step = len(rewards)-1
        next_q_true = 0

        if not terminals[last_step]:
            next_q_true = q_preds[last_step]
            q_preds = q_preds[:-1]
            last_step -= 1

        while last_step >= 0:
            q_true = rewards[last_step]
            if terminals[last_step]:
                pass
            else:
                q_true += discount * next_q_true
                if entropies_flag:
                    q_true -= discount * entropies[last_step + 1]

            q_trues = [q_true] + q_trues

            next_q_true = q_true
            last_step -= 1

        q_trues = np.array(q_trues)

        assert len(q_trues) == len(q_preds), '{} {}'.format(len(q_trues), len(q_preds))

        q_predss.append(q_preds)
        q_truess.append(q_trues)
        q_gaps.append(q_preds - q_trues)

    return np.vstack(q_predss), np.vstack(q_truess), np.vstack(q_gaps)


def get_average_returns(paths):
    returns = [sum(path["rewards"]) for path in paths]
    return np.mean(returns)


def create_stats_ordered_dict(
        name,
        data,
        stat_prefix=None,
        always_show_all_stats=True,
        exclude_max_min=False,
):
    if stat_prefix is not None:
        name = "{}{}".format(stat_prefix, name)
    if isinstance(data, Number):
        return OrderedDict({name: data})

    if len(data) == 0:
        return OrderedDict()

    if isinstance(data, tuple):
        ordered_dict = OrderedDict()
        for number, d in enumerate(data):
            sub_dict = create_stats_ordered_dict(
                "{0}_{1}".format(name, number),
                d,
            )
            ordered_dict.update(sub_dict)
        return ordered_dict

    if isinstance(data, list):
        try:
            iter(data[0])
        except TypeError:
            pass
        else:
            data = np.concatenate(data)

    if (isinstance(data, np.ndarray) and data.size == 1
            and not always_show_all_stats):
        return OrderedDict({name: float(data)})

    stats = OrderedDict([
        (name + ' Mean', np.mean(data)),
        (name + ' Std', np.std(data)),
    ])
    if not exclude_max_min:
        stats[name + ' Max'] = np.max(data)
        stats[name + ' Min'] = np.min(data)
    return stats
