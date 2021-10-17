import json
import os
import os.path as osp
import numpy as np
from time import time
import sys

from lifelong_rl.core.logging.logging import logger


def setup_logger(
        variant=None,
        text_log_file="stdout.log",
        variant_log_file="variant.json",
        tabular_log_file="progress.csv",
        log_to_tensorboard=False,
        snapshot_mode="all",
        snapshot_gap=1000,
        log_tabular_only=False,
        log_dir=None,
        base_log_dir='results',
        git_infos=None,
        script_name=None,
        **create_log_dir_kwargs
):
    log_dir = osp.join(base_log_dir, log_dir)
    os.makedirs(log_dir, exist_ok=True)
    logger.log_dir = log_dir

    print("logging to:", log_dir)

    text_log_path = osp.join(log_dir, text_log_file)
    tabular_log_path = osp.join(log_dir, tabular_log_file)

    logger.set_text_output(text_log_path)
    logger.set_tabular_output(tabular_log_path)

    logger.set_snapshot_dir(log_dir)

    logger.set_snapshot_mode(snapshot_mode)
    logger.set_snapshot_gap(snapshot_gap)
    logger.set_log_tabular_only(log_tabular_only)

    exp_name = log_dir.split("/")[-1]
    logger.push_prefix("[%s] " % exp_name)

    if variant is not None:
        logger.log("Variant:")
        logger.log(json.dumps(dict_to_safe_json(variant), indent=2))
        variant_log_path = osp.join(log_dir, variant_log_file)
        logger.log_variant(variant_log_path, variant)

    if log_to_tensorboard:
        logger.set_log_to_tensorboard(log_to_tensorboard)

    if script_name is not None:
        with open(osp.join(log_dir, "script_name.txt"), "w") as f:
            f.write(script_name)

    return log_dir


def dict_to_safe_json(d):
    new_d = {}
    for key, item in d.items():
        if safe_json(item):
            new_d[key] = item
        else:
            if isinstance(item, dict):
                new_d[key] = dict_to_safe_json(item)
            else:
                new_d[key] = str(item)
    return new_d


def safe_json(data):
    if data is None:
        return True
    elif isinstance(data, (bool, int, float)):
        return True
    elif isinstance(data, (tuple, list)):
        return all(safe_json(x) for x in data)
    elif isinstance(data, dict):
        return all(isinstance(k, str) and safe_json(v) for k, v in data.items())
    return False
