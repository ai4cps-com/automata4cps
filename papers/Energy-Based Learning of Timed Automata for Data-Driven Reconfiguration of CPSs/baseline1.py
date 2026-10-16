import json
import random
import numpy as np


def conveyor_system_random_selection_baseline(current_config):
    return random.choice([x for x in ["Config1", "Config2", "Config3", "Config4", None] if x != current_config])


if __name__ == "__main__":
    random.seed(123)
    np.random.seed(123)

    with open('conveyor_system_reconfig_data.json', 'r') as f:
        data = json.load(f)

    for exp_ind in range(1, 11):
        sampled_configs = data['current_configs']
        sampled_target_configs = data['target_configs']

        correct = 0
        for current_config, target in zip(sampled_configs, sampled_target_configs):
            predicted_config = conveyor_system_random_selection_baseline(current_config)
            if predicted_config in target:
                correct += 1
            elif len(target) == 0 and predicted_config is None:
                correct += 1
        print("Random reconfiguration baseline {}: Number correct: {}".format(exp_ind, correct))
