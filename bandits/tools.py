import json
import gzip
import numpy as np


ARRAY_KEYS = ['chosen_arm', 'reward', 'expected_reward', 'expected_best_reward']

def record_zip(filename, logs):
    # remove arrays
    for log in logs:
        for key in log.keys():
            if key in ARRAY_KEYS:
                log[key] = log[key].tolist()

    print('file', filename)
    json_str = json.dumps(logs)
    json_bytes = json_str.encode('utf-8')     
    with gzip.GzipFile(filename, 'w') as fout:  
        fout.write(json_bytes)
    return 'done'


def retrieve_data_from_zip(file_name):
    with gzip.GzipFile(file_name, 'r') as fin:
        json_bytes = fin.read()

    json_str = json_bytes.decode('utf-8')            # 2. string (i.e. JSON)
    logs = json.loads(json_str)

    # add arrays
    for log in logs:
        for key in log.keys():
            if key in ARRAY_KEYS:
                log[key] = np.array(log[key])

    return logs
