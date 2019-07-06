import numpy as np
import json

def max_min_normalization(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))

if __name__ == '__main__':
    result = np.array(1)

    # result = result.tolist()
    # print(result)
    # print(type(result))
    # data = {"name": result, "age": 18}
    #
    # with open("json文件路径", 'w') as json_file:
    #     json.dump(data, json_file)

