import tensorize
import os
import json
import util

if __name__ == '__main__':
    data_path = '/users/sxl180006/research/codi2021/conllua_dir'
    data_fn = 'all_except_trains_91.entity_type_train.jsonlines'
    path = os.path.join(data_path, data_fn)

    config = util.initialize_config('train_spanbert_large_ml1_d1_for_light')
    tokenizer = util.get_tokenizer(config['bert_tokenizer_name'])

    tensorizer = tensorize.Tensorizer(config, tokenizer)
    with open(path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            sample = json.loads(line)
            one_tensor_sample = tensorizer.tensorize_example(sample, 0)
            print(max(one_tensor_sample[1][-2]))
    # for i in one_tensor_sample[0]:
    #     if isinstance(i, int):
    #         print(i)
    #         continue
    #     if isinstance(i, list):
    #         print(len(i))
    #         continue
    #     print(i.shape)