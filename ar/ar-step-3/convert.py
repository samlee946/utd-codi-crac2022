import helper
import os

path = '../conllua_dir'
_, _, files = next(os.walk(path))
for fn in files:
    UA_PATH = os.path.join(path, fn)
    JSON_PATH = os.path.join(path, fn.split('.')[0] + '.jsonlines')
    helper.convert_coref_ua_to_json(UA_PATH, JSON_PATH, MODEL="coref-hoi", SEGMENT_SIZE=512, TOKENIZER_NAME="bert-base-cased")
