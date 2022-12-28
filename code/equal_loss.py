import json
import pdb


with open('/home2/pengyifan/pyf/hypergraph_cluster/log/graduate/vgg/mute_max.json','r',encoding='utf8')as fp:
    json_data = json.load(fp)
max_value = []
for k, v in json_data.items():
    max_value.append(v[1])

with open('/home2/pengyifan/pyf/hypergraph_cluster/log/graduate/vgg/mute_min.json','r',encoding='utf8')as fp:
    json_data = json.load(fp)
min_value = []
for k, v in json_data.items():
    min_value.append(v[1])

pdb.set_trace()