import pickle
import os

gt_file = '../data/crossval_gt.pkl'
with open(gt_file, 'rb') as f:
    data = pickle.load(f)


ALL_GT = {}
folds = list(data.keys())
for fold in folds:
    for split in ['train', 'test']:
        cur_data = data[fold][split]
        for x in cur_data:
            name = x['name']
            label = x['class']
            duration = x['seg']
            if name not in ALL_GT:
                ALL_GT[name] = {"label": label, "duration": duration}
            else:
                pass

# save the groundtruth
with open('../data/eval_gt.pkl', 'wb') as f:
    pickle.dump(ALL_GT, f)