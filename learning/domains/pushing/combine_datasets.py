import os 
import pickle 
to_combine = ["one_object_large", "one_object_large2", "one_object_large3", "one_object_large4", "one_object_large5", "one_object_large6", "one_object_large7"] 
new_dataset = "one_object_final" 

def combine_datasets(comb, new): 
    total_train = []
    total_val = []
    total_test = []

    for dataset in comb: 
        with open(os.path.join('learning', 'data', 'pushing', dataset, 'train_dataset.pkl'), 'rb') as f: 
            train = pickle.load(f) 
        with open(os.path.join('learning', 'data', 'pushing', dataset, 'samegeo_test_dataset.pkl'), 'rb') as f: 
            val = pickle.load(f) 
        with open(os.path.join('learning', 'data', 'pushing', dataset, 'test_dataset.pkl'), 'rb') as f: 
            test = pickle.load(f) 

        total_train.extend(train)
        total_val.extend(val)
        total_test.extend(test) 
    os.makedirs(os.path.join('learning', 'data', 'pushing', new_dataset), exist_ok=True) 

    print(len(total_train), len(total_val), len(total_test))
    with open(os.path.join('learning', 'data', 'pushing', new_dataset, 'train_dataset.pkl'), 'wb') as f:
        pickle.dump(total_train, f)
    with open(os.path.join('learning', 'data', 'pushing', new_dataset, 'samegeo_test_dataset.pkl'), 'wb') as f:
        pickle.dump(total_val, f)
    with open(os.path.join('learning', 'data', 'pushing', new_dataset, 'test_dataset.pkl'), 'wb') as f:
        pickle.dump(total_test, f)
combine_datasets(to_combine, new_dataset)




