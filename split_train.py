import numpy as np
import pandas as pds

def split_train_validation(num_val=3000):
    """
    Save train image names and validation image names to csv files
    """
    train_image_idx = np.sort(np.random.choice(40479, 40479-num_val, replace=False))
    all_idx = np.arange(40479)
    validation_image_idx = np.zeros(num_val, dtype=np.int32)
    val_idx = 0
    train_idx = 0
    for i in all_idx:
        if i not in train_image_idx:
            validation_image_idx[val_idx] = i
            val_idx += 1
        else:
            train_idx += 1
    # save train
    train = []
    for name in train_image_idx:
        train.append('train_%s' % name)

    eval = []
    for name in validation_image_idx:
        eval.append('train_%s' % name)

    df = pds.DataFrame(train)
    df.to_csv('dataset/train-%s' % (40479 - num_val), index=False, header=False)

    df = pds.DataFrame(eval)
    df.to_csv('dataset/validation-%s' % num_val, index=False, header=False)


split_train_validation(num_val=3000)
