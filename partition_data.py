import random
import numpy as np

def create_folds(data, n): 
    '''splits dataset into balanced folds by sorting targets based on number of examples
    and filling the n-bins in a back-and-forth manner. returns folds which is a list containing
    n lists of target names in each bin'''
    
    targets = list(data.keys())
    sorted_targets = []
    for target in targets:
        sorted_targets.append((target, len(data[target]['y'])))
    sorted_targets.sort(key=lambda tup: tup[1], reverse=True)

    index = -1
    forward = True
    folds = [[] for i in range(n)]
    for tuple in sorted_targets:
        if forward:
            index += 1
            if index == n:
                index -= 1
                forward = False
        else:
            index -= 1
            if index == -1:
                index += 1
                forward = True
        target = tuple[0]
        folds[index].append(target)
    
#     print_amt_in_each_bin(data, folds)
            
    return folds
    
def leave_one_target_out(data, targetnum):
    targets = list(data.keys())
    left_out = [targets.pop(targetnum)]
    return assemble_data(data, left_out, targets) #leftout is testing targ, rest of targets are training


def bootstrap_sampling(data, x, seed):
    '''given the data dictionary, x% and a value to use for the random number seed (so results
    are reproducible), returns testing sets x/y which is x% of the data, randomly chosen, and
    training sets x/y which is everything else'''
    
    testing_targets = []
    training_targets = []
    
    targets = list(data.keys())
    sample_size = int(round(float(x)/100 * len(targets)))
    random.seed(seed) #for reproducibility
    indices = random.sample(xrange(len(targets)), sample_size)
    for index in sorted(indices, reverse=True): #remove indices in reverse order
        testing_targets.append(targets.pop(index))
    training_targets = targets #the ones that are left
    
#     print_testing_and_training(testing_targets, training_targets)
    
    return assemble_data(data, testing_targets, training_targets)

def split_into_sets(data, folds, withheld_fold):
    '''returns testing sets x/y and training sets x/y given the target names split into folds 
    and the int value of the fold to withhold (as the testing set)'''
    
    folds = list(folds)
    testing_targets = folds.pop(withheld_fold)
    training_targets = [target for sublist in folds for target in sublist] #flattens list of lists
    
#     print_testing_and_training(testing_targets, training_targets)
    
    return assemble_data(data, testing_targets, training_targets)

def assemble_data(data, testing_targets, training_targets):
    '''given the data dictionary, a list of testing target names, and a list of training target
    names, return 4 lists: train_x, train_y, test_x, test_y'''
    
    train_x = []
    train_y = []
    test_x = []
    test_y = []

    for target in training_targets:
        targetdata_x = data[target]['x']
        targetdata_y = data[target]['y']
        for features in targetdata_x:
            train_x.append(features)
        for val in targetdata_y:
            train_y.append(val)

    for target in testing_targets:
        targetdata_x = data[target]['x']
        targetdata_y = data[target]['y']
        for features in targetdata_x:
            test_x.append(features)
        for val in targetdata_y:
            test_y.append(val)
    
    return test_x, test_y, train_x, train_y

  
def create_dict(file):
    '''opens the scoredata file and processes it into a dictionary called 'data' with the form:
    {'targetname' -> {x: numpy.array, y: numpy.array}, ..... }
    where x is a list of features and y is the corresponding inactive/active value (0 or 1)'''
    
    f = open(file)
    data = dict()
    for line in f:
        tokens = line.rstrip().split()
        target = tokens[1]
        y = int(tokens[0])
        x = tokens[4:]
        if target in data:
            data[target]['x'].append(x)
            data[target]['y'].append(y)
    
        else:
            data[target] = {'x' : [x], 'y' : [y]}

    for target in data: #convert to numpy
        data[target]['x'] = np.array(data[target]['x'], np.float64)
        data[target]['y'] = np.array(data[target]['y'])
    
    data.pop("fgfr1", None) #throw out this target bc no neg data
        
    return data
    

################ TESTING STUFF ################
def print_amt_in_each_bin(data, folds):
    '''prints the amount of targets and examples in each fold - used to verify the behavior
    of create_folds and check for an evenly distributed partition'''
    sum = 0
    targets = 0
    for fold in folds:
        for target in fold:
            sum += len(data[target]['y'])
            targets += 1
        print str(targets) + " targets and " + str(sum) + " examples "
        sum = 0
        targets = 0
        
def print_testing_and_training(testing_targets, training_targets):
    '''prints names of all targets in current testing set and all targets in current training set'''
    print str(len(testing_targets)) + " TESTING TARGETS:", testing_targets
    print str(len(training_targets)) + " TRAINING TARGETS:", training_targets
    

    
    