# https://github.com/szagoruyko/functional-zoo/blob/master/visualize.py
# https://github.com/szagoruyko/functional-zoo/blob/master/resnet-18-export.ipynb
# https://discuss.pytorch.org/t/print-autograd-graph/692
#


from net.common import *
from graphviz import Digraph
from torch.autograd import Variable
import builtins
import time

# log ------------------------------------
def remove_comments(lines, token='#'):
    """ Generator. Strips comments and whitespace from input lines.
    """

    l = []
    for line in lines:
        s = line.split(token, 1)[0].strip()
        if s != '':
            l.append(s)
    return l


def open(file, mode=None, encoding=None):
    if mode == None: mode = 'r'

    if '/' in file:
        if 'w' or 'a' in mode:
            dir = os.path.dirname(file)
            if not os.path.isdir(dir):  os.makedirs(dir)

    f = builtins.open(file, mode=mode, encoding=encoding)
    return f


def remove(file):
    if os.path.exists(file): os.remove(file)


def empty(dir):
    if os.path.isdir(dir):
        shutil.rmtree(dir, ignore_errors=True)
    else:
        os.makedirs(dir)


# http://stackoverflow.com/questions/34950201/pycharm-print-end-r-statement-not-working
class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout  #stdout
        self.file = None

    def open(self, file, mode=None):
        if mode is None: mode ='w'
        self.file = open(file, mode)

    def write(self, message, is_terminal=1, is_file=1 ):
        if '\r' in message: is_file=0

        if is_terminal == 1:
            self.terminal.write(message)
            self.terminal.flush()
            #time.sleep(1)

        if is_file == 1:
            self.file.write(message)
            self.file.flush()

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass


# net ------------------------------------

## this is broken !!!
def make_dot(var):
    node_attr = dict(style='filled',
                     shape='box',
                     align='left',
                     fontsize='12',
                     ranksep='0.1',
                     height='0.2')
    dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))
    seen = set()

    def add_nodes(var):
        if var not in seen:
            if isinstance(var, Variable):
                value = '('+(', ').join(['%d'% v for v in var.size()])+')'
                dot.node(str(id(var)), str(value), fillcolor='lightblue')
            else:
                dot.node(str(id(var)), str(type(var).__name__))
            seen.add(var)
            if hasattr(var, 'previous_functions'):
                for u in var.previous_functions:
                    dot.edge(str(id(u[0])), str(id(var)))
                    add_nodes(u[0])
    add_nodes(var.creator)
    return dot


# https://github.com/pytorch/examples/blob/master/imagenet/main.py ###############
def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def get_learning_rate(optimizer):
    lr=[]
    for param_group in optimizer.param_groups:
       lr +=[ param_group['lr'] ]
    return lr

# https://discuss.pytorch.org/t/how-to-load-part-of-pre-trained-model/1113/3 ######

#params, stats = state_dict['params'], state_dict['stats']
#https://github.com/szagoruyko/attention-transfer/blob/master/imagenet.py
def load_valid(model, pretrained_dict, skip_list=[], log=None):

    model_dict = model.state_dict()

    # 1. filter out unnecessary keys
    pretrained_dict1 = {k: v for k, v in pretrained_dict.items() if k in model_dict and k not in skip_list }

    ## debug
    if 0:
        print('model_dict.keys()')
        print(model_dict.keys())
        print('pretrained_dict.keys()')
        print(pretrained_dict.keys())
        print('pretrained_dict1.keys()')
        print(pretrained_dict1.keys())

    #pring missing keys
    if log is not None:
        log.write('--missing keys at load_valid():--\n')
        for k in model_dict.keys():
            if k not in pretrained_dict1.keys():
                log.write('\t %s\n'%k)

        log.write('------------------------\n')

    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict1)
    model.load_state_dict(model_dict)




# test-time augment, ensemble etc ------------------------------------
def change_images(images, agument):

    num = len(images)
    h,w = images[0].shape[0:2]
    if agument == 'left-right' :
        for n in range(num):
            image = images[n]
            images[n] = cv2.flip(image,1)

    if agument == 'up-down' :
        for n in range(num):
            image = images[n]
            images[n] = cv2.flip(image,0)


    if agument == 'center' :
        for n in range(num):
            b=8
            image = images[n, b:h-b,b:w-b,:]
            images[n]  = cv2.resize(image,(w,h),interpolation=cv2.INTER_LINEAR)


    if agument == 'transpose' :
        for n in range(num):
            image = images[n]
            images[n] = image.transpose(1,0,2)


    if agument == 'rotate90' :
        for n in range(num):
            image = images[n]
            image = image.transpose(1,0,2)  #cv2.transpose(img)
            images[n]  = cv2.flip(image,1)


    if agument == 'rotate180' :
        for n in range(num):
            image = images[n]
            images[n] = cv2.flip(image,-1)


    if agument == 'rotate270' :
        for n in range(num):
            image = images[n]
            image = image.transpose(1,0,2)  #cv2.transpose(img)
            images[n]  = cv2.flip(image,0)

    return images



# f2 score etc ------------------------------------

#https://www.kaggle.com/paulorzp/planet-understanding-the-amazon-from-space/find-best-f2-score-threshold/code
def find_f_measure_threshold_fix(probs, labels, beta=2):

    #f0 = fbeta_score(labels, probs, beta=2, average='samples')
    def f_measure(probs, labels, threshold=0.5, beta=beta ):

        SMALL = 1e-12 #0  #1e-12
        batch_size, num_classes = labels.shape[0:2]

        l = labels
        p = probs>threshold

        num_pos     = p.sum(axis=1) + SMALL
        num_pos_hat = l.sum(axis=1)
        tp          = (l*p).sum(axis=1)
        precise     = tp/num_pos
        recall      = tp/num_pos_hat

        fs = (1+beta*beta)*precise*recall/(beta*beta*precise + recall + SMALL)
        f  = fs.sum()/batch_size
        return f


    batch_size, num_classes = labels.shape[0:2]
    thresholds = np.arange(0,1,0.005)
    ##thresholds = np.unique(probs)

    best_threshold =  0
    best_score     = -1

    N = len(thresholds)
    scores = np.zeros(N,np.float32)
    for n in range(N):
        t = thresholds[n]
        #score = f_measure(probs, labels, threshold=t)
        score = fbeta_score(labels, probs>t, beta=beta, average='samples')
        scores[n] = score

    i = np.argmax(scores)
    best_threshold, best_score = thresholds[i], scores[i]
    best_threshold = np.ones(num_classes,np.float32)*best_threshold

    return best_threshold, best_score


def find_f_measure_threshold_per_class (probs, labels, num_iters=100, init_thresholds=0.235, beta=2):

    batch_size, num_classes = labels.shape[0:2]
    if isinstance(init_thresholds, numbers.Number) :
        init_thresholds = np.ones(num_classes,np.float32)*init_thresholds

    best_thresholds = init_thresholds.copy()

    print('-----------------------------------------------')
    print('\tscore, t, best_thresholds[t]')
    for t in range(num_classes):
        thresholds = init_thresholds.copy()
        score = 0
        for i in range(num_iters):
            thresholds[t] = i / float(num_iters)
            f2 = fbeta_score(labels, probs > thresholds, beta=beta, average='samples')
            if  f2 > score:
                score = f2
                best_thresholds[t] =  thresholds[t]
        print('\t%0.5f, %2d, %0.3f'%(score, t, best_thresholds[t]))
    print('')

    best_score = fbeta_score(labels, probs > best_thresholds, beta=beta, average='samples')
    return best_thresholds, best_score

