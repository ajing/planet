## https://github.com/colesbury/examples/blob/8a19c609a43dc7a74d1d4c71efcda102eed59365/imagenet/main.py
## https://github.com/pytorch/examples/blob/master/imagenet/main.py




from net.common import *
from net.dataset.kgforest import *

from net.rates import *
from net.util import *


# from net.model.pyramidnet import PyNet_12  as Net
#from net.model.fusenet1 import FuseNet as Net
#from net.model.pyramidnet1 import pyresnet34 as Net
# --------------------------------------------

#from net.model.resnet import resnet18 as Net
#from net.model.resnet import resnet34 as Net
from net.model.resnet import resnet50 as Net

#from net.model.densenet import densenet121 as Net
#from net.model.densenet import densenet169 as Net


#from net.model.inceptionv2 import Inception2 as Net
#from net.model.inceptionv3 import Inception3 as Net
#from net.model.inceptionv4 import Inception4 as Net


#from net.model.vggnet import vgg16 as Net
#from net.model.vggnet import vgg16_bn as Net



## global setting ################
SIZE =  288   #256   #128  #112
EXT  = 'jpg'  #'jpg'

'''
299
288 9
256	8
224	7
192	6
160	5
128	4

'''


# write csv
def write_submission_csv(csv_file, predictions, thresholds, split):

    class_names = CLASS_NAMES
    num_classes = len(class_names)

    with open(KAGGLE_DATA_DIR +'/split/'+ split) as f:
        names = f.readlines()
    names = [x.strip() for x in names]
    num_test = len(names)


    assert((num_test,num_classes) == predictions.shape)
    with open(csv_file,'w') as f:
        f.write('image_name,tags\n')
        for n in range(num_test):
            shortname = names[n].split('/')[-1].replace('.<ext>','')

            prediction = predictions[n]
            s = score_to_class_names(prediction, class_names, threshold=thresholds)
            f.write('%s,%s\n'%(shortname,s))



# loss ----------------------------------------
def multi_criterion(logits, labels):
    loss = nn.MultiLabelSoftMarginLoss()(logits, Variable(labels))
    return loss

#https://www.kaggle.com/paulorzp/planet-understanding-the-amazon-from-space/find-best-f2-score-threshold/code
#f  = fbeta_score(labels, probs, beta=2, average='samples')
def multi_f_measure( probs, labels, threshold=0.235, beta=2 ):

    SMALL = 1e-6 #0  #1e-12
    batch_size = probs.size()[0]

    #weather
    l = labels
    p = (probs>threshold).float()

    num_pos     = torch.sum(p,  1)
    num_pos_hat = torch.sum(l,  1)
    tp          = torch.sum(l*p,1)
    precise     = tp/(num_pos     + SMALL)
    recall      = tp/(num_pos_hat + SMALL)

    fs = (1+beta*beta)*precise*recall/(beta*beta*precise + recall + SMALL)
    f  = fs.sum()/batch_size
    return f



## main functions ############################################################
def predict(net, test_loader):

    test_dataset = test_loader.dataset
    num_classes  = len(test_dataset.class_names)
    predictions  = np.zeros((test_dataset.num,num_classes),np.float32)

    test_num  = 0
    for iter, (images, indices) in enumerate(test_loader, 0):

        # forward
        logits, probs = net(Variable(images.cuda(),volatile=True))

        batch_size = len(images)
        test_num  += batch_size
        start = test_num-batch_size
        end   = test_num
        predictions[start:end] = probs.data.cpu().numpy().reshape(-1,num_classes)

    assert(test_dataset.num==test_num)
    return predictions



def evaluate( net, test_loader ):

    test_num  = 0
    test_loss = 0
    test_acc  = 0
    for iter, (images, labels, indices) in enumerate(test_loader, 0):

        # forward
        logits, probs = net(Variable(images.cuda(),volatile=True))
        loss  = multi_criterion(logits, labels.cuda())

        batch_size = len(images)
        test_acc  += batch_size*multi_f_measure(probs.data, labels.cuda())
        test_loss += batch_size*loss.data[0]
        test_num  += batch_size

    assert(test_num == test_loader.dataset.num)
    test_acc  = test_acc/test_num
    test_loss = test_loss/test_num
    return test_loss, test_acc



def evaluate_and_predict(net, test_loader):

    test_dataset = test_loader.dataset
    num_classes  = len(test_dataset.class_names)
    predictions  = np.zeros((test_dataset.num,num_classes),np.float32)

    test_num  = 0
    test_loss = 0
    test_acc  = 0
    for iter, (images, labels, indices) in enumerate(test_loader, 0):

        # forward
        logits, probs = net(Variable(images.cuda(),volatile=True))
        loss  = multi_criterion(logits, labels.cuda())

        batch_size = len(images)
        test_acc  += batch_size*multi_f_measure(probs.data, labels.cuda())
        test_loss += batch_size*loss.data[0]
        test_num  += batch_size

        start = test_num-batch_size
        end   = test_num
        predictions[start:end] = probs.data.cpu().numpy().reshape(-1,num_classes)

    assert(test_dataset.num==test_num)
    test_acc  = test_acc/test_num
    test_loss = test_loss/test_num
    return test_loss, test_acc, predictions


#-----------------------------------------------------------
def augment(x, u=0.75):
    if random.random()<u:
        if random.random()>0.5:
            x = randomDistort1(x, distort_limit=0.35, shift_limit=0.25, u=1)
        else:
            x = randomDistort2(x, num_steps=10, distort_limit=0.2, u=1)
        x = randomShiftScaleRotate(x, shift_limit=0.0625, scale_limit=0.10, rotate_limit=45, u=1)

    x = randomFlip(x, u=0.5)
    x = randomTranspose(x, u=0.5)
    x = randomContrast(x, limit=0.2, u=0.5)
    #x = randomSaturation(x, limit=0.2, u=0.5),
    x = randomFilter(x, limit=0.5, u=0.2)
    return x


def do_training():

    out_dir ='/root/share/project/pytorch/results/kaggle-forest/resnet50-train-40479-pretrain-288-aug-new-00'
    initial_checkpoint = None #'/root/share/project/pytorch/results/kaggle-forest/inception_v3-train-40479-pretrain-288-aug-new-18/checkpoint/030.pth'  #None
    initial_model      = None

    #pretrained_file = '/root/share/project/pytorch/data/pretrain/densenet/densenet121-241335ed.pth'
    #pretrained_file = '/root/share/project/pytorch/data/pretrain/densenet/densenet169-6f0f7f60.pth'
    #pretrained_file = '/root/share/project/pytorch/data/pretrain/vgg/vgg16-397923af.pth'
    #pretrained_file = '/root/share/project/pytorch/data/pretrain/inception/inception_v3_google-1a9a5a14.pth'
    #pretrained_file = None #'/root/share/project/pytorch/data/pretrain/inception/inceptionv4-58153ba9.pth'
    #pretrained_file = '/root/share/project/pytorch/data/pretrain/resnet/resnet18-5c106cde.pth'
    #pretrained_file = '/root/share/project/pytorch/data/pretrain/resnet/resnet34-333f7ec4.pth'
    pretrained_file = '/root/share/project/pytorch/data/pretrain/resnet/resnet50-19c8e357.pth'
    #pretrained_file = None #

    skip_list = ['fc.weight', 'fc.bias']
    #skip_list = ['fc.0.weight', 'fc.0.bias', 'fc.3.weight', 'fc.3.bias', 'fc.6.weight', 'fc.6.bias'] # #for vgg16

    ## ------------------------------------
    os.makedirs(out_dir +'/snap', exist_ok=True)
    os.makedirs(out_dir +'/checkpoint', exist_ok=True)

    log = Logger()
    log.open(out_dir+'/log.train.txt',mode='a')
    log.write('\n--- [START %s] %s\n\n' % (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '-' * 64))
    log.write('** some experiment setting **\n')
    log.write('\tSEED    = %u\n' % SEED)
    log.write('\tfile    = %s\n' % __file__)
    log.write('\tout_dir = %s\n' % out_dir)
    log.write('\n')


    ## dataset ----------------------------------------
    log.write('** dataset setting **\n')
    num_classes = len(CLASS_NAMES)
    batch_size  = 48  #96 #96  #80 #96 #96   #96 #32  #96 #128 #

    train_dataset = KgForestDataset('train-40479',  #'train-32479',#'train-32479-unsuper-31936' , #'train-3000',#'train-8000',#'debug-32', #'train-3000',#'train-8000',###
                                    transform=[
                                        lambda x: augment(x),
                                        lambda x: img_to_tensor(x),
                                    ],
                                    height=SIZE, width=SIZE, ext=EXT,
                                    is_preload=False,
                                    label_csv='train_v2-unsuper-31936.csv' #'train_v2.csv'
                                    )
    train_loader  = DataLoader(
                        train_dataset,
                        sampler = RandomSampler(train_dataset),  ##
                        batch_size  = batch_size,
                        drop_last   = True,
                        num_workers = 3,
                        pin_memory  = True)

    test_dataset = KgForestDataset('valid-8000', #'debug-32', #'train-3000',#'valid-8000',
                                    transform=[
                                        lambda x: img_to_tensor(x),
                                    ],
                                    height=SIZE, width=SIZE, ext=EXT,
                                    is_preload=True,
                                    label_csv='train_v2-unsuper-31936.csv' #'train_v2.csv'
                                    )
    test_loader  = DataLoader(
                        test_dataset,
                        sampler     = SequentialSampler(test_dataset),  #None,
                        batch_size  = batch_size,
                        drop_last   = False,
                        num_workers = 2,
                        pin_memory  = True)


    height, width , in_channels   =  test_dataset.height, test_dataset.width, test_dataset.channel
    log.write('\t(height,width) = (%d, %d)\n'%(height,width))
    log.write('\tin_channels    = %d\n'%(in_channels))
    log.write('\ttrain_dataset.split  = %s\n'%(train_dataset.split))
    log.write('\ttrain_dataset.num    = %d\n'%(train_dataset.num))
    log.write('\ttest_dataset.split   = %s\n'%(test_dataset.split))
    log.write('\ttest_dataset.num     = %d\n'%(test_dataset.num))
    log.write('\tbatch_size           = %d\n'%batch_size)
    log.write('\ttrain_loader.sampler = %s\n'%(str(train_loader.sampler)))
    log.write('\ttest_loader.sampler  = %s\n'%(str(test_loader.sampler)))
    log.write('\n')

    log.write(inspect.getsource(augment)+'\n')
    log.write('\n')

    # if 0:  ## check data
    #     check_kgforest_dataset(train_dataset, train_loader)
    #     exit(0)


    ## net ----------------------------------------
    log.write('** net setting **\n')
    net = Net(in_shape = (in_channels, height, width), num_classes=num_classes)

    net.cuda()
    #log.write('\n%s\n'%(str(net)))
    log.write('%s\n\n'%(type(net)))
    log.write(inspect.getsource(net.__init__)+'\n')
    log.write(inspect.getsource(net.forward)+'\n')
    log.write('\n')


    ## optimiser ----------------------------------
    # LR = StepLR([ (0,0.1),  (10,0.01),  (25,0.005),  (35,0.001), (40,0.0001), (43,-1)])
    ## fine tunning
    LR = StepLR([ (0, 0.01),  (10, 0.005),  (23, 0.001),  (35, 0.0001), (38,-1)])
    #LR = CyclicLR(base_lr=0.001, max_lr=0.01, step=5., mode='triangular', gamma=1., scale_fn=None, scale_mode='cycle')

    num_epoches = 50  #100
    it_print    = 20  #20
    epoch_test  = 1
    epoch_save  = 5

    #optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=0.001)
    optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0005)  ###0.0005

    #only fc
    #optimizer = optim.SGD(net.fc.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0005)
    #print some params for debug
    ## print(net.features.state_dict()['0.conv.weight'][0,0])


    ## resume from previous ----------------------------------
    start_epoch=0
    if initial_checkpoint is not None:
        checkpoint  = torch.load(initial_checkpoint)
        start_epoch = checkpoint['epoch']
        optimizer.load_state_dict(checkpoint['optimizer'])
        pretrained_dict = checkpoint['state_dict']
        load_valid(net, pretrained_dict,skip_list=[], log=log)

    elif pretrained_file is not None:  #pretrain
        pretrained_dict = torch.load(pretrained_file)
        load_valid(net, pretrained_dict, skip_list=skip_list, log=log)

    elif initial_model is not None:
        net = torch.load(initial_model)



    ## start training here! ##############################################3
    log.write('** start training here! **\n')

    log.write(' optimizer=%s\n'%str(optimizer) )
    log.write(' LR=%s\n\n'%str(LR) )
    log.write(' epoch   iter   rate  |  smooth_loss   |  train_loss  (acc)  |  valid_loss  (acc)  | min\n')
    log.write('----------------------------------------------------------------------------------------\n')

    smooth_loss = 0.0
    train_loss  = np.nan
    train_acc   = np.nan
    test_loss   = np.nan
    test_acc    = np.nan
    time = 0

    start0 = timer()
    for epoch in range(start_epoch, num_epoches):  # loop over the dataset multiple times
        #print ('epoch=%d'%epoch)
        start = timer()

        #---learning rate schduler ------------------------------
        lr =  LR.get_rate(epoch, num_epoches)
        if lr<0 :break

        adjust_learning_rate(optimizer, lr)
        rate =  get_learning_rate(optimizer)[0] #check
        #--------------------------------------------------------

        sum_smooth_loss = 0.0
        sum = 0
        net.cuda().train()
        num_its = len(train_loader)
        for it, (images, labels, indices) in enumerate(train_loader, 0):

            logits, probs = net(Variable(images.cuda()))
            loss  = multi_criterion(logits, labels.cuda())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            #additional metrics
            sum_smooth_loss += loss.data[0]
            sum += 1

            #print some params for debug
            ##print(net.features.state_dict()['0.conv.weight'][0,0])

            # print statistics
            if it % it_print == it_print-1:
                smooth_loss = sum_smooth_loss/sum
                sum_smooth_loss = 0.0
                sum = 0

                train_acc  = multi_f_measure(probs.data, labels.cuda())
                train_loss = loss.data[0]

                print('\r%5.1f   %5d    %0.4f   |  %0.4f  | %0.4f  %6.4f | ... ' % \
                        (epoch + it/num_its, it + 1, rate, smooth_loss, train_loss, train_acc),\
                        end='',flush=True)


        #---- end of one epoch -----
        end = timer()
        time = (end - start)/60

        if epoch % epoch_test == epoch_test-1  or epoch == num_epoches-1:
            net.cuda().eval()
            test_loss,test_acc = evaluate(net, test_loader)

            print('\r',end='',flush=True)
            log.write('%5.1f   %5d    %0.4f   |  %0.4f  | %0.4f  %6.4f | %0.4f  %6.4f  |  %3.1f min \n' % \
                    (epoch + 1, it + 1, rate, smooth_loss, train_loss, train_acc, test_loss,test_acc, time))

        if epoch % epoch_save == epoch_save-1 or epoch == num_epoches-1:
            torch.save(net, out_dir +'/snap/%03d.torch'%(epoch+1))
            torch.save({
                'state_dict': net.state_dict(),
                'optimizer' : optimizer.state_dict(),
                'epoch'     : epoch,
            }, out_dir +'/checkpoint/%03d.pth'%(epoch+1))
            ## https://github.com/pytorch/examples/blob/master/imagenet/main.py


    #---- end of all epoches -----
    end0  = timer()
    time0 = (end0 - start0) / 60

    ## check : load model and re-test
    torch.save(net,out_dir +'/snap/final.torch')
    if 1:
        net = torch.load(out_dir +'/snap/final.torch')
        net.cuda().eval()
        test_loss, test_acc, predictions = evaluate_and_predict( net, test_loader )

        log.write('\n')
        log.write('%s:\n'%(out_dir +'/snap/final.torch'))
        log.write('\tall time to train=%0.1f min\n'%(time0))
        log.write('\ttest_loss=%f, test_acc=%f\n'%(test_loss,test_acc))



##to determine best threshold etc ... ## -----------------------------------------------------------

def do_thresholds():

    out_dir ='/root/share/project/pytorch/results/kaggle-forest/resnet50-train-40479-pretrain-288-aug-new-00'
    model_file = out_dir +'/snap/020.torch'  #final

    ## ------------------------------------
    log = Logger()
    log.open(out_dir+'/thresholds/log.thresholds.txt',mode='a')
    log.write('\n--- [START %s] %s\n\n' % (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '-' * 64))
    log.write('\n')

    ## dataset ----------------------------
    log.write('** dataset setting **\n')
    batch_size = 64

    test_dataset = KgForestDataset('train-40479',  #'train-32479', #'train-40479',  #'valid-8000',
                                    transform=[
                                         lambda x: img_to_tensor(x),
                                    ],
                                    height=SIZE, width=SIZE, ext=EXT,
                                    is_preload=True,
                                    label_csv='train_v2.csv')
    test_loader  = DataLoader(
                        test_dataset,
                        sampler     = SequentialSampler(test_dataset),
                        batch_size  = batch_size,
                        drop_last   = False,
                        num_workers = 4,
                        pin_memory  = True)

    height, width , in_channels = test_dataset.images[0].shape
    log.write('\t(height,width)     = (%d, %d)\n'%(height,width))
    log.write('\tin_channels        = %d\n'%(in_channels))
    log.write('\ttest_dataset.split = %s\n'%(test_dataset.split))
    log.write('\ttest_dataset.num   = %d\n'%(test_dataset.num))
    log.write('\tbatch_size         = %d\n'%batch_size)
    log.write('\n')


    ## net ----------------------------------------
    log.write('** net setting **\n')
    log.write('\tmodel_file = %s\n'%model_file)
    log.write('\n')

    net = torch.load(model_file)
    net.cuda().eval()

    # do testing here ###########
    aguments = ['default', 'left-right', 'up-down', 'transpose',  'rotate90', 'rotate180', 'rotate270', ]
    num_augments = len(aguments)
    num_classes  = len(test_dataset.class_names)
    test_num     = test_dataset.num
    test_dataset_images = test_dataset.images.copy()

    all_predictions = np.zeros((num_augments+3, test_num, num_classes),np.float32)
    for a in range(num_augments):
        agument = aguments[a]
        log.write('** predict @ agument = %s **\n'%agument)

        test_dataset.images = change_images(test_dataset_images,agument)
        test_loss, test_acc, predictions = evaluate_and_predict( net, test_loader )
        all_predictions[a] = predictions

        log.write('\t\ttest_loss=%f, test_acc=%f\n\n'%(test_loss,test_acc))

    # add ensemble average, etc ...
    log.write('\n')
    aguments = aguments + ['average']
    predictions = all_predictions.sum(axis=0)/num_augments
    all_predictions[num_augments+0] = predictions

    aguments = aguments + ['max']
    predictions = all_predictions.max(axis=0)
    all_predictions[num_augments+1] = predictions

    aguments = aguments + ['vote']
    # all_predictions[num_augments+2] += (predictions>best_thresholds)/num_augments

    # find thresholds and save all
    labels = test_dataset.labels
    for a in range(num_augments+3):
        agument = aguments[a]
        predictions = all_predictions[a]

        test_dir = out_dir +'/thresholds/'+ agument
        os.makedirs(test_dir, exist_ok=True)
        log.write('** thresholding @ test_dir = %s **\n'%test_dir)

        #find threshold --------------------------

        #fixed threshold
        best_thresholds, best_score  = find_f_measure_threshold_fix(predictions, labels)
        f2 = fbeta_score(labels, predictions > best_thresholds, beta=2, average='samples')

        log.write('*best_threshold (fixed)*\n')
        log.write(np.array2string(best_thresholds, formatter={'float_kind':lambda x: ' %.3f' % x}, separator=','))
        log.write('\n')
        log.write('*best_score*\n')
        log.write('%0.4f (%0.4f)\n'%(best_score, f2))

        #per class threshold
        init_thresholds = best_thresholds.copy()
        best_thresholds,  best_score  = find_f_measure_threshold_per_class( predictions, labels, init_thresholds=init_thresholds)
        f2 = fbeta_score(labels, predictions > best_thresholds, beta=2, average='samples')

        log.write('*best_threshold (per class)*\n')
        log.write(np.array2string(best_thresholds, formatter={'float_kind':lambda x: ' %.3f' % x}, separator=','))
        log.write('\n')
        log.write('*best_score*\n')
        log.write('%0.4f (%0.4f)\n\n'%(best_score, f2))


        #save
        assert(type(test_loader.sampler)==torch.utils.data.sampler.SequentialSampler)
        np.save(test_dir +'/predictions.npy',predictions)
        np.save(test_dir +'/labels.npy',labels)
        np.savetxt(test_dir +'/best_threshold.txt', best_thresholds,fmt='%.5f' )
        np.savetxt(test_dir +'/best_score.txt',np.array([best_score]),fmt='%.5f')

        #- voting ------------
        if a<num_augments:
            all_predictions[num_augments+2] += (predictions>best_thresholds)/num_augments

    # pass


##-----------------------------------------
def do_submissions():


    out_dir ='/root/share/project/pytorch/results/kaggle-forest/resnet50-train-40479-pretrain-288-aug-new-00'
    model_file = out_dir +'/snap/020.torch'  #final

    ## ------------------------------------

    log = Logger()
    log.open(out_dir+'/submissions/log.submissions.txt',mode='a')
    log.write('\n--- [START %s] %s\n\n' % (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '-' * 64))
    log.write('\n')

    ## dataset ----------------------------
    log.write('** dataset setting **\n')
    batch_size    = 96  #128

    test_dataset = KgForestDataset('test-61191',  #'valid-8000', #
                                    transform=[
                                         lambda x: img_to_tensor(x),
                                    ],
                                    height=SIZE,width=SIZE,
                                    label_csv=None)
    test_loader  = DataLoader(
                        test_dataset,
                        sampler     = SequentialSampler(test_dataset),  #None,
                        batch_size  = batch_size,
                        drop_last   = False,
                        num_workers = 4,
                        pin_memory  = True)

    height, width , in_channels = test_dataset.images[0].shape
    log.write('\t(height,width)    = (%d, %d)\n'%(height,width))
    log.write('\tin_channels       = %d\n'%(in_channels))
    log.write('\ttest_dataset.split = %s\n'%(test_dataset.split))
    log.write('\ttest_dataset.num  = %d\n'%(test_dataset.num))
    log.write('\tbatch_size        = %d\n'%batch_size)
    log.write('\n')


    ## net ----------------------------------------
    log.write('** net setting **\n')
    log.write('\tmodel_file = %s\n'%model_file)
    log.write('\n')

    net = torch.load(model_file)
    net.cuda().eval()


    # do testing here ###
    aguments = ['default', 'left-right', 'up-down', 'transpose', 'rotate90', 'rotate180', 'rotate270', ]
    num_augments = len(aguments)
    num_classes  = len(test_dataset.class_names)
    test_num     = test_dataset.num
    test_dataset_images = test_dataset.images.copy()

    all_predictions = np.zeros((num_augments+3,test_num,num_classes),np.float32)
    for a in range(num_augments):
        agument = aguments[a]
        log.write('** predict @ agument = %s **\n'%agument)

        ## perturb here for test argumnetation  ## ----
        test_dataset.images = change_images(test_dataset_images,agument)
        predictions = predict( net, test_loader )
        all_predictions[a] = predictions

    # add average case, etc ...
    aguments = aguments + ['average']
    predictions = all_predictions.sum(axis=0)/num_augments
    all_predictions[num_augments] = predictions
    log.write('\n')

    aguments = aguments + ['max']
    predictions = all_predictions.max(axis=0)
    all_predictions[num_augments+1] = predictions

    aguments = aguments + ['vote']
    # all_predictions[num_augments+2] += (predictions>best_thresholds)/num_augments


    # apply thresholds and save all
    for a in range(num_augments+3):
        thresholds = np.loadtxt(out_dir +'/thresholds/'+ agument+'/best_threshold.txt')

        agument = aguments[a]
        predictions = all_predictions[a]

        test_dir = out_dir +'/submissions/'+ agument
        os.makedirs(test_dir, exist_ok=True)

        #save
        assert(type(test_loader.sampler)==torch.utils.data.sampler.SequentialSampler)
        np.save(test_dir +'/predictions.npy',predictions)
        np.savetxt(test_dir +'/thresholds.txt',thresholds)
        write_submission_csv(test_dir + '/results.csv', predictions, thresholds, test_dataset.split )

        #- voting ------------
        if a<num_augments:
            all_predictions[num_augments+2] += (predictions>thresholds)/num_augments
    pass



#
#
#
# # averaging over many models ################################################
# def list_to_weight(list, class_names):
#     num_classes = len(class_names)
#     weight = np.zeros(num_classes, np.float32)
#     for l in list:
#         idx = class_names.index(l)
#         weight[idx] = 1
#     return weight
#
#
# def do_find_thresholds_by_averaging():
#
#     out_dir  = '/root/share/project/pytorch/results/kaggle-forest/ensemble-02'
#     test_dir = out_dir +'/thresholds/average11'
#     os.makedirs(test_dir, exist_ok=True)
#
#
#     ## ------------------------------------
#     class_names = CLASS_NAMES
#     num_classes = len(class_names)
#
#     log = Logger()
#     log.open(out_dir+'/log.thresholds.txt',mode='a')
#     log.write('\n--- [START %s] %s\n\n' % (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '-' * 64))
#     log.write('\n')
#
#
#     lists = [
#         [*class_names],
#         ['cloudy','agriculture','water','bare_ground','artisinal_mine',	]
#     ]
#     predict_files = [
#         '/root/share/project/pytorch/results/kaggle-forest/fusenet-xxx/thresholds/average/predictions.npy',
#         '/root/share/project/pytorch/results/kaggle-forest/fusenet-xxx1/thresholds/average/predictions.npy',
#     ]
#     label_file = '/root/share/project/pytorch/results/kaggle-forest/fusenet-xxx//thresholds/average/labels.npy'
#
#     # averaging -------------------------------------------------------------
#     labels = np.load(label_file)
#
#     num         = len(predict_files)
#     predictions = None
#
#     for n in range(num):
#         prediction = np.load(predict_files[n])
#         weight     = list_to_weight(lists[n], class_names)
#
#         if predictions is None:
#             predictions = np.zeros(prediction.shape, np.float32)
#             weights    = np.zeros(num_classes, np.float32)
#
#         predictions += prediction*weight
#         weights     += weight
#
#     predictions = predictions/weights
#
#     #find threshold --------------------------
#     if 1:
#         log.write('\tmethod1:\n')
#
#         thresholds, scores = find_f_measure_threshold1(predictions, labels)
#         i = np.argmax(scores)
#         best_threshold, best_score = thresholds[i], scores[i]
#
#         log.write('\tbest_threshold=%f, best_score=%f\n\n'%(best_threshold, best_score))
#         #plot_f_measure_threshold(thresholds, scores)
#         #plt.pause(0)
#
#     if 1:
#         log.write('\tmethod2:\n')
#
#         seed = best_threshold  #0.21  #best_threshold
#         best_thresholds,  best_scores = find_f_measure_threshold2(predictions, labels, num_iters=100, seed=seed)
#
#         log.write('\tbest_threshold\n')
#         log.write (str(best_thresholds)+'\n')
#         log.write('\tbest_scores\n')
#         log.write (str(best_scores)+'\n')
#         log.write('\n')
#
#     #--------------------------------------------
#     np.save(test_dir +'/predictions.npy',predictions)
#     np.save(test_dir +'/labels.npy',labels)
#     np.savetxt(test_dir +'/best_threshold.txt', np.array(best_thresholds),fmt='%.5f' )
#     np.savetxt(test_dir +'/best_scores.txt',np.array(best_scores),fmt='%.5f')
#
#     with open(test_dir +'/predict_files.txt', 'w') as f:
#         for file in predict_files:
#             f.write('%s\n'%file)
#
#     with open(test_dir +'/lists.txt', 'w') as f:
#         for l in lists:
#             f.write('%s\n'%str(l))
#     # pass
#
#
#
# def do_submission_by_averaging():
#
#     out_dir  = '/root/share/project/pytorch/results/kaggle-forest/ensemble-02'
#     test_dir = out_dir +'/submissions/average11'
#     os.makedirs(test_dir, exist_ok=True)
#
#     thresholds = \
#         [0.32, 0.16, 0.17, 0.05, 0.37, 0.27, 0.19, 0.22, 0.16, 0.29, 0.09, 0.06, 0.2, 0.14, 0.11, 0.11, 0.05]
#
#
#     ## ------------------------------
#     class_names = CLASS_NAMES
#     num_classes = len(class_names)
#
#     log = Logger()
#     log.open(out_dir+'/log.submission.txt',mode='a')
#     log.write('\n--- [START %s] %s\n\n' % (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '-' * 64))
#     log.write('\n')
#
#     lists = [
#         [*class_names],
#         ['cloudy','agriculture','water','bare_ground','artisinal_mine',	]
#     ]
#     predict_files = [
#         '/root/share/project/pytorch/results/kaggle-forest/fusenet-xxx/submissions/average/predictions.npy',
#         '/root/share/project/pytorch/results/kaggle-forest/fusenet-xxx1/submissions/average/predictions.npy',
#    ]
#
#
#
#     # averaging -------------------------------------------------------------
#     num         = len(predict_files)
#     predictions = None
#
#     for n in range(num):
#         prediction = np.load(predict_files[n])
#         weight     = list_to_weight(lists[n], class_names)
#
#         if predictions is None:
#             predictions = np.zeros(prediction.shape, np.float32)
#             weights    = np.zeros(num_classes, np.float32)
#
#         predictions += prediction*weight
#         weights     += weight
#
#     predictions = predictions/weights
#
#
#     # -------------------
#
#     write_submission_csv(test_dir + '/results.csv', predictions, thresholds )
#     np.save(test_dir +'/predictions.npy',predictions)
#     np.savetxt(test_dir +'/thresholds.txt',thresholds)
#
#     with open(test_dir +'/predict_files.txt', 'w') as f:
#         for file in predict_files:
#             f.write('%s\n'%file)
#




# majority voting over many models ################################################
# https://mlwave.com/kaggle-ensembling-guide/
# http://www.kdnuggets.com/2015/06/ensembles-kaggle-data-science-competition-p1.html


## f2 score on train+validation data ...  ####
def do_thresholds_by_voting():

    out_dir='/root/share/project/pytorch/results/ensemble'
    test_dir = out_dir +'/ensemble17/thresholds'
    os.makedirs(test_dir, exist_ok=True)

    ## ------------
    log = Logger()
    log.open(test_dir+'/log.validation-ensemble.txt',mode='a')
    log.write('\n--- [START %s] %s\n\n' % (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '-' * 64))

    base_dir ='/root/share/project/pytorch/results/kaggle-forest'
    dirs = [

        '__old_3__/resnet34-40479-256-jpg-mix-pool-0/<replace>/average',  #//results-0.92858.csv
        '__old_3__/resnet50-pretrain-40479-jpg-1/<replace>/average',  #//results-0.92998.csv',
        '__old_3__/densenet121-pretrain-40479-jpg-0/<replace>0/average',  #//results-0.92913.csv',
        '__old_3__/densenet121-pretrain-40479-jpg-0/<replace>/average',  #//results-0.9299.csv',
        '__old_3__/inception_v3-pretrain-40479-jpg-1/<replace>/average',  #//results-0.92987.csv
        '__old_3__/inception_v3-pretrain-40479-jpg-1/<replace>0/average',  #//results-0.92952.csv
        '__old_3__/resNet34-40479-new-02/<replace>/average',  #//results-0.92990.csv',
        '__old_3__/PyResNet34-40479-add-03/<replace>/average',  #//results-0.92995.csv',


        'resnet34-pretrain-40479-jpg-0/<replace>/average',  #//results-0.93015.csv',
        'inception_v3-train-40479-pretrain-288-aug-new-18/<replace>-025/average',  #//results-0.93088.csv',
        'inception_v3-train-40479-pretrain-288-aug-new-18/<replace>-final/average',  #//results-0.93012.csv',


        'densenet169-train-40479-pretrain-224-aug-new-00/<replace>/average',  #/results-0.92966.csv',
        'resnet34-train-40479-pretrain-256-aug-new-00/<replace>/average',   #//results.csv',
        'resnet50-train-40479-pretrain-288-aug-new-00/<replace>-025/average',   #//rresults-0.92991.csv',

   ]


    # majority vote  -----------
    predictions = None
    labels      = None

    class_names = CLASS_NAMES
    num_classes = len(class_names)

    num=len(dirs)
    for n in range(num):
        dir = (base_dir + '/' + dirs[n]).replace('<replace>','thresholds')
        label      = np.load(dir+'/labels.npy')
        prediction = np.load(dir+'/predictions.npy')
        threshold  = np.loadtxt(dir+'/best_threshold.txt')

        if predictions is None:
            N =len(label)
            predictions = np.zeros((N,num_classes),dtype=np.float32)
            labels      = label

        l = prediction > threshold
        predictions = predictions + l

    predictions = predictions/num

    # -------------------
    if 1:
        beta = 2  #2.048

        #fixed threshold
        best_thresholds, best_score  = find_f_measure_threshold_fix(predictions, labels, beta=beta)
        f2 = fbeta_score(labels, predictions > best_thresholds, beta=beta, average='samples')

        log.write('*best_threshold (fixed)*\n')
        log.write(np.array2string(best_thresholds, formatter={'float_kind':lambda x: ' %.3f' % x}, separator=','))
        log.write('\n')
        log.write('*best_score*\n')
        log.write('%0.4f (%0.4f)\n'%(best_score, f2))

        #per class threshold
        init_thresholds = best_thresholds.copy()
        best_thresholds,  best_score  = find_f_measure_threshold_per_class( predictions, labels, init_thresholds=init_thresholds, beta=beta)
        f2 = fbeta_score(labels, predictions > best_thresholds, beta=beta, average='samples')

        log.write('*best_threshold (per class)*\n')
        log.write(np.array2string(best_thresholds, formatter={'float_kind':lambda x: ' %.3f' % x}, separator=','))
        log.write('\n')
        log.write('*best_score*\n')
        log.write('%0.4f (%0.4f)\n\n'%(best_score, f2))


    #save
    with open(test_dir +'/dirs.txt', 'w') as f:
        for d in dirs:
            f.write('%s\n'%d)


    np.save(test_dir +'/predictions.npy',predictions)
    np.save(test_dir +'/labels.npy',labels)
    np.savetxt(test_dir +'/init_threshold.txt', init_thresholds,fmt='%.5f' )
    np.savetxt(test_dir +'/best_threshold.txt', best_thresholds,fmt='%.5f' )
    np.savetxt(test_dir +'/best_score.txt',np.array([best_score]),fmt='%.5f')



def do_submission_by_voting():


    out_dir ='/root/share/project/pytorch/results/ensemble'
    test_dir = out_dir +'/ensemble17/submissions'
    os.makedirs(test_dir, exist_ok=True)


    ## ------------
    log = Logger()
    log.open(test_dir+'/log.submission-ensemble.txt',mode='a')
    log.write('\n--- [START %s] %s\n\n' % (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '-' * 64))
    log.write('\n')

    base_dir ='/root/share/project/pytorch/results/kaggle-forest'
   #  dirs = [
   #
   #      ##'/root/share/project/pytorch/results/kaggle-forest/resnet34-32479-pretrain-256-aug-old-15/submissions-32479/average/results-0.92993.csv',
   #      ##'/root/share/project/pytorch/results/kaggle-forest/resnet34-32479-pretrain-256-aug-old-15/submissions-32479/vote/results-0.92966.csv',
   #
   #      'densenet169-train-40479-pretrain-224-aug-new-00/<replace>/average',  #/results-0.92966.csv',
   #      'resnet34-train-40479-pretrain-256-aug-new-00/<replace>/average',  #//results.csv',
   #
   #      'resnet34-pretrain-40479-jpg-0/<replace>/average',  #//results-0.93015.csv',
   #      '__old_3__/resnet50-pretrain-40479-jpg-1/<replace>/average',  #//results-0.92998.csv',
   #      '__old_3__/densenet121-pretrain-40479-jpg-0/<replace>/average',  #//results-0.9299.csv',
   #      '__old_3__/resNet34-40479-new-02/<replace>/average',  #//results-0.92990.csv',
   #      '__old_3__/PyResNet34-40479-add-03/<replace>/average',  #//results-0.92995.csv',
   #
   #      'inception_v3-train-40479-pretrain-288-aug-new-18/<replace>-025/average',  #//results-0.93088.csv',
   #      'inception_v3-train-40479-pretrain-288-aug-new-18/<replace>-final/average',  #//results-0.93012.csv',
   #      '__old_3__/densenet121-pretrain-40479-jpg-0/<replace>0/average',  #//results-0.92913.csv',
   # ]
    with open((test_dir +'/dirs.txt').replace('submissions','thresholds')) as f:
        dirs = f.readlines()
    dirs = [x.strip()for x in dirs]


    # majority vote  -----------
    class_names = CLASS_NAMES
    num_classes = len(class_names)

    csv_files=[]
    num=len(dirs)
    for n in range(0,num):
        csv_file =  glob.glob((base_dir + '/' + dirs[n]).replace('<replace>','submissions') + '/*.csv')[0]
        csv_files.append(csv_file)

        df = pd.read_csv(csv_file)
        for c in class_names:
            df[c] = df['tags'].apply(lambda x: 1 if c in x.split(' ') else 0)

        if n ==0:
            names  = df.iloc[:,0].values
            N = df.shape[0]
            predictions = np.zeros((N,num_classes),dtype=np.float32)

        l = df.iloc[:,2:].values.astype(np.float32)
        predictions = predictions+l

    predictions = predictions/num

    init_threshold = np.loadtxt((test_dir +'/init_threshold.txt').replace('submissions','thresholds'))    ##1/3
    best_threshold = np.loadtxt((test_dir +'/best_threshold.txt').replace('submissions','thresholds'))    ##1/3
    # -------------------

    write_submission_csv(test_dir + '/init_results.csv', predictions, init_threshold,'test-61191')
    write_submission_csv(test_dir + '/best_results.csv', predictions, best_threshold,'test-61191')

    np.save(test_dir +'/predictions.npy',predictions)
    np.savetxt(test_dir +'/init_threshold.txt',init_threshold,fmt='%.5f')
    np.savetxt(test_dir +'/best_threshold.txt',best_threshold,fmt='%.5f')

    #for t in [0.60, 0.55, 0.50, 0.45, 0.40, 0.35]:
    for t in [0.40, 0.35, 0.325, 0.30]:
        threshold = t*np.ones(num_classes)
        write_submission_csv(test_dir + '/results-th=%f.csv'%t, predictions, threshold,'test-61191')
        np.savetxt(test_dir +'/threshold.txt',threshold,fmt='%.5f')

    with open(test_dir +'/csv_files.txt', 'w') as f:
        for file in csv_files:
            f.write('%s\n'%file)





# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    #do_training()
    #do_thresholds()
    #do_submissions()


    #do_thresholds_by_voting()
    do_submission_by_voting()


    print('\nsucess!')