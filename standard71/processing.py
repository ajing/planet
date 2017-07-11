## do alignment for tif and jpg

from net.common import *
from net.dataset.kgforest import *
from net.util import *


def create_image(image, width=256, height=256):
    h,w,c = image.shape

    if c==3:
        jpg_src=0
        tif_src=None

        M=1
        jpg_dst=0

    if c==4:
        jpg_src=None
        tif_src=0

        M=2
        tif_dst=0

    if c==7:
        jpg_src=0
        tif_src=3

        M=4
        jpg_dst=0
        tif_dst=1


    img = np.zeros((h,w*M,3),np.uint8)
    if jpg_src is not None:
        jpg_blue  = image[:,:,jpg_src  ] *255
        jpg_green = image[:,:,jpg_src+1] *255
        jpg_red   = image[:,:,jpg_src+2] *255

        img[:,jpg_dst*w:(jpg_dst+1)*w] = np.dstack((jpg_blue,jpg_green,jpg_red)).astype(np.uint8)

    if tif_src is not None:
        tif_blue  = np.clip(image[:,:,tif_src  ] *4095*255/65536.0*6 -25-30,a_min=0,a_max=255)
        tif_green = np.clip(image[:,:,tif_src+1] *4095*255/65536.0*6    -30,a_min=0,a_max=255)
        tif_red   = np.clip(image[:,:,tif_src+2] *4095*255/65536.0*6 +25-30,a_min=0,a_max=255)
        tif_nir   = np.clip(image[:,:,tif_src+3] *4095*255/65536.0*4,a_min=0,a_max=255)

        img[:,tif_dst*w:(tif_dst+1)*w] = np.dstack((tif_blue,tif_green,tif_red)).astype(np.uint8)
        img[:,(tif_dst+1)*w:(tif_dst+2)*w ] = np.dstack((tif_nir,tif_nir,tif_nir)).astype(np.uint8)

    if M==4:
        im1 = img[:,jpg_dst*w:(jpg_dst+1)*w]
        im2 = img[:,tif_dst*w:(tif_dst+1)*w]
        im1_gray = cv2.cvtColor(im1,cv2.COLOR_BGR2GRAY)
        im2_gray = cv2.cvtColor(im2[:,:,0:3],cv2.COLOR_BGR2GRAY)
        zz = np.zeros((h,w),np.uint8)
        img[:,3*w: ] = np.dstack((im1_gray,zz,im2_gray)).astype(np.uint8)


    if height!=h or width!=w:
        img = cv2.resize(img,(width*M,height))

    return img

def norm_channel(data):
    h,b  = np.histogram(data, bins=100)

    dmin=b[ 10]
    dmax=b[-10]
    data = (data-dmin)/(dmax-dmin)
    data = np.clip(data,a_min=0,a_max=1)


    step = 16
    data = ((data*255).astype(np.int32)//step)*step /255
    data = np.clip(data,a_min=0,a_max=1)


    return data





def align_tif_to_jpg(image_jpg, image_tif):
    # http://www.learnopencv.com/image-alignment-ecc-in-opencv-c-python/
    # “Parametric Image Alignment using Enhanced Correlation Coefficient Maximization” - Evangelidis, G.D.
    #     and Psarakis E.Z, PAMI 2008
    #

    # Convert images to grayscale
    im1= image_jpg
    im2= image_tif
    im1_gray = cv2.cvtColor(im1,cv2.COLOR_BGR2GRAY)
    im2_gray = cv2.cvtColor(im2[:,:,0:3],cv2.COLOR_BGR2GRAY)

    sz = im1.shape
    warp_mode = cv2.MOTION_TRANSLATION # Define the motion model
    warp_matrix = np.eye(2, 3, dtype=np.float32)

    # Define termination criteria
    number_of_iterations = 100
    termination_eps = 1e-5
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations,  termination_eps)

    # Run the ECC algorithm. The results are stored in warp_matrix.
    try:
        correlation, warp_matrix = cv2.findTransformECC (im1_gray,im2_gray,warp_matrix, warp_mode, criteria)
    except cv2.error:
        return im2, 0, 0, -1

    im2_aligned = cv2.warpAffine(im2, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP, borderMode = cv2.BORDER_CONSTANT, borderValue = (0, 0, 0))
    tx = warp_matrix[0,2]
    ty = warp_matrix[1,2]

    return im2_aligned, tx, ty, correlation

def get_rect(tx,ty,width=256,height=256):

    tx = int(round(tx))
    ty = int(round(ty))
    x1 = max(tx,0)
    y1 = max(ty,0)
    x2 = min(tx+width,width)-1
    y2 = min(ty+height,height)-1

    return x1,y1,x2,y2


def run_one():


    width,height = 64,64  #256, 256


    #jpg_file  = '/root/share/data/kaggle-forest/classification/dummy/train_4.jpg'
    #tif_file  = '/root/share/data/kaggle-forest/classification/dummy/train_4.tif'
    jpg_file  = '/root/share/data/kaggle-forest/classification/image/train-jpg/train_4.jpg'  #9,4,66  ## 21
    tif_file  = '/root/share/data/kaggle-forest/classification/image/train-tif/train_4.tif'

    image_jpg = cv2.imread(jpg_file,1)
    image_tif = io.imread(tif_file)
    image = np.zeros((height,width,7),dtype=np.float32)


    h,w = image_jpg.shape[0:2]
    if height!=h or width!=w:
        image_jpg = cv2.resize(image_jpg,(height,width))
        image_tif = cv2.resize(image_tif,(height,width))

        #cv2.circle(image_jpg, (0,0), 20,(0,0,255),-1) #mark orgin for debug
        #cv2.circle(image_tif, (0,0), 20,(0,0,255),-1)

    image[:,:,:3] = image_jpg.astype(np.float32)/255.0
    image[:,:,3:] = image_tif.astype(np.float32)/4095.0  #2^12=4096
    img = create_image(image)
    im_show('img_before',img,1)
    cv2.waitKey(1)


    ## nomalised and cut
    im1 = image[:,:,:3]  #image_jpg.astype(np.float32)
    im2 = image[:,:,3:]  #image_tif.astype(np.float32)
    im2_aligned, tx, ty, correlation = align_tif_to_jpg(im1, im2)

    image[:,:,3:] = im2_aligned
    img = create_image(image)
    #draw alignment
    if correlation >=0.5:
        x1,y1,x2,y2 = get_rect(-tx,-ty,width=256,height=256)
        cv2.rectangle(img, (x1,y1), (x2,y2), (0,0,255),1 )
        cv2.rectangle(img, (x1+256,y1), (x2+256,y2), (0,0,255),1 )
        cv2.rectangle(img, (x1+2*256,y1), (x2+2*256,y2), (0,0,255),1 )


    im_show('img_after',img,1)
    cv2.waitKey(1)

    # im1 = im1.astype(np.uint8)
    # im2 = im2.astype(np.uint8)
    # im2_aligned = im2_aligned.astype(np.uint8)

    # Show final results
    print('correlation=%f'%correlation)
    # cv2.imshow("im1", im1)
    # cv2.imshow("im2", im2)
    # cv2.imshow("im2_aligned", im2_aligned)
    cv2.waitKey(0)


def run_one():


    width,height = 256, 256


    #jpg_file  = '/root/share/data/kaggle-forest/classification/dummy/train_4.jpg'
    #tif_file  = '/root/share/data/kaggle-forest/classification/dummy/train_4.tif'
    jpg_file  = '/root/share/data/kaggle-forest/classification/image/train-jpg/train_4.jpg'  #9,4,66  ## 21
    tif_file  = '/root/share/data/kaggle-forest/classification/image/train-tif/train_4.tif'

    image_jpg = cv2.imread(jpg_file,1)
    image_tif = io.imread(tif_file)
    image = np.zeros((height,width,7),dtype=np.float32)


    h,w = image_jpg.shape[0:2]
    if height!=h or width!=w:
        image_jpg = cv2.resize(image_jpg,(height,width))
        image_tif = cv2.resize(image_tif,(height,width))

        #cv2.circle(image_jpg, (0,0), 20,(0,0,255),-1) #mark orgin for debug
        #cv2.circle(image_tif, (0,0), 20,(0,0,255),-1)

    image[:,:,:3] = image_jpg.astype(np.float32)/255.0
    image[:,:,3:] = image_tif.astype(np.float32)/4095.0  #2^12=4096
    img = create_image(image)
    im_show('img_before',img,1)
    cv2.waitKey(1)


    ## nomalised and cut
    im1 = image[:,:,:3]  #image_jpg.astype(np.float32)
    im2 = image[:,:,3:]  #image_tif.astype(np.float32)
    im2_aligned, tx, ty, correlation = align_tif_to_jpg(im1, im2)

    image[:,:,3:] = im2_aligned
    img = create_image(image)
    #draw alignment
    if correlation >=0.5:
        x1,y1,x2,y2 = get_rect(-tx,-ty,width=256,height=256)
        cv2.rectangle(img, (x1,y1), (x2,y2), (0,0,255),1 )
        cv2.rectangle(img, (x1+256,y1), (x2+256,y2), (0,0,255),1 )
        cv2.rectangle(img, (x1+2*256,y1), (x2+2*256,y2), (0,0,255),1 )


    im_show('img_after',img,1)
    cv2.waitKey(1)

    # im1 = im1.astype(np.uint8)
    # im2 = im2.astype(np.uint8)
    # im2_aligned = im2_aligned.astype(np.uint8)

    # Show final results
    print('correlation=%f'%correlation)
    # cv2.imshow("im1", im1)
    # cv2.imshow("im2", im2)
    # cv2.imshow("im2_aligned", im2_aligned)
    cv2.waitKey(0)

def run_many():


    width,height = 256, 256
    image = np.zeros((height,width,7),dtype=np.float32)

    for n in range(1000):
        jpg_file  = '/root/share/data/kaggle-forest/classification/image/train-jpg/train_%d.jpg'%n
        tif_file  = '/root/share/data/kaggle-forest/classification/image/train-tif/train_%d.tif'%n

        image_jpg = cv2.imread(jpg_file,1)
        image_tif = io.imread(tif_file)


        h,w = image_jpg.shape[0:2]
        if height!=h or width!=w:
            image_jpg = cv2.resize(image_jpg,(height,width))
            image_tif = cv2.resize(image_tif,(height,width))

            #cv2.circle(image_jpg, (0,0), 20,(0,0,255),-1) #mark orgin for debug
            #cv2.circle(image_tif, (0,0), 20,(0,0,255),-1)

        image[:,:,:3] = image_jpg.astype(np.float32)/255.0
        image[:,:,3:] = image_tif.astype(np.float32)/4095.0  #2^12=4096
        img = create_image(image)
        im_show('img_before',img,1)
        cv2.waitKey(1)


        ## nomalised and cut
        im1 = image[:,:,:3]  #image_jpg.astype(np.float32)
        im2 = image[:,:,3:]  #image_tif.astype(np.float32)
        im2_aligned, tx, ty, correlation = align_tif_to_jpg(im1, im2)

        image[:,:,3:] = im2_aligned
        img = create_image(image)
        #draw alignment
        if correlation >=0.5:
            x1,y1,x2,y2 = get_rect(-tx,-ty,width=256,height=256)
            cv2.rectangle(img, (x1,y1), (x2,y2), (0,0,255),1 )
            cv2.rectangle(img, (x1+256,y1), (x2+256,y2), (0,0,255),1 )
            cv2.rectangle(img, (x1+2*256,y1), (x2+2*256,y2), (0,0,255),1 )
        else:
            pass

        print('correlation=%f'%correlation)
        im_show('img_after',img,1)
        cv2.waitKey(0)

########################################################################
## precision curve

# http://eso-python.github.io/ESOPythonTutorials/ESOPythonDemoDay5_matplotlib_BerndHusemann_part1.html
def draw_recall_precision(fig, recall, precise, f2, threshold, idx,
                          title='',
                          color=None,
                          size=1,
                          xmin=0,xmax=1,dx=0.1,
                          ymin=0,ymax=1,dy=0.1):

    recall_star    = recall   [idx]
    precise_star   = precise  [idx]
    f2_star        = f2       [idx]
    threshold_star = threshold[idx]



    ax = fig.gca()
    ax.set_axisbelow(True)
    ax.minorticks_on()
    ax.set_xticks(np.arange(xmin,xmax+0.0001, dx))
    ax.set_yticks(np.arange(ymin,ymax+0.0001, dy))
    ax.set_xlim(xmin,xmax+0.0001)
    ax.set_ylim(ymin,ymax+0.0001)
    ax.grid(b=True, which='minor', color='black', alpha=0.3, linestyle='dashed')
    ax.grid(b=True, which='major', color='black', alpha=0.5, linestyle='dashed')
    ax.set_xlabel('recall')
    ax.set_ylabel('precise')

    if color is None:
        ax.scatter(recall, precise, c=f2, s=size, marker='o',  cmap = matplotlib.cm.jet)
    else:
        ax.scatter(recall, precise, c=color, s=size, marker='o',  cmap = matplotlib.cm.jet)

    ax.scatter(recall_star, precise_star, marker='o', facecolors='white', edgecolors='black')
    ax.set_title('%s\nbest recall=%0.3f, precise=%0.3f\n f2=%0.3f @ th=%0.3f'%(title,recall_star,precise_star,f2_star,threshold_star))


def run_make_curve():

    predictions = np.load('/root/share/project/pytorch/results/kaggle-forest/resnet34-32479-pretrain-256-aug-old-15/thresholds-40479/default/predictions.npy')
    labels      = np.load('/root/share/data/kaggle-forest/classification/split/train-40479-labels.npy')  #valid-8000-labels.npy')

    dump_dir='/root/share/project/pytorch/results/kaggle-forest/resnet34-32479-pretrain-256-aug-old-15/thresholds-40479/default/curve'
    os.makedirs(dump_dir, exist_ok=True)

    class_names = CLASS_NAMES
    num_classes = len(class_names)


    fig = plt.figure(figsize=(7,7))
    for n in range(num_classes):
        label      = labels     [:,n]
        prediction = predictions[:,n]

        precise, recall, threshold = sklearn.metrics.precision_recall_curve(label, prediction)
        f2 = (1+4)*precise*recall/(4*precise + recall + 1e-12)
        idx = np.argmax(f2)

        fig.clear()
        draw_recall_precision(fig, recall, precise, f2, threshold, idx,
                              title='target=%d (%s)'%(n,class_names[n]),
                              size=1,
                              xmin=0.0,xmax=1,dx=0.1,
                              ymin=0.0,ymax=1,dy=0.1)

        dump_dir1 = dump_dir +'/%02d_%s'%(n,class_names[n])
        os.makedirs(dump_dir1, exist_ok=True)
        fig.savefig(dump_dir1 +'/curve.png')
        np.savetxt (dump_dir1 +'/precise.txt',precise,fmt='%.5f')
        np.savetxt (dump_dir1 +'/recall.txt',recall,fmt='%.5f')
        np.savetxt (dump_dir1 +'/threshold.txt',threshold,fmt='%.5f')


        #plt.show()
        plt.pause(1)


    pass


def run_make_combined_curve():

    dirs= [
        '/root/share/project/pytorch/results/kaggle-forest/resnet34-32479-pretrain-256-aug-old-15/submissions-8000/default/curve',
        '/root/share/project/pytorch/results/kaggle-forest/resnet34-32479-pretrain-256-aug-old-15/submissions-32479-valid/default/curve',
   ]
    colors=[
        (1,0,0),
        (0,0,1),
        (0,1,0),
        (0.5,1,0),
        (0.5,0,0.5),
    ]
    dump_dir='/root/share/project/pytorch/results/kaggle-forest/resnet34-32479-pretrain-256-aug-old-15/more'
    os.makedirs(dump_dir, exist_ok=True)

    selected_thresholds = np.array([
        [0.2,	0.2,	0.11,	0.04,	0.25,	0.19,	0.24,	0.22,	0.13,	0.18,	0.2,	0.11,	0.09,	0.08,	0.16,	0.1,	0.07,],
        [0.17,	0.19,	0.19,	0.11,	0.22,	0.22,	0.2,	0.23,	0.2,	0.26,	0.24,	0.17,	0.13,	0.14,	0.24,	0.21,	0.06,]
    ])


    class_names = CLASS_NAMES
    num_classes = len(class_names)

    fig = plt.figure(figsize=(7,7))
    for n in range(num_classes):
        L=len(dirs)

        fig.clear()
        for i in range(L):
            recall    = np.loadtxt(dirs[i] + '/%02d_%s/recall.txt'%(n,class_names[n]))
            precise   = np.loadtxt(dirs[i] + '/%02d_%s/precise.txt'%(n,class_names[n]))
            threshold = np.loadtxt(dirs[i] + '/%02d_%s/threshold.txt'%(n,class_names[n]))
            f2 = (1+4)*precise*recall/(4*precise + recall + 1e-12)
            idx = np.argmax(f2)

            ## -----------------

            # draw_recall_precision(fig, recall, precise, f2, threshold, idx,
            #                   title='target=%d (%s)'%(n,class_names[n]),
            #                   color=colors[i],
            #                   size=1,
            #                   xmin=0.0,xmax=1,dx=0.1,
            #                   ymin=0.0,ymax=1,dy=0.1)


            ## -----------------
            color = (0,0,0)  #colors[i]
            size=1
            xmin=0.0; xmax=1; dx=0.1;
            ymin=0.0; ymax=1; dy=0.1;

            ax = fig.gca()
            ax.set_axisbelow(True)
            ax.minorticks_on()
            ax.set_xticks(np.arange(xmin,xmax+0.0001, dx))
            ax.set_yticks(np.arange(ymin,ymax+0.0001, dy))
            ax.set_xlim(xmin,xmax+0.0001)
            ax.set_ylim(ymin,ymax+0.0001)
            ax.grid(b=True, which='minor', color='black', alpha=0.3, linestyle='dashed')
            ax.grid(b=True, which='major', color='black', alpha=0.5, linestyle='dashed')
            ax.set_xlabel('recall')
            ax.set_ylabel('precise')
            ax.set_title('target=%d (%s)'%(n,class_names[n]))
            ax.scatter(recall, precise, c=color, s=size, marker='o',  cmap = matplotlib.cm.jet)

        for i in range(L):
            threshold = np.loadtxt(dirs[i] + '/%02d_%s/threshold.txt'%(n,class_names[n]))
            precise   = np.loadtxt(dirs[i] + '/%02d_%s/precise.txt'%(n,class_names[n]))
            threshold = np.loadtxt(dirs[i] + '/%02d_%s/threshold.txt'%(n,class_names[n]))
            f2 = (1+4)*precise*recall/(4*precise + recall + 1e-12)

            ## load selected ---
            selected = selected_thresholds[i][n]
            idx = np.argmin(np.fabs(threshold-selected))


            recall_star    = recall   [idx]
            precise_star   = precise  [idx]
            f2_star        = f2       [idx]
            threshold_star = threshold[idx]


            color = colors[i]
            ax.scatter(recall_star, precise_star, marker='o', facecolors=color, edgecolors='black')
            ax.text(0.02, (L-1-i)*0.025, 'threshold=%f'%selected, fontsize = 11, color = color)



        #plt.show()
        fig.savefig(dump_dir +'/%02d_%s_curve.png'%(n,class_names[n]))
        plt.pause(3)

####fix tif jpg filename mismatch ####
def run_fix_tif_gpg_filename_mismatch():
    src_dir='/media/ssd/[data]/kaggle-forest/classification/image/__temp__/test-tif----bug!!!'
    #dst_dir='/media/ssd/[data]/kaggle-forest/classification/image/test-tif'
    dst_dir='/root/share/data/kaggle-forest/classification/image/test-tif'

    cvs_file='/root/share/data/kaggle-forest/classification/image/test_v2_file_mapping.csv'
    df = pd.read_csv(cvs_file)

    for index, row in df.iterrows():
        print(index)
        old = src_dir + '/' +row['old']
        new = dst_dir + '/' +row['new']
        shutil.copy(old, new)




## kaggle evaluations  -----------------------------------

def run_f2_from_csv():
    predict_csv = '/root/share/project/pytorch/results/kaggle-forest/resnet34-32479-pretrain-256-aug-new-15/submissions/transpose/results.csv'
    true_csv = KAGGLE_DATA_DIR + '/image/train_v2.csv'

    class_names = CLASS_NAMES
    num_classes = len(class_names)

    true_df = pd.read_csv(true_csv)
    predict_df = pd.read_csv(predict_csv)
    for c in class_names:
        true_df   [c] = true_df   ['tags'].apply(lambda x: 1 if c in x.split(' ') else 0)
        predict_df[c] = predict_df['tags'].apply(lambda x: 1 if c in x.split(' ') else 0)


    #get true labels
    num = predict_df.shape[0]
    labels = np.zeros((num,num_classes),dtype=np.float32)

    names  = predict_df.iloc[:,0].values
    df1 = true_df.set_index('image_name')
    for n in range(num):
        shortname = names[n]
        labels[n] = df1.loc[shortname].values[1:]

    #get predict
    predictions = predict_df.values[:,2:].astype(np.float32)

    #f2 score
    f2 = sklearn.metrics.fbeta_score(labels, predictions, beta=2, average='samples')
    print('predictions.shape=%s'%str(predictions.shape))
    print('f2=%f'%f2)


def run_prob_ambiguity():

    prediction_file = '/root/share/project/pytorch/results/kaggle-forest/resnet34-32479-pretrain-256-aug-old-15/submissions-32479/average/predictions.npy'
    predictions = np.load(prediction_file)

    csv_file = '/root/share/data/kaggle-forest/classification/split/unsupervised/xxx.csv'
    split_file = '/root/share/data/kaggle-forest/classification/split/unsupervised/xxx'


    class_names = CLASS_NAMES
    num_classes = len(class_names)

    list = KAGGLE_DATA_DIR +'/split/'+ 'test-61191'
    with open(list) as f:
        names = f.readlines()
    names = [x.strip()for x in names]
    num   = len(names)


    threshold=0.10
    c_top    = (predictions >(1-threshold)).sum(axis=1)
    c_bottom = (predictions <(threshold)  ).sum(axis=1)
    c_all    = c_top+c_bottom

    idx = np.where (c_all>=num_classes-1)[0]
    with open(split_file,'w') as f:
        for n in idx:
            f.write('%s\n'%(names[n]))

    with open(csv_file,'w') as f:
        f.write('image_name,tags\n')
        for n in idx:
            shortname = names[n].split('/')[-1].replace('.<ext>','')

            prediction = predictions[n]
            s = score_to_class_names(prediction, class_names, threshold=0.5)
            f.write('%s,%s\n'%(shortname,s))
            print(c_all[n])

    print(len(idx))
    print(len(idx)/num)

# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    run_prob_ambiguity()

