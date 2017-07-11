from net.common import *
from net.dataset.tool import *
import pandas as pd

KAGGLE_DATA_DIR ='/media/ssd/[data]/kaggle-forest/classification'
#KAGGLE_DATA_DIR ='/root/share/data/kaggle-forest/classification'
CLASS_NAMES=[
    'clear',    	 # 0
    'haze',	         # 1
    'partly_cloudy', # 2
    'cloudy',	     # 3
    'primary',	     # 4
    'agriculture',   # 5
    'water',	     # 6
    'cultivation',	 # 7
    'habitation',	 # 8
    'road',	         # 9
    'slash_burn',	 # 10
    'conventional_mine', # 11
    'bare_ground',	     # 12
    'artisinal_mine',	 # 13
    'blooming',	         # 14
    'selective_logging', # 15
    'blow_down',	     # 16
]

# helper functions -------------
def score_to_class_names(prob, class_names, threshold = 0.5, nil=''):

    N = len(class_names)
    if not isinstance(threshold,(list, tuple, np.ndarray)) : threshold = [threshold]*N

    s=nil
    for n in range(N):
        if prob[n]>threshold[n]:
            if s==nil:
                s = class_names[n]
            else:
                s = '%s %s'%(s, class_names[n])
    return s


def draw_class_names(image,  prob, class_names, threshold=0.5):

    weather = CLASS_NAMES[:4]
    s = score_to_class_names(prob, class_names, threshold, nil=' ')
    for i, ss in enumerate(s.split(' ')):
        if ss in weather:
            color = (255,255,0)
        else:
            color = (0, 255,255)

        draw_shadow_text(image, ' '+ss, (5,30+(i)*15),  0.5, color, 1)



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
        tif_blue  = np.clip((image[:,:,tif_src  ] *255*6   -25)*0.90,a_min=0,a_max=255)
        tif_green = np.clip((image[:,:,tif_src+1] *255*9   -50)*0.95,a_min=0,a_max=255)
        tif_red   = np.clip((image[:,:,tif_src+2] *255*9   -25)*0.95,a_min=0,a_max=255)
        tif_nir   = np.clip(image[:,:,tif_src+3] *255*4,a_min=0,a_max=255)

        img[:,tif_dst*w:(tif_dst+1)*w] = np.dstack((tif_blue,tif_green,tif_red)).astype(np.uint8)
        img[:,(tif_dst+1)*w:(tif_dst+2)*w ] = np.dstack((tif_nir,tif_nir,tif_nir)).astype(np.uint8)

    if jpg_src is not None and tif_src is not None:
        im1 = img[:,jpg_dst*w:(jpg_dst+1)*w]
        im2 = img[:,tif_dst*w:(tif_dst+1)*w]
        im1_gray = cv2.cvtColor(im1,cv2.COLOR_BGR2GRAY)
        im2_gray = cv2.cvtColor(im2[:,:,0:3],cv2.COLOR_BGR2GRAY)
        zz = np.zeros((h,w),np.uint8)
        img[:,3*w: ] = np.dstack((im1_gray,zz,im2_gray)).astype(np.uint8)


    if height!=h or width!=w:
        img = cv2.resize(img,(width*M,height))

    return img






## custom data loader -----------------------------------
def align_tif_to_jpg(image_jpg, image_tif):
    raise ValueError('this function needs to be updated!')
    # http://www.learnopencv.com/image-alignment-ecc-in-opencv-c-python/
    # “Parametric Image Alignment using Enhanced Correlation Coefficient Maximization” - Evangelidis, G.D.
    #     and Psarakis E.Z, PAMI 2008
    #

    # Convert images to grayscale
    im1= image_jpg.astype(np.float32)/255.0
    im2= image_tif.astype(np.float32)/4095.0
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
        image_tif_aligned = cv2.warpAffine(image_tif, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,borderMode=cv2.BORDER_REFLECT_101)  #, borderMode = cv2.BORDER_CONSTANT, borderValue = (0, 0, 0))
        tx = warp_matrix[0,2]
        ty = warp_matrix[1,2]
        return image_tif_aligned, tx, ty, correlation

    except cv2.error:
        return image_tif, 0, 0, -1


def load_one_image(name, width, height, ext, data_dir=KAGGLE_DATA_DIR):

    if ext =='tif':
        image = np.zeros((height,width,4),dtype=np.uint16)  #

        img_file  = data_dir + '/image/' + name
        tif_file  = img_file.replace('<ext>','tif')
        image_tif = io.imread(tif_file)

        h,w = image_tif.shape[0:2]
        if height!=h or width!=w:
            image_tif = cv2.resize(image_tif,(height,width))

        image[:,:,:4] = image_tif

    elif ext =='jpg':
        image = np.zeros((height,width,3),dtype=np.uint16)  #

        img_file  = data_dir + '/image/' + name
        jpg_file  = img_file.replace('<ext>','jpg')
        image_jpg = cv2.imread(jpg_file,1)

        h,w = image_jpg.shape[0:2]
        if height!=h or width!=w:
            image_jpg = cv2.resize(image_jpg,(height,width))

        image = image_jpg*256


    elif ext =='all':
        image = np.zeros((height,width,7),dtype=np.uint16)  #

        img_file  = data_dir + '/image/' + name
        jpg_file  = img_file.replace('<ext>','jpg')
        tif_file  = img_file.replace('<ext>','tif')
        image_jpg = cv2.imread(jpg_file,1)
        image_tif = io.imread(tif_file)

        h,w = image_jpg.shape[0:2]
        if height!=h or width!=w:
            image_jpg = cv2.resize(image_jpg,(height,width))
            image_tif = cv2.resize(image_tif,(height,width))

        ##image_tif, _, _, _ = align_tif_to_jpg(image_jpg, image_tif)
        image[:,:,:3] = image_jpg*256
        image[:,:,3:] = image_tif

    return image


class KgForestDataset(Dataset):

    def __init__(self, split, transform=None, height=64, width=64, ext='jpg', label_csv='train_v2.csv', is_preload=True):
        data_dir    = KAGGLE_DATA_DIR
        class_names = CLASS_NAMES
        num_classes = len(class_names)

        # read names
        list = data_dir +'/split/'+ split
        with open(list) as f:
            names = f.readlines()
        names = [x.strip()for x in names]
        num   = len(names)


        #read images
        if ext =='tif':
            channel=4
        elif ext =='jpg':
            channel=3
        elif ext =='all':
            channel=7
        else:
            raise ValueError('KgForestDataset() : unknown ext !?')

        images  = None
        if is_preload==True:
            images = np.zeros((num,height,width,channel),dtype=np.uint16)
            for n in range(num):
                images[n] = load_one_image(names[n], width, height, ext)
                #cv2.circle(image_jpg, (0,0), 20,(0,0,255),-1) #mark orgin for debug
                #cv2.circle(image_tif, (0,0), 20,(0,0,255),-1)

                if 0: #debug
                    image = create_image(images[n])
                    im_show('image',image,1)
                    cv2.waitKey(0)
                pass


        #read labels
        df     = None
        labels = None
        if label_csv is not None:
            labels = np.zeros((num,num_classes),dtype=np.float32)

            csv_file  = data_dir + '/image/' + label_csv   # read all annotations
            df = pd.read_csv(csv_file)
            for c in class_names:
                df[c] = df['tags'].apply(lambda x: 1 if c in x.split(' ') else 0)

            df1 = df.set_index('image_name')
            for n in range(num):
                shortname = names[n].split('/')[-1].replace('.<ext>','')
                labels[n] = df1.loc[shortname].values[1:]

                if 0: #debug
                    image = create_image(images[n])
                    draw_shadow_text  (image, shortname, (5,15),  0.5, (255,255,255), 1)
                    draw_class_names(image, labels[n], class_names)
                    im_show('image', image)
                    cv2.waitKey(0)

                    #images[n]=cv2.resize(image,(height,width)) ##mark for debug
                    pass
        #save
        self.transform = transform
        self.num       = num
        self.split     = split
        self.names     = names
        self.images    = images
        self.ext       = ext
        self.is_preload = is_preload
        self.width  = width
        self.height = height
        self.height = height
        self.channel = channel

        self.class_names = class_names
        self.df     = df
        self.labels = labels


    #https://discuss.pytorch.org/t/trying-to-iterate-through-my-custom-dataset/1909
    def __getitem__(self, index):
        #print ('\tcalling Dataset:__getitem__ @ idx=%d'%index)

        if self.is_preload==True:
            img = self.images[index]
        else:
            img = load_one_image(self.names[index], self.width, self.height, self.ext)

        img = img.astype(np.float32)/65536  #2**16
        if self.transform is not None:
            for t in self.transform:
                img = t(img)

        if self.labels is None:
            return img, index

        else:
            label = self.labels[index]
            return img, label, index


    def __len__(self):
        #print ('\tcalling Dataset:__len__')
        return len(self.names)




def check_kgforest_dataset(dataset, loader):

    class_names = dataset.class_names
    names  = dataset.names

    if dataset.labels is not None:
        for i, (images, labels, indices) in enumerate(loader, 0):
            print('i=%d: '%(i))

            # get the inputs
            num = len(images)
            for n in range(num):
                label = labels[n].numpy()
                image = tensor_to_img(images[n], mean=0, std=1, dtype=np.float32)

                s = score_to_class_names(label, class_names)
                print('%32s : %s %s'%  (names[indices[n]], label.T, s))

                #if label[13] < 0.5: continue

                image = create_image(image)
                shortname = names[indices[n]].split('/')[-1].replace('.<ext>','')
                draw_shadow_text(image, shortname, (5,15),  0.5, (255,255,255), 1)
                draw_class_names(image, label, class_names)
                im_show('image',image)
                cv2.waitKey(0)
                #print('\t\tlabel=%d : %s'%(label,classes[label]))
                #print('')

    if dataset.labels is None:
        for i, (images, indices) in enumerate(loader, 0):
            print('i=%d: '%(i))

            # get the inputs
            num = len(images)
            for n in range(num):
                image = tensor_to_img(images[n], mean=0, std=1, dtype=np.float32)

                print('%32s : nil'% (names[indices[n]]))

                image = create_image(image)
                shortname = names[indices[n]].split('/')[-1].replace('.<ext>','')
                draw_shadow_text  (image, shortname, (5,15),  0.5, (255,255,255), 1)
                im_show('image',image)
                cv2.waitKey(0)


# fit sigmoid curve for displaying tif
def run_fit():
    dataset = KgForestDataset('debug-32', #'valid-8000', ##'debug-32', ###'train-40479',  ##'train-ordered-20', ##
                                transform=[
                                    lambda x: randomShiftScaleRotate(x, u=0.75, shift_limit=16, scale_limit=0.1, rotate_limit=45),
                                    #lambda x: randomFlip(x),
                                    #lambda x: randomTranspose(x),
                                    #lambda x: randomContrast(x, limit=0.3, u=0.5),
                                    #lambda x: randomSaturation(x, limit=0.3, u=0.5),
                                    #lambda x: randomDistort1(x, limit=0.5, u=0.5),
                                    lambda x: randomFilter(x, limit=0.5, u=0.5),
                                    lambda x: img_to_tensor(x),
                                ],
                                width=256,height=256,
                                ext='all',
                                is_preload=True,
                                #label_csv=None,
                              )
    num =32
    jpg_b = np.zeros(num*256*256,np.float32)
    tif_b = np.zeros(num*256*256,np.float32)
    for n in range(num):
        jpg_b[n*256*256:(n+1)*256*256]=dataset.images[n,:,:,0+0].reshape(-1)/65536
        tif_b[n*256*256:(n+1)*256*256]=dataset.images[n,:,:,3+0].reshape(-1)/65536

    fig = plt.figure(figsize=(7,7))
    ax = fig.gca()
    ax.set_axisbelow(True)
    ax.minorticks_on()
    ax.set_xticks(np.arange(0,1.0001, 0.1))
    ax.set_yticks(np.arange(0,1.0001, 0.1))
    ax.set_xlim(0,1.0001)
    ax.set_ylim(0,1.0001)
    ax.grid(b=True, which='minor', color='black', alpha=0.3, linestyle='dashed')
    ax.grid(b=True, which='major', color='black', alpha=0.5, linestyle='dashed')
    ax.scatter(tif_b, jpg_b)
    plt.show()


#test dataset
def run_check_dataset():
    dataset = KgForestDataset('testv2-20522', #'valid-8000', ##'debug-32', ###'train-40479',  ##'train-ordered-20', ##
                                transform=[
                                    #lambda x: randomShiftScaleRotate(x, u=0.75, shift_limit=16, scale_limit=0.1, rotate_limit=45),
                                    #lambda x: randomFlip(x),
                                    #lambda x: randomTranspose(x),
                                    #lambda x: randomContrast(x, limit=0.3, u=0.5),
                                    #lambda x: randomSaturation(x, limit=0.3, u=0.5),
                                    #lambda x: randomDistort1(x, limit=0.5, u=0.5),
                                    #lambda x: randomFilter(x, limit=0.5, u=0.5),
                                    lambda x: img_to_tensor(x),
                                ],
                                width=256,height=256,
                                ext='all',
                                is_preload=False,
                                label_csv=None,
                              )
    sampler = SequentialSampler(dataset)  #FixedSampler(dataset,[5,]*100  )    #RandomSampler
    loader  = DataLoader(dataset, batch_size=4, sampler=sampler,  drop_last=False, pin_memory=True)


    for epoch in range(100):
        print('epoch=%d -------------------------'%(epoch))
        check_kgforest_dataset(dataset, loader)

    print('sucess')

# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    run_check_dataset()
    #run_fit()
    #run_check_dataset()
