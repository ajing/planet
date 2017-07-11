# http://stackoverflow.com/questions/42630994/gimp-python-fu-how-to-create-and-save-multilayer-xcf-from-1-jpg-and-1-png
# http://stackoverflow.com/questions/12662676/writing-a-gimp-python-script


# in python 2.7
from gimpfu import *
import glob
import os

#--------------------------------------------------------------------------------------

image_dir  = '/root/share/data/kaggle-forest/_mark_/by_class/road'
gimp_dir   = '/root/share/data/kaggle-forest/_mark_/by_class/road-gimp'
mask_file='/root/share/data/kaggle-forest/_mark_/by_class/mask0.png'

if not os.path.exists(gimp_dir):
    os.makedirs(gimp_dir)

for file in sorted(glob.glob(image_dir +'/*.jpg')):
    name = os.path.basename(file).replace('.jpg','')  #'%06d'%t

    print ('running file = %s'%name)
    img_file  = image_dir  + '/' + name + '.jpg'
    gimp_file = gimp_dir   + '/' + name + '.xcf'

    image=pdb.gimp_file_load(img_file,img_file)
    layer=pdb.gimp_file_load_layer(image,mask_file)

    layer.opacity = 30. #1-100
    image.add_layer(layer,0)
    pdb.gimp_xcf_save(0,image,layer,gimp_file,gimp_file)


#https://www.gimp.org/tutorials/Basic_Batch/


'''
 
gimp -idf --batch-interpreter python-fu-eval -b 'execfile("/root/share/project/pytorch/build/forest-1/__gimp__/add_mask.py"); pdb.gimp_quit(1)'


  
'''