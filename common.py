import sys
import os
import glob
import PIL.Image as Image
import numpy as np
import errno
import subprocess

def im2ar(x):
    y = [[[0]*len(x)]*len(x)]*3
    y[0] = x[:,:,0]
    y[1] = x[:,:,1]
    y[2] = x[:,:,2]
    return y

def ar2im(x):
    y = [[[0]*3]*len(x)]*len(x)
    y[:,:,0] = x[0,:,:] 
    y[:,:,1] = x[1,:,:]
    y[:,:,2] = x[2,:,:]
    return y

def np_ar2im(x):
    rows = x.shape[0]
    width = x.shape[2]
    height = x.shape[3]
    return x.transpose(0, 2, 3, 1).reshape(x.shape[0], width, height, 3)

def does_dir_exist(path):
    try:
        os.stat(path)
    except OSError, e:
        if e.errno == errno.ENOENT:
            return False
        else:
            raise e
    return True


def load_dataset_mscoco():
    #Path

    tmp_dir = "/Tmp"
    if not does_dir_exist(tmp_dir):
        tmp_dir="/tmp"
    mscoco = os.path.join(tmp_dir, "inpainting")
    if not does_dir_exist(mscoco):
        project_dataset_url="http://lisaweb.iro.umontreal.ca/transfert/lisa/datasets/mscoco_inpaiting/inpainting.tar.bz2"
        print("Could not find mscoco dataset. Downloading dataset from: {}".format(project_dataset_url))
        subprocess.call(["wget", "--continue", "-P", tmp_dir, project_dataset_url])
        print("Extracting dataset into directory: {}".format(mscoco))
        subprocess.call(["tar", "xf", os.path.join(tmp_dir, "inpainting.tar.bz2"), "-C", tmp_dir])

    split="train2014"
    data_path = os.path.join(mscoco, split)
    imgs = glob.glob(data_path + "/*.jpg")
    print("cantidad de imagenes = "+str(len(imgs)))
    
    MAXIMAS_train = 100000
    MAXIMAS_val   = 1000
    TESTSETsize   =  100
    
    X_train = []
    y_train = []

    
    
            
    for i, img_path in enumerate(imgs):
        
        if i == MAXIMAS_train: #If the number of images exceeds the limit
            print("limit of "+str(i)+" images for training was achieved")    
            break
        
        img = Image.open(img_path)
        img_array = np.divide(np.array(img,dtype='float32'),255)

        if len(img_array.shape) == 3:
            temp = np.copy(img_array)
            input = np.copy(img_array)
            input[16:48, 16:48,:] = 0
            target = img_array[16:48, 16:48,:]
        else:
            input[:,:,0] = np.copy(img_array)
            input[:,:,1] = np.copy(img_array)
            input[:,:,2] = np.copy(img_array)
            target = input[16:48, 16:48,:]
            input[16:48, 16:48,:] = 0
        
        X_train.append(im2ar(input))
        y_train.append(im2ar(target))
    
    split="val2014"
    data_path = os.path.join(mscoco, split)
    imgs = glob.glob(data_path + "/*.jpg")

    X_original = []
    X_val = []
    y_val = []
            
    for i, img_path in enumerate(imgs):
        
        if i == MAXIMAS_val: #If the number of images exceeds the limit
            print("limit of "+str(i)+" images for validation was achieved")    
            break
        
        img = Image.open(img_path)
        img_array = np.divide(np.array(img,dtype='float32'),255)

        if len(img_array.shape) == 3:
            input = np.copy(img_array)
            original = np.copy(img_array)
            input[16:48, 16:48,:] = 0
            target = img_array[16:48, 16:48,:]
        else:
            input[:,:,0] = np.copy(img_array)
            input[:,:,1] = np.copy(img_array)
            input[:,:,2] = np.copy(img_array)
            original[:,:,0] = np.copy(img_array)
            original[:,:,1] = np.copy(img_array)
            original[:,:,2] = np.copy(img_array)
            target = input[16:48, 16:48,:]
            input[16:48, 16:48,:] = 0

        X_original.append(im2ar(original))
        X_val.append(im2ar(input))
        y_val.append(im2ar(target))
    # We reserve the last TESTSETsize training examples for testing. (was 10000)
    print("From the validation set, the last  "+str(TESTSETsize)+" images are chosen for testing")
    X_original_val, X_original_test = X_original[:-TESTSETsize], X_original[-TESTSETsize:]
    X_val, X_test = X_val[:-TESTSETsize], X_val[-TESTSETsize:]
    y_val, y_test = y_val[:-TESTSETsize], y_val[-TESTSETsize:]
    
    return (np.array(X_train),np.array(y_train),np.array(X_val),np.array(y_val),np.array(X_test),np.array(y_test),np.array(X_original_test))
    

def save_jpg_results(assets_dir, preds, X_test, y_test, X_original_test):
    if not os.path.exists(assets_dir):
        os.makedirs(assets_dir)

    # Denormalize images datasets
    preds = (preds*255).clip(0, 255).astype('uint8')
    X_test = (X_test*255).clip(0, 255).astype('uint8')
    y_test = (y_test*255).clip(0, 255).astype('uint8')
    X_original_test = (X_original_test*255).clip(0, 255).astype('uint8')

    # Save the 100 predictions to JPG files within the 'assets' subdirectory
    X_test = np_ar2im(X_test)
    preds = np_ar2im(preds)
    y_test = np_ar2im(y_test)
    X_original_test = np_ar2im(X_original_test)
    for index in range(preds.shape[0]):
        Image.fromarray(X_test[index]).save(os.path.join(assets_dir, 'images_outer2d_' + str(index) + '.jpg'))
        Image.fromarray(preds[index]).save(os.path.join(assets_dir, 'images_pred_' + str(index) + '.jpg'))
        Image.fromarray(y_test[index]).save(os.path.join(assets_dir, 'images_inner2d_' + str(index) + '.jpg'))
        Image.fromarray(X_original_test[index]).save(os.path.join(assets_dir, 'fullimages_' + str(index) + '.jpg'))
        fullimg_pred = np.copy(X_original_test[index])
        center = (int(np.floor(fullimg_pred.shape[0] / 2.)), int(np.floor(fullimg_pred.shape[1] / 2.)))
        fullimg_pred[center[0]-16:center[0]+16, center[1]-16:center[1]+16, :] = preds[index, :, :, :]
        Image.fromarray(fullimg_pred).save(os.path.join(assets_dir, 'fullimages_pred_' + str(index) + '.jpg'))

def create_html_results_page(filename, assets_dir, num_images):
        # Write a file called 'results.html' that display the image predictions versus the true images in a convenient way
    img_src = assets_dir
    if not img_src[-1] == '/':
        img_src += '/'
    html_file = filename
    with open(html_file, 'w') as fd:
        fd.write("""
<table>
  <tr>
    <th style="width:132px">Input</th>
    <th style="width:68px">Model prediction</th>
    <th style="width:68px">Correct output</th> 
    <th style="width:132px">Input + prediction</th>
    <th style="width:132px">Input + correct output</th>
  </tr>
""")

        for index in range(num_images):
            fd.write("  <tr>\n")
            fd.write("    <td><img src='%s/images_outer2d_%i.jpg' width='128' height='128'></td>\n" % (img_src, index))
            fd.write("    <td><img src='%s/images_pred_%i.jpg' width='64' height='64'></td>\n" % (img_src, index))
            fd.write("    <td><img src='%s/images_inner2d_%i.jpg' width='64' height='64'></td>\n" % (img_src, index))
            fd.write("    <td><img src='%s/fullimages_pred_%i.jpg' width='128' height='128'></td>\n" % (img_src, index))
            fd.write("    <td><img src='%s/fullimages_%i.jpg' width='128' height='128'></td>\n" % (img_src, index))
            fd.write('</tr>\n')
        
        fd.write('</table>')
        
