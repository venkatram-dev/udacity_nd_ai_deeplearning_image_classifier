import argparse
import torch
from torchvision import datasets, transforms

def get_input_args_for_train():
    # Create Parse using ArgumentParser
    
    parser = argparse.ArgumentParser(description="Arguments for Training Image classifier")
    
    # Create 3 command line arguments as mentioned above using add_argument() from ArguementParser method
    
    parser.add_argument("data_directory",default="../../../data/flowers/",help=" Specify Data Directory")
    parser.add_argument("--save_directory",dest="save_directory",default="saved_checkpoints/",help=" Directory to save checkpoints")
    parser.add_argument("--arch",dest="arch",default="vgg13",help=" Specify Model Architecture")
    parser.add_argument("--learning_rate",dest="learning_rate",default="0.001",help=" Specify Model Architecture")
    parser.add_argument("--hidden_units",dest="hidden_units",default="1000",help=" Specify Hidden layer units")
    parser.add_argument("--epochs",dest="epochs" ,default="5",help=" Specify number of epochs")
    parser.add_argument("--gpu",action="store_true", default=False,help=" Specify device as cpu or gpu")
   
    args=parser.parse_args()
    # Replace None with parser.parse_args() parsed argument collection that 
    # you created with this function 
    return args


def data_transforms(train_dir,test_dir,valid_dir):
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.RandomVerticalFlip(), # check if these work
                                           #transforms.Resize(224),# check if these work
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], 
                                                                [0.229, 0.224, 0.225])])


    test_transforms = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], 
                                                               [0.229, 0.224, 0.225])])


    validation_transforms = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], 
                                                               [0.229, 0.224, 0.225])])


    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)
    validation_data=datasets.ImageFolder(valid_dir, transform=validation_transforms)

    # TODO: Load the datasets with ImageFolder
    #image_datasets = 'a'

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    #dataloaders = 'b'
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=32)
    validationloader = torch.utils.data.DataLoader(validation_data, batch_size=32)
    
    #print ('in funct',trainloader,testloader ,validationloader)
    return trainloader,testloader ,validationloader,train_data


def get_input_args_for_predict():
    # Create Parse using ArgumentParser
    
    parser = argparse.ArgumentParser(description="Arguments for Predicting Image class")
        
    parser.add_argument("image_location",default="../../../data/flowers/test/11/image_03177.jpg",help=" Specify Image File name")
    parser.add_argument("checkpoint_file",default="saved_checkpoints/checkpoint3.pth",help=" Specify Checkpoint File name")
    parser.add_argument("--top_k",dest="top_k",default="5",help=" Topk to predict")
    parser.add_argument("--category_names",dest="category_names",default="cat_to_name.json",help=" Specify category json")
    parser.add_argument("--gpu",action="store_true", default=False,help=" Specify device as cpu or gpu")
   
    args=parser.parse_args()
    # Replace None with parser.parse_args() parsed argument collection that 
    # you created with this function 
    return args


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    print ('input image',image)
    
    
    from matplotlib.pyplot import imshow
    import numpy as np
    
    from PIL import Image
    
    im = Image.open(image)
   
    
    #%matplotlib inline
    im = Image.open(image)
    #print('initial shape')

    size = 270, 256

    im.thumbnail(size)
    
    #print('thumbnail shape')
    #print( np.asarray(im).shape)

    orig_width, orig_height = im.size   

    desired_width,desired_height=224,224

    left = (orig_width - desired_width)/2  
    top = (orig_height - desired_height)/2
    right = (orig_width + desired_width)/2
    bottom = (orig_height + desired_height)/2
    
    #print ('crop sizes',left,top,right,bottom)
    new_im=im.crop((left, top, right, bottom))

    #print('after crop')
    #print (np.asarray(new_im).shape)

    np_image = np.array(new_im)
    
    #print('np_image.shape',np_image.shape)
      
    np_image_div = np.divide(np_image,255)
      
    mean_array=[0.485, 0.456, 0.406]
    
    np_image_mean=np_image_div-mean_array
    
    std_array=[0.229, 0.224, 0.225]
    
    np_image_std=np_image_mean/std_array
    
    np_image_trans=np.transpose(np_image_std, (2, 0, 1))
        
    print('shape', np_image_trans.shape)

    
    return np_image_trans
    
def imshow(image, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax    
