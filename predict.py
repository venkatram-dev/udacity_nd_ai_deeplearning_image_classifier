from helper_functions import get_input_args_for_predict
from helper_functions import process_image
from helper_functions import imshow

import torch
import numpy as np
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models

#python predict.py ../../../data/flowers/test/10/image_07104.jpg saved_checkpoints/checkpoint3.pth --gpu --top_k 5 --category_names cat_to_name.json

#python predict.py ../../../data/flowers/test/10/image_07104.jpg saved_checkpoints/checkpoint4.pth --gpu --top_k 7 --category_names cat_to_name.json

def build_model_from_checkpoint(filepath):
    filepath=filepath
    checkpoint_ld = torch.load(filepath)
    state_dict = checkpoint_ld['state_dict_saved']

    print(state_dict.keys())
    

    print(checkpoint_ld['classifier_dict_saved'])
    print('checkpoint_ld',checkpoint_ld)
    model_name=checkpoint_ld['model_name']
    learning_rate=checkpoint_ld['learning_rate']
    
    if model_name=='vgg16':
        
        newb_model= models.vgg16(pretrained=True)
        newb_model.classifier=nn.Sequential(checkpoint_ld['classifier_dict_saved'])
        print('newb_model',newb_model)
        newb_model.load_state_dict(checkpoint_ld['state_dict_saved'])
        optimizer = optim.Adam(newb_model.classifier.parameters(), lr=learning_rate)
        optimizer.load_state_dict(checkpoint_ld['optimizer_state_dict_saved'])
        print('optimizer',optimizer)
        newb_model.class_to_idx = checkpoint_ld['class_to_idx_saved']
        #newb_model.class_to_idx
    elif model_name=='resnet50':
        newb_model= models.resnet50(pretrained=True)
        classifier_dict=checkpoint_ld['classifier_dict_saved']
        features = classifier_dict['in_features']
        out_class = classifier_dict['out_class']
        newb_model.fc=nn.Linear(features,out_class)
        print('newb_model',newb_model)
        newb_model.load_state_dict(checkpoint_ld['state_dict_saved'])
        optimizer = optim.Adam(newb_model.fc.parameters(), lr=learning_rate)
        optimizer.load_state_dict(checkpoint_ld['optimizer_state_dict_saved'])
        print('optimizer',optimizer)
        newb_model.class_to_idx = checkpoint_ld['class_to_idx_saved']
        
    return newb_model,model_name


def predict_class(image_path, model, topk,device,idx_to_class,cat_to_name,model_name):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    
    t=process_image(image_path)
    v = torch.from_numpy(t)
    #print(v.shape)
    v.unsqueeze_(0)
    #print(v.shape)
    v=v.float()
    if device =='gpu':
        v = v.cuda()
    model.eval()

    
    if model_name =='vgg16':
            with torch.no_grad():
                prediction = model(v)
                ps = torch.exp(prediction)
    else:
            with torch.no_grad():
                preds = nn.LogSoftmax()(model(v))
                print('preds',preds)
                #res = np.argmax(preds)
                #print('res',res)
                ps = torch.exp(preds)

                print('ps',ps)
    #stop
    k=topk
    c2,v2=ps.topk(k)

    #print (c2,v2)
    probablities=c2.cpu().numpy()
    classes=v2.cpu().numpy()

    j=0
    class_list=[]
    cat_list=[]
    probability_list=[]
    for i in classes[0]:  
        class_list.append(idx_to_class[i])
        cat_list.append(cat_to_name[idx_to_class[i]])
    for i in probablities[0]:
        probability_list.append(i)

    #print(class_list)
    #print (probability_list)
    #print(cat_list)
    return probability_list,class_list,cat_list
    


def main():
    
    
    predict_args=get_input_args_for_predict()
    
    image_location=predict_args.image_location
    checkpoint_file_location=predict_args.checkpoint_file
    
    print('image_location',image_location)
    print('checkpoint_file_location',checkpoint_file_location)
    
    newb_model,model_name=build_model_from_checkpoint(checkpoint_file_location)
    
    # invert the class_to_idx dictionary and create new dictionary to store index to class mapping
    idx_to_class={}
    for key, value in newb_model.class_to_idx.items():
        idx_to_class[value]= key
    
    if predict_args.gpu:
        device='gpu'
    else:
        device='cpu'
        
    top_k=int(predict_args.top_k)
    print('top_k',top_k )
    
    #im1='../../../data/flowers/test/11/image_03141.jpg.jpg'
    im1=image_location

    im1
    
    import json

    if predict_args.category_names != None:
        category_names_file=predict_args.category_names
    else:
        category_names_file='cat_to_name.json'
        
    with open(category_names_file, 'r') as f:
        cat_to_name = json.load(f)
    print (cat_to_name)

    if device=='gpu':
        # model to cuda
        newb_model.to('cuda')

    #im1=test_dir+'/10/image_07090.jpg'
    probs, classes,cat_list = predict_class(im1, newb_model, top_k,device,idx_to_class,cat_to_name,model_name)
    print('probabilities',probs)
    print('classes',classes)
    print ('cat_list',cat_list)
        
    
main()