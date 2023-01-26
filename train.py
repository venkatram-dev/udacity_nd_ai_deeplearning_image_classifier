
from  helper_functions import get_input_args_for_train
from  helper_functions import data_transforms
from collections import OrderedDict
# Imports here
#%matplotlib inline
#%config InlineBackend.figure_format = 'retina'

#python train.py ../../../data/flowers/ --arch "vgg16" --hidden_units 1000 --epochs 2 --gpu --save_directory saved_checkpoints/checkpoint3.pth --learning_rate 0.001

#python train.py ../../../data/flowers/ --arch "resnet50" --hidden_units 1000 --epochs 5 --gpu --save_directory saved_checkpoints/checkpoint4.pth --learning_rate 0.007

import matplotlib.pyplot as plt

import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict

def build_classifier(in_features,out_class,model_name,hidden_units):
    
    input_layer_units=in_features
    output_layer_units=out_class
    hidden_layer_units=[1000]
    if hidden_units:
        #units_layer_list=hidden_units.split(',')
        units_layer_list=[ int(q) for q in hidden_units.split(',')]
        print('split',units_layer_list)
        
        #input_layer_units=int(units_layer_list[0].strip('[').strip(']'))
        #output_layer_units=int(units_layer_list[-1].strip('[').strip(']'))
        hidden_layer_units=units_layer_list   
        print ('input_units',input_layer_units)
        print ('output_units',output_layer_units)
        print ('hidden_units',hidden_layer_units)
    
    layer_string=''
    first_layer="""('fc1', nn.Linear({0}, {1})),\
    ('relu1', nn.ReLU()),\
    ('drp1',nn.Dropout()),""".format(input_layer_units,int(hidden_layer_units[0]))
    print('first_layer',first_layer)
    
    layer_string+=first_layer
    print ('layer_list before',layer_string)
    #g1 = [x.strip('\'"') for x in layer_list]
    #print ('g1',g1)
    #stop
    print('hidden layers')
    inp=hidden_layer_units[0]
    j=1
    for i in hidden_layer_units[1:]:
        out=int(i)
        print('out',out)
        j+=1
        intermediate_layer="""('fc{2}', nn.Linear({0}, {1})),\
                 ('relu{2}', nn.ReLU()),\
                 ('drp{2}',nn.Dropout()),""".format(inp,i,j)
        print('intermediate_layer',intermediate_layer)
        
        inp=out
        
        layer_string+=intermediate_layer
        
    #stop    
    
    print ('layer_list inter',layer_string)
    j+=1
    last_layer="""('fc{2}', nn.Linear({0}, {1})),\
                          ('output', nn.LogSoftmax(dim=1))""".format(int(hidden_layer_units[-1]),output_layer_units,j)
    print ('last_layer',last_layer)

    layer_string+=last_layer
    #layer_list.append(last_layer)
    print ('layer_list',layer_string)

    return layer_string

def validation(model, testloader, criterion,device='cpu'):
    test_loss = 0
    accuracy = 0
    for images, labels in testloader:

        #images.resize_(images.shape[0], 784)
        #print ('device in validation func',device)
        if device =='gpu':
            images, labels = images.to('cuda'), labels.to('cuda')
        else: 
            images, labels = images.to('cpu'), labels.to('cpu')
        
        output = model.forward(images)
        test_loss += criterion(output, labels).item()

        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    
    return test_loss, accuracy

def check_accuracy_on_test(model,testloader,device='cpu'):    
    correct = 0
    total = 0
    if device=='gpu':
        model.to('cuda')
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        model.to('cpu')
        torch.set_default_tensor_type('torch.FloatTensor')
    
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            if device=='gpu':
                images, labels = images.to('cuda'), labels.to('cuda')
            else:
                images, labels = images.to('cpu'), labels.to('cpu')
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

def do_deep_learning2(model, trainloader, validationloader, testloader, epochs, print_every, criterion, optimizer, device='cpu'):
    epochs = epochs
    print_every = print_every
    steps = 0

    print('device',device)
    # change to cuda
    if device=='gpu':
        model.to('cuda')
    else:
        model.to('cpu')

    for e in range(epochs):
        running_loss = 0
        for ii, (inputs, labels) in enumerate(trainloader):
            steps += 1
            if device=='gpu':
                inputs, labels = inputs.to('cuda'), labels.to('cuda')
            else: 
                inputs, labels = inputs.to('cpu'), labels.to('cpu')

            optimizer.zero_grad()

            # Forward and backward passes
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                model.eval()
                
                with torch.no_grad():
                     test_loss, accuracy = validation(model, validationloader, criterion,device)
            
                print("Epoch: {}/{}.. ".format(e+1, epochs),
                  "Training Loss: {:.3f}.. ".format(running_loss/len(trainloader)),
                 #"Training Loss1: {:.3f}.. ".format(running_loss/print_every),
                  "Test Loss: {:.3f}.. ".format(test_loss/len(testloader)),
                  "Test Accuracy: {:.3f}".format(accuracy/len(testloader)))
                
                running_loss = 0
                
                model.train()
    
def save_checkpoint(save_directory,epochs,optimizer,model,classifier_dict,class_to_idx,model_name,learning_rate):

    print("Our model: \n\n", model, '\n')
    print("The state dict keys: \n\n", model.state_dict().keys())
    model.class_to_idx = class_to_idx
    checkpoint = {'model_name':model_name,
                  'epoch_saved': epochs,
                  'state_dict_saved': model.state_dict(),
                  'optimizer_state_dict_saved': optimizer.state_dict(),
                   'classifier_dict_saved': classifier_dict,
                  'class_to_idx_saved': class_to_idx,
                  'learning_rate': learning_rate}
    checkpoint_file=save_directory 
    #+ '/checkpoint3.pth'
    torch.save(checkpoint, checkpoint_file)


def main():

    training_args=get_input_args_for_train()


    print ('training_args',training_args)

    data_dir = training_args.data_directory
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    print (train_dir)
    
    
    trainloader,testloader ,validationloader,train_data=data_transforms(train_dir,valid_dir,test_dir)
    print(trainloader)

    model_name=training_args.arch
    model=eval("""models.{0}(pretrained=True)""".format(model_name))
    #model=model_name
    print ('model',model)
    out_class=102
    
    learning_rate=float(training_args.learning_rate)
    print ('learning_rate',learning_rate)
    for param in model.parameters():
        param.requires_grad = False
    if model_name =='resnet50':
        features=model.fc.in_features
        print('features',features) 
        model.fc=nn.Linear(features,out_class)
        print ('model after',model)
        classifier_dict={}
        classifier_dict['in_features']=features
        classifier_dict['out_class']=out_class
        
        #criterion = torch.nn.CrossEntropyLoss()
        #optimizer = torch.optim.SGD(model.fc.parameters(), lr=0.001, momentum=0.9)
        #exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
        
        # to try
        #criterion = nn.NLLLoss()
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.fc.parameters(), lr=learning_rate)
    
        print ('optimizer',optimizer)
    if model_name =='vgg16':
        in_features=model.classifier[0].in_features
        print('in_features',in_features) 
        print ('hidden_units',training_args.hidden_units)    
        classifier_string=build_classifier(in_features,out_class,model_name,training_args.hidden_units)
        print ('classifier_string',classifier_string)

        classifier_dict=OrderedDict(eval(classifier_string))
        print ('classifier_dict',classifier_dict)

        classifier = nn.Sequential(classifier_dict)

        model.classifier = classifier

        print(model)
        criterion = nn.NLLLoss()
        optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
        print ('optimizer',optimizer)



    

    
    epochs=int(training_args.epochs)
    print ('epochs',epochs)
    
    print (training_args.gpu)
    
    
    if training_args.gpu== True:
        device='gpu'
    else:
        device='cpu'
    steps=40
    do_deep_learning2(model, trainloader, validationloader, testloader, epochs, steps, criterion, optimizer, device)
    
    check_accuracy_on_test(model,testloader,device)
    
    save_directory=training_args.save_directory
    # will do save later
    print('save_directory',save_directory)
    save_checkpoint(save_directory,epochs,optimizer,model,classifier_dict,train_data.class_to_idx,model_name,learning_rate)
    
main()