import torch

from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader 


from train import Train

class ModelHelper: 
    '''
    interface to interact with driver.py 
    note that all interactions with directory will be done by driver, not here 

    this handles the train and predict modes too 
        - this involves continuing to train based on a given model, where the params of the model have to be checked 
    '''
    def __init__(self, args, loaded_model= None): 
        self.args= args 
        self.loaded_model = loaded_model

        if loaded_model: 
            self.loaded_model_check() 

    def loaded_model_check(self): 
        '''
        if there is a loaded model specified, ensure that the model is correct to the parameters set in the current model 
        stop all training if the models do not match 

        the items that have to match are: 
        - model_type 
        - hidden_sizes 
        - k_value 
        '''
        matches= {}
        if self.args.target_mode != self.loaded_model['args'].target_mode:
            matches['target_mode'] = False 
        if self.args.model_type != self.loaded_model['args'].model_type: 
            matches['model_type'] = False 
        if self.args.hidden_sizes != self.loaded_model['args'].hidden_sizes:
            matches['hidden_sizes'] = False 
        if self.args.k_value != self.loaded_model['args'].k_value:
            matches['k_value'] = False  

        if not all(matches.values()): # if any one of them is false 
            raise Exception("current specified model and loaded model params don't match: ", matches )
        else: 
            print('model to be loaded has matching params to args')
    
    def load_data(self, dataset): # returns dataloader objects 
        '''
        given the data and args, prepare the data to be fed into the model
            since using dataloader, the data can be placed onto gpu in training loop, iterating through dataloader (dataloader object 
            itself is on the cpu, but references the batch , which can be placed ont gpu)
        
        this function shouldn't change based on the model
        however, it should change based on the specification of the args
        '''
        # drop column of IDs 
        data= dataset.drop(columns= ['ID'], axis = 1 , inplace= False ) # create copy 

        # split into x and y 
        x, y = data.iloc[:, :-1], data.iloc[:, -1] # array, and series
        print('x and y are created, the shapes are: ', x.shape, y.shape)

        # convert to torch
        x= torch.tensor(x.values, dtype= torch.float32 ) # note pytorch always intialized model to dtype float32 
        if self.args.target_mode == 'classification': 
            y= torch.tensor(y, dtype= torch.long) # crossentropyloss expects dtype long
        else : 
            y= torch.tensor(y, dtype= torch.float32) # crossentropyloss expects dtype float32


        # train test split 
        xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size =.30, shuffle= True, random_state = 1 )
            # random state makes it replicable 
            # batching not done here, done in dataloader 
            # shuffle rows 

        # zip into tensordataset: train and test (for train loop parts)
        train= TensorDataset(xtrain, ytrain) # training set 
        test= TensorDataset(xtest, ytest) # testing set 

        # use dataloader to iterate through tensordatasets 
        train_loader = DataLoader(train, batch_size = self.args.batch_size, shuffle= True)
        test_loader= DataLoader(test, batch_size = self.args.batch_size, shuffle= True)
            # shuffle batches, smaller batch size = less memory requirement, but more noisy updates within epoch 
            # epoch goes over entire dataset 
        
        return train_loader, test_loader 

    def training(self, train_test_dataloaders): 
        ''' 
        this calls the training loop in train.py  
            this can vary based on the model chosen 

        plotting can be called from here too to visualize the training progress 

        the model state will be passed back to driver so that it can be saved 
        '''

        train = Train(self.args, self.loaded_model) # model given is guaranteed to be valid, checked already  

        if self.args.target_mode =='classification': 
            model_state = train.training_loop_classification(train_test_dataloaders) # can specify existimg model to train from if wanted with optional argument... 
        else: # regression 
            model_state = train.training_loop_regression(train_test_dataloaders)

        return model_state 




