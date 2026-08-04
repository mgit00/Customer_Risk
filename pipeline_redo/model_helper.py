from utils import Utils 
from models import SLP
from train import Train 

import torch 
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader 

import logging 

class ModelHelper: 
    '''
    model helper class to implement needed functions for train.py 
    - load + split data into x and y 
    - create dataloader objects 
    - determined by the model specified 

    the sequence of the functions called are dependent on the model selected 
    - fnn and slp 
        load data 
        make dataloader
        train.py -> model.py

    - kmeans + fnn 
        # load data 
        # train kmeans on xtrain --> xtrain+clusters 
        # make dataloader with xtrain+clusters 
        # train.py -> model.py 
    
        # or 
        # do below, makes code easier and more modular 
            # easy to build upon later on 

        load data 
        train kmeans on xtrain -> trained kmeans model 
        make dataloader with xtrain
        train.py -> model.py 
            model gets xtrain data, predicts on trained kmeans, passes to nn layers 
    
    - autoencoder + kmeans + fnn 
        load data 
        make dataloader 
        train.py -> model.py 
            autoencoder 
        
        predict on entire xtrain data (to get compressed embeddings)
            -> use to train kmeans model 

        return to xtrain data! 
        make dataloader 
        train.py -> model.py 
            fnn (pred on AE for embeddings, pred on kmeans for clusters, pred on fnn )


    RETHINKING APPROACH: 
    this means that pred will be called for number of epochs * data divided by num batches 
    which means that it will be computationally expensive 
    instead seperate out pipelines for each model 
        so you dont have to predict on trained autoencoder and kemans model for each train epoch of the fnn 
    
    however, for the predicting portion of the pipeline, i will have to do this 

    so, for training, do: 
    - fnn and slp 
        load data 
        make dataloader
        train.py -> model.py

    - kmeans + fnn 
        load data 
        train kmeans on xtrain --> xtrain+clusters 
        make dataloader with xtrain+clusters 
        train.py -> model.py 

    - autoencoder + kmeans + fnn 
        load data 
        make dataloader 
        train.py -> model.py 
            autoencoder (training)
        
        predict xtrain with trained autoencoder (full data, to get compressed embeddings) -> xtrain_embeddings
            -> use to train kmeans model 
        kmeans model training -> xtrain_embeddings+clusters 

        make dataloader with xtrain_embeddings+clusters 
        train.py -> model.py 
            fnn 

    
    for predicting do: 
        load data 
        make dataloader 
        train.py -> model.py 
            call predict() for each model to call on the different steps for each model 
    
        


    '''

    def __init__(self, args): 
        self.args= args 
        self.logger= logging.getLogger(self.args.logger_name)

        self.train= Train(self.args)

    def load_data(self) -> dict: # return train, test for x, y 
        '''
        load data for the purpose of model creation 
        split data into x,y 
        make tensors 
        do train test split 
        '''
        # read data 
        data = Utils.read_data(self.args.data_filename)
        self.logger.debug('data is loaded ')

        x,y = data.iloc[:, :-1], data.iloc[:, -1] # df, and series 

        # y has to be long if using crossentropy loss (classification) or float32 for mse loss (regression)
        x = torch.tensor(x.values, dtype = torch.float32) # float32 = default for pytorch models/model layers 
        y = torch.tensor(y, dtype= torch.long if self.args.output_mode == 'classification' else torch.float32)

        trainx, testx, trainy, testy = train_test_split(x, y, test_size=.3, random_state= 1, shuffle = True)
            # random state makes it replicable 
            # batching not done here, done in dataloader 
            # shuffle rows 

        self.logger.info('train test split for data is done')
        
        return {'trainx':trainx, 'testx':testx, 'trainy':trainy, 'testy':testy}

    def make_dataloader(self, data: dict) -> tuple: # of dataloaders 
        # zip into tensordataset: train and test (for train loop parts)

        train = TensorDataset(data['trainx'], data['trainy'])
        test = TensorDataset(data['testx'], data['testy'])

        # use dataloader to iterate through tensordatasets 
        train_loader = DataLoader(train, batch_size = self.args.batch_size, shuffle = True)
        test_loader = DataLoader(test, batch_size  = self.args.batch_size, shuffle = True )
            # shuffle batches, smaller batch size = less memory requirement, but more noisy updates within epoch 
            # epoch goes over entire dataset 

        self.logger.info('tensordataset is batched and shuffled with dataloader')
        return train_loader, test_loader 

    def model_chooser(self) -> None: 
        '''
        choose the model to train 
        each one will be done uniquely to reduce computation cost 
        otherwise, the model in models.py will have to be called to predict multiple times (number of epochs), which is unncessarily costly
        '''

        self.logger.debug('model chosen: '+ self.args.model_type)

        if self.args.model_type =='slp': 
            self.slp_chosen()
            pass 
        elif self.args.model_type == 'fnn': 
            pass 
        elif self.args.model_type == 'kmeans+fnn': 
            pass 
        elif self.args.model_type == 'autoencoder+kmeans+fnn': 
            pass 

        return 
    


    def slp_chosen(self): # -> return model state 
        '''
        single layer perceptron model 
        mainly used for debugging code
        '''
        # get data 
        data = self.load_data()
        # make data loader 
        train, test = self.make_dataloader(data)
        # train model with dataloader 

        # print(next(iter(train)))
        input_size = next(iter(train))[0].shape[1]
        output_size = 5 if self.args.output_mode == 'classification' else 1  # depends on output type 
        
        self.logger.info('batched x,y data size: '+ str([data.shape for data in next(iter(train))]))
        self.logger.info('model output shape: '+ str(output_size))


        model = SLP(input_size, output_size)

        # pass to training loop 
        model_state = self.train.train((train, test), model)




