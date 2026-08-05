from utils import Utils 
from models import SLP, FNN, KMeansFNN, AutoencoderKmeansFNN
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
    

    better workflow done below :)


    '''

    def __init__(self, args): 
        self.args= args 
        self.logger= logging.getLogger(self.args.logger_name)

        self.train_class= Train(self.args)

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

    def model_chooser(self) -> dict: # model state  
        '''
        choose the model to train 
        each one will be done uniquely to reduce computation cost 
        otherwise, the model in models.py will have to be called to predict multiple times (number of epochs), which is unncessarily costly
        '''

        self.logger.debug('model chosen: '+ self.args.model_type)


        if self.args.model_type =='slp': 
            model_state= self.slp_chosen()
            pass 
        elif self.args.model_type == 'fnn': 
            model_state= self.fnn_chosen()
            pass 
        elif self.args.model_type == 'kmeans+fnn': 
            model_state = self.kmeans_fnn_chosen()
            pass 
        elif self.args.model_type == 'autoencoder+kmeans+fnn': 
            model_state = self.autoencoder_kmeans_fnn_chosen()
            pass 
        
        self.logger.info('completed training for: '+self.args.model_type)
        return model_state 
    
    def autoencoder_kmeans_fnn_chosen(self) -> dict: # model state  
        '''
        this is for the autoencoder + kmeans + fnn model
        try to contain all within 1 model for more modularity in code  
        '''
        # get data 
        data = self.load_data()

        # turn data in dataloader objects 
        train, test = self.make_dataloader(data)

        input_size = next(iter(train))[0].shape[1] 
        hidden_sizes = self.args.hidden_sizes 
        output_size = 5 if self.args.output_mode == 'classification' else 1  # depends on output type 
        model = AutoencoderKmeansFNN(self.args.k_value, input_size, hidden_sizes, output_size)
        self.logger.info('autoencoder means fnn model is made: '+ str(model))


        ### autoencoder section (get the sub model though the main model)
        self.logger.debug('train autoencoder')
        # initalize the model 
        autoencoder_model = model.autoencoder
        # pass train dataloader to train loop to train the model 
        autoencoder_model_state = self.train_class.train(data= (train, test), model = autoencoder_model )
        self.logger.info('autoencoder trained')

        ### kmneans section 
        self.logger.debug('train kmeans')
        # initalize model 
        kmeans_model = model.kmeans
         # get embeddings from autoencoder 
        embeddings = autoencoder_model.predict_embedding(data['trainx'])
        # train model on entire data embeddings 
        # kmeans_model.fit(data['trainx'])
        kmeans_model.fit(embeddings.detach().numpy()) # want numpy array for sklearn kmeans  
        self.logger.info('kmeans trained on autoencoder embeddings')
        

        ### fnn model 
        self.logger.debug('train fnn portion')
        # initialize model- done above 
        # freeze params of autoencoder so backprop doesn't update layers 
        self.train_class.freeze_params(autoencoder_model)
        # train loop applied to dataloader data 
        model_state = self.train_class.train(data = (train, test), model = model )
        # within the model, forward is used to get predictions from autoencoder and kmeans 
        self.logger.info('autoencoder kmeans fnn model trained')

        return model_state 


    def kmeans_fnn_chosen(self)-> dict: 
        '''
        kmeans + fnn model 
        in order to do this, the kmeans model has to be trained already and integrated into the data 
            - this is the proper way to 
        i want the class for this model to be defined on its own
            this means that the class will use a saved kmeans model during the forward pass
        '''
        # get data 
        data= self.load_data()

        # get dataloader from data 
        train,test = self.make_dataloader(data)

        # get layer sizes 
        input_size = next(iter(train))[0].shape[1] + 1 # since cluster labels will be added in model 
        hidden_sizes = self.args.hidden_sizes 
        output_size = 5 if self.args.output_mode == 'classification' else 1  # depends on output type 
         
        # initialize model 
        model = KMeansFNN(self.args.k_value, input_size, hidden_sizes, output_size)
        self.logger.info('kmeans fnn model is made: '+ str(model))

        # use data to train the kmeans model 
        model.train_kmeans(data['trainx']) # train the kmeans model 
        self.logger.info('kmeans model is trained')

        # call training loop  
        model_state = self.train_class.train((train, test), model) # dataloaders passed in with model which already has trained kmeans 
        self.logger.debug('fnn is now trained')

        return model_state

    # def driver(self): 
    #     # call model_chooser, returns model_state 
    #     # plot the model_state training info 
    #     pass 

    def slp_chosen(self) -> dict: # model state 
        '''
        single layer perceptron model 
        mainly used for debugging code
        '''
        # get data 
        data = self.load_data()
        # make data loader 
        train, test = self.make_dataloader(data)
        # train model with dataloader 

        input_size = next(iter(train))[0].shape[1]
        output_size = 5 if self.args.output_mode == 'classification' else 1  # depends on output type 
        
        self.logger.info('batched x,y data size: '+ str([data.shape for data in next(iter(train))]))
        self.logger.info('model output shape: '+ str(output_size))


        model = SLP(input_size, output_size)

        # pass to training loop 
        model_state = self.train_class.train((train, test), model)
        return model_state 

    def fnn_chosen(self) -> dict: 
        '''
        fnn 
        hidden size determines the number of layers
        ''' 

        # get data 
        data = self.load_data()
        # make dataloader 
        train, test= self.make_dataloader(data)

        # get layer sizes 
        input_size = next(iter(train))[0].shape[1]
        hidden_sizes = self.args.hidden_sizes 
        output_size = 5 if self.args.output_mode == 'classification' else 1  # depends on output type 
        

        # define model 
        model = FNN(input_size, hidden_sizes, output_size)
        self.logger.info('fnn model is made: '+ str(model))

        # pass to training loop 
        model_state = self.train_class.train(data = (train, test), model = model )

        return model_state 




