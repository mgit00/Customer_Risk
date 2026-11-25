from .utils import Utils 
from .models import SLP, FNN, KMeansFNN, AutoencoderKmeansFNN
from .train import Train 
from .plotting import Plotting
from sklearn.preprocessing import StandardScaler, PowerTransformer
import torch 
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader   
import pickle 
import logging 
import numpy as np 
from sklearn.utils.class_weight import compute_class_weight

from utils.schemas import ModelTrainingBasemodel
from utils.constants import ConstantFilenames, ModelType

from torch.utils.data import WeightedRandomSampler

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

class ModelHelper: 
    '''
    model helper class to implement needed functions for train.py 
    - load + split data into x and y 
    - create dataloader objects 
        - determined by the model specified 

    the sequence of the functions called are dependent on the model selected 
    '''

    def __init__(self, args: ModelTrainingBasemodel): 
        self.args= args 
        self.logger= logging.getLogger('main_logger')
        self.output_mode = 'classification'

        self.train_class= Train(self.args)
        self.plotting_class = Plotting()

    def driver(self) -> None: 
        # call model_chooser, returns model_state 
        model_state = self.model_chooser()

        # save model state 
        Utils.save_model(location = self.args.model_filename, model_state= model_state)

        # plot the model_state training info 
        self.plotting_class.nn_plotting(self.args, model_state)
        return
    

    def load_data(self) -> dict: # return train, test for x, y 
        '''
        load data for the purpose of model creation 
        split data into x,y 
        make tensors 
        do train test split 
        '''
        # read data 
        data = Utils.read_data(ConstantFilenames.data_filename.value)
        self.logger.debug('data is loaded ')

        # x,y = data.iloc[:, :-1], data.iloc[:, -1:] # df, and df (not series, makes mode code more complicated to switch between classificaiton and regression later on) 
        y = data[['risk_level']]
        x = data.drop(columns = ['risk_level'])

        # print(x.columns, y.columns)
        scaler = StandardScaler() 
        # x = scaler.fit_transform(x) # add values backto x call below!!!!! 
        x = PowerTransformer(method='yeo-johnson', standardize=True).fit_transform(x) # fixme have to split to train test and to do on ytest transform only 
        # with open("model_info/data_processing/scaler.pkl", "wb") as f:
        #     pickle.dump(scaler, f)

        # print("NaNs in trainx:", np.isnan(x).sum(), "Infs:", np.isinf(x).sum())

        # y has to be long if using crossentropy loss (classification) or float32 for mse loss (regression)
        x = torch.tensor(x, dtype = torch.float32) # float32 = default for pytorch models/model layers 
        y = torch.tensor(y.values.ravel(), dtype= torch.long if self.output_mode == 'classification' else torch.float32)


        trainx, testx, trainy, testy = train_test_split(x, y, test_size=.3, random_state= 1, shuffle = True)
            # random state makes it replicable 
            # batching not done here, done in dataloader 
            # shuffle rows 

         # get weights for each class 
        ytrain= trainy.numpy().ravel()
        classes = np.unique(ytrain)
        weights = compute_class_weight(class_weight='balanced', classes=classes, y=ytrain)
        class_weights = torch.tensor(weights, dtype=torch.float32)
        self.logger.info(f'class weights are {class_weights}')

        # # do sampler -- this made things worse ... 
        # sample_weights = class_weights[trainy.cpu().view(-1)]
        # self.sampler = WeightedRandomSampler(
        #     weights=sample_weights,
        #     num_samples=len(sample_weights),
        #     replacement=True
        # )


        '''  
        rf = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=1)
        rf.fit(trainx.numpy(), trainy.numpy())
        preds = rf.predict(testx.numpy())

        print("RF Test Accuracy:", accuracy_score(testy.numpy(), preds))
        print(classification_report(testy.numpy(), preds))

        return '''
    

        self.logger.info('train test split for data is done')
        
        return {'trainx':trainx, 'testx':testx, 'trainy':trainy, 'testy':testy}

    
    def make_dataloader(self, data: dict) -> tuple: # of dataloaders 
        # # zip into tensordataset: train and test (for train loop parts)

        train = TensorDataset(data['trainx'], data['trainy'])
        test = TensorDataset(data['testx'], data['testy'])

        # use dataloader to iterate through tensordatasets 
        train_loader = DataLoader(train, batch_size = self.args.batch_size, shuffle = True)
        test_loader = DataLoader(test, batch_size  = self.args.batch_size, shuffle = True )
            # shuffle batches, smaller batch size = less memory requirement, but more noisy updates within epoch 
            # epoch goes over entire dataset 

        # # zip into tensordataset: train and test (for train loop parts)
        
        # train = TensorDataset(data['trainx'], data['trainy'])
        # test = TensorDataset(data['testx'], data['testy'])

        # # use dataloader to iterate through tensordatasets 
        # train_loader = DataLoader(train, batch_size = self.args.batch_size, sampler=self.sampler,  shuffle = False)
        # test_loader = DataLoader(test, batch_size  = self.args.batch_size, shuffle = True )
        #     # shuffle batches, smaller batch size = less memory requirement, but more noisy updates within epoch 
        #     # epoch goes over entire dataset 

        self.logger.info('tensordataset is batched and shuffled with dataloader')
        return train_loader, test_loader 

    def model_chooser(self) -> dict: # model state  
        '''
        choose the model to train 
        each one will be done uniquely to reduce computation cost 
        otherwise, the model in models.py will have to be called to predict multiple times (number of epochs), which is unncessarily costly
        '''

        self.logger.debug('model chosen: '+ self.args.model_type)


        if self.args.model_type.name =='slp': 
            model_state= self.slp_chosen()
            pass 
        elif self.args.model_type.name == 'fnn': 
            model_state= self.fnn_chosen()
            pass 
        elif self.args.model_type.name == 'kmeans_fnn': 
            model_state = self.kmeans_fnn_chosen()
            pass 
        elif self.args.model_type.name == 'autoencoder_kmeans_fnn': 
            model_state = self.autoencoder_kmeans_fnn_chosen()
            pass 
        
        self.logger.info('completed training for: '+self.args.model_type)
        return model_state 
    
    #### -- models are below, calls train() from these functoins 
    
    def autoencoder_kmeans_fnn_chosen(self) -> dict: # model state  
        '''
        this is for the autoencoder + kmeans + fnn model
        try to contain all within 1 model for more modularity in code  
        '''
        # get data 
        data = self.load_data()
        autoencoder_data = {
            'trainx':data['trainx'], 
            'testx':data['testx'], 
            'trainy':data['trainx'], 
            'testy':data['testx']}

        # turn data in dataloader objects 
            # the dataloader objects will be different for the autoencoder model and fnn model 
        train, test = self.make_dataloader(data) # target as the risk group or score 
        train_autoencoder, test_autoencoder = self.make_dataloader(autoencoder_data) # target as the input itself

        # initalize model 
        input_size = next(iter(train))[0].shape[1] 
        hidden_sizes = self.args.hidden_sizes 
        output_size = 3 if self.output_mode == 'classification' else 1  # depends on output type 
        model = AutoencoderKmeansFNN(self.args.k_value, input_size, hidden_sizes, output_size)
        self.logger.info('autoencoder means fnn model is made: '+ str(model))


        ### autoencoder section (get the sub model though the main model)
        # initalize the model 
        autoencoder_model = model.autoencoder
        # no need to train on second pass 
        if not self.args.load_model: # only if not loading model 
            self.logger.debug('train autoencoder')
            # pass train dataloader to train loop to train the model 
                # the target is the same as the input for autoencoder
                # loss func should be mse 
            autoencoder_model_state = self.train_class.train(data= (train_autoencoder, test_autoencoder), model = autoencoder_model )
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
        # save model 
        Utils.save_kmeans_model(location= self.args.kmeans_model_filename, model_state=kmeans_model) 
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
        output_size = 3 if self.output_mode == 'classification' else 1  # depends on output type 
         
        # initialize model 
        model = KMeansFNN(self.args.k_value, input_size, hidden_sizes, output_size)
        self.logger.info('kmeans fnn model is made: '+ str(model))

        # use data to train the kmeans model 
        model.train_kmeans(data['trainx']) # train the kmeans model 
        # save model 
        Utils.save_kmeans_model(location= self.args.kmeans_model_filename, model_state=model.kmeans) 
        self.logger.info('kmeans model is trained')

        # call training loop  
        model_state = self.train_class.train((train, test), model) # dataloaders passed in with model which already has trained kmeans 
        self.logger.debug('fnn is now trained')

        return model_state

      

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
        output_size = 3 if self.output_mode == 'classification' else 1  # depends on output type 
        
        self.logger.info('batched x,y data size: '+ str([data.shape for data in next(iter(train))]))
        self.logger.info('model output shape: '+ str(output_size))


        model = SLP(input_size, output_size)
        self.logger.info('slp model is made: '+ str(model))

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
        output_size = 3 if self.output_mode == 'classification' else 1  # depends on output type 
        

        # define model 
        model = FNN(input_size, hidden_sizes, output_size)
        self.logger.info('fnn model is made: '+ str(model))

        # pass to training loop 
        model_state = self.train_class.train(data = (train, test), model = model )

        return model_state 




