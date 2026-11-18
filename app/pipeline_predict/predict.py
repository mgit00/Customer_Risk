import torch 

from pipeline.utils import Utils
from pipeline.models import SLP, FNN, KMeansFNN, AutoencoderKmeansFNN

from sklearn.metrics import (
    f1_score,
)

from torch.optim import Adam 

class Predict: 
    '''
    this predicts the customer risk for a given customer id 
    '''
    def __init__(self, logger): 
        self.logger = logger

        self.level_options= {0: 'no risk', 
                 1: 'low', 
                 2: 'medium_low', 
                 3: 'medium_high', 
                 4: 'high'} # ('none', 'low', 'medium_low', 'medium_high', 'high')
        

    def process_data(self, data) -> torch.Tensor : 
        ''' transform through data processing pipleine '''
        pipeline = Utils.load_pipeline('./model_info/data_processing/application_record_pipeline.pkl')
        data = pipeline.transform(data)
        data.drop(columns = ['ID'], axis = 1 , inplace= True )
        # data is now ready for passing into the models 
        self.logger.info('data is now put through the previously created data processing pipeline')

        x = torch.tensor(data.values, dtype = torch.float32)
        self.logger.debug('data now transformed to tensor')

        return x 
    
    def predict_step(self, data: torch.Tensor) -> str: 
        '''
        load model 
        pass data through model 

        do this for each of the 4 classification models
        '''
    
        self.logger.debug('loading model ')
        self.logger.debug('predict from model')

        models ={'slp': './model_info/saved_models/model_classification_slp.pt', 
                 'fnn': './model_info/saved_models/model_classification_fnn.pt', 
                 'kmeans_fnn': './model_info/saved_models/model_classification_kmeans+fnn.pt',  
                 'autoencoder_kmeans_fnn': './model_info/saved_models/model_classification_autoencoder+kmeans+fnn.pt'
                 }
        

        results = ''

        for name, loc in models.items(): 
            if name == 'slp': 
                level = self.slp_model(data, loc)
            elif name == 'fnn': 
                level = self.fnn_model(data, loc)
            elif name == 'kmeans_fnn': 
                level = self.kmeans_fnn_model(data, loc)
            else: 
                level = self.autoencoder_kmeans_fnn_model(data, loc)


            results =  results + name + ' states ' + level.upper() + ', '
        
        self.logger.info('the results are: ' + results)
        return results 


    ### models --- 
    def slp_model(self,data, loc, input_size= 35, output_size=5) -> str: 
        # load model 
        loaded =Utils.load_model(location= loc)

        model = SLP(input_size, output_size) 
        # note no need for optimizer, only for training to calc gradients and update weights

        # return self.do_predict(data, model, loaded)
        model.load_state_dict(loaded['model_state_dict'])
        self.logger.debug(' slp model loaded in ')

        model.eval()
        self.logger.debug('prediction done')

        pred =model(data) # forward pass 

        level = torch.argmax(pred).item()
        return self.level_options[level]


    def fnn_model(self,data, loc, input_size= 35, output_size=5)-> str: 
        # load model 
        loaded =Utils.load_model(location= loc)
        hidden_sizes= loaded['args'].hidden_sizes 

        model = FNN(input_size, hidden_sizes, output_size) 
        # note no need for optimizer, only for training to calc gradients and update weights
        
        # return self.do_predict(data, model, loaded)
        model.load_state_dict(loaded['model_state_dict'])
        self.logger.debug(' fnn model loaded in ')

        model.eval()
        self.logger.debug('prediction done')

        pred =model(data) # forward pass 

        level = torch.argmax(pred).item()
        return self.level_options[level]
    
    
    def kmeans_fnn_model(self,data, loc, input_size= 35, output_size=5)-> str: 
        # load model 
        loaded =Utils.load_model(location= loc)

        hidden_sizes= loaded['args'].hidden_sizes 
        k_value= loaded['args'].k_value
        kmeans_model_filename= loaded['args'].kmeans_model_filename

        model = KMeansFNN(k_value, input_size + 1 , hidden_sizes, output_size) 
        # note no need for optimizer, only for training to calc gradients and update weights
        
        # return self.do_predict(data, model, loaded)
        model.load_state_dict(loaded['model_state_dict'])
        model.kmeans = Utils.load_kmeans_model(kmeans_model_filename)
        self.logger.debug('kmeans fnn model loaded in ')

        model.eval()
        self.logger.debug('prediction done')

        pred =model(data) # forward pass 

        level = torch.argmax(pred).item()
        return self.level_options[level]
    
    
    def autoencoder_kmeans_fnn_model(self,data, loc, input_size= 35, output_size=5)-> str: 
        # load model 
        loaded =Utils.load_model(location= loc)
        hidden_sizes= loaded['args'].hidden_sizes 
        k_value= loaded['args'].k_value
        autoencoder_kmeans_model_filename= loaded['args'].autoencoder_kmeans_model_filename

        model = AutoencoderKmeansFNN(k_value, input_size, hidden_sizes, output_size) 
        # note no need for optimizer, only for training to calc gradients and update weights
        
        # return self.do_predict(data, model, loaded)
        model.load_state_dict(loaded['model_state_dict'])
        model.kmeans = Utils.load_kmeans_model(autoencoder_kmeans_model_filename)
        self.logger.debug('autoencoder kmeans fnn model loaded in ')

        model.eval()
        self.logger.debug('prediction done')

        pred =model(data) # forward pass 

        level = torch.argmax(pred).item()
        return self.level_options[level]





class PredictEvaluation(Predict) : # subclass 
    # subclass, no need to make init 

    def predict_step_evalset(self, data: torch.Tensor) : # returns y pred for each of the models   
        res = {}

        models ={'slp': './model_info/saved_models/model_classification_slp.pt', 
                'fnn': './model_info/saved_models/model_classification_fnn.pt', 
                'kmeans_fnn': './model_info/saved_models/model_classification_kmeans+fnn.pt',  
                'autoencoder_kmeans_fnn': './model_info/saved_models/model_classification_autoencoder+kmeans+fnn.pt'
                }


        ### slp 
        input_size= 35
        output_size=5
        # load model 
        loaded =Utils.load_model(location= models['slp'])

        model = SLP(input_size, output_size) 
        # note no need for optimizer, only for training to calc gradients and update weights

        # return self.do_predict(data, model, loaded)
        model.load_state_dict(loaded['model_state_dict'])
        self.logger.debug(' slp model loaded in ')

        model.eval()
        self.logger.debug('prediction done')

        pred =model(data) # forward pass 

        level = torch.argmax(pred, dim = 1)
        res['slp'] = level 


        ### fnn 
        # load model 
        loaded =Utils.load_model(location= models['fnn'])
        hidden_sizes= loaded['args'].hidden_sizes 

        model = FNN(input_size, hidden_sizes, output_size) 
        # note no need for optimizer, only for training to calc gradients and update weights
        
        # return self.do_predict(data, model, loaded)
        model.load_state_dict(loaded['model_state_dict'])
        self.logger.debug(' fnn model loaded in ')

        model.eval()
        self.logger.debug('prediction done')

        pred =model(data) # forward pass 

        level = torch.argmax(pred, dim = 1)
        res['fnn'] = level 


        ### kmeans_fnn
        # load model 
        loaded =Utils.load_model(location= models['kmeans_fnn'])

        hidden_sizes= loaded['args'].hidden_sizes 
        k_value= loaded['args'].k_value
        kmeans_model_filename= loaded['args'].kmeans_model_filename

        model = KMeansFNN(k_value, input_size + 1 , hidden_sizes, output_size) 
        # note no need for optimizer, only for training to calc gradients and update weights
        
        # return self.do_predict(data, model, loaded)
        model.load_state_dict(loaded['model_state_dict'])
        model.kmeans = Utils.load_kmeans_model(kmeans_model_filename)
        self.logger.debug('kmeans fnn model loaded in ')

        model.eval()
        self.logger.debug('prediction done')

        pred =model(data) # forward pass 

        level = torch.argmax(pred, dim = 1)
        res['kmeans_fnn'] = level 

        ### autoencoder_kmeans_fnn
        # load model 
        loaded =Utils.load_model(location= models['autoencoder_kmeans_fnn'])
        hidden_sizes= loaded['args'].hidden_sizes 
        k_value= loaded['args'].k_value
        autoencoder_kmeans_model_filename= loaded['args'].autoencoder_kmeans_model_filename

        model = AutoencoderKmeansFNN(k_value, input_size, hidden_sizes, output_size) 
        # note no need for optimizer, only for training to calc gradients and update weights
        
        # return self.do_predict(data, model, loaded)
        model.load_state_dict(loaded['model_state_dict'])
        model.kmeans = Utils.load_kmeans_model(autoencoder_kmeans_model_filename)
        self.logger.debug('autoencoder kmeans fnn model loaded in ')

        model.eval()
        self.logger.debug('prediction done')

        pred =model(data) # forward pass 

        level = torch.argmax(pred, dim = 1)
        res['autoencoder_kmeans_fnn'] = level 

        # res now holds y_pred results 
        return res 
    

    def score_predictions(self, y_true, predictions : dict)-> str : # returns string of scores 
        y_true = torch.tensor(y_true.to_numpy())

        scores = []

        for model, y_pred in predictions.items(): 

            print(y_true.shape, y_pred.shape)
            # get scores for each model
            score = f"Model {model} scores are: \n Macro F1:{f1_score(y_true, y_pred, average='macro'):.4f} \n Weighted F1:{f1_score(y_true, y_pred, average='weighted'):.4f}"

            scores.append(score)

        return 'scores are: ' + '\n\n\n'.join(scores)

        
