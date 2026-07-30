
from data_processing import DataProcessing 
from utils import Utils 

from model_helper import ModelHelper
from clustering import Clustering
from plotting import Plotting

class Driver: 
    '''
    interface which interacts with the directory 
    this keeps interactions with directory easy to follow
    '''
    def __init__(self, args): 
        self.args= args 

    def do_mode(self): 

        # full pipeline 
        if self.args.mode == 'full': 
            self.process_data()
            self.building_model()
            # .... 
        
        elif self.args.mode == 'kmeans_only': 
            # self.process_data()
            self.kmeans()

        # sections of the pipeline
        elif self.args.mode == 'data_processing': 
            self.process_data()
        
        elif self.args.mode == 'build_model': 
            self.building_model()
    
    def process_data(self): 
        '''
        this will process application_record.csv and credit_record.csv
        and it will save it to the correct files 

        -> should save to dataset/customer_risk.csv == credit risk of each customer 
        -> dataset/applications_processed.csv == processed applications of credit cards 
        -> dataset/JOINED???? .... 
      
        # original data 
        filenames.add_argument('--application_record', default='dataset/application_record.csv', help='original data ')
        filenames.add_argument('--credit_record', default='dataset/credit_record.csv', help='original data ')
        # processed data 
        filenames.add_argument('--customer_risk', default='dataset/customer_risk.csv', help= 'processed data ')
        filenames.add_argument('--applications_processed', default='dataset/applications_processed.csv', help='processed data ')

        '''
        dp= DataProcessing(self.args) # initalize 
        # cusomter risk 
        credit_record = Utils.read_data(self.args.credit_record)
        customer_risk = dp.customer_risk_processing(credit_record)
        Utils.save_df(customer_risk, self.args.customer_risk)
        print('customer credit risk processed from: ', self.args.credit_record, ' and saved to: ', self.args.customer_risk)

        # applications processed 
        application_record = Utils.read_data(self.args.application_record)
        applications_processed = dp.cc_applications_processing(application_record)
        Utils.save_df(applications_processed, self.args.applications_processed)
        print('application record processed from: ', self.args.application_record, ' and saved to: ', self.args.applications_processed)


        # save the pipeline for the predict step and for ease of use 
        apps_pipeline= dp.cc_applications_pipeline
        Utils.save_pipeline(pipeline=apps_pipeline, location= self.args.application_record_pipeline)
        print('application_record pipeline saved to: ', self.args.application_record_pipeline)

        # merge the customer_risk and applications data so that it can be fed to the model 
        data= dp.join_data(customer_risk, applications_processed)
        Utils.save_df(data, self.args.data)
        print('complete joined data saved to: ', self.args.data)


    def kmeans(self): 
        '''
        kmeans_only was called
        the processed data is guanteed to exist 

        if kmeans_only, then the mode doesn't matter at all 
            there should be processed data in the directory 
            just produce the kmeans clustering elbow graph 
        '''
        data = Utils.read_data(self.args.data)

        # do kmeans clustering models for different k 
        clustering= Clustering(self.args)
        intertias= clustering.get_intertias(data)

        # plot the results 
        plotting= Plotting(self.args)
        plotting.kmeans_elbow_curve(intertias)



    def building_model(self) : 
        '''
        here will be different types of models which will be built: 
        - regular fnn 
        - k means (clusters) + fnn 
        - autoencoder + k means (clusters) + fnn 

        the building pipeline for each model will be as follows: 
        - driver (checks right hyperparamters with lodaded model, initalizes model, calls training loop) 
        - model (contians the 3 models that you can call )
        - training (contains the training loop)
        - plotting 
        
        loading and saving model will be done here in pipeline (keeps interactions with directory easy to follow)
        plots will be saved here too, since this is used to interact with the directory 
        '''

        modelhelper= ModelHelper(self.args )

        # read data from directory based on args 
        data = Utils.read_data(self.args.data)
        train_test_dataloaders= modelhelper.load_data(data)

        # check if there is an existing model which should be trained upon .... 

        # train model 
        model_state= modelhelper.training(train_test_dataloaders)
        
        # save model 
        # plot, and save plot 


        # layout how the code structure for the model building should be in .md file ... 
        pass 

    def predicting(self): 
        '''
        passes data through processing step 
        calls predict.py 
        takes data and passes it through all 3 saved/specified models 
        - returns predictions (set of 3)
        '''
        # data processing 
        # predict 

        pass 
        





