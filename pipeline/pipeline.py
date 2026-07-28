import argparse 
 
parser= argparse.ArgumentParser(description='model creation and pipeline')

'''
can run through full pipeline, calls all functions 

if you call steps like data_processing, then only that function will be executed 
'''
parser.add_argument('--mode', type= str, 
                    choices=['predict', 'build_model', 'data_processing', 'full'], 
                    required=True, 
                    help= 'pick the entire pipeline or section of the pipeline to do'  )


filenames= parser.add_argument_group(title= 'filenames and paths for datasets' )
# original data 
filenames.add_argument('--application_record', default='dataset/application_record.csv', help='original data ')
filenames.add_argument('--credit_record', default='dataset/credit_record.csv', help='original data ')
# processed data 
filenames.add_argument('--customer_risk', default='dataset/customer_risk.csv', help= 'processed data ')
filenames.add_argument('--applications_processed', default='dataset/applications_processed.csv', help='processed data ')
filenames.add_argument('--application_record_pipeline', default= 'pipeline/application_record_pipeline.pkl', help= 'application pipeline for processing')


args= parser.parse_args()

print(args)


from data_processing import DataProcessing 
from utils import Utils 


class Pipeline: 
    '''
    interface which interacts with the directory 
    '''
    def __init__(self, args): 
        self.args= args 

        # full pipeline 
        if args.mode == 'full': 
            self.process_data()
            self.building_model()
            # .... 


        # sections of the pipeline
        if args.mode == 'data_processing': 
            self.process_data()
        
        if args.mode == 'build_model': 
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
        plots will be saved here too 
        '''
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
        






        




pipeline_class= Pipeline(args)



