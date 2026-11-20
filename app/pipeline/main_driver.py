'''
this file will remain in pipeline for use
'''

import argparse 
from .driver import Driver 

import logging 

from .utils import Utils

'''
this parser will have subgrouping associated with the file they are used in 
this makes the parsing easier with the redone pipeline 

create full pipeline
when testing, use helper functions to debug

note: argparse automatically picks up on sys.argv (global from the file that is run)
'''

# FIXME to be deprecated 
class MainDriver: 
    '''
    initializing automatically starts running the pipeline 
    does: argparse, Driver set up 
    '''
    def __init__(self, args_list): 
        # ---- 
        self.args = self.init_parser(args_list) # final set of args 
        self.init_driver(self.args)
    
    def init_parser(self, args_list): 
        parser= argparse.ArgumentParser(description='model creation and pipeline', 
                                        fromfile_prefix_chars='@')
        parser.add_argument('--logger_level', choices=['info', 'debug'], default = 'debug', help= 'logger level setting')
        parser.add_argument('--automatic_file_naming', default= True, choices=[True, False], help= 'automatic file naming, which doesnt have to be specified in commandline' )

        mode = parser.add_mutually_exclusive_group()
        # store_true means that mode is T/F indicator in an exclusive group 
        mode.add_argument('--run_pipeline',  action= 'store_true', help='this runs the full pipeline')
        mode.add_argument('--build_code',  action='store_true', help= 'this calls the helper functions used to debug code')


        data_processing =  parser.add_argument_group(title='args for data_processing.py')
            # original data 
        data_processing.add_argument('--application_record', default='dataset/application_record.csv', help='original data ')
        data_processing.add_argument('--credit_record', default='dataset/credit_record.csv', help='original data ')
            # saving filenames 
        data_processing.add_argument('--data_pipeline', default = 'model_info/data_processing/application_record_pipeline.pkl', help='location to save the application_record sklearn pipeline')
        data_processing.add_argument('--data_filename', default = 'model_info/data_processing/data.csv', help='location to save the application_record sklearn pipeline')
            # classification or regression 
        data_processing.add_argument('--output_mode', default = 'classification' , choices=['classification', 'regression'], dest='output_mode')

        modelhelper = parser.add_argument_group(title='args for modelhelper.py')
        # read data from data_filename 
        modelhelper.add_argument('--batch_size', default = 1000, type = int, help = 'batch size for dataloader- smaller batches mean less memory, but noisy ')
        modelhelper.add_argument('--model_type', choices =['slp', 'fnn', 'kmeans+fnn', 'autoencoder+kmeans+fnn'], default = 'slp', help = 'model type to choose from')
        modelhelper.add_argument('--hidden_sizes', nargs= '+', type= int, default= [30, 20], help= 'list of hidden sizes for the hidden layers')
        modelhelper.add_argument('--model_filename', default = 'model_info/saved_models/model.pt', help= 'location of saved model')
        modelhelper.add_argument('--kmeans_model_filename', default = 'model_info/saved_models/clustering/kmeans_model.pkl')
        modelhelper.add_argument('--autoencoder_kmeans_model_filename', default = 'model_info/saved_models/clustering/autoencoder_kmeans_model.pkl')


        train= parser.add_argument_group(title = 'args for train.py training loop ')
        train.add_argument('--lr', default = .0001, type = float)
        train.add_argument('--epochs', default = 3, type= int )
        train.add_argument('--load_model', action= 'store_true', help= 'call if wanting to load in model_filename, continue training, and save model to model_filename')

        models = parser.add_argument_group(title = 'args for models.py')
        models.add_argument('--k_value', default = 11, type = int, help = 'int > 0, number of clusters for kmeans models')

        plotting= parser.add_argument_group(title='args for plotting')
        plotting.add_argument('--plot_filepath', default = 'images/nn_plots/', help = 'base location')
        plotting.add_argument('--elbow_curve_for_kmeans', default = 'images/kmeans_plots/elbow_curve_for_kmeans.png', help = 'full location ')

        clustering = parser.add_argument_group(title='args for clustering')
        clustering.add_argument('--k_values' , nargs= 2, default = [2, 30], type= int, help= 'the starting and ending k values to plot the elbow curve for. min of 1 cluster' )

        args= parser.parse_args(args= args_list) # from the API 


        # automatic file naming 
        if args.automatic_file_naming: 
            # args.data_pipeline stays the same 
            args.data_filename = 'model_info/data_processing/data' + '_'+ args.output_mode + '.csv'
            args.model_filename= 'model_info/saved_models/model' + '_' + args.output_mode + '_' + args.model_type + '.pt'
            args.plot_filepath = 'images/nn_plots/' + args.model_type + '/' + args.output_mode+ '_'  # the last filename is ignored, so add this for correct path 

        # if filenames/paths don't exist, make them
        Utils.make_path([args.data_pipeline,
                        args.data_filename, 
                        args.model_filename, 
                        args.plot_filepath, 
                        args.elbow_curve_for_kmeans, 
                        args.kmeans_model_filename, 
                        args.autoencoder_kmeans_model_filename])


        for arg in args.__dict__.items(): 
            print(arg)


        # constant args which cannot be changed in the command line 
        args.logger_name = 'main_logger'

        # additional args to be collected as the pipeline is run 
        args.others = {'plot_files': []}

        return args 



    def init_driver(self, args): 
        '''
        call driver.py to begin pipeline 
        '''
        driver_class = Driver(args)

        if args.run_pipeline: 
            driver_class.begin_pipeline()
        elif args.build_code: 
            driver_class.build_code()






