from data_processing import DataProcessing
from model_helper import ModelHelper

import logging 

class Driver: 
    '''
    this class interacts with the argument parser to begin the pipeline
    '''

    def __init__(self, args): 
        self.args = args

        self.logger =logging.getLogger(self.args.logger_name)

        # initialize classes as needed 
        self.dataprocessingclass = DataProcessing(self.args)
        self.modelhelper = ModelHelper(self.args)
        



    def begin_pipeline(self): 
        '''
        this does: 
        data processing 
        kmeans k choosing <--- TBD 
        model training 
        prediction 
        '''
        # this starts off the pipeline 
        self.logger.debug('begin pipeline')

        # first, do data processing 
            # process the application record (x)
            # process the credit record (y)
            # get the joined data (x and y together)
            # save the joined data (split as needed further down the pipeline )
        self.dataprocessingclass.complete_data_processing()


        # then, do model building 
            # this is where the branching off has to be done 
            # build modular code so that i can reuse sections as needed 
        self.modelhelper.model_chooser()

        self.logger.debug('pipeline for data and model finished')

    def build_code(self): 
        self.logger.debug('work on section - NOT full pipeline: modelhelper')

        self.modelhelper.model_chooser()

        pass 