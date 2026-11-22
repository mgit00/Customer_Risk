
# # to prevent crashes for fastapi 
# import matplotlib 
# matplotlib.use('Agg')
# # ---- 

# import uvicorn 
# from pipeline.main_driver import MainDriver 
# from pipeline.driver import Driver
# from pipeline.utils import Utils 

# from utils.schemas import config_settings 
# import logging 


# from api_framework.fastapi_setup import app # runs the script first before func below 


# # FIXME delete 
# deprecate 

# if __name__ == '__main__': 
#     # get the api set up <- one in fastapi_setup import 
#         # connects to argument_items class to get list for argparse 
#     # achieved with import app 

#     '''
#     initialize logger
#     allow consol for logger 

#     this should be done in main file, otherwise duplicate streamhandlers do duplicate printing 
#     '''
#     logger= logging.getLogger('main_logger') # keep logger name instead of __name__, this won't be a problem when imported - where __main__ becomes main
#     # level_attr = getattr(logging, 'debug'.upper()) # similar tp logging.DEBUG
#     level_attr = getattr(logging, config_settings.logger_level.upper())
#     logging.basicConfig(filename ='model_info/logger.log', filemode='w',  level = level_attr)
        
#     # needed for printing 
#     console = logging.StreamHandler()
#     console.setLevel(level_attr) # logging.DEBUG
#     logger.addHandler(console)

#     logger.debug('finshed logger initialization')
    
#     # unzip the data files 
#     Utils.check_loaded_dataset(logger)

#     # for docker image 
#     uvicorn.run(__name__+":app",host="0.0.0.0", port=8000,  log_config = None,  reload=False) # current name # logging from uvicorn 
#     # uvicorn.run(__name__+":app", log_config = None,  reload=False) # current name # logging from uvicorn 
    
#     logger.debug('finished api run')