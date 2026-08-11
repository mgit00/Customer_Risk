from fastapi import FastAPI 
from fastapi.responses import FileResponse

from .argument_items import ArgumentItems
from pipeline.driver import Driver # same direcotry level, no need
from pipeline.main_driver import MainDriver 

import logging 

'''
fastapi set up 

note: i can extract data out of state without request 
but, writing or mutating app.state directly requires access to request.app.state
doing this when there are multiple workers means that only 1 of the workers will be updated, which can be a problem
'''


app = FastAPI(title = 'Customer Risk API') # import this in main file 

@app.get('/')
def starting(): 
    return 'base api is set up and accessible'

@app.put('/model/{model_verson}')
def model_pipeline(argument_items: ArgumentItems, model_version: int = 1 ): # argument_items inferred or new list[str] accepted
    print('model version', model_version)
    # argument_items.args = list of args to be passed to argparse 
    args_list  =argument_items.args # this should be passed to args 

    # pass to argument parser 
    pipeline_driver = MainDriver(args_list)# initalized argparse and runs driver 
        # sets off pipeline 

    # at this point, the image for the args is guaranteed to be in the directory 
        # can be accessed by pipeline_driver.args.plot_filepath
    return FileResponse(pipeline_driver.args.others['plot_files'][0], media_type= 'image/png')



