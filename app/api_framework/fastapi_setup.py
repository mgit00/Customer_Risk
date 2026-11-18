from fastapi import FastAPI , HTTPException
from fastapi.responses import FileResponse, PlainTextResponse

from .argument_items import ArgumentItems
from pipeline.driver import Driver # same direcotry level, no need
from pipeline.main_driver import MainDriver 

# for prediction 
from pipeline_predict.driver import DriverPrediction
from pipeline_predict.customer_attributes import CustomerAttributes 

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



# for prediction
@app.put('/customer_risk/') # {customer} inferred 
def prediction_pipeline(customer: CustomerAttributes): 
    '''
    this is called for the prediction step of the fastapi 
    the prediction will be done for the customer id specified 
    it will return the customer risk to the bank
    then, the bank can either approve or deny the credit card application using that information 
    '''
    # note logger will initialized in main.py, use that to update during the prediction process 

    driver= DriverPrediction(customer)

    if not driver.check_requirements(): 
        raise HTTPException(status_code=404, detail="not all models are made")
    
    if not driver.check_customer(): # check if customer exists in df 
        raise HTTPException(status_code=404, detail="customer id not found in dataset")

    # pass onto rest of pipeline 
    pred = driver.get_prediction()
        # sets off pipeline 
 
    return 'The customer risk level for cutomer ' + str(customer.id_number) + ' is: ' + pred 


@app.post('/score_models', response_class=PlainTextResponse)
def scoring_models_evaluation(): 
    driver= DriverPrediction()
    
    if not driver.check_requirements(): 
        raise HTTPException(status_code=404, detail="not all models are made")

    if not driver.check_customer(): # check if customer exists in df 
        raise HTTPException(status_code=404, detail="customer id not found in dataset")

    # pass onto rest of pipeline 
    pred = driver.evaluate_model()

    return pred 