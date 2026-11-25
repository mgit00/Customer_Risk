

'''
to be used for fastapi 

connected to src/main.py 
''' 

# NOTE important call below 
# to prevent crashes for fastapi 
import matplotlib 
matplotlib.use('Agg') # switches it to a non-interactive image-only backend so it generates plots in memory
# this doesn't do pop up windows 

from fastapi import FastAPI, Request , HTTPException
from fastapi.responses import FileResponse

from contextlib import asynccontextmanager

from src.main_driver import MainDriver
from utils.schemas import CustomerAttributes, KMeansExplorationBasemodel, ModelTrainingBasemodel
from utils.constants import ConstantFilenames




@asynccontextmanager
async def project_lifespan(app:FastAPI): 
    print('lifespan starting')
    # get through request.app.state.class calls 
    app.state.main_driver = MainDriver()

    # start logging, unzip files, make file paths 
    app.state.main_driver.start_app()

    # spin up api 
    yield 

    # if needed, close driver 

    print('lifespan closing ')




app = FastAPI(title = 'Customer Risk API', 
              lifespan= project_lifespan) # import this in main file 


# calls 
'''
calcualte customer risk - get 
process data - post 
explore k values - post 
train model - put 
add to data # TODO - put 

get = retrieve, no request body 
post = create new or trigger action 
put = replace existing item 
'''

@app.get('/')
def starting(): 
    return 'base api is set up and accessible'

# for prediction
@app.put('/customer_risk/') # {customer} inferred 
def prediction_pipeline(request: Request, customer: CustomerAttributes): 
    '''
    this is called for the prediction step of the fastapi 
    the prediction will be done for the customer id specified 
    it will return the customer risk to the bank
    then, the bank can either approve or deny the credit card application using that information 
    '''
    # note logger will initialized in main.py, use that to update during the prediction process 

    success, value = request.app.state.main_driver.calculate_customer_risk(customer)
    if not success: 
        raise HTTPException(status_code= 404, 
                            detail = value )
 
    return value 

@app.post('/data_management/process_data/')
def processing_data(request: Request): 
    '''
    this is called for putting the data through the processing pipeline 
    '''
    request.app.state.main_driver.process_data()

    return 'current data is processed'

@app.post('/data_management/add_data/')
def adding_data(): 
    '''
    # TODO add to pd df 
    '''

    # print('added data, data is now in shape: {}')
    pass 

@app.post('/models/explore_kvalues/') # kmeans_args inferred 
def exploring_kvalues(request: Request, kmeans_args: KMeansExplorationBasemodel): 
    '''
    view kmeans algorithm at different k values 
    '''
    request.app.state.main_driver.explore_kvalues(kmeans_args)

    # at this point, the image for the args is guaranteed to be in the directory 
        # this is in constant filenames 
    image= ConstantFilenames.elbow_curve_for_kmeans
    return FileResponse(image, media_type= 'image/png')


@app.post('/models/train_model/') # args inferred 
def training_model(request: Request, training_args: ModelTrainingBasemodel): 
    '''
    train the model 
    '''
    # run the training pipeline 
    request.app.state.main_driver.train_model(training_args)

    # return the loss image 
    return FileResponse(training_args.plot_filepath + 'losses.png', media_type = 'image/png')





