from  fastapi import FastAPI 
from pydantic import BaseModel 

'''
this file is the main file to be called 
if the command line args are passed in, then it will run the pipeline
also, if the args are passed in as a string to FastAPI, it will also run the pipeline

this file allows for dual usage
'''

class Arguments(BaseModel): 
    args: list[str]  # takes in list of strings from the put request


'''
fastapi set up 
'''
app = FastAPI(title = 'Customer Risk API')

@app.get('/')
def starting(): 
    return 'base api is set up and accessible'


with open("../args.txt" 'r') as file: 
    # read args.txt- usually run by python argument_parser @args.txt
    lines = file.read().splitlines() # list of strings
default_args= Arguments(**{'args': lines}) # for unpacking dictionary


@app.get('/run_pipeline/{args_list}')
def running(args_list: Arguments = default_args): 
    # run pipleine 
    # sets 
    return # return image of trained model 


if __name__ == '__main__': 
    # can either call the argparse through the command line or through the api 

    if 

'''
the pipleine should be ideally: 
    get the args from the api call 
    pass into argparser 
    continue the pipeline 

1 way is: 
make basemodel 
get the info from basemodel and pass it into argparser 
    can make basemodel complicated or easy, either way it gets passed into basemodel 
this is a bit messy though 

can branch off git and make a better solution where fastapi interacts with pipeline_starter.py
'''