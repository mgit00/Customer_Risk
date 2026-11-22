

'''
file to be called by the DockerFile
spins up the api- this is main driver code now 

connected to src/api.py 
'''

# matplotlib agg called by fastapi code 

import uvicorn 
# from api_framework.fastapi_setup import app # runs the script first before func below 

if __name__ == '__main__': 
    # working directory is app, due to containerization 

    # for local host, default host 
    uvicorn.run("src.api:app", port=8000,  log_config = None,  reload=False) # current name # logging from uvicorn 

    # for docker image 
    # uvicorn.run("src.api:app",host="0.0.0.0", port=8000,  log_config = None,  reload=False) # current name # logging from uvicorn 
        # uvicorn.run(__name__+":app", log_config = None,  reload=False) # current name # logging from uvicorn 
    
    

# CALL with python -m src.main