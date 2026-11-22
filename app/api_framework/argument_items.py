
# FIXME delete 

# from pydantic import BaseModel, Field

# from pathlib import Path 

# '''
# called by fastapi_setup for the model training 
# '''

# # DONE FIXME deprecated 

# def get_default_args() : # in function, don't want to run it automatically when importing 
#         parentpath = Path(__file__).resolve().parent # -> api_framework folder 
#         argspath = parentpath.parent / 'args.txt' # navigate up and get args.txt 

#         with open(argspath, 'r') as file: 
#             # read args.txt- usually run by python argument_parser @args.txt
#             args_list_default = file.read().splitlines() # list of strings

#         return args_list_default


# class ArgumentItems(BaseModel): 
#     '''
#     usually used for data validation when interacting with api 
#     passes info from pipeline/argument_parser.py
#     ''' 

#     # read in args from args.txt as the default
#     args: list[str] = Field(default = get_default_args(),
#                 description = 'the arguments to be passed to argument_parser.py')



