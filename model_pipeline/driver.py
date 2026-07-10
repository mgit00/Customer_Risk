import argparse 

parser= argparse.ArgumentParser()
'''
this is the parser 
there are 2 ways to use it 
  give in a list (from file) with all the arguments 
  or use the command line 
'''

parser.add_argument('--command_file', action= 'store_true') # True if called, else False 



# datafile 
parser.add_argument('--application_file', default= 'dataset/application_record_processed.csv')
parser.add_argument('--risk_file', default= 'dataset/credit_risk.csv')
parser.add_argument('--application_risk_file', default= 'dataset/customer_application_risk.csv')
# for creating elbow curve for k means algo 
  # example: --kmeans_evaluation --k_values 1 30
parser.add_argument('--kmeans_evaluation', action='store_true')
parser.add_argument('--k_values', nargs= 2, default= (1, 5),  type= int )

# for model training 
parser.add_argument('--chosen_k_value', default=10, type= int  )
parser.add_argument('--epochs', default= 3, type = int )
parser.add_argument('--batch_size', default = 5, type = int )
parser.add_argument('--hidden_sizes', nargs= '+', type= int , default = [16, 32, 16])
parser.add_argument('--lr', default= .001, type= float, required=False)

# model saving and loading 
  # save model automatically if the filename is given 
  # load model if filename is given 
parser.add_argument('--save_model_kmeans', const= 'saved_models/kmeans.pkl', nargs= '?', type = str )
parser.add_argument('--load_model_kmeans', const= 'saved_models/kmeans.pkl', nargs= '?', type = str)
parser.add_argument('--save_modek_knn', const = 'saved_models/kmeans.pkl', nargs='?')
parser.add_argument('--load_modek_knn', const = 'saved_models/kmeans.pkl', nargs='?')


# for predicting 
parser.add_argument('--predicting', default = False , type = str)



args= parser.parse_args()
# if command_line is given, then all other args get overridden by the file args 
from args_commands import commands 


# commands= []
if args.command_file: 
  args= parser.parse_args(commands)
else: 
  pass # keep original args from command line 


print(args)


from modelhelper import ModelHelper
from clustering_model import ClusteringModel 

mh= ModelHelper(args) # initialize 



# if looking to choose k, this code is run
# this will stop and allow looking at the graph to choose an appropriate k 
# once k is chosen, continue onto training
if args.kmeans_evaluation: 
  data= mh.get_data(args.application_file)
  # apply to clustering algo
  ClusteringModel(args, data).driver()
  raise Exception('Choose a k and move to train.')



# training and predicting 
if not args.predicting: # training calls  
  train= mh.training()



else: # predicting calls 
  pass 







# import argparse 

# parser= argparse.ArgumentParser()

# # datafile 
# parser.add_argument('--application_file', default= 'dataset/application_record_processed.csv')
# parser.add_argument('--risk_file', default= 'dataset/credit_risk.csv')
# parser.add_argument('--application_risk_file', default= 'dataset/customer_application_risk.csv')
# # for creating elbow curve for k means algo 
#   # example: --kmeans_evaluation --k_values 1 30
# parser.add_argument('--kmeans_evaluation', action='store_true')
# parser.add_argument('--k_values', nargs= 2, default= (1, 5),  type= int )

# # for model training 
# parser.add_argument('--chosen_k_value', default=10, type= int  )
# parser.add_argument('--epochs', default= 3, type = int )
# parser.add_argument('--batch_size', default = 5, type = int )
# parser.add_argument('--hidden_sizes', nargs= '+', type= int , default = [16, 32, 16])
# parser.add_argument('--lr', default= .001, type= float, required=False)

# # model saving and loading 
#   # save model automatically if the filename is given 
#   # load model if filename is given 
# parser.add_argument('--save_model_kmeans', const= 'saved_models/kmeans.pkl', nargs= '?', type = str )

# # for predicting 
# parser.add_argument('--predicting', default = False , type = str)

# args= parser.parse_args()

# print(args)

# raise Exception('here')


# from modelhelper import ModelHelper
# from clustering_model import ClusteringModel 

# mh= ModelHelper(args) # initialize 



# # if looking to choose k, this code is run
# # this will stop and allow looking at the graph to choose an appropriate k 
# # once k is chosen, continue onto training
# if args.kmeans_evaluation: 
#   data= mh.get_data(args.application_file)
#   # apply to clustering algo
#   ClusteringModel(args, data).driver()
#   raise Exception('Choose a k and move to train.')



# # training and predicting 
# if not args.predicting: # training calls  
#   train= mh.training()



# else: # predicting calls 
#   pass 
