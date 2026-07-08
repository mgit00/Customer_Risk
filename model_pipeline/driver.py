import argparse 

parser= argparse.ArgumentParser()

# datafile 
parser.add_argument('--application_file', default= 'dataset/application_record_processed.csv')

# for creating elbow curve for k means algo 
  # example: --kmeans_evaluation --k_values 1 30
parser.add_argument('--kmeans_evaluation', action='store_true')
parser.add_argument('--k_values', nargs= 2, default= (1, 5),  type= int )

# for model training 
parser.add_argument('--chosen_k_value', default=15, type= int  )
parser.add_argument('--lr', default= .001, type= float, required=False)

# for predicting 
parser.add_argument('--predicting', default = False , type = str)

args= parser.parse_args()

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
  data_processed= mh.get_data(args.application_file)
  # training= mh.training(data_processed)



else: # predicting calls 
  pass 
