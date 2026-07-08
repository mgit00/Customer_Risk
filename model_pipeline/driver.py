import argparse 

parser= argparse.ArgumentParser()

parser.add_argument('--application_file', default= 'dataset/application_record_processed.csv')
parser.add_argument('--kmeans_evaluation', action='store_true')
parser.add_argument('--k_values', nargs= 2, default= (1, 5),  type= int )
parser.add_argument('--predicting', default = False , type = str)
parser.add_argument('--lr', default= .001, type= float, required=False)

args= parser.parse_args()

print(args)


from modelhelper import ModelHelper
from clustering_model import ClusteringModel 

mh= ModelHelper(args) # initialize 


if args.kmeans_evaluation: 
  data= mh.get_data()
  # apply to clustering algo



  raise Exception('choose k and continue to train')

if not args.predicting: # training calls  
  data_processed= mh.get_data(args.application_file)
  # training= mh.training(data_processed)



else: # predicting calls 
  pass 
