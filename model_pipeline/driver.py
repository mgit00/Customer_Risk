import argparse 

parser= argparse.ArgumentParser()

parser.add_argument('--application_file', default= 'dataset/application_record_processed.csv')
parser.add_argument('--predicting', default = False , type = str)
parser.add_argument('--lr', default= .001, type= float, required=False)

args= parser.parse_args()

print(args)


from modelhelper import ModelHelper

mh= ModelHelper(args) # initialize 

if not args.predicting: # training calls  
  data_processed= mh.get_data(args.application_file)
  # training= mh.training(data_processed)



else: # predicting calls 
  pass 
