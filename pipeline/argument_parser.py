import argparse 
 
parser= argparse.ArgumentParser(description='model creation and pipeline')

'''
can run through full pipeline, calls all functions 


if you call steps like data_processing, then only that function will be executed 
'''
parser.add_argument('--mode', type= str, 
                    choices=['predict', 'build_model', 'data_processing', 'full'], 
                    required=True, 
                    help= 'pick the entire pipeline or section of the pipeline to do'  )


filenames= parser.add_argument_group(title= 'filenames and paths for datasets' )
# original data 
filenames.add_argument('--application_record', default='dataset/application_record.csv', help='original data ')
filenames.add_argument('--credit_record', default='dataset/credit_record.csv', help='original data ')
# processed data 
filenames.add_argument('--customer_risk', default='dataset/customer_risk.csv', help= 'processed data ')
filenames.add_argument('--applications_processed', default='dataset/applications_processed.csv', help='processed data ')
filenames.add_argument('--application_record_pipeline', default= 'pipeline/application_record_pipeline.pkl', help= 'application pipeline for processing')
filenames.add_argument('--data', default='dataset/data.csv', help='the customer_risk and applications_processed joined dataset')

model= parser.add_argument_group(title= 'args for model')
model.add_argument('--model_type', default= ['fnn'], 
                   choices=['slp', 'fnn', 'kmeans_fnn', 'autoencoder_kmeans_knn'],
                   required=False , help= 'pick the model to work on')
# data to read in is: --data 
model.add_argument('--batch_size', default = 1000, type = int, help='smaller batches mean less memory, but noisy')
# --device arg created in train.training_loop 
model.add_argument('--lr', default = .0001, type = float)
model.add_argument('--epochs', default = 5, type= int )
model.add_argument('--hidden_sizes', nargs= '+', type= int, default= [30, 20], help= 'list of hidden sizes for the hidden layers')
model.add_argument('--k_value', default = 7, type= int, help= 'k value to determine the number of clusters for the KMeans model')

clustering = parser.add_argument_group(title='args to deal with various k for kmeans(clustering)')
clustering.add_argument('--kmeans_only',  action='store_true', help = 'kmeans_only means that the kmeans graph for multiple k values is generated. this allows to see the elbow curve, and pick k ')



args= parser.parse_args()

print(args)

raise Exception('here')


#---- 
from driver import Driver

driver_class= Driver(args)
driver_class.do_mode()