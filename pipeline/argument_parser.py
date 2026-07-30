import argparse 
 
parser= argparse.ArgumentParser(description='model creation and pipeline')

'''
can run through full pipeline, calls all functions 


if you call steps like data_processing, then only that function will be executed 
'''
parser.add_argument('--mode', type= str, 
                    choices=['predict', 'build_model', 'data_processing', 'full', 'kmeans_only'], 
                    required=True, 
                    help= 'pick the entire pipeline or section of the pipeline to do. kmeans_only means that data will be processed and the kmeans graph for multiple k values is generated. this allows to see the elbow curve, and pick k '  )


filenames= parser.add_argument_group(title= 'filenames and paths for datasets' )
# original data 
filenames.add_argument('--application_record', default='dataset/application_record.csv', help='original data ')
filenames.add_argument('--credit_record', default='dataset/credit_record.csv', help='original data ')
# processed data 
filenames.add_argument('--customer_risk', default='dataset/customer_risk.csv', help= 'processed data ')
filenames.add_argument('--applications_processed', default='dataset/applications_processed.csv', help='processed data ')
filenames.add_argument('--application_record_pipeline', default= 'pipeline/application_record_pipeline.pkl', help= 'application pipeline for processing')
filenames.add_argument('--data', default='dataset/data.csv', help='the customer_risk and applications_processed joined dataset')
filenames.add_argument('--elbow_curve_for_kmeans', default='images/elbow_curve_for_kmeans.png', help= 'location to save the elbow curve')

dataprocessing= parser.add_argument_group(title = 'args for mode data_processing')
dataprocessing.add_argument('--target_mode', options= ['classification', 'regression'], default = 'classification', title= 'decides if the dataset created will have credit risk as a continuous feature')

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
clustering= parser.add_argument('--k_values' , nargs= 2, default = [1, 5], type= int, help= 'the starting and ending k values to plot the elbow curve for. min of 1 cluster' )


args= parser.parse_args()

print(args)


#---- 
from driver import Driver

driver_class= Driver(args)
driver_class.do_mode()