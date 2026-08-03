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
filenames.add_argument('--application_record_pipeline', default= 'model_info/application_record_pipeline.pkl', help= 'application pipeline for processing')
filenames.add_argument('--data', default='dataset/data.csv', help='the customer_risk and applications_processed joined dataset')
filenames.add_argument('--elbow_curve_for_kmeans', default='images/elbow_curve_for_kmeans.png', help= 'location to save the elbow curve')
filenames.add_argument('--loaded_model', nargs ='?', const= 'model_info/saved_model.pt', help = 'filename for loading in model')
filenames.add_argument('--saved_model', default= 'model_info/saved_model.pt', help = 'filename for saving model')
filenames.add_argument('--nn_plot', default= 'images/nn_plots', dest= 'nn_plot_file', help='base name for the nn plot, will be appended below...')
filenames.add_argument('--kmeans_model', default='model_info/kmeans_model.pkl', help='file to save the trained kmeans model')

dataprocessing= parser.add_argument_group(title = 'args for mode data_processing')
dataprocessing.add_argument('--target_mode', choices= ['classification', 'regression'], default = 'classification', help= 'decides if the dataset created will have credit risk as a continuous feature')

model= parser.add_argument_group(title= 'args for model')
model.add_argument('--model_type', default= 'fnn', 
                   choices=['slp', 'fnn', 'kmeans_fnn', 'autoencoder_kmeans_knn'],
                   required=False , help= 'pick the model to work on')
# data to read in is: --data 
model.add_argument('--batch_size', default = 1000, type = int, help='smaller batches mean less memory, but noisy')
# --device arg created in train.training_loop 
model.add_argument('--lr', default = .0001, type = float)
model.add_argument('--epochs', default = 5, type= int )
model.add_argument('--hidden_sizes', nargs= '+', type= int, default= [30, 20], help= 'list of hidden sizes for the hidden layers')
model.add_argument('--k_value', default = 11, type= int, help= 'k value to determine the number of clusters for the KMeans model')
# 11 clusters decided by the kmeans_only option in the pipeline 

clustering = parser.add_argument_group(title='args to deal with various k for kmeans(clustering)')
clustering= parser.add_argument('--k_values' , nargs= 2, default = [2, 5], type= int, help= 'the starting and ending k values to plot the elbow curve for. min of 1 cluster' )




args= parser.parse_args()
# dynamic changes to filename based on args that are passed in 
args.nn_plot_file = args.nn_plot_file + '/' + args.target_mode + '/'+  args.model_type + '/' # + '.png' <- done at end 


print(args)

#---- 
from driver import Driver

driver_class= Driver(args)
driver_class.do_mode()

