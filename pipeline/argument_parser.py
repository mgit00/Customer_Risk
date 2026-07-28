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
model.add_argument('--model_type', default= ['fnn'], choices=['fnn', 'kmeans_fnn', 'autoencoder_kmeans_knn'], required=False , 
                   help= 'pick the model to work on')
# data to read in is: --data 
model.add_argument('--batch_size', default = 1, type = int, help='smaller batches mean less memory, but noisy')
# --device arg created in train.training_loop 
model.add_argument('--lr', default = .001, type = float)
model.add_argument('--epochs', default = 3, type= int )

args= parser.parse_args()

print(args)



#---- 
from driver import Driver

driver_class= Driver(args)
driver_class.do_mode()