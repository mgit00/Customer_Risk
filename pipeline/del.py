from utils import Utils 
from data_processing import DataProcessing

data= Utils.read_data('dataset/application_record.csv')
dp = DataProcessing(None).cc_applications_processing(data)
# Utils.save_df('dataset/del.csv')
print(dp)
print(dp.shape)