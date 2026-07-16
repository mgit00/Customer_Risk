commands= [
  # '--epochs=5', 
  # '--lr=.001',

  
  # # '--save_model_kmeans',
  # '--load_model_kmeans' , 
  # '--save_model_knn', 
  # # '--load_model_knn', 

  # # '--model_training_loss_plot=images/knn2.png', 

  # # if loading in model, ensure that the hidden sizes and other hyperparams are the saem 
  # '--hidden_sizes=64', 
  # '--hidden_sizes=128', 
  # '--hidden_sizes=64', 



  '--epochs=5', 
  '--lr=.001',
  '--batch_size=100',

  '--application_file=dataset_redo/application_record_processed.csv',
  '--application_risk_file=dataset_redo/customer_application_risk.csv',


  # '--kmeans_evaluation', 
  
  '--save_model_kmeans=saved_models/redone_data_kmeans',
  # '--load_model_kmeans' , 
  '--save_model_fnn=saved_models/redone_data_fnn.pt', 
  # '--load_model_knn', 

  '--model_training_loss_plot=images/redone_data_fnn2.png', 

  # if loading in model, ensure that the hidden sizes and other hyperparams are the saem 
  '--hidden_sizes=64', 
  '--hidden_sizes=32', 




]