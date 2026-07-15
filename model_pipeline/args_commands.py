commands= [
  '--epochs=5', 
  '--lr=.01',

  
  # '--save_model_kmeans',
  '--load_model_kmeans' , 
  '--save_model_knn', 
  # '--load_model_knn', 

  # '--model_training_loss_plot=images/knn2.png', 

  # if loading in model, ensure that the hidden sizes and other hyperparams are the saem 
  '--hidden_sizes=64', 
  '--hidden_sizes=128', 
  '--hidden_sizes=64', 


]