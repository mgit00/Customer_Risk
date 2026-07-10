from model import FNN 
from torch.nn import HuberLoss, MSELoss
from torch.optim import Adam
import torch 


class Train: 
  '''
  this implements the train loop 
  '''
  def __init__(self, args): 
    self.args= args 

  def training_loop(self, train_loader, test_loader): 
    '''
    train and test loader objects contain the tensors which are already 
    on the gpu (if available)
    using batching, but the entire tensors are already on the gpu memory since they are small 
    batching is used for better backpropogation updates 

    - do work with saving or loading model 
    '''


    input_size= self.args.input_size # 55 
    hidden_sizes= self.args.hidden_sizes # list 
    output_size= self.args.output_size # 1 


    model = FNN(input_size, hidden_sizes, output_size)
    model.to(self.args.device) # place on gpu if available 
    
    loss_metric= MSELoss() #HuberLoss()
    optimizer= Adam(model.parameters(), lr= self.args.lr)

    train_loss_data= []
    test_loss_data=[]
    epochs_completed = 0 

    # load in model if needed 
    if self.args.load_model_knn: 
      # get model state 
      model_state= torch.load(self.args.load_model_knn)

      model.state_dict= model_state['model_state_dict']
      optimizer.state_dict= model_state['optimizer_state_dict']

      epochs_completed= model_state['epochs_completed']
      train_loss_data= model_state['train_loss_data']
      test_loss_data= model_state['test_loss_data']


    for epoch in range(self.args.epochs): 
      model.train()
      loss_epoch = 0 

      for xbatch, ybatch in train_loader:
        # make forward pass 
        pred = model(xbatch) # forward 
        loss = loss_metric(pred, ybatch)

        loss_epoch+= loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


      model.eval()
      loss_epoch_test= 0
      for xbatch, ybatch in test_loader: 
        # do prediction 
        pred = model(xbatch)
        loss= loss_metric(pred, ybatch)
        loss_epoch_test+= loss.item()

      train_loss_data.append(loss_epoch)
      test_loss_data.append(loss_epoch_test)
      print(epoch +epochs_completed , loss_epoch, loss_epoch_test )

       

    # if saving model, save the current state of the model 
      # model state dict 
      # optimizer state dict 

      # epochs 
      # train and test loss data 
      # args if wanted 

    # model state 
      model_state = {
        'model_state_dict': model.state_dict(), 
        'optimizer_state_dict': optimizer.state_dict(), 

        'epochs_completed' : self.args.epochs + epochs_completed, 
        'train_loss_data': train_loss_data, 
        'test_loss_data': test_loss_data
      }

    if self.args.save_model_knn: 
      torch.save(model_state, self.args.save_model_knn)
      print('Model state saved')
      
    



