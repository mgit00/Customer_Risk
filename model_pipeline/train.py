from model import FNN 


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
    '''

    input_size= 
    hidden_sizes= 
    output_size= 

    model = FNN(input_size, hidden_sizes, output_size)
    loss= 

    for epoch in range(self.args.epochs): 
      model.train()


      for xbatch, ybatch in train_loader:
        # make forward pass 

