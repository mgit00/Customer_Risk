from model import FNN 
from torch.nn import HuberLoss
from torch.optim import Adam


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
    loss_metric= HuberLoss()
    optimizer= Adam(model.parm)

    for epoch in range(self.args.epochs): 
      model.train()
      loss_epoch = 0 

      for xbatch, ybatch in train_loader:
        # make forward pass 
        pred = model(xbatch) # forward 
        loss = loss_metric()

      model.eval()

      for xbatch, ybatch in test_loader: 
        # do prediction 
