from model import FNN 
from torch.nn import HuberLoss, MSELoss
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


    input_size= self.args.input_size # 55 
    hidden_sizes= self.args.hidden_sizes # list 
    output_size= self.args.output_size # 1 

    model = FNN(input_size, hidden_sizes, output_size)
    model.to(self.args.device) # place on gpu if available 
    
    loss_metric= MSELoss() #HuberLoss()
    optimizer= Adam(model.parameters(), lr= self.args.lr)

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

      print(epoch, loss_epoch, loss_epoch_test )

    



