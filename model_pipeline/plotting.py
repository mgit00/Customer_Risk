import seaborn as sns 
import matplotlib.pyplot as plt 


class ClusteringPlotting: 
  def __init__(self, args): 
    self.args= args 

  def elbow_curve(self, k_vals, intertias): 
    plt.plot(k_vals, intertias)
    plt.title('Elbow Curve for KMeans')
    plt.savefig('images/elbow_curve.png')
    plt.show()

class TrainingPlotting: 
  def __init__(self, args): 
    self.args= args 

  def plotting(self, epochs, train, test): 
    '''
    need number of epochs, and train test losses 
    '''
    print('here')

    plt.plot(range(epochs), train, label= 'Train Losses' )
    plt.plot(range(epochs), test, label= 'Test Losses' )

    plt.title('Train and Test Losses for Epochs')
    plt.legend()
    plt.show()


# tp= TrainingPlotting(args= None)

# import torch 

# model_state= torch.load('saved_models/knn.pkl') 

# tp.plotting(model_state['epochs_completed'], 
#                               model_state['train_loss_data'], 
#                               model_state['test_loss_data'], )
