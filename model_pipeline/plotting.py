import seaborn as sns 
import matplotlib.pyplot as plt 


class ClusteringPlotting: 
  def __init__(self, args): 
    self.args= args 

  def elbow_curve(self, k_vals, intertias): 
    plt.plot([k_vals, intertias])
    plt.title('Elbow Curve for KMeans')
    plt.show()