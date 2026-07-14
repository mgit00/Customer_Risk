from sklearn.cluster import KMeans
import pandas as pd 

from plotting import ClusteringPlotting

'''
do k means for multiple k values 
plot the intertia against k 
pick the best k, and save model 
'''


class ClusteringModel: 
  def __init__(self, args, data: pd.DataFrame): 
    self.data = data 
    self.args = args 

  def driver(self): 
    k_vals, interias = self.apply_kmeans()

    # plot them 
    ClusteringPlotting(self.args).elbow_curve(k_vals, interias )


  def apply_kmeans(self): 
    intertias= []
    k_vals= []
    for k in range(self.args.k_values[0], self.args.k_values[1]): 
      # apply kmeans 

      kmeans= KMeans(n_clusters= k , random_state=1 )
      clusters= kmeans.fit_predict(self.data)
      # need to map intertia, so keep track of that 
      k_vals.append(k)
      intertias.append(kmeans.inertia_)

    return k_vals, intertias 








  

  








