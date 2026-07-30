import matplotlib.pyplot as plt 
import seaborn as sns 

class Plotting: 
    def __init__(self, args): 
        self.args = args 

    def kmeans_elbow_curve(self, intertias): 
        kstart, kend = self.args.k_values 
        x = range(kstart, kend + 1)
        y  = intertias 

        plt.plot(x, y)
        plt.title('Elbow Curve for KMeans')
        plt.xticks(x)
        plt.savefig(self.args.elbow_curve_for_kmeans)
        plt.show ()





