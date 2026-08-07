import matplotlib.pyplot as plt 
import seaborn as sns 
from pathlib import Path

class Plotting: 
    def __init__(self, args): 
        self.args = args  

    def kmeans_elbow_curve(self, intertias): 
        kstart, kend = self.args.k_values 
        x = range(kstart, kend + 1)
        y  = intertias 

        plt.plot(x, y)
        plt.title('Elbow Curve for KMeans')
        plt.xticks(x, rotation= 45)
        plt.savefig(self.args.elbow_curve_for_kmeans)
        plt.show ()

    def nn_plotting(self, model_state): 
        epochs = model_state['epochs_completed']

        plt.plot(range(epochs), model_state['train_losses'], label = 'Train Losses')
        plt.plot(range(epochs), model_state['test_losses'], label = 'Test Losses')

        plt.title('Train and Test Losses for: ' + self.args.model_type + ' '  + self.args.target_mode)
        plt.xlabel('Epochs')
        plt.ylabel('CrossEntropyLoss' if self.args.target_mode == 'classification' else 'MSELoss')
        plt.xticks(rotation= 30)
        plt.legend()

        path = Path(self.args.nn_plot_file+'losses.png')
        path.parent.mkdir(parents= True, exist_ok=True )
        plt.savefig(path)
        print('plot saved to: ', path)

        plt.show()
        plt.close()
        #---- 

        plt.plot(range(epochs), model_state['train_accuracy'], label = 'Train Accuracy')
        plt.plot(range(epochs), model_state['test_accuracy'], label = 'Test Accuracy')

        plt.title('Train and Test Accuracy for: ' + self.args.model_type + ' '  + self.args.target_mode)
        plt.xlabel('Epochs')
        plt.xticks(rotation= 30)
        plt.legend()

        path = Path(self.args.nn_plot_file+'accuracy.png')
        path.parent.mkdir(parents= True, exist_ok=True )
        plt.savefig(path)
        print('plot saved to: ', path)

        plt.show()
        plt.close()
        #----

        plt.plot(range(epochs), model_state['train_f1'], label = 'Train F1')
        plt.plot(range(epochs), model_state['test_f1'], label = 'Test F1')

        plt.title('Train and Test F1 for: ' + self.args.model_type + ' '  + self.args.target_mode)
        plt.xlabel('Epochs')
        plt.xticks(rotation= 30)
        plt.legend()

        path = Path(self.args.nn_plot_file+'f1.png')
        path.parent.mkdir(parents= True, exist_ok=True )
        plt.savefig(path)
        print('plot saved to: ', path)

        plt.show()
        plt.close()





