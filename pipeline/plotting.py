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
        plt.xticks(x, rotation= 45)
        plt.savefig(self.args.elbow_curve_for_kmeans)
        plt.show ()

    def nn_plotting(self, model_state): 
        print(model_state['train_losses'], model_state['train_losses'])
        epochs = model_state['epochs_completed']

        plt.plot(range(epochs), model_state['train_losses'], label = 'Train Losses')
        plt.plot(range(epochs), model_state['test_losses'], label = 'Test Losses')

        plt.title('Train and Test Losses for: ' + self.args.model_type + ' '  + self.args.target_mode)
        plt.xlabel('Epochs')
        plt.ylabel('CrossEntropyLoss' if self.args.target_mode == 'classification' else 'MSELoss')
        plt.xticks(rotation= 30)
        plt.legend()

        plt.savefig(self.args.nn_plot_file)
        print('plot saved to: ', self.args.nn_plot_file)
        plt.show()





