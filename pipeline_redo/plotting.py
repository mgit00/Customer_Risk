import logging

from utils import Utils 
import matplotlib.pyplot as plt 
import seaborn 

class Plotting: 
    '''
    this class is used for plotting 
    '''

    def __init__(self,args): 
        self.args= args 

        self.logger= logging.getLogger(self.args.logger_name)

    def nn_plotting(self, state): 
        '''
        neural net plotting 
        info needed: 
            - epochs 
            losses 
            testlosses 

            if loss_func_mode == cel: 
                accuracy 
                testaccuracy 
                f1 
                testf1 
        '''

        self.logger.debug('plotting started')

        # create the plot 
        plt.plot(range(state['epochs']), state['losses'], label = 'Train Losses' )
        plt.plot(range(state['epochs']), state['testlosses'], label = 'Test Losses' )
        
        plt.title('Train and Test Losses for: ' + self.args.model_type + ' '  + self.args.output_mode)
        plt.xlabel('Epochs')
        plt.xticks(rotation= 30)
        plt.legend()

        filename = self.args.plot_filepath + 'losses.png'
        plt.savefig(filename) # can also do in utils if wanted
        self.logger.info('plot saved to: ' + filename)
        plt.show()


        if state['loss_func_mode'] =='cel': 
            self.plot_accuracy(state)
            self.plot_f1score(state)


        return 
    
    def plot_accuracy(self, state): 
        ''' for cel loss '''
        # create the plot 
        plt.plot(range(state['epochs']), state['losses'], label = 'Train Losses' )
        plt.plot(range(state['epochs']), state['testlosses'], label = 'Test Losses' )
        
        plt.title('Train and Test Accuracy for: ' + self.args.model_type + ' '  + self.args.output_mode)
        plt.xlabel('Epochs')
        plt.xticks(rotation= 30)
        plt.legend()

        filename= self.args.plot_filepath + 'accuracy.png'
        plt.savefig(filename) # can also do in utils if wanted
        self.logger.info('plot saved to: ' + filename)
        plt.show()
        return 

    def plot_f1score(self,state): 
        ''' for cel loss '''
        # create the plot 
        plt.plot(range(state['epochs']), state['losses'], label = 'Train Losses' )
        plt.plot(range(state['epochs']), state['testlosses'], label = 'Test Losses' )
        
        plt.title('Train and Test F1 Score for: ' + self.args.model_type + ' '  + self.args.output_mode)
        plt.xlabel('Epochs')
        plt.xticks(rotation= 30)
        plt.legend()

        filename= self.args.plot_filepath + 'f1score.png'
        plt.savefig(filename) # can also do in utils if wanted
        self.logger.info('plot saved to: ' + filename)
        plt.show()
        return 