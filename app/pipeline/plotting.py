import logging

from utils.constants import ConstantFilenames

import matplotlib.pyplot as plt 

from utils.schemas import ModelTrainingBasemodel

plt.set_loglevel('warning')

class Plotting: 
    '''
    this class is used for plotting 
    '''

    def __init__(self): 
        # self.args= args 

        self.logger= logging.getLogger('main_logger')
    
    def kmeans_elbow_curve(self, kstart, kend, inertias): 
        x = range(kstart, kend + 1)
        y  = inertias 

        plt.plot(x, y)
        plt.title('Elbow Curve for KMeans')
        plt.xlabel('Number of Clusters')
        plt.xticks(x, rotation= 45)
        plt.savefig(ConstantFilenames.elbow_curve_for_kmeans)
        self.logger.info('kmeansplot saved to: ' + ConstantFilenames.elbow_curve_for_kmeans)
        # plt.show ()
        # # add to args 
        # self.args.others['plot_files'].append(self.args.elbow_curve_for_kmeans)

        plt.close('all')

    def nn_plotting(self, args: ModelTrainingBasemodel , state: dict ): 
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
        
        plt.title('Train and Test Losses for: ' + args.model_type.name + ' '  + 'classification')
        plt.xlabel('Epochs')
        plt.xticks(rotation= 30)
        plt.legend()

        filename = args.plot_filepath + 'losses.png'
        plt.savefig(filename) # can also do in utils if wanted
        self.logger.info('plot saved to: ' + filename)
        # # plt.show()
        # self.args.others['plot_files'].append(filename)


        if state['loss_func_mode'] =='cel': 
            self.plot_accuracy(args, state)
            self.plot_f1score(args, state)

        plt.close('all')
        return 
    
    def plot_accuracy(self, args,  state): 
        ''' for cel loss '''
        # create the plot 
        plt.plot(range(state['epochs']), state['accuracy'], label = 'Train Accuracy' )
        plt.plot(range(state['epochs']), state['testaccuracy'], label = 'Test Accuracy' )
        
        plt.title('Train and Test Accuracy for: ' + args.model_type.name + ' '  + 'classification')
        plt.xlabel('Epochs')
        plt.xticks(rotation= 30)
        plt.legend()

        filename= args.plot_filepath + 'accuracy.png'
        plt.savefig(filename) # can also do in utils if wanted
        self.logger.info('plot saved to: ' + filename)
        # plt.show()
        return 

    def plot_f1score(self, args, state): 
        ''' for cel loss '''
        # create the plot 
        plt.plot(range(state['epochs']), state['f1'], label = 'Train F1 Score' )
        plt.plot(range(state['epochs']), state['testf1'], label = 'Test F1 Score' )
        
        plt.title('Train and Test F1 Score for: ' + args.model_type.name + ' '  + 'classification')
        plt.xlabel('Epochs')
        plt.xticks(rotation= 30)
        plt.legend()

        filename= args.plot_filepath + 'f1score.png'
        plt.savefig(filename) # can also do in utils if wanted
        self.logger.info('plot saved to: ' + filename)
        # plt.show()
        return 