import os
import numpy
import pandas
import time
import sys                                      #to write out stuff printed to screen
import pickle
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils.visualize_util import plot
from keras.optimizers import SGD
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping,LearningRateScheduler

from sklearn.preprocessing import RobustScaler
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix



def create_model(input_model, input_scale):
        #input_model = Sequential()
        input_model.add(Dense(100, input_dim=input_scale, init='uniform', activation='relu')) #Input layer
        input_model.add(Dense(100, init='uniform', activation='relu'))               #Hidden Layer 02
        input_model.add(Dense(100, init='uniform', activation='relu'))               #Hidden Layer 03
#        model.add(Dense(input_scale, init='uniform', activation='relu'))               #Hidden Layer 04
#        model.add(Dense(input_scale, init='uniform', activation='relu'))               #Hidden Layer 05
#        model.add(Dense(input_scale, init='uniform', activation='relu'))               #Hidden Layer 06
#        model.add(Dense(input_scale, init='uniform', activation='relu'))               #Hidden Layer 07
#        model.add(Dense(input_scale, init='uniform', activation='relu'))               #Hidden Layer 08
#        model.add(Dense(input_scale, init='uniform', activation='relu'))               #Hidden Layer 09
#        model.add(Dense(input_scale, init='uniform', activation='relu'))               #Hidden Layer 10
        input_model.add(Dense(1, init='uniform', activation='sigmoid'))
        return


def plot_accuracy(history_arg, x, show_toggle, save_toggle):
        # summarize history for accuracy
        plt.plot(history_arg.history['acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        if show_toggle == True:
                plt.show()
        if save_toggle == True:
                plt.savefig('./plots/27D_not1000_accuracy_truemass_%02d.png' % x)
        plt.close()
        return

def plot_loss(history_arg, x, show_toggle, save_toggle):
        # summarize history for loss
        plt.plot(history_arg.history['loss'])
        #plt.plot(history_arg.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        if show_toggle == True:
                plt.show()
        if save_toggle == True:
                plt.savefig('./plots/27D_not1000_loss_truemass_%02d.png' % x)
        plt.close()
        return

def plot_predictions(feature_arg, x_test_data, predictions, x, show_toggle, save_toggle):
        # Setup plot
        bins_array = numpy.arange(-5,5,0.1)
        plt.xlabel('Feature %d' % feature_arg)
        plt.ylabel('Events')
        plt.title('Distro of ttWW Reconstructed Mass')

        # Plot predictions
        plt.hist(x_test_data[numpy.nonzero(predictions.T==0)[1]][:, feature_arg-1], bins_array, alpha=0.3, color='r', label=['bkg'])
        plt.hist(x_test_data[(numpy.nonzero((predictions.T) & (x_test_data[:,27]==1500))[1])][:,feature_arg-1], bins_array, alpha=0.3, color='y', label=['1500'])
        plt.hist(x_test_data[(numpy.nonzero((predictions.T) & (x_test_data[:,27]==1250))[1])][:,feature_arg-1], bins_array, alpha=0.3, color='c', label=['1250'])
        plt.hist(x_test_data[(numpy.nonzero((predictions.T) & (x_test_data[:,27]==1000))[1])][:,feature_arg-1], bins_array, alpha=0.3, color='g', label=['1000'])
        plt.hist(x_test_data[(numpy.nonzero((predictions.T) & (x_test_data[:,27]==750))[1])][:,feature_arg-1], bins_array, alpha=0.3, color='b', label=['750'])
        plt.hist(x_test_data[(numpy.nonzero((predictions.T) & (x_test_data[:,27]<500))[1])][:,feature_arg-1], bins_array, alpha=0.3, color='m', label=['500'])
        plt.legend()
        if save_toggle == True:
                plt.savefig('./plots/27D_not1000_predictions_truemass_%02d.png' % x)
        if show_toggle == True:
                plt.show()
        plt.close()

def plot_ROC(y_test_data, predictions, x, show_toggle, save_toggle):
        #Plot ROC curve
        fpr, tpr, _ = roc_curve(y_test_data, predictions)
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.05])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic curve')
        if show_toggle == True:
                plt.show()
        print('\n')
        print('AUC: %f' % roc_auc)
        if save_toggle == True:
                plt.savefig('./plots/27D_not1000_ROC_truemass_%02d.png' % x)
        numpy.save('./roc_info/fpr_not1000_%02d' %x, fpr)
        numpy.save('./roc_info/tpr_not1000_%02d' %x, tpr)
        plt.close()
