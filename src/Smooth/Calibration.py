import numpy as np
import matplotlib.pyplot as plt
import calibration as cal
from sklearn.isotonic import IsotonicRegression
from sklearn import preprocessing
from sklearn.calibration import  calibration_curve
from sklearn.metrics import precision_score, f1_score, recall_score, roc_auc_score

"""
This script uses library uncertianity-calibration to estimate calibration error from 
@inproceedings{kumar2019calibration,
  author = {Ananya Kumar and Percy Liang and Tengyu Ma},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  title = {Verified Uncertainty Calibration},
  year = {2019},
}
"""

class Calibrator():

    def __init__(self, n_classes, method="Isotonic"):
        self.method = method
        self.n_classes = n_classes
        self.models = [None]*n_classes

    def normalize(self, proba):
        """ Normalize the probabilities
        """
        if self.n_classes == 2:
            proba[:, 0] = 1. - proba[:, 1]
        else:
            proba /= np.sum(proba, axis=1)[:, np.newaxis]

        # handle nan probs
        proba[np.isnan(proba)] = 1. / self.n_classes

        # handle when predicted probaability minimally exceeds 1.0
        proba[(1.0 < proba) & (proba <= 1.0 + 1e-5)] = 1.0
        
        return proba

    def fit(self, proba, y):
        
        if self.method == 'Platt':
            print("Not implemented yet. Using Isotonic regression..")
            self.method = "Isotonic"
            
        if self.method =='Isotonic':

            # binarize class labels
            lb = preprocessing.LabelBinarizer().fit(y)
            y_cal_ohe = lb.transform(y)
            
            for i in range(self.n_classes):
                self.models[i] = IsotonicRegression(out_of_bounds = 'clip').fit(proba[:,i], y_cal_ohe[:,i])

    def transform(self, proba):

        if np.any([model is None for model in self.models]):
            print("Warning: No trained calibrator found. Returning original probabilities.")
            return proba
        
        shape = proba.shape
        proba_flatten = proba.reshape(-1,self.n_classes)
        iso_prob = np.zeros((proba_flatten.shape[0],self.n_classes))
        for i in range(self.n_classes):    
            iso_prob[:,i] = self.models[i].transform(proba_flatten[:,i])
        iso_prob = self.normalize(iso_prob).reshape(*shape)
        return iso_prob
 
def plot_reliability_curve(pred_prob,y_cal,pop_order,method='Uncalibrated',bins=10, legend=True):
    
    fig, (ax1) = plt.subplots(nrows = 1, ncols=1, figsize = (8,6))
                                              
    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    ax1.set_title(method + ' Reliability Plot')
        
    class_metrics = {}
    n_classes = len(pop_order)
    for i in range(n_classes):
        y_label = np.where(y_cal==i,1,0)
        est_prob = pred_prob[:,:,i]
        true_prob, pred_proba = calibration_curve(y_label.flatten(),est_prob.flatten(), n_bins = bins)
        ax1.plot(pred_proba, true_prob, label = pop_order[i])
        
        ax1.set_xlabel('Est. Prob/ mean predicted value')
        ax1.set_ylabel('True Prob/fraction_of_positives')
        
        y_test = y_label.flatten()
        y_pred =np.where(est_prob.flatten()>1/n_classes,1,0)
        
        class_metrics[pop_order[i]]={}
        class_metrics[pop_order[i]]['Precision'] = float("%1.3f" %precision_score(y_test, y_pred))
        class_metrics[pop_order[i]]['Recall'] = float("%1.3f" %recall_score(y_test, y_pred))
        class_metrics[pop_order[i]]['F1'] = float("%1.3f" %f1_score(y_test, y_pred))
        class_metrics[pop_order[i]]['AUC'] = float("%1.3f" %roc_auc_score(y_test, y_pred))

    if legend:
        ax1.legend(loc="lower right")
    fig.tight_layout()
    plt.show()
    return class_metrics

def comparison(uncalibrated_zs,calibrated_zs,ys,pop_type,pop_order):
    
    """
    Compare probabilities of a population type before and after calibration and plot binned probabilities histogram   """
    # pop_type (int between 0 and len(pop_order) for which population the comparison needs to be
    
    fig, (ax1,ax2) = plt.subplots(nrows = 2, ncols=1, figsize = (10,10))
    
    y_label = np.where(ys==pop_type,1,0)
    uncal_est_prob = uncalibrated_zs[:,:,pop_type]
    cal_est_prob = calibrated_zs[:,:,pop_type]
    uncal_true_prob, uncal_pred_proba = calibration_curve(y_label.flatten(),uncal_est_prob.flatten(), n_bins = 10)
    ax1.plot(uncal_pred_proba, uncal_true_prob, label = pop_order[pop_type] + ' uncalib.')
    
    cal_true_prob, cal_pred_proba = calibration_curve(y_label.flatten(),cal_est_prob.flatten(), n_bins = 10)
    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    ax1.plot(cal_pred_proba, cal_true_prob, label = pop_order[pop_type] + ' calib.')
    
    ax1.set_xlabel('Est. Prob/ mean predicted value')
    ax1.set_ylabel('True Prob/fraction_of_positives')
    ax1.legend(loc="lower right")
    
    ax2.hist(uncal_est_prob.flatten(), range=(0, 1), bins=20, histtype="step", label = 'uncalib.', log=True, lw=3)
    ax2.hist(cal_est_prob.flatten(), range=(0, 1), bins=20, histtype="step", label = 'calib.', log=True, lw=3)
    ax2.legend(loc="lower right")
        
    calibration_error = cal.get_ece(uncal_est_prob.flatten(), y_label.flatten())
    print("Scaling-binning L2 calibration error for uncalib. probs is %.2f%%" % (100 * calibration_error))
    
    calibration_error = cal.get_ece(cal_est_prob.flatten(), y_label.flatten())
    print("Scaling-binning L2 calibration error for calib. probs is %.2f%%" % (100 * calibration_error))
        
    fig.tight_layout()
    plt.show()
