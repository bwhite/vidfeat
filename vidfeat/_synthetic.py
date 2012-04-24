import vidfeat
import imfeat
import sklearn.svm
import kernels
import random


class SyntheticFrameFeature(vidfeat.ClassifierFrameFeature):

    def __init__(self, *args, **kw):
        feature = imfeat.MetaFeature(imfeat.GradientHistogram(), imfeat.Histogram('lab'))
        #classifier = sklearn.svm.SVC(kernel=kernels.histogram_intersection)
        classifier = sklearn.svm.LinearSVC()
        super(SyntheticFrameFeature, self).__init__(classifier=classifier,
                                                    feature=feature,
                                                    *args, **kw)

if __name__ == '__main__':
    data_root = '/home/brandyn/playground/synthetic_data'
    c = SyntheticFrameFeature().train(vidfeat.load_label_frames(data_root))
    c.save_module('models/synthetic_frame_model0.py')

    
