import vidfeat
import imfeat
import sklearn.svm
import kernels


class SyntheticFrameFeature(vidfeat.ClassifierFrameFeature):

    def __init__(self, *args, **kw):
        feature = imfeat.MetaFeature(imfeat.GradientHistogram(), imfeat.Histogram('lab'))
        classifier = sklearn.svm.SVC(kernel=kernels.histogram_intersection)
        super(SyntheticFrameFeature, self).__init__(classifier=classifier,
                                                    feature=feature,
                                                    *args, **kw)
        #sklearn.svm.LinearSVC(),

if __name__ == '__main__':
    data_root = '/home/brandyn/playground/synthetic_data'
    #SyntheticFrameFeature().train(vidfeat.load_label_frames(data_root))
    print(SyntheticFrameFeature().xval(vidfeat.load_label_frames(data_root)))
    
