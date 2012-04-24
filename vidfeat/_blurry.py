import vidfeat
import imfeat
import sklearn.svm
import kernels


class BlurryFrameFeature(vidfeat.ClassifierFrameFeature):

    def __init__(self, *args, **kw):
        #classifier = sklearn.svm.SVC(kernel=kernels.histogram_intersection)
        classifier = sklearn.svm.LinearSVC()
        feature = imfeat.GradientHistogram()
        super(BlurryFrameFeature, self).__init__(classifier=classifier,
                                                 feature=feature,
                                                 *args, **kw)

if __name__ == '__main__':
    data_root = '/home/brandyn/playground/blurry_data'
    c = BlurryFrameFeature().train(vidfeat.load_label_frames(data_root))
    c.save_module('models/blurry_frame_model0.py')
