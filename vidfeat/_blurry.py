import vidfeat
import imfeat
import sklearn.svm
import kernels


class BlurryFrameFeature(vidfeat.ClassifierFrameFeature):

    def __init__(self, *args, **kw):
        classifier = sklearn.svm.SVC(kernel=kernels.histogram_intersection)
        #classifier = sklearn.svm.LinearSVC()
        super(BlurryFrameFeature, self).__init__(classifier=classifier,
                                                 feature=imfeat.GradientHistogram(),
                                                 *args, **kw)

if __name__ == '__main__':
    data_root = '/home/brandyn/playground/blurry_data'
    print(BlurryFrameFeature().xval(vidfeat.load_label_frames(data_root)))
