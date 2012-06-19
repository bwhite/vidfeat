import vidfeat
import imfeat
import sklearn.svm


class LowLightFrameFeature(vidfeat.ClassifierFrameFeature):
    feature = imfeat.MetaFeature(imfeat.GradientHistogram(), imfeat.Histogram('lab', num_bins=4), imfeat.Moments('lab', 2), max_side=128)

    def __init__(self, *args, **kw):
        classifier = sklearn.svm.LinearSVC(class_weight='auto')
        self.svm_parameters = [{'C': [10 ** x for x in range(0, 12, 3)]}]
        super(LowLightFrameFeature, self).__init__(classifier=classifier,
                                                   *args, **kw)
    
    def _feature(self, image):
        return self.feature(image)


if __name__ == '__main__':
    vidfeat._frame_feature_main('low_light', vidfeat.LowLightFrameFeature)
