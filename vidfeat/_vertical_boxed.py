import vidfeat
import imfeat
import sklearn.svm


class VerticalBoxedFrameFeature(vidfeat.ClassifierFrameFeature):
    feature = imfeat.BlackBars()

    def __init__(self, *args, **kw):
        classifier = sklearn.svm.LinearSVC(class_weight='auto')
        self.svm_parameters = [{'C': [10 ** x for x in range(0, 12, 3)]}]
        super(VerticalBoxedFrameFeature, self).__init__(classifier=classifier,
                                                        *args, **kw)

    def _feature(self, image):
        return self.feature(image)


if __name__ == '__main__':
    vidfeat._frame_feature_main('vertical_boxed', vidfeat.VerticalBoxedFrameFeature, remove_bars=True)
