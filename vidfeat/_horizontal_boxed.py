import vidfeat
import imfeat
import sklearn.svm


class HorizontalBoxedFrameFeature(vidfeat.ClassifierFrameFeature):
    feature = imfeat.BlackBars()

    def __init__(self, *args, **kw):
        classifier = sklearn.svm.LinearSVC(class_weight='auto')
        self.svm_parameters = [{'C': [10 ** x for x in range(0, 12, 3)]}]
        super(HorizontalBoxedFrameFeature, self).__init__(classifier=classifier,
                                                          *args, **kw)
    def remove_bars(self, image):
        pass

    def _feature(self, image):
        return self.feature(image)

if __name__ == '__main__':
    vidfeat._frame_feature_main('horizontal_boxed', vidfeat.HorizontalBoxedFrameFeature, remove_bars=False)
