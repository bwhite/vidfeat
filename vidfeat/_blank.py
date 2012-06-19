import vidfeat
import imfeat
import sklearn.svm


class BlankFrameFeature(vidfeat.ClassifierFrameFeature):
    feature = imfeat.BlackBars()
    
    def __init__(self, *args, **kw):
        classifier = sklearn.svm.LinearSVC(class_weight='auto')
        self.svm_parameters = [{'C': [10 ** x for x in range(0, 12, 3)]}]
        super(BlankFrameFeature, self).__init__(classifier=classifier, *args, **kw)

    def _feature(self, image):
        return self.feature(image)

    def predict(self, frame):
        if not frame.size:
            return 0 if self._invert else 1
        else:
            return super(BlankFrameFeature, self).predict(frame)

if __name__ == '__main__':
    vidfeat._frame_feature_main('blank', vidfeat.BlankFrameFeature, remove_bars=False)
    
