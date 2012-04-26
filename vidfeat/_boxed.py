import vidfeat
import imfeat
import sklearn.svm


class BoxedFrameFeature(vidfeat.ClassifierFrameFeature):

    def __init__(self, *args, **kw):
        feature = imfeat.GridStats()
        classifier = sklearn.svm.LinearSVC()
        super(BoxedFrameFeature, self).__init__(classifier=classifier,
                                                feature=feature,
                                                *args, **kw)

if __name__ == '__main__':
    data_root = '/home/brandyn/playground/boxed_data'
    c = BoxedFrameFeature().train(vidfeat.load_label_frames(data_root))
    c.save_module('models/boxed_frame_model0.py')
