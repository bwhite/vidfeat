import vidfeat
import imfeat
import sklearn.svm


class VerticalBoxedFrameFeature(vidfeat.ClassifierFrameFeature):

    def __init__(self, *args, **kw):
        feature = imfeat.MetaFeature(imfeat.GridStats(), imfeat.Histogram('lab', num_bins=4))
        classifier = sklearn.svm.LinearSVC()
        super(VerticalBoxedFrameFeature, self).__init__(classifier=classifier,
                                                          feature=feature,
                                                          *args, **kw)

if __name__ == '__main__':
    data_root = '/home/brandyn/playground/vertical_boxed_data'
    c = VerticalBoxedFrameFeature().train(vidfeat.load_label_frames(data_root))
    c.save_module('models/vertical_boxed_frame_model0.py')
