import vidfeat
import imfeat
import sklearn.svm


class HorizontalBoxedFrameFeature(vidfeat.ClassifierFrameFeature):

    def __init__(self, *args, **kw):
        feature = imfeat.MetaFeature(imfeat.GridStats(), imfeat.Histogram('lab', num_bins=4))
        classifier = sklearn.svm.LinearSVC()
        super(HorizontalBoxedFrameFeature, self).__init__(classifier=classifier,
                                                          feature=feature,
                                                          *args, **kw)

if __name__ == '__main__':
    data_root = '/home/brandyn/playground/horizontal_boxed_data'
    c = HorizontalBoxedFrameFeature().train(vidfeat.load_label_frames(data_root))
    c.save_module('models/horizontal_boxed_frame_model0.py')
