import vidfeat
import imfeat
import sklearn.svm


class VisibleLensFrameFeature(vidfeat.ClassifierFrameFeature):

    def __init__(self, *args, **kw):
        feature = imfeat.MetaFeature(imfeat.TinyImage())
        classifier = sklearn.svm.LinearSVC()
        super(VisibleLensFrameFeature, self).__init__(classifier=classifier,
                                                      feature=feature,
                                                      *args, **kw)

if __name__ == '__main__':
    data_root = '/home/brandyn/playground/visible_lens_data'
    c = VisibleLensFrameFeature().train(vidfeat.load_label_frames(data_root))
    c.save_module('models/visible_lens_frame_model0.py')
