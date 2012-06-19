import vidfeat
import imfeat
import sklearn.svm


class VisibleLensFrameFeature(vidfeat.ClassifierFrameFeature):
    feature = imfeat.MetaFeature(imfeat.TinyImage())

    def __init__(self, *args, **kw):
        classifier = sklearn.svm.LinearSVC(class_weight='auto')
        super(VisibleLensFrameFeature, self).__init__(classifier=classifier,
                                                      *args, **kw)

    def _feature(self, image):
        return self.feature(image)


if __name__ == '__main__':
    data_root = '/home/brandyn/playground/visible_lens_data'
    c = VisibleLensFrameFeature().train(vidfeat.load_label_frames(data_root))
    vidfeat.save_to_py('models/visible_lens_frame_model0.py', classifier=c)
