import vidfeat
import imfeat
import sklearn.svm
import numpy as np


def color_stats(image):
    image_hsv = imfeat.convert_image(image, {'mode': 'hsv', 'type': 'numpy', 'dtype': 'float32'})
    image_hsv = image_hsv.reshape((image_hsv.shape[0] * image_hsv.shape[1], image_hsv.shape[2]))
    image_hsv[:, 0] /= 360  # Rescale
    return np.hstack([np.min(image_hsv, 0), np.max(image_hsv, 0), np.mean(image_hsv, 0),
                      np.median(image_hsv, 0), np.std(image_hsv, 0), imfeat.UniqueColors()(image)])


class GrayFrameFeature(vidfeat.ClassifierFrameFeature):

    def __init__(self, *args, **kw):
        classifier = sklearn.svm.LinearSVC(class_weight='auto')
        self.svm_parameters = [{'C': [10 ** x for x in range(0, 12, 3)]}]
        super(GrayFrameFeature, self).__init__(classifier=classifier,
                                               *args, **kw)

    def _feature(self, image):
        out = color_stats(imfeat.resize_image_max_side(image, 128))
        return out


if __name__ == '__main__':
    vidfeat._frame_feature_main('gray', vidfeat.GrayFrameFeature)
