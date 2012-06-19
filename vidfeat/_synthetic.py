import vidfeat
import imfeat
import sklearn.svm
from vidfeat.models.synthetic_bovw_clusters import clusters
#import kernels
HOG = imfeat.HOGLatent(8, 2)


class SyntheticFrameFeature(vidfeat.ClassifierFrameFeature):
    #feature = imfeat.MetaFeature(imfeat.GradientHistogram(), imfeat.Histogram('lab'))
    feature = imfeat.MetaFeature(imfeat.BoVW(lambda x: HOG.make_bow_mask(x, clusters), clusters.shape[0], 3), imfeat.Histogram('lab', num_bins=4), imfeat.UniqueColors())
    
    def __init__(self, *args, **kw):
        classifier = sklearn.svm.LinearSVC(class_weight='auto')
        self.svm_parameters = [{'C': [10 ** x for x in range(0, 12, 3)]}]
        super(SyntheticFrameFeature, self).__init__(classifier=classifier,
                                                    *args, **kw)

    def _feature(self, image):
        import time
        st = time.time()
        out = self.feature(imfeat.resize_image_max_side(image, 160))
        print time.time() - st
        return out

if __name__ == '__main__':
    vidfeat._frame_feature_main('synthetic', vidfeat.SyntheticFrameFeature)

