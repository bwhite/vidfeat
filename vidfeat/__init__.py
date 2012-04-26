import numpy as np
import glob
import os
import cv2
import base64
import zlib
import cPickle as pickle
import sklearn.cross_validation


class Feature(object):

    def __init__(self, *args, **kw):
        super(Feature, self).__init__(*args, **kw)


class FrameFeature(Feature):

    def __init__(self, *args, **kw):
        super(FrameFeature, self).__init__(*args, **kw)

    def check_frame(self, frame):
        """Compute a feature on a single frame

        Args:
            frame: numpy array (height, width, 3) bgr with dtype=np.uint8
        """
        assert isinstance(frame, np.ndarray)
        assert frame.ndim == 3
        assert frame.shape[2] == 3
        assert frame.dtype == np.uint8


class ClassifierFrameFeature(FrameFeature):

    def __init__(self, classifier=None, feature=None, load_module=None, binary=True,
                 *args, **kw):
        super(ClassifierFrameFeature, self).__init__(*args, **kw)
        self._binary = binary
        self._feature = feature
        if load_module is None:
            self._classifier = classifier
        else:
            mod = __import__(load_module, fromlist=['blah'])
            self._classifier = mod.classifier

    def save_module(self, output_path):
        save_to_py(output_path,
                   classifier=self._classifier)

    def train(self, label_frames):
        """Compute feature and train classifier

        Args:
            label_frames: Iterator of (label, frame)

        Return:
            Trained classifier
        """
        self._classifier.fit(*self._compute_features_labels(label_frames))
        return self

    def _compute_features_labels(self, label_frames):
        labels = []
        features = []
        for label, frame in label_frames:
            labels.append(label)
            features.append(self._feature(frame))
        return np.array(features), np.array(labels)

    def xval(self, label_frames):
        """Compute feature and train classifier

        Args:
            label_frames: Iterator of (label, frame)

        Return:
            Trained classifier
        """
        return sklearn.cross_validation.cross_val_score(self._classifier, *self._compute_features_labels(label_frames))

    def __call__(self, frame):
        super(ClassifierFrameFeature, self).check_frame(frame)
        if self._binary:
            return self._classifier.predict(self._feature(frame)) == 1
        else:
            return np.array(self._classifier.decision_function(self._feature(frame))[0])


class FrameRegionFeature(Feature):
    
    def __init__(self, *args, **kw):
        super(FrameRegionFeature, self).__init__(*args, **kw)

    def __call__(self, num_time_frames):
        """Compute a feature on a single frame

        Args:
            frame: numpy array (height, width, 3) bgr with dtype=np.uint8

        Returns:
        
        """


class SequenceFeature(Feature):
    
    def __init__(self, *args, **kw):
        super(SequenceFeature, self).__init__(*args, **kw)

    def __call__(self, num_time_frames):
        """Compute a feature across the whole video

        Args:
            num_time_frames: Iterator of (frame_num, frame_time, frame)  where
                frame is a numpy array (height, width, 3) bgr with dtype=np.uint8

        Returns:

        """


class VideoFeature(Feature):
    
    def __init__(self, *args, **kw):
        super(VideoFeature, self).__init__(*args, **kw)

    def __call__(self, num_time_frames):
        """Compute a feature across the whole video

        Args:
            num_time_frames: Iterator of (frame_num, frame_time, frame)  where
                frame is a numpy array (height, width, 3) bgr with dtype=np.uint8

        Returns:

        """


### Helper Functions
def load_label_frames(path):
    if path[-1] != '/':
        path += '/'
    for label_dir in glob.glob(path + '*'):
        label = int(os.path.basename(label_dir))
        for frame_fn in glob.glob(label_dir + '/*'):
            yield label, cv2.imread(frame_fn, cv2.CV_LOAD_IMAGE_COLOR)


def save_to_py(output_path, **kw):
    with open(output_path, 'w') as fp:
        fp.write('import zlib, base64, cPickle\n')
        for name, val in kw.items():
            fp.write(name)
            fp.write(' = cPickle.loads(zlib.decompress(base64.b64decode("')
            fp.write(base64.b64encode(zlib.compress(pickle.dumps(val, -1))))
            fp.write('")))\n')


from _synthetic import SyntheticFrameFeature
from _blurry import BlurryFrameFeature
from _boxed import BoxedFrameFeature
from _horizontal_boxed import HorizontalBoxedFrameFeature
from _vertical_boxed import VerticalBoxedFrameFeature
from _visible_lens import VisibleLensFrameFeature
import models

