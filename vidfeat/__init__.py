import numpy as np
import glob
import os
import cv2
import base64
import random
import zlib
import cPickle as pickle
import sklearn.cross_validation
import sklearn.grid_search
import sklearn.metrics
import copy
import imfeat

DATA_ROOT = '/home/brandyn/playground/aladdin_data/'


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
    
    def __init__(self, classifier,
                 *args, **kw):
        super(ClassifierFrameFeature, self).__init__(*args, **kw)
        self._invert = False
        self._classifier = classifier
        self._sparse = None
        self.raw_coef_ = None

    def train(self, label_frames):
        """Compute feature and train classifier

        Args:
            label_frames: Iterator of (label, frame)

        Return:
            Trained classifier
        """
        try:
            parameters = self.svm_parameters
        except AttributeError:
            parameters = [{'kernel': ['rbf'],
                           'gamma': [10 ** x for x in range(-8, -2, 2)],
                           'C': [10 ** x for x in range(0, 9, 3)]},
                          {'kernel': ['linear'],
                           'C': [10 ** x for x in range(0, 9, 3)]}]
        clf = sklearn.grid_search.GridSearchCV(self._classifier, parameters)
        features_labels = self._compute_features_labels(label_frames)
        clf.fit(*features_labels)
        self._classifier = clf.best_estimator_
        self._sparse = self._classifier._sparse
        self.raw_coef_ = self._classifier.raw_coef_
        try:
            print(sorted(clf.grid_scores_, key=lambda x: x[1], reverse=True)[:10])
        except AttributeError:
            pass
        y_true, y_pred = zip(*[(int(l), int(self._classifier.predict(f)))
                               for f, l in zip(*features_labels)])
        print(sklearn.metrics.confusion_matrix(y_true, y_pred))
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
        #self._classifier._sparse = self._sparse
        #self._classifier.raw_coef_ = self.raw_coef_
        return np.array(self._classifier.decision_function(self._feature(frame))[0])

    def predict(self, frame):
        if not frame.size:
            return 1 if self._invert else 0
        out = self._classifier.predict(np.ascontiguousarray(self._feature(frame))) == 1
        return not out if self._invert else out

    @property
    def I(self):
        c = copy.copy(self)
        c._invert = True
        return c


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
REMOVE_BARS = imfeat.BlackBars()
def load_label_frames(path, num_per_label=None, remove_bars=True, balance_classes=False):
    if path[-1] != '/':
        path += '/'
    if num_per_label is None:
        num_per_label = float('inf')
    if balance_classes:
        num_per_label = int(min([num_per_label] + [len(glob.glob(label_dir + '/*'))
                                                   for label_dir in glob.glob(path + '*')]))
    for label_dir in glob.glob(path + '*'):
        label = int(os.path.basename(label_dir))
        image_paths = glob.glob(label_dir + '/*')
        random.shuffle(image_paths)
        for frame_num, frame_fn in enumerate(image_paths):
            if num_per_label is not None and frame_num >= num_per_label:
                break
            frame = cv2.imread(frame_fn, cv2.CV_LOAD_IMAGE_COLOR)
            if remove_bars:
                sz = REMOVE_BARS.find_bars(frame)
                frame = frame[sz[0]:sz[1], sz[2]:sz[3], :]
            if not frame.size:
                continue
            yield label, frame


def save_to_py(output_path, **kw):
    with open(output_path, 'w') as fp:
        fp.write('import zlib, base64, cPickle\n')
        for name, val in kw.items():
            fp.write(name)
            fp.write(' = cPickle.loads(zlib.decompress(base64.b64decode("')
            fp.write(base64.b64encode(zlib.compress(pickle.dumps(val, -1))))
            fp.write('")))\n')


def _frame_feature_main(name, cls, remove_bars=True):
    data_root = DATA_ROOT + name
    c = cls().train(load_label_frames(data_root, remove_bars=remove_bars))
    save_to_py('models/%s_frame_model0.py' % name, classifier=c)


PREDICATES = [('synthetic', ([], ['blank'])),
              ('screenshot', ([], ['blank'])),
              ('text', ([], ['blank'])),
              ('poor_quality', ([], ['blank', 'synthetic'])),
              ('low_light', ([], ['blank', 'synthetic'])),
              ('gray', ([], ['blank', 'synthetic'])),
              ('fence', ([], ['blank', 'synthetic'])),
              ('water_outdoor', ([], ['blank', 'synthetic'])),
              ('blank', ([], [])),
              ('fire', ([], ['blank', 'synthetic'])),
              ('grass', ([], ['blank', 'synthetic'])),
              ('person', ([], ['blank', 'synthetic'])),
              ('person_hand', ([], ['blank', 'synthetic'])),
              ('person_upper', ([], ['blank', 'synthetic'])),
              ('snow', ([], ['blank', 'synthetic'])),
              ('road', ([], ['blank', 'synthetic'])),
              ('underwater', ([], ['blank', 'synthetic']))]
            #('sky_outdoor', ([], ['blank', 'synthetic'])),
            #('snow_outdoor', ([], ['blank', 'synthetic'])),
            #('grass_outdoor', ([], ['blank', 'synthetic'])),
            #('close_foreground', ([], ['blank', 'synthetic']))]


def _features_to_true_predicates(features):
    feature_names = [x[0] for x in features]
    features = [(y, z) for x, (y, z) in features]
    n2n = lambda x: [feature_names.index(y) for y in x]
    return [(n2n(y), n2n(z)) for y, z in features]
PREDICATES_IND = _features_to_true_predicates(PREDICATES)
PREDICATE_NAMES = [x[0] for x in PREDICATES]


from _synthetic import SyntheticFrameFeature
from _poor_quality import PoorQualityFrameFeature
from _horizontal_boxed import HorizontalBoxedFrameFeature
from _vertical_boxed import VerticalBoxedFrameFeature
from _visible_lens import VisibleLensFrameFeature
from _blank import BlankFrameFeature
from _meta import MetaFrameFeature
from _screenshot import ScreenshotFrameFeature
from _text import TextFrameFeature
from _low_light import LowLightFrameFeature
from _gray import GrayFrameFeature
from _fire import FireFrameFeature
from _water_outdoor import WaterOutdoorFrameFeature
from _fence import FenceFrameFeature
from _grass import GrassFrameFeature
from _road import RoadFrameFeature
from _snow import SnowFrameFeature
from _underwater import UnderwaterFrameFeature
from _person import PersonFrameFeature
from _person_upper import PersonUpperFrameFeature
from _person_hand import PersonHandFrameFeature
