import vidfeat


class MetaFrameFeature(vidfeat.ClassifierFrameFeature):
    def __init__(self, *frame_features, **kw):
        self._frame_features = frame_features
        super(MetaFrameFeature, self).__init__(classifier=None, **kw)

    def _feature(self, image):
        return self.feature(image)

    def train(self, label_frames):
        raise NotImplementedError
        
    def __call__(self, frame):
        raise NotImplementedError

    def predict(self, frame):
        return all(f.predict(frame) for f in self._frame_features)
