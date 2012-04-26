import zlib, base64, cPickle
classifier = cPickle.loads(zlib.decompress(base64.b64decode("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")))