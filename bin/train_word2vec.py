import io
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SentenceGenerator(object):
    def __init__(self, filename):
        self.fname = filename
 
    def __iter__(self):
        for line in io.open(self.fname, "r", encoding="utf-8"):
            yield line.rstrip().split(" ")

def train(model_name, training_data, size=100, window=5, negative=5, min_count=0, workers=4):
    if path.isfile(model_name):
        input("%s already exists and will be overwritten. Press Enter to proceed." % model_name)

    logger.info("Training the model")
    sentences = SentenceGenerator(training_data)
    model = Word2Vec(sentences, size=size, window=window, negative=negative, min_count=min_count, workers=workers)
    logger.info("Saving the model")
    model.init_sims(replace=True) # trim unneeded model memory = use (much) less RAM. 
    # model.save_word2vec_format(model_name, binary=True)
    model.save(model_name)
    logger.info("Trained model was saved to %s." % model_name)
