
from logging import Logger
from gensim.models import Word2Vec
from typing import Union

from DeepRefiner.Network.dataset import BytecodeDataset
from SDAC.Network.dataset import APIDataset
import time

def word2vec_train(model_path: str, train_data: Union[BytecodeDataset, APIDataset], logger: Logger, **kwargs):
    """
    Train a Word2Vec model.
    :param model_path: Path to save the trained model.
    :param train_data: Training dataset.
    :param logger: Logger.
    :param kwargs: Other parameters.
    """
    logger.info('Training Word2Vec model...')
    sentences = train_data.get_data()
    start_time = time.time()

    vector_size = kwargs['vector_size']
    window = kwargs['window']
    workers = kwargs['workers']
    epochs = kwargs['w2v_epochs']
    sg = kwargs['sg']
    min_count = kwargs['min_count']
    model = Word2Vec(sentences, vector_size=vector_size, window=window, workers=workers, epochs=epochs, sg=sg, min_count=min_count)

    # Add unknown word
    model.wv['<UNK>'] = model.wv.vectors.mean(axis=0)
    end_time = time.time()
    # We only save word vocabulary.
    model.wv.save(model_path)

    # Print wv size
    logger.info(f'Word vocab size: {len(model.wv)}')
    logger.debug(f'Word2vec training time: {end_time - start_time}s')
    