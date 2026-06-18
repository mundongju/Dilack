import logging

import os

def get_logger(log_file='experiment.log', log_dir='./logs'):
    logger = logging.getLogger(name='DPS')
    logger.setLevel(logging.INFO)
    
    formatter = logging.Formatter("%(asctime)s [%(name)s] >> %(message)s")
    
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    file_handler = logging.FileHandler(os.path.join(log_dir, log_file))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger