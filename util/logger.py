import logging

import os

# 4월4일 logger 파일 함수 수정
def get_logger(log_file='experiment.log', log_dir='./logs'):
    logger = logging.getLogger(name='DPS')
    logger.setLevel(logging.INFO)
    
    # 로그 포맷터 설정
    formatter = logging.Formatter("%(asctime)s [%(name)s] >> %(message)s")
    
    # 스트림 핸들러 설정
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    
    # 로그 디렉토리 생성 (존재하지 않는 경우)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # 파일 핸들러 설정
    file_handler = logging.FileHandler(os.path.join(log_dir, log_file))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger