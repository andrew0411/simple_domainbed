'''
Singleton pattern Logger
    : 클래스가 단 하나의 인스턴스만 가지게 제한
    : 중요한 자원을 관리할 때 다수의 인스턴스가 생기지 않게 하기 위함
'''
import sys
import logging


def levelize(levelname):
    if isinstance(levelname, str):
        return logging.getLevelName(levelname)
    else:
        return levelname


class ColorFormatter(logging.Formatter):
    color_dic = {
        'INFO': 36,  #cyan
        'WARNING' : 33,
        'ERROR': 31,
        'CRITICAL' : 41
    }

    def format(self, record):
        color = self.color_dic.get(record.levelname, 37) # default : white
        record.levelname = f'\033[{color}m{record.levelname}\033[0m'
        return logging.Formatter.format(self, record)


class Logger(logging.Logger):
    NAME = 'SingletonLogger'

    @classmethod
    def get(cls, file_path=None, level='INFO', track_code=False):
        logging.setLoggerClass(cls)
        logger = logging.getLogger(cls.NAME)
        logging.setLoggerClass(logging.Logger)
        logger.setLevel(level)

        if logger.hasHandlers(): # 만약 처리기가 있고, 그 수가 두개라면 logger 반환, 아니면 처리기 reset
            if len(logger.handlers) == 2:
                return logger
            
            logger.handlers.clear()

        log_format = '%(levelname)s %(asctime)s | %(message)s'
        if track_code:
            log_format = (
                '%(levelname)s::%(asctime)s | [%(filename)s] [%(funcName)s:%(lineno)d] '
                '%(message)s'
            )
        
        date_format = '%m/%d %H:%M:%S'
        formatter = ColorFormatter(log_format, date_format)

        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        if file_path:
            file_handler = logging.FileHandler(file_path)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        logger.propagate = False # 로깅 메세지가 조상 처리기로 전달되지 않게

        return logger


    def nofmt(self, msg, *args, level='INFO', **kwargs):
        level = levelize(level)
        formatters = self.remove_formats()
        super().log(level, msg, *args, **kwargs)
        self.set_formats(formatters)


    def remove_formats(self):
        '''
        Logger의 모든 format을 삭제
        '''
        formatters = []
        for handler in self.handlers:
            formatters.append(handler.formatter)
            handler.setFormatter(logging.Formatter('%(message)s'))

        return formatters


    def set_formats(self, formatters):
        '''
        Logger의 모든 처리기에 format을 설정
        '''
        for handler, formatter in zip(self.handlers, formatters):
            handler.setFormatter(formatter)

    
    def set_file_handler(self, file_path):
        file_handler = logging.FileHandler(file_path)
        formatter = self.handlers[0].formatter
        file_handler.setFormatter(formatter)
        self.addHandler(file_handler)


