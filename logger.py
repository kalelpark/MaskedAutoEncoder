import sys

class Logger(object):
    def __init__(self, local_rank = 0, no_save = False):
        self.terminal = sys.stdout
        self.file = None
        self.local_rank = local_rank
        self.no_save = no_save
    
    def open(self, fp, mode = None):
        if mode is None:
            mode = "a"
        if self.local_rank == 0:
            self.file = open(fp, mode)
    
    def write(self, msg, is_terminal = 1, is_file = 1):
        if msg[-1] != "\n":
            msg += "\n"
        
        if self.local_rank == 0:
            if "\r" in msg:
                is_file = 0
            
            if is_terminal == 1:
                self.terminal.write(msg)
                self.terminal.flush()
            if is_file == 1 and not self.no_save:
                self.file.write(msg)
                self.file.flush()

    def flush(self):
        pass
            
# https://rlawjdghek.github.io/pytorch%20&%20tensorflow%20&%20coding/utils/

logger = Logger(local_rank = 0)
logger.open("logging.txt")
# logger.write("hello")
logger.write("hello1")
# logger.write("hello2")