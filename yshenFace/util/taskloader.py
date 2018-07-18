import zmq

class taskloader:
    def __init__(self, socket, batch_size=1):
        self.task_socket = socket
        self.batch_size = batch_size
        self.curindex = 0
    
    def __iter__(self):
        return self
    
    def __next__(self):
        return self.next()
        
    def next(self):
        if self.curindex < self.batch_size:
            ret = self.get_batch()
            if ret == 0:
                return self.data
            else:
                raise StopIteration
        else:
            self.curindex = 0
            raise StopIteration

    def get_batch(self):
        try:
            self.data = self.task_socket.recv_string(zmq.NOBLOCK)
            self.curindex += 1
        except:
            return -1

        return 0


