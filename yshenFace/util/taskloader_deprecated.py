import zmq

class taskloader:
    def __init__(self, socket):
        self.task_socket = socket
    
    def __iter__(self):
        return self
    
    def __next__(self):
        return self.next()
        
    def next(self):
        ret = self.get_batch()
        if ret == 0:
            return self.data
        else:
            raise StopIteration

    def get_batch(self):
        try:
            self.data = self.task_socket.recv_string()
        except:
            return -1

        return 0

