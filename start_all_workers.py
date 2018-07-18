from subprocess import Popen
from time import sleep
import os
import signal

WORKER_SIZE = 2


if __name__ == "__main__":

    cmd_str_list = []
    proc_pool = []

    for i in range(WORKER_SIZE): 
        cmd_str = ["python","/root/server/worker.py","/gpu:"+str(i)]
        cmd_str_list.append(cmd_str)

    for j in range(WORKER_SIZE):
        proc = Popen(cmd_str_list[j], shell=False,
                     stdin=None, stdout=None, stderr=None, close_fds=True)
        proc_pool.append(proc)

    
    # os.killpg(os.getpgid(proc_pool[0].pid), signal.SIGTERM)

    try:
        while True:
            for l in range(WORKER_SIZE):
                if proc_pool[l].poll() is not None:
                    proc_pool[l] = Popen(cmd_str_list[l], shell=False,
                                        stdin=None, stdout=None, stderr=None, close_fds=True)
            sleep(60)
    except KeyboardInterrupt:
        for k in range(WORKER_SIZE):
            os.killpg(os.getpgid(proc_pool[k].pid), signal.SIGTERM)
        print('KeyboardInterrupt!')
        exit(1)
