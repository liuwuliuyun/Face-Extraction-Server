from subprocess import Popen
from time import sleep
import os
import signal

cmd_str_1 = "python /root/server/worker_0.py"

cmd_str_2 = "python /root/server/worker_1.py"

proc_1 = Popen([cmd_str_1], shell=True,
             stdin=None, stdout=None, stderr=None, close_fds=True)

proc_2 = Popen([cmd_str_2], shell=True,
             stdin=None, stdout=None, stderr=None, close_fds=True)

try:
    while True:
        sleep(1)
except KeyboardInterrrupt:
    os.killpg(os.getpgid(proc_1.pid), signal.SIGTERM)
    os.killpg(os.getpgid(proc_2.pid), signal.SIGTERM)
    print('KeyboardInterrupt!')
    exit(1)

