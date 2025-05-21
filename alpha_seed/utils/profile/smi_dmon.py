import subprocess
import threading
import traceback
from datetime import datetime
import socket
from typing import Optional, List

from alpha_seed.utils.profile.timeline import CounterEvent, Tracer, CombinedEvents


class NvidiaSmiDmon(object):

    def __init__(self, interval: int = 1, index: Optional[int] = None):
        self.last_buffer = {}  # gpu_id -> ('power', 'temp', 'util', 'mem_util', 'pclk')
        self.running = False
        self.mutex = threading.Lock()
        self.interval = interval  # interval seconds
        self.gpu_index = index

    def start(self):
        if self.running:
            return

        self.running = True
        t = threading.Thread(target=self.feed)
        t.daemon = True
        t.start()

    def stop(self):
        self.running = False

    def feed(self):
        cmd = ['nvidia-smi', 'dmon', '--delay', f'{self.interval}sec']
        if self.gpu_index is not None:
            cmd += ['-i', f'{self.gpu_index}']
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        while self.running:
            try:
                l = proc.stdout.readline().decode().rstrip()
                if not l:
                    break
                if l.startswith('#'):
                    continue
                l = [word for word in l.split(' ') if word]
                if len(l) != 10:
                    # should be 10 fields
                    continue
                gpu_id, power, temp, _, util, mem_util, enc, dec, mclk, pclk = l
                gpu_id = int(gpu_id)
                with self.mutex:
                    self.last_buffer[gpu_id] = (power, temp, util, mem_util, pclk)
            except:
                traceback.print_exc()
                proc.terminate()
                return
        proc.terminate()

    def get(self, gpu_id=0):
        with self.mutex:
            if gpu_id not in self.last_buffer:
                return tuple([''] * 5)
            return self.last_buffer[gpu_id]


DEFAULT_QUERY_FIELDS = [
    'index',
    'timestamp',
    'utilization.gpu',
    'utilization.memory',
    'power.draw',
    'clocks.sm',
    'temperature.gpu',
    'memory.used',
]


class NvidiaSmiQueryGPUTracer(object):

    def __init__(self, interval_ms: int, query_fields: List[str] = None):
        self.interval_ms = interval_ms
        self.query_fields = query_fields or DEFAULT_QUERY_FIELDS
        if 'timestamp' not in self.query_fields:
            self.query_fields.insert(0, 'timestamp')
        if 'index' not in self.query_fields:
            self.query_fields.insert(0, 'index')
        self.running = False
        self.stopped = threading.Event()
        self.smi_proc = None

    def start(self):
        if self.running:
            return

        self.running = True
        t = threading.Thread(target=self.feed, name='nvidia-smi-dmon-feed')
        t.daemon = True
        t.start()

    def stop(self):
        if self.smi_proc is not None:
            try:
                self.smi_proc.terminate()
            except:
                pass
        self.running = False
        # wait at most 2 seconds
        self.stopped.wait(timeout=2)

    def feed(self):
        hostname = self._hostname()
        tracer = Tracer.get_instance()
        query_fields_str = ','.join(self.query_fields)
        cmd = [
            'nvidia-smi', '--format=csv,nounits,noheader', '-lms', f'{self.interval_ms}',
            f'--query-gpu={query_fields_str}'
        ]

        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

        events = []
        while self.running:
            try:
                l = proc.stdout.readline().decode().strip()
                if not l:
                    break
                l = [word.strip() for word in l.split(',')]
                if len(l) != len(self.query_fields):
                    continue
                metrics = dict(zip(self.query_fields, l))
                datetime_str = metrics.pop('timestamp')
                gpu_index = metrics.pop('index')
                ts_micro = datetime.strptime(datetime_str, '%Y/%m/%d %H:%M:%S.%f').timestamp() * 1e6
                metrics = {k: float(v) for k, v in metrics.items()}
                for k, v in metrics.items():
                    e = CounterEvent(name=f'gpu({gpu_index}) {k}', pid=hostname, ts=ts_micro, data={'current': v})
                    events.append(e)
                # flush every small batch
                if len(events) > 64:
                    tracer.trace(CombinedEvents(events))
                    events = []
            except:
                traceback.print_exc()
                proc.terminate()
                break
        try:
            proc.terminate()
        except:
            pass

        tracer.trace(CombinedEvents(events))
        self.stopped.set()

    def _hostname(self):
        try:
            # try ipv4 first
            return socket.gethostbyname(socket.gethostname())
        except socket.gaierror:
            # 获取主机名
            host = socket.gethostname()
            # 使用 getaddrinfo 获取地址信息
            addr_info = socket.getaddrinfo(host, None, socket.AF_INET6)
            # 提取第一个 IPv6 地址
            ipv6_address = addr_info[0][4][0]
            return ipv6_address
