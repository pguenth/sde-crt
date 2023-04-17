import time
from threading import Thread
from multiprocessing import connection

class Supervisor:
    """
    Collects information from threads via pipes and prints the collected information regularly.

    :param logger: the logger on which to print on.
    :type logger: `logging.Logger`

    :param log_every: amount of seconds between each log entry.
    :type log_every: `float`

    :param fields_cumulate: names of fields in the submitted data that should be cumulated during the runtime.
    :type fields_cumulate: `list` of `str`. 

    :param fields_rates: names of cumulated fields that a rate (increase per second) should be calculated for.
    :type fields_rates: `list` of `str`.

    :param fields_total: names of fields in the submitted data that summed up for all threads.
    :type fields_total: `list` of `str`. Default: same as `fields_cumulate`

    :param fields_total_rates: names of fields of totals that a rate (increase per second) should be calculated for.
    :type fields_total_rates: `list` of `str`. Default: same as `fields_rates` 
    """
    def __init__(self, logger=None, log_every=3, fields_cumulate=None, fields_rates=None, fields_total=None, fields_total_rates=None):
        self.pipes = []
        self.threads = {}

        if logger is None:
            import logging
            self.logger = logging.getLogger("Supervisor")
        else:
            self.logger = logger

        self._interrupt = False
        self.log_every = log_every
        self._last_log_time = time.perf_counter()

        if fields_cumulate is None:
            fields_cumulate = ['time_cpp', 'time_phys', 'particles', 'splits']
        self.fields_cumulate = fields_cumulate

        if fields_rates is None:
            fields_rates = ['time_phys', 'particles', 'splits']
        self.fields_rates = fields_rates

        if fields_total is None:
            fields_total = self.fields_cumulate
        self.fields_total = fields_total

        if fields_total_rates is None:
            fields_total_rates = self.fields_rates
        self.fields_total_rates = fields_total_rates

        self.totals = {t: 0 for t in self.fields_total}

    def attach(self, pipe):
        self.pipes.append(pipe)

    def store_data(self, data):
        """
        store retrieved data thread-wise
        """
        thread_id = data['thread_id']
        if not thread_id in self.threads:
            self.threads[thread_id] = {c : 0 for c in self.fields_cumulate} 

        for c in self.fields_cumulate:
            self.threads[thread_id][c] += data[c]

        # rates are updated in log to save computational time

    def _update_totals(self):
        """
        update the total values (across all threads)
        """
        for k in self.fields_total:
            self.totals[k] = 0

        for p, info in self.threads.items():
            for e in self.fields_total:
                self.totals[e] += info[e]

        self._update_rates(self.fields_total_rates, self.totals)

    def _get_current_rate(self, current_value, last_state=None):
        """
        calculates the current rate of some extensive variable

        last_state: the tuple returned in the last call or None on the first call. 
            (tuple(last_time, last_value))
        current_value: some number

        returns current particles per second and new last_state tuple
        """

        if last_state is None:
            last_state = (0, 0)

        tdiff = time.perf_counter() - last_state[0]
        if not tdiff == 0:
            pdiff = current_value - last_state[1]
            return pdiff / tdiff, (time.perf_counter(), current_value)
        else:
            return 0, (0, 0)

    def _update_rates(self, names, store_dict):
        """
        update the rates of entries stored in store_dict with key in names

        rates are called [NAME]_rate and a opaque object required for
        updating the rate is stored in [NAME]_rate_last_state
        """
        for i in names:
            ls_name = i + '_rate_last_state'
            if not ls_name in store_dict:
                store_dict[ls_name] = None

            rate, ls = self._get_current_rate(store_dict[i], store_dict[ls_name])
            store_dict[i + '_rate'] = rate
            store_dict[ls_name] = ls

    def _update_all_thread_rates(self):
        for thread_id in self.threads.keys():
            self._update_rates(self.fields_rates, self.threads[thread_id])

    def interrupt(self):
        self._interrupt = True

    def _data_loop(self):
        while not self._interrupt:
            if len(self.pipes) > 0:
                for p in connection.wait(self.pipes):
                    try:
                        data = p.recv()
                    except (EOFError):
                        p.close()
                        self.pipes.remove(p)
                    else:
                        self.store_data(data)

            time.sleep(0.1)

    def loop(self):
        """
        Starts the data recieve loop in a separate thread and then
        prints the log regularly.
        """
        self.logger.info("Supervisor started.")

        dt = Thread(target=self._data_loop)
        dt.start()

        while not self._interrupt:
            self.log()
            time.sleep(self.log_every)

        dt.join()

        self.log()
        self.logger.info("Supervisor has shut down.")
            
    def log(self):
        self._update_totals()
        self._update_all_thread_rates()
        self.logger.info(f"Supervisor: {len(self.pipes)} pipes attached and {len(self.threads)} threads tracked.")

        threadstr = ""
        for k in self.fields_cumulate:
            threadstr += k + ": {" + k + ":g}, "
        for r in self.fields_rates:
            threadstr += r + " per second: {" + r + "_rate:g}, "

        totalstr = ""
        for k in self.fields_total:
            totalstr += k + ": {" + k + ":g}, "
        for r in self.fields_total_rates:
            totalstr += r + " per second: {" + r + "_rate:g}, "

        for p, info in self.threads.items():
            self.logger.info(("Thread {p}: " + threadstr).format(p=p, **info))
        self.logger.info(("Total: " + totalstr).format(**self.totals))

    @property
    def data(self):
        self._update_totals()
        self._update_all_thread_rates()
        return self.threads, self.totals
