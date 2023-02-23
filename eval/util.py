import sys
import os
import json
import numpy as np
import jax.numpy as jnp
""" Custom Logger """
import sys


class Logger:
    def __init__(self, filename):
        self.console = sys.stdout
        self.file = open(filename, 'w')

    def write(self, message):
        self.console.write(message)
        self.file.write(message)

    def flush(self):
        self.console.flush()
        self.file.flush()


""" Async executer """
import multiprocessing


class AsyncExecutor:
    def __init__(self, n_jobs=1):
        self.num_workers = n_jobs if n_jobs > 0 else multiprocessing.cpu_count(
        )
        self._pool = []
        self._populate_pool()

    def run(self, target, *args_iter, verbose=False):
        workers_idle = [False] * self.num_workers
        tasks = list(zip(*args_iter))
        n_tasks = len(tasks)

        while not all(workers_idle):
            for i in range(self.num_workers):
                if not self._pool[i].is_alive():
                    self._pool[i].terminate()
                    if len(tasks) > 0:
                        if verbose:
                            print(n_tasks - len(tasks))
                        next_task = tasks.pop(0)
                        self._pool[i] = _start_process(target, next_task)
                    else:
                        workers_idle[i] = True

    def _populate_pool(self):
        self._pool = [
            _start_process(_dummy_fun) for _ in range(self.num_workers)
        ]


def _start_process(target, args=None):
    if args:
        p = multiprocessing.Process(target=target, args=args)
    else:
        p = multiprocessing.Process(target=target)
    p.start()
    return p


def _dummy_fun():
    pass


""" Command generators """


def generate_base_command(module, flags=None, unbuffered=True):
    """ Module is a python file to execute """
    interpreter_script = sys.executable
    base_exp_script = os.path.abspath(module.__file__)
    if unbuffered:
        base_cmd = interpreter_script + ' -u ' + base_exp_script
    else:
        base_cmd = interpreter_script + ' ' + base_exp_script
    if flags is not None:
        assert isinstance(flags, dict), "Flags must be provided as dict"
        for flag, setting in flags.items():
            if type(setting) == bool or type(setting) == np.bool_:
                if setting:
                    base_cmd += f" --{flag}"
            else:
                base_cmd += f" --{flag}={setting}"
    return base_cmd


def generate_run_commands(command_list,
                          num_cpus=1,
                          num_gpus=0,
                          dry=False,
                          n_hosts=1,
                          mem=6000,
                          long=False,
                          mode='local',
                          promt=True,
                          output_file=None):

    if mode == 'local':
        if promt:
            answer = input(
                f"About to run {len(command_list)} jobs in a loop. Proceed? [yes/no]"
            )
        else:
            answer = 'yes'

        if answer == 'yes':
            for cmd in command_list:
                if dry:
                    print(cmd)
                else:
                    os.system(cmd)

    elif mode == 'local_async':
        if promt:
            answer = input(
                f"About to launch {len(command_list)} commands in {num_cpus} local processes. Proceed? [yes/no]"
            )
        else:
            answer = 'yes'

        if answer == 'yes':
            if dry:
                for cmd in command_list:
                    print(cmd)
            else:
                exec = AsyncExecutor(n_jobs=num_cpus)
                cmd_exec_fun = lambda cmd: os.system(cmd)
                exec.run(cmd_exec_fun, command_list)
    else:
        raise NotImplementedError


""" Hashing and Encoding dicts to JSON """


class NumpyArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer) or isinstance(obj, jnp.integer):
            return int(obj)
        elif isinstance(obj, np.floating) or isinstance(obj, jnp.floating):
            return float(obj)
        elif isinstance(obj, np.bool_) or isinstance(obj, jnp.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray) or isinstance(obj, jnp.ndarray):
            return obj.tolist()
        else:
            return super(NumpyArrayEncoder, self).default(obj)


def hash_dict(d):
    return str(
        abs(json.dumps(d, sort_keys=True, cls=NumpyArrayEncoder).__hash__()))
