from pathlib import Path
from result import Err, Ok
import numpy as np
import re
import logging
logger = logging.getLogger(__name__)
import subprocess
import string
import time
import tempfile
from typing import IO
import os
import threading
import signal 
from result import Err, Ok


from ada_verona import VerificationContext
from ada_verona.verification_module.abcrown_yaml_config import AbcrownYamlConfig

from ada_verona.database.verification_result import (
    CompleteVerificationResult,
    VerificationResultString,
    CompleteVerificationData
)

_tempfiles_to_clean: list[str] = []

def _reg_for_clean(f: str | Path):
    _tempfiles_to_clean.append(str(f))

def tmp_file(extension: str) -> IO[str]:
    """Return a new tempfile with the given extension."""
    f = tempfile.NamedTemporaryFile("w", suffix=extension, delete=False)
    _reg_for_clean(f.name)
    return f

# Credits: @jfs
def pid_exists(pid: int) -> bool:  # pragma: no cover
    """Returns if the given PID exists."""
    if pid < 0:
        return False  # NOTE: pid == 0 returns True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:  # errno.ESRCH
        return False  # No such process
    except PermissionError:  # errno.EPERM
        return True  # Operation not permitted (i.e., process exists)
    else:
        return True  # no error, we can send a signal to the process


class abcrownModule():

    def __init__(self, timeout: float, config: Path = None) -> None:
        self.timeout = timeout
        self.config = config
        self.name = "abCrown"
    

    def _get_run_cmd(self):
        return f"python /home/annelot/alpha-beta-CROWN/complete_verifier/abcrown.py --config {str(self.config)}"

    def verify(self, verification_context: VerificationContext, epsilon: float) -> str| CompleteVerificationData: 

        image  = verification_context.data_point.data.reshape(-1).detach().numpy()
        vnnlib_property = verification_context.property_generator.create_vnnlib_property(image, verification_context.data_point.label, epsilon)
        verification_context.save_vnnlib_property(vnnlib_property)

        if not self.config:
           raise NotImplementedError
           self.config = create_config_file(verification_context.network.path, vnnlib_property.path,timeout=self.timeout)
        else:
            self.config = update_config_file(self.config, verification_context.network.path, vnnlib_property.path,timeout=self.timeout)



        #make different definition for this. 
        run_cmd = self._get_run_cmd()
        output_lines: list[str] = []
        result_file = Path(tmp_file(".txt").name)
        process = subprocess.Popen(
                run_cmd,
                executable="/bin/bash",
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                shell=True,
                universal_newlines=True,
                preexec_fn=os.setsid,
                encoding="utf-8",  # Add this line
                errors="replace",
            )
        
        logging.info(f"{process.stderr} ")

        before_t = time.time()
        self._timeout_event: threading.Event | None = threading.Event()

        def _terminate(timeout_sec):
            assert self._timeout_event
            on_time = self._timeout_event.wait(timeout_sec)

            if not on_time:
                global result
                result = "TIMEOUT"  # type: ignore

            if pid_exists(process.pid):
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)

        t = threading.Thread(target=_terminate, args=[self.timeout])
        t.start()

        assert process.stdout
        logging.info(f"{process.stderr} ")
        logging.info(f"{process.stdout} ")

        for line in iter(lambda: process.stdout.readline().encode("utf-8", errors="replace"), b""):
            output_lines.append(line.decode("utf-8", errors="replace"))


        process.stdout.close()
        return_code = process.wait()
        took_t = time.time() - before_t
        self._timeout_event.set()

        output_str = "".join(output_lines)

    
        if return_code > 0:
            result = "ERR"
        else:
            data_result = self._create_completeverificationdata(result_file = result_file, took_t = took_t, output_str=output_str)
            return data_result
        

        counter_example: str | None = None
        return CompleteVerificationData(
                result,  # type: ignore
                took_t,
                counter_example,
                "",  # TODO: Remove err field; its piped it to stdout
                output_str,
            )
   

    def _parse_result(
            self,
            output: str,
            result_file: Path | None,
    ) -> tuple[VerificationResultString, str | None]:
            if find_substring("Result: sat", output):
                with open(str(result_file), "r") as f:
                    counter_example = f.read()

                return "SAT", counter_example
            elif find_substring("Result: unsat", output):
                return "UNSAT", None
            elif find_substring("Result: timeout", output):
                return "TIMEOUT", None

            return "TIMEOUT", None

    def _create_completeverificationdata(self, result_file:Path, took_t:float, output_str:str)-> str|CompleteVerificationData:
        result,counter_example = self._parse_result(output_str,result_file)

        if result == "TIMEOUT":
            took_t = self.timeout

        return CompleteVerificationData(
              result,  # type: ignore
                took_t,
                counter_example,
                "",  # TODO: Remove err field; its piped it to stdout
                output_str,
        )



def parse_counter_example(result: Ok):
    string_list_without_sat = [x for x in result.unwrap().counter_example.split("\n") if not "sat" in x]
    numbers = [x.replace("(", "").replace(")", "") for x in string_list_without_sat if "Y" not in x]
    counter_example_array = np.array([float(re.sub(r'X_\d*', '', x).strip()) for x in numbers])

    return counter_example_array.reshape(28,28)

def parse_counter_example_label(result: Ok):
    string_list_without_sat = [x for x in result.unwrap().counter_example.split("\n") if not "sat" in x]
    numbers = [x.replace("(", "").replace(")", "") for x in string_list_without_sat if "X" not in x]
    counter_example_array = np.array([float(re.sub(r'Y_\d*', '', x).strip()) for x in numbers])

    return np.argmax(counter_example_array)

# def create_config_file(network, property, timeout) -> Path:
#     config = from_scratch(network, property, timeout)
#     return config 

def update_config_file(config, network, property, timeout) -> Path:
    return AbcrownYamlConfig.from_yaml(config, network, property, timeout).get_yaml_file_path()

# credits: @aaronasterling
def find_substring(needle: str, haystack: str) -> bool:
    """Finds a whole word in a substring.

    Args:
        needle: The word to find.
        haystack: The text to find the word in.

    Returns:
        bool: True if the whole word was found, false otherwise.
    """
    index = haystack.find(needle)

    if index == -1:
        return False

    if index != 0 and haystack[index - 1] not in string.whitespace:
        return False

    L = index + len(needle)
    if L < len(haystack) and haystack[L] not in string.whitespace:
        return False

    return True