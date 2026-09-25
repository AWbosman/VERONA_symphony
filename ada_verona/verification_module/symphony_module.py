# Copyright 2025 ADA Reseach Group and VERONA council. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import logging
import re
from pathlib import Path
import os, time
import subprocess
import sys
import numpy as np
from result import Err, Ok
from typing import Optional, Tuple


from ada_verona.database.verification_context import VerificationContext
from ada_verona.database.verification_result import CompleteVerificationData,  CompleteVerificationResult
from ada_verona.verification_module.verification_module import VerificationModule

logger = logging.getLogger(__name__)


_re_time = re.compile(r"^\s*Total User Time\s+([0-9]*\.?[0-9]+)\s*$", re.MULTILINE)
_re_found = re.compile(r"Solution Found:\s*Node\s+(\d+),\s*Level\s+(\d+)")
_re_infeas = re.compile(r"Infeasible", re.IGNORECASE)
_re_timelimit = re.compile(r"Time Limit Reached", re.IGNORECASE)


class SymphonyModule(VerificationModule):
    """
    A module for automatically verifying the robustness of a model using a specified verifier.
    """

    def __init__(self,timeout: float, symphony_path: Path = None, config: Path = None) -> None:
        """
        Initialize the SymphonyModule with a specific verifier, timeout, and optional configuration.
        Args:
            timeout (float): The timeout for the verification process.
            config (Path, optional): The configuration file for the verifier.
        """

        self.timeout = timeout
        self.config = config
        self.symphony_path = symphony_path
        self.name = f"Symphony" 
        
    def reformulate_mps_to_bigM(self, mps_path):
        subprocess.run(
            [sys.executable, "/home/annelot/WARMSTART_PROJECT/pipeline/mps_reformulate.py", mps_path, mps_path],
            check=True,
        )

    def create_mps_using_marabou(self, verification_context: VerificationContext,vnnlib, epsilon):
        env = os.environ.copy()
        marabou_path = Path("/home/annelot/Marabou/Marabou")
        mps_path = Path(verification_context.tmp_path/f"{verification_context.network.name}_{verification_context.data_point.id}_{str(epsilon).replace('.', '_')}.mps")

        vnnlib_path = verification_context.tmp_path / f"{vnnlib.name}.vnnlib"

        start = time.time()
        marabou_cmd = [
        str(marabou_path),
        str(verification_context.network.path),
        str(vnnlib_path),
        "--milp",
        "--dump-mps",
        "--dump-mps-path", str(mps_path),
        ]

        result = subprocess.run(
            marabou_cmd,
            capture_output=True,
            text=True,
            env=env,
            timeout=self.timeout,
            cwd=str(marabou_path.parent) 
        )
        took = time.time() - start
        stdout = result.stdout or ""
        stderr = result.stderr or ""
        log = (stdout + "\n" + stderr).lower()

        if "unsat" in log:
            return CompleteVerificationData(result="UNSAT", took=took, err=stderr, stdout=stdout)
        if "sat" in log:
            return CompleteVerificationData(result="SAT", took=took, err=stderr, stdout=stdout)
        
        if not mps_path.exists():
            logging.error("Marabou reported success but MPS not found at: %s\nSTDOUT:\n%s\nSTDERR:\n%s",
                        mps_path, result.stdout, result.stderr)
            raise FileNotFoundError(mps_path)
       
        self.reformulate_mps_to_bigM(mps_path) 
        return mps_path
    
    
    def parse_symphony_log(self,text: str) -> Tuple[
            Optional[float],
            bool,
            bool,
            bool,
        ]:
    
        user_time: Optional[float] = None
        solution_found = False
        infeasible = False
        time_limit = False

        for line in text.splitlines():
            if _re_found.search(line):
                solution_found = True
            if _re_infeas.search(line):
                infeasible = True
            if _re_timelimit.search(line):
                time_limit = True

            m = _re_time.search(line)
            if m:
                user_time = float(m.group(1))


        return user_time, solution_found, infeasible, time_limit

    
    def run_symphony_script(self,symphony_script, mps_path) -> CompleteVerificationData:
        symphony_script = Path(symphony_script)
        symphony_cmd = [str(symphony_script), "-F", str(mps_path)]
        env = os.environ.copy()

        start = time.time()
        try:
            result = subprocess.run(
                symphony_cmd,
                capture_output=True,
                text=True,
                env=env,
                timeout=self.timeout,
                cwd=str(symphony_script.parent)
            )
        except subprocess.TimeoutExpired as e:
            return CompleteVerificationData(
                result="TIMEOUT",
                took=time.time() - start,
                err=e.stderr or "",
                stdout=e.stdout or "",
            )


        user_times, solution_found, infeasible, time_limit = self.parse_symphony_log(result.stdout)
        print(user_times, solution_found, infeasible, time_limit)

        if solution_found:
            final= "SAT"
        elif infeasible:
            final = "UNSAT"
        elif time_limit:
            final = "TIMEOUT"
        elif result.returncode != 0:
            final = "ERR"
        else:
            final = "ERR"

        took = user_times 

        return CompleteVerificationData(
            result=final,
            took=took,
            err=result.stderr,
            stdout=result.stdout,
        )
        
        
    def verify(self, verification_context: VerificationContext, epsilon: float) -> str | CompleteVerificationData:
        """
        Verify the robustness of the model within the given epsilon perturbation.
        Args:
            verification_context (VerificationContext): The context for verification,
            including the model and data point.
            epsilon (float): The perturbation magnitude for the attack.

        Returns:
            str | CompleteVerificationData: The result of the verification,
            either SAT or UNSAT, along with the duration.
        """
        image = verification_context.data_point.data.reshape(-1).detach().numpy()
        vnnlib_property = verification_context.property_generator.create_vnnlib_property(
            image, verification_context.data_point.label, epsilon
        )

        verification_context.save_vnnlib_property(vnnlib_property)
        
        mps_path = self.create_mps_using_marabou(verification_context,vnnlib_property, epsilon)
        
        if isinstance(mps_path, CompleteVerificationData):
            return mps_path

        if self.config:
            raise NotImplementedError("TODO: implement with configurations")
        else:
           return self.run_symphony_script(self.symphony_path,mps_path)
            
       

def parse_counter_example(result: Ok, verification_context: VerificationContext) -> np.ndarray:
    """
    Parse the counter example from the verification result.

    Args:
        result (Ok): The verification result containing the counter example.

    Returns:
        np.ndarray: The parsed counter example as a numpy array.
    """
    string_list_without_sat = [x for x in result.unwrap().counter_example.split("\n") if "sat" not in x]
    numbers = [x.replace("(", "").replace(")", "") for x in string_list_without_sat if "Y" not in x]
    counter_example_array = np.array([float(re.sub(r'X_\d*', '', x).strip()) for x in numbers if x.strip()])
    

    return counter_example_array.reshape(verification_context.data_point.data.shape)

def parse_counter_example_label(result: Ok) -> int:
    """
    Parse the counter example label from the verification result.

    Args:
        result (Ok): The verification result containing the counter example.

    Returns:
        int: The parsed counter example label.
    """
    string_list_without_sat = [x for x in result.unwrap().counter_example.split("\n") if "sat" not in x]
    numbers = [x.replace("(", "").replace(")", "") for x in string_list_without_sat if "X" not in x]
    counter_example_array = np.array([float(re.sub(r'Y_\d*', '', x).strip()) for x in numbers if x.strip()])

    return int(np.argmax(counter_example_array))