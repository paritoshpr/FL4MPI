# This project enables large scale Federated Learning for HPC using MPI
# Copyright (C) 2022 Paritosh Ramanan
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

from mpi4py import MPI
from ConfigReader import ConfigReader
from Learner import Learner
import numpy as np
mpiObj = MPI.COMM_WORLD
device_id = mpiObj.Get_rank()  #the individual rank number
num_devices = mpiObj.Get_size() #the number of epsilon X instances

cr = ConfigReader()
learner = Learner(device_id,num_devices)

for rounds in range(cr.max_rounds):

  #aggregate using MPI here
  mpiObj.allreduce(learner.model_weight_buffer, op=MPI.SUM)

  learner.setGlobalWeight()

