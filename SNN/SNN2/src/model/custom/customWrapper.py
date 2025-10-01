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
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# Copyright (C) 2020 Mattia Milani <mattia.milani@nokia.com>

import functools

from SNN2.src.model.custom.siamese import SiameseModel
from SNN2.src.decorators.decorators import models, cmodel

@cmodel
def none_customModel(*args, **kwargs) -> None:
    pass

def CustomModel_selector(obj, *args, **kwargs):
    if obj in models.keys():
        return models[obj](*args, **kwargs)
    else:
        raise ValueError(f"Model \"{obj}\" not available")


