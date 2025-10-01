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


from datetime import datetime
import tensorflow as tf
import pytest

from SNN2.src.io.commandInterpolation import fn as CI

class TestInterpolation():


    def test_dateTime(self) -> None:
        assert CI("!datetime") == datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
