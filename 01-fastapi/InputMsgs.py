# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 17:22:04 2021

@author: acer
"""

from pydantic import BaseModel

class InputMsg(BaseModel):
    msg: str