import os
import sys
import itertools
import math
import logging
import json
import re
import random
from collections import OrderedDict
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as lines
from matplotlib.patches import Polygon
from mrcnn.config import Config

ROOT_DIR = os.getcwd()

sys.path.append(ROOT_DIR)
import mrcnn.utils as utils
import mrcnn.visualize as visualize
from mrcnn.visualize import display_images
import mrcnn.model as modellib

import soybean

config = soybean.SoyConfig()
SOYBEAN_DIR = os.path.join(ROOT_DIR, "datasets/soybean")

dataset = soybean.SoyDataset()
dataset.load_soybean(SOYBEAN_DIR, "train")

dataset.prepare()