"""frost-night-forecaster: spring frost early warning for vineyards and orchards."""

import os

# The models are small (a few thousand rows): one OpenMP thread is fastest and avoids oversubscription
# when other jobs share the machine. Must be set before scikit-learn is imported.
os.environ.setdefault("OMP_NUM_THREADS", "1")

__version__ = "0.1.0"
