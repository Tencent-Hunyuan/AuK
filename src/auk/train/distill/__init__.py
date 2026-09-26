"""Two-stage distillation for AuK.

Stage one (initializer) trains a few-step consistency student from the
teacher; stage two (dmd) runs the three-role DMD objective, warm-started
from that artifact:

    auk.train.distill.initializer.train   # stage one
    auk.train.distill.dmd.train           # stage two
"""

__all__: list[str] = []
