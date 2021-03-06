from search.objective import obj_func
from samplers.markov_jump_hmc import ControlHMC
from misc.distributions import MultimodalGaussian

def main(job_id, params):
    print "job id: {}, params: {}".format(job_id, params)
    return obj_func(ControlHMC,
                    MultimodalGaussian(ndims=5, separation=1),
                    job_id,
                    **params)