from src.algos.base import BaseAlgorithm
from src.algos.baseline_utils import future_origin_scores, solve_rebalance_to_scores


class INF(BaseAlgorithm):
    def __init__(self, **kwargs):
        """
        :param cplexpath: Path to the CPLEX solver.
        """

        self.max_reb = kwargs.get('max_reb')
        self.roh = kwargs.get('roh')
        self.cplexpath = kwargs.get("cplexpath")
        self.directory = kwargs.get("directory")
        self.horizon = int(kwargs.get("horizon", 10))
    
    def select_action(self, env):
        scores = future_origin_scores(env, horizon=self.horizon, start_offset=1, price_weight=False)
        return solve_rebalance_to_scores(env, self.directory, self.cplexpath, scores)
