import json
import logging

import hyperopt
from churnchall.constants import TUNING_DIR
from wax_toolbox import Timer

logger = logging.getLogger(__name__)


class HyperParamsTuningMixin:
    """Base class for hyper parameters tuning using hyperopt."""

    int_params = ()
    float_params = ()

    @property
    def hypertuning_space(self):
        raise NotImplementedError

    @property
    def cv(self):
        raise NotImplementedError

    def _ensure_type_params(self, params):
        """Sanitize params according to their type."""
        for k in self.int_params:
            if k in params:
                params[k] = int(params[k])

        for k in self.float_params:
            if k in params:
                params[k] = round(params[k], 3)

        return params

    def _hypertuning_save_results(self, best_params, trials):
        # Store eval_hist for best score:
        fpath = TUNING_DIR / "eval_hist_best_score.json"
        print("Saving {}".format(fpath))

        # Best score idx:
        best_score = 0
        for i, d in enumerate(trials.results):
            if best_score < d["loss"]:
                idx = i
            best_score = max(best_score, d["loss"])

        with open(fpath, "w") as file:
            eval_hist = trials.trial_attachments(
                trials.trials[idx])["eval_hist"]
            file.write(json.dumps(eval_hist))

        fpath = TUNING_DIR / "best_params.json"
        print("Saving {}".format(fpath))
        with open(fpath, "w") as file:
            file.write(json.dumps(best_params))

    def hypertuning_objective(self, params):
        params = self._ensure_type_params(params)
        msg = "-- HyperOpt -- CV with {}\n".format(params)
        params = {
            **self.common_params,
            **params
        }  # recombine with common params

        # Fix learning rate:
        params["learning_rate"] = 0.04

        with Timer(msg, at_enter=True):
            eval_hist = self.cv(params_model=params, nfold=5)

        metric_name_mean = "{}-mean".format(self.metric_name)
        score = max(eval_hist[metric_name_mean])

        print("{}: {}".format(self.metric_name, score))

        result = {
            "loss": score,
            "status": hyperopt.STATUS_OK,
            # -- store other results like this
            # "eval_time": time.time(),
            # 'other_stuff': {'type': None, 'value': [0, 1, 2]},
            # -- attachments are handled differently
            "attachments": {
                "eval_hist": eval_hist
            },
        }

        return result

    def tuning(self, max_evals=3, metric_name='AUC Lift'):
        trials = hyperopt.Trials()
        self.metric_name = metric_name

        # https://github.com/hyperopt/hyperopt/wiki/FMin
        best_params = hyperopt.fmin(
            fn=self.hypertuning_objective,
            space=self.hypertuning_space,
            algo=hyperopt.tpe.suggest,
            max_evals=max_evals,
            trials=trials,  # store results
        )

        # Save some results:
        self._hypertuning_save_results(best_params, trials)

        return best_params
