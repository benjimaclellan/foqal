from foqal.model import ModelBase


class GeneralizedProbabilityTheory(ModelBase):

    def __init__(self, num_states: int, num_effects: int, rank: int, **kwargs):
        """

        :param kwargs:
        """
        super().__init__(**kwargs)
        self.num_states = num_states
        self.num_effects = num_effects
        self.rank = rank

        self.terms = {
            "S": dict(
                shape=(self.num_states, self.rank),
                bounds=(-1, 1),
            ),
            "E": dict(
                shape=(self.num_effects, self.rank),
                bounds=(-1, 1),
            ),
        }

        self.initialize_params()
        return

    def forward(self):
        self.clip_params()
        pred = self.params["S"] @ self.params["E"]
        return pred
