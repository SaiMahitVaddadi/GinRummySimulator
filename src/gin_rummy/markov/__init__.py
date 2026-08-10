"""Markov machinery for the rummy simulator.

This subpackage bundles two complementary strands of stochastic-process
modelling:

* **Hidden Markov Models** (:mod:`gin_rummy.markov.hmm`,
  :mod:`gin_rummy.markov.opponent_hand_hmm`, :mod:`gin_rummy.markov.utils`)
  — pure-Python discrete-observation HMM plus a concrete application to
  latent opponent-hand inference.
* **General Markov chains, absorbing chains, deck-depletion processes,
  and probabilistic response cascades** (:mod:`gin_rummy.markov.chain`,
  :mod:`gin_rummy.markov.absorbing`, :mod:`gin_rummy.markov.deck_process`,
  :mod:`gin_rummy.markov.cascade`).

The two strands are independent: nothing in the general-chain code
depends on the HMM, and vice versa.
"""

from gin_rummy.markov.absorbing import AbsorbingChain, expected_turns_to_gin
from gin_rummy.markov.cascade import (
    CascadeState,
    CascadeStep,
    ResponseModel,
    cascade_value,
    greedy_response_model,
    probabilistic_cascade,
)
from gin_rummy.markov.chain import (
    MarkovChain,
    estimate_chain_from_sequences,
)
from gin_rummy.markov.deck_process import (
    CLASS_NAMES,
    DeckDepletionModel,
    RANK_CLASSES,
)
from gin_rummy.markov.hmm import HMM, ProgressCallback
from gin_rummy.markov.opponent_hand_hmm import (
    DISCARD_RANK_CLASSES,
    DRAW_SOURCES,
    HAND_STRENGTH_CLASSES,
    N_OBS,
    OBSERVATION_ALPHABET,
    OpponentHandHMM,
    decode_observation,
    describe_state,
    encode_observation,
    extract_observations,
    hand_strength_class,
    hand_strength_index,
)
from gin_rummy.markov.utils import logsumexp, normalise

__all__ = [
    "AbsorbingChain",
    "CLASS_NAMES",
    "CascadeState",
    "CascadeStep",
    "DISCARD_RANK_CLASSES",
    "DRAW_SOURCES",
    "DeckDepletionModel",
    "HAND_STRENGTH_CLASSES",
    "HMM",
    "MarkovChain",
    "N_OBS",
    "OBSERVATION_ALPHABET",
    "OpponentHandHMM",
    "ProgressCallback",
    "RANK_CLASSES",
    "ResponseModel",
    "cascade_value",
    "decode_observation",
    "describe_state",
    "encode_observation",
    "estimate_chain_from_sequences",
    "expected_turns_to_gin",
    "extract_observations",
    "greedy_response_model",
    "hand_strength_class",
    "hand_strength_index",
    "logsumexp",
    "normalise",
    "probabilistic_cascade",
]
