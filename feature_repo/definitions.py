"""Discovery shim so the Feast CLI finds the views defined in
fraud.features.registry."""

from fraud.features.registry import default_objects

_objects = default_objects()

card = _objects.card
card_velocity = _objects.card_velocity
transaction_dynamics = _objects.transaction_dynamics
card_activity = _objects.card_activity
