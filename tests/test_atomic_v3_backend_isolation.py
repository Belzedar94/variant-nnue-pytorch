import inspect

import atomic_v2
import feature_set
import model as legacy_model

import atomic_v3
from atomic_v3.dataset import create_role_provider


def test_v3_is_not_registered_in_legacy_or_v2_dispatch():
    assert atomic_v3 is not atomic_v2
    assert atomic_v3.BACKEND_KEY == "atomic-nnue-v3"
    assert "atomic_v3" not in inspect.getsource(feature_set)
    assert "atomic_v3" not in inspect.getsource(legacy_model)


def test_v3_campaign_seam_has_no_implicit_native_or_legacy_loader_import():
    source = inspect.getsource(create_role_provider)
    assert "provider_factory" in source
    assert "nnue_dataset" not in source
    assert "atomic_v2" not in source
