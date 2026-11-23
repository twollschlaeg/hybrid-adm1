
"""The parameter container the campaigns' ``*-parameters.json`` files load into."""

from __future__ import annotations

import json

from adm1_package.utils import parameters as sf


def test_parameters_container_is_custom_dict_like_and_round_trips(tmp_path):
    params = sf.Parameters()
    params.add('k_hyd_ch', 0.5, vary=True, min=0.0, max=2.0)
    params.add('f_ch', 0.4, vary=True, min=0.0, max=1.0)

    assert 'lmfit' not in sf.Parameters.__module__
    assert isinstance(params, dict)
    assert params['k_hyd_ch']['value'] == 0.5

    save_path = tmp_path / 'custom-params.json'
    sf.save_params(params, str(save_path))
    loaded = sf.load_params(str(save_path))

    assert isinstance(loaded, dict)
    assert loaded['k_hyd_ch']['value'] == 0.5
    assert loaded['k_hyd_ch']['vary'] is True
    assert loaded['f_ch']['max'] == 1.0
    assert loaded.valuesdict()['f_ch'] == 0.4


def test_load_params_accepts_lmfit_serialized_json(tmp_path):
    legacy_payload = {
        'params': [
            ['k_hyd_ch', 0.5, True, None, 0.0, 2.0, None, None, None, 1.0, None],
            ['f_ch', 0.4, True, None, 0.0, 1.0, None, None, None, 1.0, None],
        ],
        'unique_symbols': {},
    }
    legacy_string = json.dumps(legacy_payload)

    save_path = tmp_path / 'legacy-params.json'
    save_path.write_text(json.dumps(legacy_string))

    loaded = sf.load_params(str(save_path))

    assert loaded['k_hyd_ch']['value'] == 0.5
    assert loaded['k_hyd_ch']['vary'] is True
    assert loaded['f_ch']['min'] == 0.0
    assert loaded['f_ch']['max'] == 1.0
