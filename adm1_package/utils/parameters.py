"""Mechanistic model parameters: container, bounds and JSON persistence.

``Parameters`` replaces ``lmfit.Parameters`` with a plain dict-backed container,
so the package no longer depends on lmfit at import time. It keeps an
lmfit-compatible surface (``add``, ``valuesdict``, ``pretty_print``) because the
calibration scripts still hand parameter objects to lmfit's optimizers, and it
reads the legacy lmfit JSON layout that the campaign parameter files were
originally written in.
"""

from __future__ import annotations

import json
import os
from typing import Any


class Parameter(dict):
    """Dictionary-backed parameter object with lmfit-like attribute access."""

    def __init__(self, name: str, value: Any, vary: bool = True, min: float | None = None,
                 max: float | None = None, expr: str | None = None, **kwargs: Any):
        super().__init__(name=name, value=value, vary=bool(vary), min=min, max=max, expr=expr)
        if kwargs:
            super().update(kwargs)

    @property
    def value(self):
        return self['value']

    @value.setter
    def value(self, new_value):
        self['value'] = new_value

    @property
    def vary(self):
        return self['vary']

    @vary.setter
    def vary(self, new_value):
        self['vary'] = bool(new_value)

    @property
    def min(self):
        return self['min']

    @min.setter
    def min(self, new_min):
        self['min'] = new_min

    @property
    def max(self):
        return self['max']

    @max.setter
    def max(self, new_max):
        self['max'] = new_max

    def set(self, value=None, vary=None, min=None, max=None, expr=None):
        if value is not None:
            self['value'] = value
        if vary is not None:
            self['vary'] = bool(vary)
        if min is not None:
            self['min'] = min
        if max is not None:
            self['max'] = max
        if expr is not None:
            self['expr'] = expr
        return self

    def __repr__(self) -> str:
        return f"Parameter(name={self['name']!r}, value={self['value']!r}, vary={self['vary']!r}, min={self['min']!r}, max={self['max']!r})"


class Parameters(dict):
    """Dictionary-based replacement for lmfit.Parameters with a compatible API."""

    def __init__(self, *args, **kwargs):
        super().__init__()
        for key, value in dict(*args, **kwargs).items():
            if isinstance(value, Parameter):
                self[key] = value
            else:
                self[key] = Parameter(key, value)

    @classmethod
    def from_dict(cls, mapping):
        params = cls()
        for name, value in mapping.items():
            if isinstance(value, Parameter):
                params[name] = value
            elif isinstance(value, dict):
                params[name] = Parameter(
                    name=name,
                    value=value.get('value', value.get('default')),
                    vary=value.get('vary', True),
                    min=value.get('min'),
                    max=value.get('max'),
                    expr=value.get('expr'),
                )
            else:
                params[name] = Parameter(name, value)
        return params

    @classmethod
    def from_lmfit(cls, lmfit_params):
        """The name, value, vary, min, max and expr of each parameter, nothing else."""
        params = cls()
        for name, item in lmfit_params.items():
            if hasattr(item, 'value'):
                params.add(
                    name,
                    value=item.value,
                    vary=getattr(item, 'vary', True),
                    min=getattr(item, 'min', None),
                    max=getattr(item, 'max', None),
                    expr=getattr(item, 'expr', None),
                )
        return params

    def add(self, name: str, value: Any = None, vary: bool = True, min: float | None = None,
            max: float | None = None, expr: str | None = None, **kwargs: Any):
        parameter = Parameter(name, value, vary=vary, min=min, max=max, expr=expr, **kwargs)
        self[name] = parameter
        return parameter

    def valuesdict(self):
        return {name: param['value'] for name, param in self.items()}

    def copy(self):
        return Parameters.from_dict({name: dict(param) for name, param in self.items()})

    def pretty_print(self):
        for name, param in self.items():
            print(f"{name}: value={param['value']}, vary={param['vary']}, min={param['min']}, max={param['max']}")

    def dumps(self, sort_keys: bool = True):
        serializable = {name: dict(param) for name, param in self.items()}
        return json.dumps(serializable, sort_keys=sort_keys)

    def loads(self, payload):
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except json.JSONDecodeError:
                payload = {}

        if isinstance(payload, dict) and 'params' in payload and isinstance(payload.get('params'), list):
            payload = _coerce_legacy_lmfit_payload(payload)

        self.clear()
        for name, item in payload.items():
            if isinstance(item, Parameter):
                self[name] = item
            elif isinstance(item, dict):
                self[name] = Parameter(
                    name=name,
                    value=item.get('value'),
                    vary=item.get('vary', True),
                    min=item.get('min'),
                    max=item.get('max'),
                    expr=item.get('expr'),
                )
                for key, value in item.items():
                    if key not in {'name', 'value', 'vary', 'min', 'max', 'expr'}:
                        self[name][key] = value
            else:
                self[name] = Parameter(name, item)
        return self

    def to_dict(self):
        return {name: dict(param) for name, param in self.items()}

    def __repr__(self):
        return f"Parameters({dict(self)!r})"


def _resolve_param_file(param_file: str) -> str:
    if os.path.splitext(param_file)[1] != '.json':
        return f'{param_file}.json'
    return param_file


def _coerce_legacy_lmfit_payload(payload):
    """Convert lmfit's JSON schema to the custom dict-based structure.

    Some parameter files under data/ are still in lmfit's layout (``{"params": [...]}``,
    stored as a JSON string).
    """
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError:
            return {}

    if not isinstance(payload, dict):
        return payload

    params_list = payload.get('params')
    if not isinstance(params_list, list):
        return payload

    converted = {}
    for param in params_list:
        if not isinstance(param, (list, tuple)) or len(param) < 2:
            continue
        name = param[0]
        value = param[1]
        vary = bool(param[2]) if len(param) > 2 and param[2] is not None else True
        min_value = param[4] if len(param) > 4 else None
        max_value = param[5] if len(param) > 5 else None
        expr = param[6] if len(param) > 6 else None

        converted[name] = {
            'name': name,
            'value': value,
            'vary': vary,
            'min': min_value,
            'max': max_value,
            'expr': expr,
        }

    return converted


def load_params(param_file):
    '''Load parameters from a .json file or an existing parameters object.'''
    if isinstance(param_file, Parameters):
        return param_file.copy()
    if isinstance(param_file, dict) and all(isinstance(v, dict) for v in param_file.values()):
        return Parameters.from_dict(param_file)

    resolved_path = _resolve_param_file(param_file)
    params = Parameters()
    with open(resolved_path, 'r') as json_file:
        payload = json.load(json_file)
        if isinstance(payload, str):
            payload = json.loads(payload)
        params.loads(payload)
    return params


def save_params(params, file_name):
    '''Save parameters to a .json file. Accepts custom Parameters or lmfit-like objects.

    A :class:`Parameters` goes through :meth:`Parameters.from_lmfit`, so each parameter
    is written with its name, value, vary, min, max and expr and nothing else.
    '''
    resolved_path = _resolve_param_file(file_name)
    if hasattr(params, 'to_dict') and not isinstance(params, dict):
        payload = params.to_dict()
    elif hasattr(params, 'valuesdict'):
        payload = Parameters.from_lmfit(params).to_dict()
    else:
        payload = Parameters.from_dict(params).to_dict() if isinstance(params, dict) else dict(params)
    with open(resolved_path, 'w') as file:
        json.dump(payload, file, sort_keys=True)
