"""Small helpers that bind Streamlit widgets to dataclass fields of DesignInputs."""
from __future__ import annotations

from typing import Optional

import streamlit as st


def key(name: str) -> str:
    return f"{name}__v{st.session_state.get('ver', 0)}"


def num(label: str, obj, attr: str, k: str, step: Optional[float] = None, fmt: Optional[str] = None,
        min_value: Optional[float] = None, max_value: Optional[float] = None, help: Optional[str] = None,
        container=st) -> float:
    kw = {} if key(k) in st.session_state else {"value": float(getattr(obj, attr))}
    v = container.number_input(label, step=step, format=fmt, min_value=min_value, max_value=max_value, key=key(k),
                               help=help, **kw)
    setattr(obj, attr, float(v))
    return v


def integer(label: str, obj, attr: str, k: str, min_value: int = 1, max_value: Optional[int] = None,
            help: Optional[str] = None, container=st) -> int:
    v = container.number_input(label, value=int(getattr(obj, attr)), step=1, min_value=min_value,
                               max_value=max_value, key=key(k), help=help)
    setattr(obj, attr, int(v))
    return v


def opt_num(label: str, obj, attr: str, k: str, default: float, help: Optional[str] = None,
            fmt: Optional[str] = None, container=st) -> Optional[float]:
    """Optional float: checkbox 'set' + number input (None if unchecked)."""
    cur = getattr(obj, attr)
    c1, c2 = container.columns([1.3, 1])
    on = c1.checkbox(label, value=cur is not None, key=key(k + "_on"), help=help)
    if on:
        v = c2.number_input(label, value=float(cur if cur is not None else default), key=key(k),
                            label_visibility="collapsed", format=fmt)
        setattr(obj, attr, float(v))
    else:
        c2.caption("not set")
        setattr(obj, attr, None)
    return getattr(obj, attr)


def select(label: str, obj, attr: str, k: str, options: dict | list, help: Optional[str] = None,
           container=st):
    opts = list(options)
    cur = getattr(obj, attr)
    idx = opts.index(cur) if cur in opts else 0
    fmt = (lambda o: f"{o}: {options[o]}") if isinstance(options, dict) else str
    v = container.selectbox(label, opts, index=idx, format_func=fmt, key=key(k), help=help)
    setattr(obj, attr, v)
    return v


def check(label: str, obj, attr: str, k: str, help: Optional[str] = None, container=st) -> bool:
    v = container.checkbox(label, value=bool(getattr(obj, attr)), key=key(k), help=help)
    setattr(obj, attr, bool(v))
    return v


def param_widgets(strategy, params: dict, k: str, container=st) -> dict:
    """Widgets for the parameters of a split strategy (from its Param declarations)."""
    out = dict(strategy.defaults())
    out.update({p: v for p, v in (params or {}).items() if p in out})
    for p in strategy.params:
        kk = key(f"{k}_{p.name}")
        cur = out[p.name]
        if p.kind == "bool":
            out[p.name] = container.checkbox(p.label, value=bool(cur), key=kk, help=p.help or None)
        elif p.kind == "int":
            out[p.name] = int(container.number_input(p.label, value=int(cur), step=1, key=kk,
                                                     min_value=int(p.min) if p.min is not None else None,
                                                     max_value=int(p.max) if p.max is not None else None,
                                                     help=p.help or None))
        elif p.kind == "choice":
            ch = list(p.choices)
            out[p.name] = container.selectbox(p.label, ch, index=ch.index(cur) if cur in ch else 0, key=kk,
                                              help=p.help or None)
        elif p.kind == "text":
            out[p.name] = container.text_area(p.label, value=str(cur), key=kk, help=p.help or None)
        else:
            out[p.name] = float(container.number_input(p.label, value=float(cur), key=kk,
                                                       min_value=p.min, max_value=p.max, help=p.help or None,
                                                       format="%.6g"))
    return out
