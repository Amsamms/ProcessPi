"""
Mixer and Splitter: the stream units a flowsheet needs around a column
(reflux split, recycle join) and between exchangers.

    from processpi.components import Water
    from processpi.equipment import Mixer, Splitter
    from processpi.streams import MaterialStream
    from processpi.units import MassFlowRate, Pressure, Temperature

    cold = MaterialStream("cold", component=Water(), temperature=Temperature(20, "C"),
                          pressure=Pressure(2, "bar"), mass_flow=MassFlowRate(1, "kg/s"))
    hot = MaterialStream("hot", component=Water(), temperature=Temperature(80, "C"),
                         pressure=Pressure(3, "bar"), mass_flow=MassFlowRate(3, "kg/s"))
    mixed = MaterialStream("mixed")

    mixer = Mixer("M1")
    mixer.connect_inlet("in1", cold)
    mixer.connect_inlet("in2", hot)
    mixer.connect_outlet("out", mixed)
    mixer.simulate()        # mixed: 4 kg/s water at about 65 C and 2 bar

The Mixer conserves mass and the moles of every species exactly. Its outlet
temperature comes from an energy balance with ideal mixing (no heat of
mixing, as in ``Mixture``): each inlet's heat capacity is integrated from its
own temperature to the outlet temperature,

    sum_i m_i * integral(cp_i dT, T_out, T_i) = 0

Every inlet must be in the same single phase (all liquid or all gas). Mixing
across phases, or an outlet that partly vaporises, is a flash calculation and
is refused here.

The Splitter divides one stream into fixed fractions; every outlet has the
inlet's temperature, pressure, composition and phase.
"""

import copy
import warnings
from math import isfinite
from typing import Dict, List, Mapping, Optional, Sequence, Union

from ..components.base import Component
from ..components.mixture import Mixture
from ..streams import MaterialStream
from ..units import MassFlowRate, MolarFlowRate, Pressure, Temperature, VolumetricFlowRate
from .base import Equipment

_GAS_PHASES = {"gas", "vapor", "vapour", "steam"}

# Composite Simpson's rule on this many intervals for each inlet's cp integral.
# The cp correlations are smooth polynomials in T, so 8 intervals reproduce a
# fine-grid integral to far better than the correlations themselves.
_SIMPSON_INTERVALS = 8


def _phase_name(phase: str) -> str:
    phase = str(phase).lower()
    if phase in _GAS_PHASES:
        return "gas"
    if phase == "liquid":
        return "liquid"
    return phase


def _species(stream: MaterialStream, unit: str) -> Dict[type, float]:
    """{component class: mole fraction} of a stream that carries a component or a Mixture."""
    comp = stream.component
    if isinstance(comp, Mixture):
        return {comp.component_class(name): x for name, x in comp.mole_fractions().items()}
    if isinstance(comp, Component):
        return {type(comp): 1.0}
    raise ValueError(
        f"{unit}: stream '{stream.name}' has no component. Give it a component "
        "(component=Water()) or a Mixture so its properties are known."
    )


def _state(species: Mapping[type, float], temperature: Temperature, pressure: Pressure,
           phase: Optional[str]) -> Component:
    """A pure component, or a Mixture, at the given state."""
    if len(species) == 1:
        (cls,) = species
        return cls(temperature=temperature, pressure=pressure, phase=phase)
    return Mixture(dict(species), temperature=temperature, pressure=pressure, phase=phase)


def _mass_flow_kg_s(stream: MaterialStream, mw_g_mol: float, unit: str) -> float:
    mass = stream.mass_flow()
    if mass is not None:
        return mass.to("kg/s").value
    molar = stream.molar_flow()
    if molar is not None:
        return molar.to("mol/s").value * mw_g_mol / 1000.0
    raise ValueError(f"{unit}: stream '{stream.name}' has no mass, molar or volumetric flow.")


def _write_state(stream: MaterialStream, component: Component, temperature: Temperature,
                 pressure: Pressure, phase: str, mass_kg_s: float) -> None:
    """Write a unit's result to an outlet stream, the way the column writes its products."""
    fractions = component.mole_fractions() if isinstance(component, Mixture) else {component.name: 1.0}
    mw = float(component.molecular_weight)
    stream.component = component
    stream.components = dict(fractions)
    stream.basis = "mole"
    stream.molecular_weights = (
        {n: float(component.component_class(n).molecular_weight) for n in fractions}
        if isinstance(component, Mixture) else {component.name: mw}
    )
    stream.temperature = temperature
    stream.pressure = pressure
    stream.phase = phase
    stream._mass_flow = MassFlowRate(mass_kg_s, "kg/s")
    stream._molar_flow = MolarFlowRate(mass_kg_s * 1000.0 / mw, "mol/s")
    stream.flow_rate = None
    # The outlet's properties come from its state, not from values given when
    # the stream object was created.
    stream.given_density = None
    stream.given_specific_heat = None
    stream.density = None
    stream.specific_heat = None
    try:
        stream.density = component.density()
        stream.specific_heat = component.specific_heat()
    except ValueError:
        # No property for this state (a liquid above its critical
        # temperature); the component explains why when it is asked.
        pass


class Mixer(Equipment):
    """
    Joins two or more material streams into one.

    Args:
        name: unit name.
        n_inlets: number of inlet ports, named ``in1``, ``in2``, ... (default 2).
        inlet_names: port names instead of ``in1``...; overrides ``n_inlets``.
        pressure: outlet pressure. Default: the lowest inlet pressure.

    Ports: the inlets above and one outlet, ``out``. Inlets left unconnected
    are skipped.
    """

    def __init__(self, name: str, n_inlets: int = 2, inlet_names: Optional[List[str]] = None,
                 pressure: Optional[Pressure] = None):
        if inlet_names is None:
            if n_inlets < 2:
                raise ValueError(f"{name}: a mixer needs at least 2 inlets, not {n_inlets}.")
            inlet_names = [f"in{i + 1}" for i in range(n_inlets)]
        elif len(inlet_names) < 2 or len(set(inlet_names)) != len(inlet_names):
            raise ValueError(f"{name}: inlet_names must be at least 2 distinct names.")
        super().__init__(name, inlet_names=list(inlet_names), outlet_names=["out"])
        self.pressure = pressure
        self.data: Dict = {}

    def simulate(self) -> Dict:
        outlet = self.outlets["out"]
        if outlet is None:
            raise ValueError(f"{self.name}: outlet port 'out' is not connected.")
        inlets = [s for s in self.inlets.values_ordered() if s is not None]
        if not inlets:
            raise ValueError(f"{self.name}: no inlet is connected.")

        moles: Dict[type, float] = {}          # mol/s of each species
        feeds = []                             # (species, T in K, P, phase, kg/s)
        phases = set()
        for s in inlets:
            species = _species(s, self.name)
            if s.temperature is None or s.pressure is None:
                raise ValueError(f"{self.name}: stream '{s.name}' needs a temperature and a pressure.")
            phase = _phase_name(s.phase or s.component.phase())
            if phase not in ("liquid", "gas"):
                raise NotImplementedError(
                    f"{self.name}: stream '{s.name}' is {phase}; the mixer takes single-phase streams only."
                )
            mw = float(s.component.molecular_weight)
            m = _mass_flow_kg_s(s, mw, self.name)
            if not isfinite(m) or m < 0:
                raise ValueError(f"{self.name}: stream '{s.name}' has mass flow {m} kg/s.")
            if m == 0:
                continue
            n = m * 1000.0 / mw
            for cls, x in species.items():
                moles[cls] = moles.get(cls, 0.0) + n * x
            feeds.append((species, s.temperature.to("K").value, s.pressure, phase, m))
            phases.add(phase)

        if not feeds:
            raise ValueError(f"{self.name}: every inlet has zero flow.")
        if len(phases) > 1:
            raise NotImplementedError(
                f"{self.name}: the inlets are liquid and gas. Mixing across phases needs a "
                "flash calculation, which the mixer does not do."
            )
        phase = phases.pop()

        if self.pressure is not None:
            p_out = self.pressure
        else:
            p_out = min((f[2] for f in feeds), key=lambda p: p.to("Pa").value)
        t_out = self._outlet_temperature(feeds, p_out, phase)

        n_total = sum(moles.values())
        species_out = {cls: n / n_total for cls, n in moles.items()}
        mass_out = sum(f[4] for f in feeds)
        temperature = Temperature(t_out, "K")
        component = _state(species_out, temperature, p_out, phase)
        self._check_phase(species_out, temperature, p_out, phase)
        _write_state(outlet, component, temperature, p_out, phase, mass_out)

        self.data = {
            "outlet_temperature": temperature,
            "outlet_pressure": p_out,
            "mass_flow": MassFlowRate(mass_out, "kg/s"),
            "molar_flow": MolarFlowRate(n_total, "mol/s"),
            "composition": dict(outlet.components),
            "phase": phase,
        }
        return self.data

    @staticmethod
    def _cp(species, t_k: float, pressure: Pressure, phase: str) -> float:
        """Heat capacity of one inlet's material at t_k, J/kg-K."""
        return _state(species, Temperature(t_k, "K"), pressure, phase).specific_heat().to("J/kgK").value

    def _outlet_temperature(self, feeds, pressure: Pressure, phase: str) -> float:
        temps = [f[1] for f in feeds]
        if max(temps) - min(temps) < 1e-12:
            return temps[0]

        def heat_released(t_out: float) -> float:
            # sum_i m_i * integral(cp_i dT) from t_out to T_i, J/s; zero at the answer.
            total = 0.0
            for species, t_in, _, _, m in feeds:
                h = (t_in - t_out) / _SIMPSON_INTERVALS
                if h == 0:
                    continue
                s = 0.0
                for k in range(_SIMPSON_INTERVALS + 1):
                    w = 1 if k in (0, _SIMPSON_INTERVALS) else (4 if k % 2 else 2)
                    s += w * self._cp(species, t_out + k * h, pressure, phase)
                total += m * s * h / 3.0
            return total

        def heat_capacity_flow(t: float) -> float:
            return sum(m * self._cp(species, t, pressure, phase) for species, _, _, _, m in feeds)

        # Start from the cp-weighted mean, then Newton: d(heat_released)/dT_out = -sum m_i cp_i(T_out).
        weights = [m * self._cp(species, t, pressure, phase) for species, t, _, _, m in feeds]
        t = sum(w * f[1] for w, f in zip(weights, feeds)) / sum(weights)
        for _ in range(50):
            step = heat_released(t) / heat_capacity_flow(t)
            t += step
            if abs(step) < 1e-9:
                return t
        raise RuntimeError(f"{self.name}: the outlet temperature did not converge.")

    def _check_phase(self, species, temperature: Temperature, pressure: Pressure, phase: str) -> None:
        """Warn when the mixed outlet would not be in the inlets' phase at its own state."""
        try:
            free = _state(species, temperature, pressure, None).phase()
        except (ValueError, OverflowError):
            return
        if _phase_name(free) != phase:
            warnings.warn(
                f"{self.name}: the outlet is taken as {phase}, but at {temperature.to('C')} "
                f"and {pressure.to('bar')} it would be {free}. A flash calculation is "
                "needed for that outlet.",
                stacklevel=3,
            )


class Splitter(Equipment):
    """
    Divides one material stream into fixed fractions.

    Args:
        name: unit name.
        fractions: the fraction of the inlet flow to each outlet, adding up to
            1. A list names the outlets ``out1``, ``out2``, ...; a dict names
            them by its keys, e.g. ``{"reflux": 0.75, "distillate": 0.25}``.

    Ports: one inlet, ``in``, and the outlets above. Every outlet has the
    inlet's temperature, pressure, composition and phase; mass, molar and
    volumetric flows are scaled by its fraction.
    """

    def __init__(self, name: str, fractions: Union[Sequence[float], Mapping[str, float]]):
        if isinstance(fractions, Mapping):
            names = list(fractions.keys())
            values = [fractions[n] for n in names]
        else:
            values = list(fractions)
            names = [f"out{i + 1}" for i in range(len(values))]
        if len(values) < 2:
            raise ValueError(f"{name}: a splitter needs at least 2 outlets.")
        for n, f in zip(names, values):
            if not isinstance(f, (int, float)) or not isfinite(f) or not 0.0 <= f <= 1.0:
                raise ValueError(f"{name}: fraction for '{n}' must be between 0 and 1, not {f!r}.")
        if abs(sum(values) - 1.0) > 1e-9:
            raise ValueError(f"{name}: the fractions add up to {sum(values):.12g}, not 1.")
        super().__init__(name, inlet_names=["in"], outlet_names=names)
        self.fractions: Dict[str, float] = dict(zip(names, (float(v) for v in values)))
        self.data: Dict = {}

    def simulate(self) -> Dict:
        inlet = self.inlets["in"]
        if inlet is None:
            raise ValueError(f"{self.name}: inlet port 'in' is not connected.")
        for port in self.fractions:
            if self.outlets[port] is None:
                raise ValueError(f"{self.name}: outlet port '{port}' is not connected.")
        mass = inlet.mass_flow()
        molar = inlet.molar_flow()
        volume = inlet.flow_rate
        if mass is None and molar is None and volume is None:
            raise ValueError(f"{self.name}: stream '{inlet.name}' has no flow to split.")

        flows = {}
        for port, f in self.fractions.items():
            out = self.outlets[port]
            out.component = copy.copy(inlet.component)
            out.components = dict(inlet.components)
            out.basis = inlet.basis
            out.molecular_weights = dict(inlet.molecular_weights)
            out.temperature = inlet.temperature
            out.pressure = inlet.pressure
            out.phase = inlet.phase
            out.density = inlet.density
            out.specific_heat = inlet.specific_heat
            out.given_density = inlet.given_density
            out.given_specific_heat = inlet.given_specific_heat
            out._mass_flow = MassFlowRate(mass.to("kg/s").value * f, "kg/s") if mass is not None else None
            out._molar_flow = MolarFlowRate(molar.to("mol/s").value * f, "mol/s") if molar is not None else None
            out.flow_rate = VolumetricFlowRate(volume.to("m3/s").value * f, "m3/s") if volume is not None else None
            flows[port] = out._mass_flow if out._mass_flow is not None else (out._molar_flow or out.flow_rate)

        self.data = {"fractions": dict(self.fractions), "flows": flows}
        return self.data
