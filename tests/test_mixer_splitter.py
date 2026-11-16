"""Mixer and Splitter: mass, species and energy balances, ports, errors, flowsheet use."""

import warnings

import pytest

from processpi.components import Benzene, Methane, Mixture, Nitrogen, Toluene, Water
from processpi.equipment import DistillationColumn, Mixer, Splitter
from processpi.integration.flowsheet import Flowsheet
from processpi.streams import MaterialStream
from processpi.units import (
    Density, MassFlowRate, MolarFlowRate, Pressure, SpecificHeat, Temperature, VolumetricFlowRate,
)

ATM = Pressure(1, "atm")


def stream(name, component, t_c, kg_s, pressure=ATM, **kwargs):
    return MaterialStream(name, component=component, temperature=Temperature(t_c, "C"),
                          pressure=pressure, mass_flow=MassFlowRate(kg_s, "kg/s"), **kwargs)


def mix(*inlets, **kwargs):
    unit = Mixer("M", n_inlets=len(inlets), **kwargs)
    for i, s in enumerate(inlets):
        unit.connect_inlet(f"in{i + 1}", s)
    out = MaterialStream("out")
    unit.connect_outlet("out", out)
    return unit, out


def pure_heat(cls, phase, pressure, t_from_k, t_to_k, steps=2000):
    """integral(cp dT) of a pure component, J/kg, by the trapezoidal rule on a fine grid."""
    h = (t_to_k - t_from_k) / steps
    cp = [cls(temperature=Temperature(t_from_k + k * h, "K"), pressure=pressure, phase=phase)
          .specific_heat().value for k in range(steps + 1)]
    return h * (sum(cp) - 0.5 * (cp[0] + cp[-1]))


def assert_energy_balance(feeds, t_out_k, phase, pressure):
    """sum_i m_i * integral(cp_i dT, T_out, T_i) = 0 for pure-component inlets [(cls, T_K, kg/s)]."""
    q = [m * pure_heat(cls, phase, pressure, t_out_k, t_k) for cls, t_k, m in feeds]
    # The outlet temperature is stored to 1e-4 K.
    capacity = sum(m * cls(temperature=Temperature(t_out_k, "K"), pressure=pressure, phase=phase)
                   .specific_heat().value for cls, _, m in feeds)
    assert abs(sum(q)) <= capacity * 2e-4 + 1e-9 * sum(abs(x) for x in q)


# ----------------------------------------------------------------------
# Mixer
# ----------------------------------------------------------------------
def test_water_streams_mix_to_the_mass_weighted_temperature():
    cold = stream("cold", Water(), 20, 1.0, pressure=Pressure(2, "bar"))
    hot = stream("hot", Water(), 80, 3.0, pressure=Pressure(3, "bar"))
    unit, out = mix(cold, hot)
    data = unit.simulate()
    # cp of water changes by under 0.2 % from 20 to 80 C, so about (1*20 + 3*80)/4 = 65 C.
    assert out.temperature.value == pytest.approx(65 + 273.15, abs=0.1)
    assert out.mass_flow().value == pytest.approx(4.0, rel=1e-12)
    assert out.pressure.value == pytest.approx(2e5)          # the lowest inlet pressure
    assert isinstance(out.component, Water)
    assert out.phase == "liquid"
    assert out.density.value == pytest.approx(out.component.density().value)
    assert out.specific_heat.value == pytest.approx(out.component.specific_heat().value)
    assert data["outlet_temperature"] is out.temperature
    assert_energy_balance([(Water, 293.15, 1.0), (Water, 353.15, 3.0)], out.temperature.value, "liquid", out.pressure)


def test_an_outlet_pressure_can_be_given():
    unit, out = mix(stream("a", Water(), 20, 1.0), stream("b", Water(), 30, 1.0), pressure=Pressure(0.5, "bar"))
    unit.simulate()
    assert out.pressure.value == pytest.approx(0.5e5)


def test_equal_temperatures_give_the_same_temperature():
    unit, out = mix(stream("a", Water(), 40, 1.0), stream("b", Water(), 40, 2.5))
    unit.simulate()
    assert out.temperature.value == pytest.approx(313.15, abs=1e-9)


def test_two_liquids_give_a_mixture_with_the_species_moles():
    b = stream("b", Benzene(), 30, 1.0)
    t = stream("t", Toluene(), 70, 2.0)
    unit, out = mix(b, t)
    data = unit.simulate()
    n_b, n_t = 1.0 * 1000 / Benzene.molecular_weight, 2.0 * 1000 / Toluene.molecular_weight
    assert isinstance(out.component, Mixture)
    assert out.components == pytest.approx({"Benzene": n_b / (n_b + n_t), "Toluene": n_t / (n_b + n_t)}, rel=1e-12)
    assert out.component.mole_fractions() == pytest.approx(out.components, rel=1e-12)
    assert out.molar_flow().value == pytest.approx(n_b + n_t, rel=1e-12)
    assert data["molar_flow"].value == pytest.approx(n_b + n_t, rel=1e-12)
    assert out.mass_flow().value == pytest.approx(3.0, rel=1e-12)
    assert out.component.temperature.value == pytest.approx(out.temperature.value)
    assert out.molecular_weights == pytest.approx({"Benzene": Benzene.molecular_weight, "Toluene": Toluene.molecular_weight})
    assert_energy_balance([(Benzene, 303.15, 1.0), (Toluene, 343.15, 2.0)], out.temperature.value, "liquid", ATM)


def test_a_mixture_inlet_adds_its_species():
    feed = MaterialStream("bt", component=Mixture({Benzene: 0.4, Toluene: 0.6}), temperature=Temperature(50, "C"),
                          pressure=ATM, molar_flow=MolarFlowRate(10, "mol/s"))
    feed._mass_flow = None
    b = MaterialStream("b", component=Benzene(), temperature=Temperature(50, "C"), pressure=ATM,
                       molar_flow=MolarFlowRate(5, "mol/s"))
    unit, out = mix(feed, b)
    unit.simulate()
    assert out.molar_flow().value == pytest.approx(15.0, rel=1e-12)
    assert out.components == pytest.approx({"Benzene": 9 / 15, "Toluene": 6 / 15}, rel=1e-12)
    mass = (10 * (0.4 * Benzene.molecular_weight + 0.6 * Toluene.molecular_weight) + 5 * Benzene.molecular_weight) / 1000
    assert out.mass_flow().value == pytest.approx(mass, rel=1e-12)
    assert out.temperature.value == pytest.approx(323.15, abs=1e-9)


def test_gases_mix_on_their_ideal_gas_heat_capacities():
    ch4 = stream("ch4", Methane(), 27, 0.2, phase="gas")
    n2 = stream("n2", Nitrogen(), 227, 0.5, phase="gas")
    unit, out = mix(ch4, n2)
    unit.simulate()
    assert out.phase == "gas"
    assert out.component.phase() == "gas"
    assert_energy_balance([(Methane, 300.15, 0.2), (Nitrogen, 500.15, 0.5)], out.temperature.value, "gas", ATM)


def test_three_inlets_and_named_ports():
    unit = Mixer("M", inlet_names=["a", "b", "c"])
    out = MaterialStream("out")
    for port, t in zip("abc", (20, 40, 60)):
        unit.connect_inlet(port, stream(port, Water(), t, 1.0))
    unit.connect_outlet("out", out)
    unit.simulate()
    assert out.mass_flow().value == pytest.approx(3.0)
    assert out.temperature.value == pytest.approx(40 + 273.15, abs=0.05)


def test_unconnected_and_zero_flow_inlets_are_skipped():
    unit = Mixer("M", n_inlets=3)
    out = MaterialStream("out")
    unit.connect_inlet("in1", stream("a", Water(), 30, 2.0))
    unit.connect_inlet("in3", stream("c", Water(), 90, 0.0))
    unit.connect_outlet("out", out)
    unit.simulate()
    assert out.mass_flow().value == pytest.approx(2.0)
    assert out.temperature.value == pytest.approx(303.15, abs=1e-9)


def test_an_outlet_above_its_bubble_point_is_flagged():
    # Benzene near its boiling point mixed with toluene near its boiling point:
    # the mixture's bubble point (about 92 C at 1 atm for this composition) is
    # below the mixed temperature, so the liquid outlet would partly boil.
    unit, out = mix(stream("b", Benzene(), 79, 1.0, phase="liquid"), stream("t", Toluene(), 110, 1.0, phase="liquid"))
    with pytest.warns(UserWarning, match="flash calculation"):
        unit.simulate()
    assert out.phase == "liquid"


def test_a_subcooled_outlet_gives_no_warning():
    unit, out = mix(stream("b", Benzene(), 25, 1.0), stream("t", Toluene(), 40, 1.0))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        unit.simulate()


@pytest.mark.parametrize("make, error, match", [
    (lambda: mix(stream("w", Water(), 20, 1.0), stream("n", Nitrogen(), 20, 1.0, phase="gas")),
     NotImplementedError, "liquid and gas"),
    (lambda: mix(stream("w", Water(), 20, 1.0),
                 MaterialStream("x", temperature=Temperature(20, "C"), pressure=ATM, mass_flow=MassFlowRate(1, "kg/s"),
                                density=Density(1000, "kg/m3"), specific_heat=SpecificHeat(4180, "J/kgK"))),
     ValueError, "has no component"),
    (lambda: mix(stream("w", Water(), 20, 1.0), MaterialStream("f", component=Water(), temperature=Temperature(20, "C"),
                                                               pressure=ATM)),
     ValueError, "no mass, molar or volumetric flow"),
    (lambda: mix(stream("w", Water(), 20, 0.0), stream("v", Water(), 30, 0.0)), ValueError, "zero flow"),
])
def test_bad_inlets_are_refused(make, error, match):
    unit, _ = make()
    with pytest.raises(error, match=match):
        unit.simulate()


def test_a_stream_given_no_temperature_is_refused():
    # A stream made without a temperature takes the component's default 25 C,
    # so the mixer only refuses one whose temperature was removed.
    s = stream("w", Water(), 20, 1.0)
    s.temperature = None
    unit, _ = mix(s, stream("v", Water(), 30, 1.0))
    with pytest.raises(ValueError, match="needs a temperature"):
        unit.simulate()


def test_mixer_port_errors():
    with pytest.raises(ValueError, match="at least 2 inlets"):
        Mixer("M", n_inlets=1)
    with pytest.raises(ValueError, match="2 distinct"):
        Mixer("M", inlet_names=["a", "a"])
    unit = Mixer("M")
    with pytest.raises(ValueError, match="'out' is not connected"):
        unit.simulate()
    unit.connect_outlet("out", MaterialStream("o"))
    with pytest.raises(ValueError, match="no inlet"):
        unit.simulate()


# ----------------------------------------------------------------------
# Splitter
# ----------------------------------------------------------------------
def split(inlet, fractions):
    unit = Splitter("S", fractions)
    unit.connect_inlet("in", inlet)
    outs = {}
    for port in unit.fractions:
        outs[port] = MaterialStream(port)
        unit.connect_outlet(port, outs[port])
    return unit, outs


def test_splitter_scales_the_flows_and_keeps_the_state():
    feed = stream("feed", Mixture({Benzene: 0.97, Toluene: 0.03}), 80, 4.0, phase="liquid")
    unit, outs = split(feed, {"reflux": 0.75, "distillate": 0.25})
    data = unit.simulate()
    assert outs["reflux"].mass_flow().value == pytest.approx(3.0, rel=1e-12)
    assert outs["distillate"].mass_flow().value == pytest.approx(1.0, rel=1e-12)
    assert sum(o.molar_flow().value for o in outs.values()) == pytest.approx(feed.molar_flow().value, rel=1e-12)
    for o in outs.values():
        assert o.temperature.value == feed.temperature.value
        assert o.pressure.value == feed.pressure.value
        assert o.phase == "liquid"
        assert o.components == feed.components
        assert o.density.value == pytest.approx(feed.density.value)
        assert o.specific_heat.value == pytest.approx(feed.specific_heat.value)
        assert o.component.mole_fractions() == pytest.approx({"Benzene": 0.97, "Toluene": 0.03})
        assert o.component is not feed.component
    assert data["fractions"] == {"reflux": 0.75, "distillate": 0.25}
    assert data["flows"]["reflux"].value == pytest.approx(3.0)


def test_splitter_list_fractions_name_the_outlets():
    unit, outs = split(stream("w", Water(), 20, 1.0), [0.2, 0.3, 0.5])
    unit.simulate()
    assert list(outs) == ["out1", "out2", "out3"]
    assert [o.mass_flow().value for o in outs.values()] == pytest.approx([0.2, 0.3, 0.5])


def test_splitter_takes_a_stream_given_by_its_properties():
    feed = MaterialStream("x", temperature=Temperature(50, "C"), pressure=ATM, density=Density(800, "kg/m3"),
                          specific_heat=SpecificHeat(2000, "J/kgK"), flow_rate=VolumetricFlowRate(0.01, "m3/s"))
    unit, outs = split(feed, [0.4, 0.6])
    unit.simulate()
    assert outs["out1"].flow_rate.value == pytest.approx(0.004)
    assert outs["out2"].mass_flow().value == pytest.approx(0.006 * 800)
    assert outs["out1"].density.value == pytest.approx(800)
    assert outs["out1"].component is None


@pytest.mark.parametrize("fractions, match", [
    ([1.0], "at least 2"),
    ([0.5, 0.6], "add up to"),
    ({"a": 1.2, "b": -0.2}, "between 0 and 1"),
    ([0.5, float("nan")], "between 0 and 1"),
])
def test_bad_fractions_are_refused(fractions, match):
    with pytest.raises(ValueError, match=match):
        Splitter("S", fractions)


def test_splitter_port_errors():
    unit = Splitter("S", [0.5, 0.5])
    with pytest.raises(ValueError, match="'in' is not connected"):
        unit.simulate()
    unit.connect_inlet("in", stream("w", Water(), 20, 1.0))
    unit.connect_outlet("out1", MaterialStream("a"))
    with pytest.raises(ValueError, match="'out2' is not connected"):
        unit.simulate()
    flowless = MaterialStream("f", temperature=Temperature(20, "C"), pressure=ATM)
    unit2, _ = split(flowless, [0.5, 0.5])
    with pytest.raises(ValueError, match="no flow"):
        unit2.simulate()


# ----------------------------------------------------------------------
# Flowsheet
# ----------------------------------------------------------------------
def test_mixer_then_splitter_in_a_flowsheet():
    fs = Flowsheet("mix-split")
    mixer, splitter = Mixer("M"), Splitter("S", {"a": 0.4, "b": 0.6})
    a, b, mid = stream("a", Water(), 20, 1.0), stream("b", Water(), 60, 1.0), MaterialStream("mid")
    out_a, out_b = MaterialStream("out_a"), MaterialStream("out_b")
    # Registered in reverse: the flowsheet orders the units from the connections.
    fs.add_equipment(splitter)
    fs.add_equipment(mixer)
    fs.connect(a, mixer, "in1")
    fs.connect(b, mixer, "in2")
    fs.connect(mid, mixer, "out", splitter, "in")
    splitter.connect_outlet("a", out_a)
    splitter.connect_outlet("b", out_b)
    fs.run()
    assert out_a.mass_flow().value + out_b.mass_flow().value == pytest.approx(2.0)
    assert out_a.temperature.value == pytest.approx(mid.temperature.value)
    assert mid.temperature.value == pytest.approx(40 + 273.15, abs=0.05)
    assert mixer.data["mass_flow"].value == pytest.approx(2.0)


def test_column_distillate_to_a_reflux_splitter():
    feed = MaterialStream("Feed", temperature=Temperature(95, "C"), pressure=Pressure(1.01325, "bar"),
                          composition={"Benzene": 0.4, "Toluene": 0.6}, molar_flow=MolarFlowRate(100, "kmol/h"))
    distillate, bottoms = MaterialStream("D"), MaterialStream("B")
    column = DistillationColumn(feed=feed, components=[Benzene(), Toluene()], light_key="Benzene",
                                heavy_key="Toluene", distillate_lk_fraction=0.97, bottoms_lk_fraction=0.02,
                                distillate=distillate, bottoms=bottoms)
    splitter = Splitter("reflux drum", {"reflux": 0.6, "product": 0.4})
    reflux, product = MaterialStream("R"), MaterialStream("P")
    fs = Flowsheet("column + split")
    fs.add_equipment(column)
    fs.connect(distillate, column, "distillate", splitter, "in")
    splitter.connect_outlet("reflux", reflux)
    splitter.connect_outlet("product", product)
    fs.run()
    d = column.results().data["distillate"]["D"].to("mol/s").value
    assert product.molar_flow().value == pytest.approx(0.4 * d, rel=1e-12)
    assert product.components == pytest.approx({"Benzene": 0.97, "Toluene": 0.03})
    assert isinstance(product.component, Mixture)
    assert product.density.value == pytest.approx(distillate.density.value)
