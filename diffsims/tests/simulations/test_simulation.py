import numpy as np
import pytest

from diffpy.structure import Structure, Atom, Lattice
from orix.crystal_map import Phase
from orix.quaternion import Rotation

from diffsims.simulations.simulation import Simulation
from diffsims.generators.simulation_generator import SimulationGenerator
from diffsims.crystallography.reciprocal_lattice_vector import ReciprocalLatticeVector


class TestSingleSimulation:
    @pytest.fixture
    def al_phase(self):
        p = Phase(
            name="al",
            space_group=225,
            structure=Structure(
                atoms=[Atom("al", [0, 0, 0])],
                lattice=Lattice(0.405, 0.405, 0.405, 90, 90, 90),
            ),
        )
        return p

    @pytest.fixture
    def single_simulation(self, al_phase):
        gen = SimulationGenerator(accelerating_voltage=200)
        rot = Rotation.from_axes_angles([1, 0, 0], 45, degrees=True)
        coords = ReciprocalLatticeVector(
            phase=al_phase, xyz=[[1, 0, 0], [0, 1, 0], [1, 1, 0]]
        )
        sim = Simulation(
            phases=al_phase, simulation_generator=gen, coordinates=coords, rotations=rot
        )
        return sim

    def test_init(self, single_simulation):
        assert isinstance(single_simulation, Simulation)
        assert isinstance(single_simulation.phases, Phase)
        assert isinstance(single_simulation.simulation_generator, SimulationGenerator)
        assert isinstance(single_simulation.rotations, Rotation)

    def test_iphase(self, single_simulation):
        with pytest.raises(ValueError):
            single_simulation.iphase[0]

    def test_plot(self, single_simulation):
        single_simulation.plot()


class TestSinglePhaseMultiSimulation:
    @pytest.fixture
    def al_phase(self):
        p = Phase(
            name="al",
            space_group=225,
            structure=Structure(
                atoms=[Atom("al", [0, 0, 0])],
                lattice=Lattice(0.405, 0.405, 0.405, 90, 90, 90),
            ),
        )
        return p

    @pytest.fixture
    def multi_simulation(self, al_phase):
        gen = SimulationGenerator(accelerating_voltage=200)
        rot = Rotation.from_axes_angles([1, 0, 0], (0, 15, 30, 45), degrees=True)
        coords = ReciprocalLatticeVector(
            phase=al_phase, xyz=[[1, 0, 0], [0, 1, 0], [1, 1, 0]]
        )
        coords = [
            coords,
        ] * 4
        sim = Simulation(
            phases=al_phase, simulation_generator=gen, coordinates=coords, rotations=rot
        )
        return sim

    def test_init(self, multi_simulation):
        assert isinstance(multi_simulation, Simulation)
        assert isinstance(multi_simulation.phases, Phase)
        assert isinstance(multi_simulation.simulation_generator, SimulationGenerator)
        assert isinstance(multi_simulation.rotations, Rotation)
        assert isinstance(multi_simulation.coordinates, np.ndarray)

    def test_iphase(self, multi_simulation):
        with pytest.raises(ValueError):
            multi_simulation.iphase[0]

    def test_irot(self, multi_simulation):
        sliced_sim = multi_simulation.irot[0]
        assert isinstance(sliced_sim, Simulation)
        assert isinstance(sliced_sim.phases, Phase)
        assert sliced_sim.rotations.size == 1
        assert sliced_sim.num_vectors == 0

    def test_irot_slice(self, multi_simulation):
        sliced_sim = multi_simulation.irot[0:2]
        assert isinstance(sliced_sim, Simulation)
        assert isinstance(sliced_sim.phases, Phase)
        assert sliced_sim.rotations.size == 2
        assert sliced_sim.num_vectors == (2,)

    def test_plot(self, multi_simulation):
        multi_simulation.plot()


class TestMultiPhaseMultiSimulation:
    @pytest.fixture
    def al_phase(self):
        p = Phase(
            name="al",
            space_group=225,
            structure=Structure(
                atoms=[Atom("al", [0, 0, 0])],
                lattice=Lattice(0.405, 0.405, 0.405, 90, 90, 90),
            ),
        )
        return p

    @pytest.fixture
    def multi_simulation(self, al_phase):
        gen = SimulationGenerator(accelerating_voltage=200)
        rot = Rotation.from_axes_angles([1, 0, 0], (0, 15, 30, 45), degrees=True)
        rot2 = rot
        coords = ReciprocalLatticeVector(
            phase=al_phase, xyz=[[1, 0, 0], [0, 1, 0], [1, 1, 0]]
        )

        coords = [
            coords,
        ] * 4
        coords2 = coords
        al_phase2 = al_phase.deepcopy()
        al_phase2.name = "al2"
        sim = Simulation(
            phases=[al_phase, al_phase2],
            simulation_generator=gen,
            coordinates=[coords, coords2],
            rotations=[rot, rot2],
        )
        return sim

    def test_init(self, multi_simulation):
        assert isinstance(multi_simulation, Simulation)
        assert isinstance(multi_simulation.phases, np.ndarray)
        assert isinstance(multi_simulation.simulation_generator, SimulationGenerator)
        assert isinstance(multi_simulation.rotations, np.ndarray)
        assert isinstance(multi_simulation.coordinates, np.ndarray)

    def test_iphase(self, multi_simulation):
        phase_slic = multi_simulation.iphase[0]
        assert isinstance(phase_slic, Simulation)
        assert isinstance(phase_slic.phases, Phase)
        assert phase_slic.rotations.size == 4

    def test_irot(self, multi_simulation):
        sliced_sim = multi_simulation.irot[0]
        assert isinstance(sliced_sim, Simulation)
        assert isinstance(sliced_sim.phases, np.ndarray)
        assert sliced_sim.rotations.size == 2
        assert sliced_sim.num_vectors == (0, 0)

    def test_irot_slice(self, multi_simulation):
        sliced_sim = multi_simulation.irot[0:2]
        assert isinstance(sliced_sim, Simulation)
        assert isinstance(sliced_sim.phases, np.ndarray)
        assert sliced_sim.rotations.size == 2
        assert sliced_sim.num_vectors == (2, 2)
        sliced_sim.plot()

    def test_plot(self, multi_simulation):
        multi_simulation.plot()

    def test_plot_rotation(self, multi_simulation):
        multi_simulation.plot_rotations()
