import numpy as np
import pytest

from diffpy.structure import Structure, Atom, Lattice
from orix.crystal_map import Phase
from orix.quaternion import Rotation

from diffsims.simulations.simulation import Simulation
from diffsims.generators.simulation_generator import SimulationGenerator
from diffsims.crystallography.reciprocal_lattice_vector import ReciprocalLatticeVector


@pytest.fixture(scope="module")
def al_phase():
    p = Phase(
        name="al",
        space_group=225,
        structure=Structure(
            atoms=[Atom("al", [0, 0, 0])],
            lattice=Lattice(0.405, 0.405, 0.405, 90, 90, 90),
        ),
    )
    return p


class TestSingleSimulation:
    @pytest.fixture
    def single_simulation(self, al_phase):
        gen = SimulationGenerator(accelerating_voltage=200)
        rot = Rotation.from_axes_angles([1, 0, 0], 45, degrees=True)
        coords = ReciprocalLatticeVector(phase=al_phase, xyz=[[1, 0, 0]])
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

    def test_irot(self, single_simulation):
        with pytest.raises(ValueError):
            single_simulation.irot[0]

    def test_iter(self, single_simulation):
        count = 0
        for sim in single_simulation:
            count += 1
            assert isinstance(sim, ReciprocalLatticeVector)
        assert count == 1

    def test_plot(self, single_simulation):
        single_simulation.plot()

    def test_polar_flatten(self, single_simulation):
        (
            r_templates,
            theta_templates,
            intensities_templates,
        ) = single_simulation.polar_flatten_simulations()
        assert r_templates.shape == (1, 1)
        assert theta_templates.shape == (1, 1)
        assert intensities_templates.shape == (1, 1)

    def test_deepcopy(self, single_simulation):
        copied = single_simulation.deepcopy()
        assert copied is not single_simulation


class TestSimulationInitFailures:
    def test_different_size(self, al_phase):
        gen = SimulationGenerator(accelerating_voltage=200)
        rot = Rotation.from_axes_angles([1, 0, 0], 45, degrees=True)
        coords = ReciprocalLatticeVector(phase=al_phase, xyz=[[1, 0, 0], [1, 1, 1]])
        with pytest.raises(ValueError):
            sim = Simulation(
                phases=al_phase,
                simulation_generator=gen,
                coordinates=[coords, coords],
                rotations=rot,
            )

    def test_different_size2(self, al_phase):
        gen = SimulationGenerator(accelerating_voltage=200)
        rot = Rotation.from_axes_angles([1, 0, 0], (0, 45), degrees=True)
        coords = ReciprocalLatticeVector(phase=al_phase, xyz=[[1, 0, 0], [1, 1, 1]])
        with pytest.raises(ValueError):
            sim = Simulation(
                phases=al_phase,
                simulation_generator=gen,
                coordinates=[coords, coords, coords],
                rotations=rot,
            )

    def test_different_size_multiphase(self, al_phase):
        gen = SimulationGenerator(accelerating_voltage=200)
        rot = Rotation.from_axes_angles([1, 0, 0], 45, degrees=True)
        coords = ReciprocalLatticeVector(phase=al_phase, xyz=[[1, 0, 0], [1, 1, 1]])
        with pytest.raises(ValueError):
            sim = Simulation(
                phases=[al_phase, al_phase],
                simulation_generator=gen,
                coordinates=[[coords, coords], [coords, coords]],
                rotations=[rot, rot],
            )

    def test_different_num_phase(self, al_phase):
        gen = SimulationGenerator(accelerating_voltage=200)
        rot = Rotation.from_axes_angles([1, 0, 0], 45, degrees=True)
        coords = ReciprocalLatticeVector(phase=al_phase, xyz=[[1, 0, 0], [1, 1, 1]])
        with pytest.raises(ValueError):
            sim = Simulation(
                phases=[al_phase, al_phase],
                simulation_generator=gen,
                coordinates=[[coords, coords], [coords, coords], [coords, coords]],
                rotations=[rot, rot],
            )

    def test_different_num_phase_and_rot(self, al_phase):
        gen = SimulationGenerator(accelerating_voltage=200)
        rot = Rotation.from_axes_angles([1, 0, 0], 45, degrees=True)
        coords = ReciprocalLatticeVector(phase=al_phase, xyz=[[1, 0, 0], [1, 1, 1]])
        with pytest.raises(ValueError):
            sim = Simulation(
                phases=[al_phase, al_phase],
                simulation_generator=gen,
                coordinates=[[coords, coords], [coords, coords], [coords, coords]],
                rotations=[rot, rot, rot],
            )


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
            phase=al_phase, xyz=[[1, 0, 0], [0, 1, 0], [1, 1, 0], [1, 1, 1]]
        )

        vectors = [coords, coords, coords, coords]

        sim = Simulation(
            phases=al_phase,
            simulation_generator=gen,
            coordinates=vectors,
            rotations=rot,
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
        assert sliced_sim.coordinates.size == 4

    def test_irot_slice(self, multi_simulation):
        sliced_sim = multi_simulation.irot[0:2]
        assert isinstance(sliced_sim, Simulation)
        assert isinstance(sliced_sim.phases, Phase)
        assert sliced_sim.rotations.size == 2
        assert sliced_sim.coordinates.size == 2

    def test_plot(self, multi_simulation):
        multi_simulation.plot()

    def test_plot_rotation(self, multi_simulation):
        multi_simulation.plot_rotations()

    def test_iter(self, multi_simulation):
        multi_simulation.phase_index = 0
        multi_simulation.rotation_index = 0
        count = 0
        for sim in multi_simulation:
            count += 1
            assert isinstance(sim, ReciprocalLatticeVector)
        assert count == 4

    def test_polar_flatten(self, multi_simulation):
        (
            r_templates,
            theta_templates,
            intensities_templates,
        ) = multi_simulation.polar_flatten_simulations()
        assert r_templates.shape == (4, 4)
        assert theta_templates.shape == (4, 4)
        assert intensities_templates.shape == (4, 4)


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
            phase=al_phase, xyz=[[1, 0, 0], [0, -1, 0], [1, -1, 0], [1, -1, -1]]
        )
        coords.intensity = 1
        vectors = [coords, coords, coords, coords]
        al_phase2 = al_phase.deepcopy()
        al_phase2.name = "al2"
        sim = Simulation(
            phases=[al_phase, al_phase2],
            simulation_generator=gen,
            coordinates=[vectors, vectors],
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

    def test_iphase_error(self, multi_simulation):
        with pytest.raises(ValueError):
            phase_slic = multi_simulation.iphase[3.1]

    def test_irot(self, multi_simulation):
        sliced_sim = multi_simulation.irot[0]
        assert isinstance(sliced_sim, Simulation)
        assert isinstance(sliced_sim.phases, np.ndarray)
        assert sliced_sim.rotations.size == 2

    def test_irot_slice(self, multi_simulation):
        sliced_sim = multi_simulation.irot[0:2]
        assert isinstance(sliced_sim, Simulation)
        assert isinstance(sliced_sim.phases, np.ndarray)
        assert sliced_sim.rotations.size == 2
        sliced_sim.plot()

    @pytest.mark.parametrize("show_labels", [True, False])
    @pytest.mark.parametrize("units", ["real", "pixel"])
    @pytest.mark.parametrize("include_zero_beam", [True, False])
    def test_plot(self, multi_simulation, show_labels, units, include_zero_beam):
        multi_simulation.phase_index = 0
        multi_simulation.rotation_index = 0
        multi_simulation.reciporical_radius = 2
        multi_simulation.coordinates[0][0].intensity = np.nan
        multi_simulation.plot(
            show_labels=show_labels,
            units=units,
            min_label_intensity=0.0,
            include_direct_beam=include_zero_beam,
            calibration=0.1,
        )

    def test_plot_rotation(self, multi_simulation):
        multi_simulation.plot_rotations()

    def test_iter(self, multi_simulation):
        multi_simulation.phase_index = 0
        multi_simulation.rotation_index = 0
        count = 0
        for sim in multi_simulation:
            count += 1
            assert isinstance(sim, ReciprocalLatticeVector)
        assert count == 8

    def test_get_diffraction_pattern(self, multi_simulation):
        pat = multi_simulation.get_diffraction_pattern(
            shape=(100, 100), calibration=0.01
        )
        assert pat.shape == (100, 100)
        assert np.max(pat.data) == 0

    def test_get_diffraction_pattern2(self, multi_simulation):
        pat = multi_simulation.get_diffraction_pattern(
            shape=(512, 512), calibration=0.01
        )
        assert pat.shape == (512, 512)
        assert np.max(pat.data) == 1

    def test_polar_flatten(self, multi_simulation):
        (
            r_templates,
            theta_templates,
            intensities_templates,
        ) = multi_simulation.polar_flatten_simulations()
        assert r_templates.shape == (8, 4)
        assert theta_templates.shape == (8, 4)
        assert intensities_templates.shape == (8, 4)

    def test_rotate_shift_coords(self, multi_simulation):
        rot = multi_simulation.rotate_shift_coordinates(angle=0.1)
        assert isinstance(rot, ReciprocalLatticeVector)
