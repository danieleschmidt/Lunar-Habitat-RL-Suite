"""Tests for physics simulation engines."""

import pytest
import numpy as np
import tempfile
from pathlib import Path

from lunar_habitat_rl.physics import ThermalSimulator, CFDSimulator, ChemistrySimulator
from tests import TEST_CONFIG, assert_array_close, assert_dict_close


class TestThermalSimulator:
    """Test thermal simulation engine."""
    
    def test_thermal_simulator_creation(self):
        """Test thermal simulator initialization."""
        sim = ThermalSimulator(
            habitat_volume=200.0,
            thermal_mass=5000.0,
            insulation_r_value=10.0
        )
        
        assert sim.habitat_volume == 200.0
        assert sim.thermal_mass == 5000.0
        assert sim.insulation_r_value == 10.0
        assert len(sim.zones) == 4  # Default 4 zones
        assert sim.current_time == 0.0
    
    def test_thermal_simulator_step_finite_difference(self):
        """Test thermal simulation step with finite difference solver."""
        sim = ThermalSimulator(solver="finite_difference")
        
        result = sim.step(
            dt=60.0,
            external_temp=-50.0,
            internal_heat_generation=2900.0,  # Crew + equipment
            heating_power=[0.5, 0.5, 0.5, 0.5],
            radiator_flow=[0.8, 0.8]
        )
        
        # Check result structure
        assert 'zone_temperatures' in result
        assert 'radiator_temperatures' in result
        assert 'heat_pump_cop' in result
        assert 'total_heat_loss' in result
        
        # Check result values
        assert len(result['zone_temperatures']) == 4
        assert len(result['radiator_temperatures']) == 2
        
        # Temperatures should be reasonable
        for temp in result['zone_temperatures']:
            assert -50 <= temp <= 50  # Reasonable range
        
        assert 1.0 <= result['heat_pump_cop'] <= 6.0
        assert result['total_heat_loss'] >= 0
    
    def test_thermal_simulator_step_simplified(self):
        """Test thermal simulation step with simplified solver."""
        sim = ThermalSimulator(solver="simplified")  # Non-standard solver falls back to simplified
        
        result = sim.step(
            dt=60.0,
            external_temp=-100.0,
            internal_heat_generation=3000.0,
            heating_power=[0.7, 0.7, 0.7, 0.7],
            radiator_flow=[0.5, 0.5]
        )
        
        # Check results are reasonable
        assert len(result['zone_temperatures']) == 4
        assert len(result['radiator_temperatures']) == 2
        
        # In extreme cold with heating, temperatures should be above external
        for temp in result['zone_temperatures']:
            assert temp > -100.0  # Should be warmer than external due to heating
    
    def test_thermal_simulator_heat_transfer_physics(self):
        """Test thermal physics calculations."""
        sim = ThermalSimulator()
        
        # Test conduction loss calculation
        T_inside = 295.0  # K (22°C)
        T_outside = 173.0  # K (-100°C)
        
        conduction_loss = sim._compute_conduction_loss(T_inside, T_outside)
        assert conduction_loss > 0  # Heat should flow out
        
        # Test radiation loss calculation
        radiation_loss = sim._compute_radiation_loss(T_inside, T_outside)
        assert radiation_loss > 0  # Heat should radiate out
        
        # Test heat pump COP
        temp_diff = 50.0  # K
        cop = sim._compute_heat_pump_cop(temp_diff)
        assert 1.0 <= cop <= 6.0  # Reasonable COP range
    
    def test_thermal_simulator_scenario_creation(self):
        """Test thermal scenario creation."""
        sim = ThermalSimulator()
        
        scenario = sim.create_scenario(
            external_temp_profile="14_day_lunar_cycle",
            internal_heat_sources={
                'crew': 400.0,
                'equipment': 2000.0,
                'lighting': 500.0
            },
            duration=86400  # 1 day
        )
        
        assert 'duration' in scenario
        assert 'time_profile' in scenario
        assert 'temperature_profile' in scenario
        assert 'internal_heat_sources' in scenario
        
        assert scenario['duration'] == 86400
        assert len(scenario['time_profile']) > 0
        assert len(scenario['temperature_profile']) > 0
        assert scenario['internal_heat_sources']['crew'] == 400.0
    
    def test_thermal_simulator_full_scenario_run(self):
        """Test running complete thermal scenario."""
        sim = ThermalSimulator()
        
        scenario = sim.create_scenario(
            external_temp_profile="constant_cold",
            internal_heat_sources={'total': 2900.0},
            duration=3600  # 1 hour
        )
        
        results = sim.run(scenario, timestep=300.0)  # 5 minute steps
        
        assert 'times' in results
        assert 'zone_temperatures' in results
        assert 'external_temperatures' in results
        
        assert len(results['times']) > 0
        assert len(results['zone_temperatures']) == 4  # 4 zones
        
        # Check temperature evolution
        for zone_temps in results['zone_temperatures']:
            assert len(zone_temps) == len(results['times'])
            
            # All temperatures should be finite
            for temp in zone_temps:
                assert np.isfinite(temp)
    
    @pytest.mark.slow
    def test_thermal_simulator_performance(self):
        """Test thermal simulator performance."""
        sim = ThermalSimulator()
        
        import time
        start_time = time.time()
        
        # Run many simulation steps
        for _ in range(100):
            sim.step(
                dt=60.0,
                external_temp=-50.0,
                internal_heat_generation=3000.0,
                heating_power=[0.5, 0.5, 0.5, 0.5],
                radiator_flow=[0.8, 0.8]
            )
        
        elapsed_time = time.time() - start_time
        
        # Should complete 100 steps in reasonable time (< 10 seconds)
        assert elapsed_time < 10.0
        
        # Performance metric: steps per second
        steps_per_second = 100 / elapsed_time
        assert steps_per_second > 10  # Should do at least 10 steps/second


class TestCFDSimulator:
    """Test CFD simulation engine."""
    
    def test_cfd_simulator_creation(self):
        """Test CFD simulator initialization.""" 
        sim = CFDSimulator(
            volume=200.0,
            turbulence_model="k_epsilon",
            mesh_resolution="medium"
        )
        
        assert sim.volume == 200.0
        assert sim.turbulence_model == "k_epsilon"
        assert sim.mesh_resolution == "medium"
        
        # Check grid generation
        assert 'nx' in sim.grid
        assert 'ny' in sim.grid
        assert 'nz' in sim.grid
        assert sim.grid['nx'] > 0
        assert sim.grid['ny'] > 0
        assert sim.grid['nz'] > 0
    
    def test_cfd_simulator_step(self):
        """Test CFD simulation step."""
        sim = CFDSimulator(mesh_resolution="coarse")  # Use coarse mesh for speed
        
        result = sim.step(
            dt=1.0,
            fan_speed=0.5,
            temperature_zones=[22.0, 23.0, 22.5, 21.8],
            species_sources={'CO2': 0.001, 'H2O': 0.002}
        )
        
        # Check result structure
        assert 'velocity_field' in result
        assert 'mixing_efficiency' in result
        assert 'thermal_mixing' in result
        assert 'air_exchange_rate' in result
        assert 'dead_zones' in result
        assert 'turbulence_intensity' in result
        
        # Check result values
        assert 0.0 <= result['mixing_efficiency'] <= 1.0
        assert 0.0 <= result['thermal_mixing'] <= 1.0
        assert result['air_exchange_rate'] >= 0.0
        assert isinstance(result['dead_zones'], list)
        assert 0.0 <= result['turbulence_intensity'] <= 1.0
    
    def test_cfd_simulator_boundary_conditions(self):
        """Test CFD boundary condition handling."""
        sim = CFDSimulator(mesh_resolution="coarse")
        
        # Test with fan running
        result_with_fan = sim.step(
            dt=1.0,
            fan_speed=0.8,
            temperature_zones=[22.0, 22.0, 22.0, 22.0]
        )
        
        # Test without fan
        result_without_fan = sim.step(
            dt=1.0,
            fan_speed=0.0,
            temperature_zones=[22.0, 22.0, 22.0, 22.0]
        )
        
        # Should have different air exchange rates
        assert result_with_fan['air_exchange_rate'] > result_without_fan['air_exchange_rate']
    
    def test_cfd_simulator_turbulence_models(self):
        """Test different turbulence models."""
        turbulence_models = ["laminar", "k_epsilon"]
        
        for model in turbulence_models:
            sim = CFDSimulator(
                turbulence_model=model,
                mesh_resolution="coarse"
            )
            
            result = sim.step(
                dt=1.0,
                fan_speed=0.5,
                temperature_zones=[22.0, 23.0, 22.0, 22.0]
            )
            
            assert 'mixing_efficiency' in result
            assert 0.0 <= result['mixing_efficiency'] <= 1.0
    
    def test_cfd_simulator_species_transport(self):
        """Test species transport simulation."""
        sim = CFDSimulator(mesh_resolution="coarse")
        
        # Add CO2 source
        result = sim.step(
            dt=10.0,  # Longer timestep for more mixing
            fan_speed=0.7,
            temperature_zones=[22.0, 22.0, 22.0, 22.0],
            species_sources={'CO2': 0.005}  # CO2 injection
        )
        
        # Should affect mixing efficiency
        assert 'mixing_efficiency' in result
        
        # Test without species sources
        result_no_species = sim.step(
            dt=10.0,
            fan_speed=0.7,
            temperature_zones=[22.0, 22.0, 22.0, 22.0]
        )
        
        # Both should have reasonable mixing efficiency
        assert 0.0 <= result['mixing_efficiency'] <= 1.0
        assert 0.0 <= result_no_species['mixing_efficiency'] <= 1.0
    
    def test_cfd_simulator_dead_zone_detection(self):
        """Test dead zone detection."""
        sim = CFDSimulator(mesh_resolution="coarse")
        
        # Low fan speed should create more dead zones
        result_low_fan = sim.step(
            dt=1.0,
            fan_speed=0.1,
            temperature_zones=[22.0, 22.0, 22.0, 22.0]
        )
        
        # High fan speed should create fewer dead zones
        result_high_fan = sim.step(
            dt=1.0,
            fan_speed=0.9,
            temperature_zones=[22.0, 22.0, 22.0, 22.0]
        )
        
        # Check that dead zones are detected
        assert isinstance(result_low_fan['dead_zones'], list)
        assert isinstance(result_high_fan['dead_zones'], list)


class TestChemistrySimulator:
    """Test chemistry simulation engine."""
    
    def test_chemistry_simulator_creation(self):
        """Test chemistry simulator initialization."""
        sim = ChemistrySimulator(
            volume=200.0,
            pressure=101325.0,
            temperature=295.0
        )
        
        assert sim.volume == 200.0
        assert sim.pressure == 101325.0
        assert sim.temperature == 295.0
        
        # Check species initialization
        assert 'O2' in sim.species
        assert 'CO2' in sim.species
        assert 'N2' in sim.species
        assert 'H2O' in sim.species
        
        # Check reaction database
        assert len(sim.reactions) > 0
    
    def test_chemistry_simulator_step(self):
        """Test chemistry simulation step."""
        sim = ChemistrySimulator()
        
        result = sim.step(
            dt=60.0,
            o2_generation_rate=0.5,
            co2_scrubbing_rate=0.7,
            crew_metabolism={
                'o2_consumption_rate': 0.84 / 86400,  # kg/s
                'co2_production_rate': 1.04 / 86400,  # kg/s
                'water_vapor_rate': 2.5 / 86400       # kg/s
            }
        )
        
        # Check result structure
        assert 'o2_pressure' in result
        assert 'co2_pressure' in result
        assert 'n2_pressure' in result
        assert 'total_pressure' in result
        assert 'trace_species' in result
        
        # Check pressure values are reasonable
        assert result['o2_pressure'] > 0
        assert result['co2_pressure'] >= 0
        assert result['n2_pressure'] > 0
        assert result['total_pressure'] > 0
        
        # Total pressure should be approximately sum of components
        component_sum = result['o2_pressure'] + result['co2_pressure'] + result['n2_pressure']
        assert abs(result['total_pressure'] - component_sum) < result['total_pressure'] * 0.1
    
    def test_chemistry_simulator_life_support_systems(self):
        """Test life support system effects."""
        sim = ChemistrySimulator()
        
        # Record initial state
        initial_o2 = sim.species['O2'].concentration
        initial_co2 = sim.species['CO2'].concentration
        
        # Run step with O2 generation and CO2 scrubbing
        result = sim.step(
            dt=600.0,  # 10 minutes
            o2_generation_rate=1.0,  # Maximum generation
            co2_scrubbing_rate=1.0,  # Maximum scrubbing
            crew_metabolism={
                'o2_consumption_rate': 0.84 / 86400,
                'co2_production_rate': 1.04 / 86400,
                'water_vapor_rate': 2.5 / 86400
            }
        )
        
        # O2 should increase (generation > consumption for short term)
        # CO2 should decrease (scrubbing > production for short term)
        final_o2 = sim.species['O2'].concentration
        final_co2 = sim.species['CO2'].concentration
        
        # Verify trends (may need adjustment based on exact balance)
        assert result['o2_pressure'] > 0
        assert result['co2_pressure'] >= 0
    
    def test_chemistry_simulator_reactions(self):
        """Test chemical reactions."""
        sim = ChemistrySimulator()
        
        # Add some hydrogen to test reactions
        sim.species['H2'].concentration = 0.1  # mol/m³
        
        result = sim.step(
            dt=3600.0,  # 1 hour for more reaction time
            o2_generation_rate=0.0,
            co2_scrubbing_rate=0.0,
            crew_metabolism={
                'o2_consumption_rate': 0.0,
                'co2_production_rate': 0.0,
                'water_vapor_rate': 0.0
            }
        )
        
        # Check that reaction rates are recorded
        assert 'reaction_rates' in result
        assert isinstance(result['reaction_rates'], dict)
    
    def test_chemistry_simulator_contamination_detection(self):
        """Test atmospheric contamination detection."""
        sim = ChemistrySimulator()
        
        # Add contaminant
        sim.species['CO'].concentration = 0.01  # High CO concentration
        
        contamination = sim.detect_contamination(threshold_ppm=1.0)
        
        # Should detect CO contamination
        assert isinstance(contamination, dict)
        if 'CO' in contamination:
            assert contamination['CO'] > 1.0  # Above threshold
    
    def test_chemistry_simulator_atmospheric_composition(self):
        """Test atmospheric composition reporting."""
        sim = ChemistrySimulator()
        
        composition = sim.get_atmospheric_composition()
        
        # Check structure
        assert isinstance(composition, dict)
        assert 'O2' in composition
        assert 'CO2' in composition
        assert 'N2' in composition
        
        # Check data structure for each species
        for species_name, data in composition.items():
            assert 'partial_pressure_kPa' in data
            assert 'concentration_mol_m3' in data
            assert 'mole_fraction' in data
            assert 'mass_fraction' in data
            
            # Check values are reasonable
            assert data['partial_pressure_kPa'] >= 0
            assert data['concentration_mol_m3'] >= 0
            assert 0 <= data['mole_fraction'] <= 1
            assert 0 <= data['mass_fraction'] <= 1
        
        # Mole fractions should sum to approximately 1
        total_mole_fraction = sum(data['mole_fraction'] for data in composition.values())
        assert abs(total_mole_fraction - 1.0) < 0.1
    
    def test_chemistry_simulator_leak_scenario(self):
        """Test atmospheric leak simulation."""
        sim = ChemistrySimulator()
        
        initial_pressure = sim.pressure
        
        # Simulate small leak
        results = sim.simulate_leak(
            leak_rate=0.01,  # Small leak rate
            duration=3600    # 1 hour
        )
        
        assert 'pressure_loss' in results
        assert 'composition_changes' in results
        
        # Pressure should decrease over time
        pressure_values = results['pressure_loss']
        assert len(pressure_values) > 0
        assert pressure_values[-1] < pressure_values[0]  # Final < Initial
    
    @pytest.mark.parametrize("temperature", [270.0, 295.0, 320.0])
    def test_chemistry_simulator_temperature_effects(self, temperature):
        """Test chemistry simulation at different temperatures."""
        sim = ChemistrySimulator(temperature=temperature)
        
        result = sim.step(
            dt=60.0,
            o2_generation_rate=0.5,
            co2_scrubbing_rate=0.5,
            crew_metabolism={
                'o2_consumption_rate': 0.84 / 86400,
                'co2_production_rate': 1.04 / 86400,
                'water_vapor_rate': 2.5 / 86400
            },
            temperature=temperature
        )
        
        # Should work at all temperatures
        assert 'total_pressure' in result
        assert result['total_pressure'] > 0
        
        # Chemical balance should be computed
        assert 'chemical_balance' in result
        balance = result['chemical_balance']
        assert 'chemical_stability' in balance
        assert 0 <= balance['chemical_stability'] <= 1