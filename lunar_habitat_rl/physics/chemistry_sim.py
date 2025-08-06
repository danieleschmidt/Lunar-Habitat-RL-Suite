"""Chemical reaction and atmospheric chemistry simulation."""

from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import logging
from dataclasses import dataclass
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class ChemicalSpecies:
    """Chemical species definition."""
    name: str
    formula: str
    molecular_weight: float  # g/mol
    partial_pressure: float  # Pa
    concentration: float  # mol/m³
    

@dataclass
class ChemicalReaction:
    """Chemical reaction definition."""
    reactants: Dict[str, float]  # species: stoichiometry
    products: Dict[str, float]   # species: stoichiometry
    rate_constant: float         # reaction rate constant
    activation_energy: float     # J/mol
    temperature_dependent: bool = True


class ChemistrySimulator:
    """
    Atmospheric chemistry simulator for lunar habitat.
    
    Models chemical reactions, gas exchanges, and atmospheric composition
    changes over time including life support system interactions.
    """
    
    def __init__(self,
                 volume: float = 200.0,  # m³
                 pressure: float = 101325.0,  # Pa
                 temperature: float = 295.0,  # K
                 database: str = "nist"):
        """
        Initialize chemistry simulator.
        
        Args:
            volume: Habitat volume
            pressure: Initial pressure
            temperature: Initial temperature
            database: Reaction database to use
        """
        
        self.volume = volume
        self.pressure = pressure
        self.temperature = temperature
        self.database = database
        
        # Gas constant
        self.R = 8.314  # J/mol·K
        
        # Initialize chemical species
        self.species = self._initialize_species()
        
        # Load reaction database
        self.reactions = self._load_reaction_database()
        
        # Simulation state
        self.current_time = 0.0
        self.reaction_rates = defaultdict(float)
        
        logger.info(f"ChemistrySimulator initialized with {len(self.species)} species, "
                   f"{len(self.reactions)} reactions")
    
    def _initialize_species(self) -> Dict[str, ChemicalSpecies]:
        """Initialize atmospheric chemical species."""
        
        species = {}
        
        # Major atmospheric components
        species['O2'] = ChemicalSpecies(
            name='Oxygen',
            formula='O2',
            molecular_weight=32.0,
            partial_pressure=21300.0,  # Pa (21.3 kPa)
            concentration=0.0  # Will be computed from partial pressure
        )
        
        species['N2'] = ChemicalSpecies(
            name='Nitrogen',
            formula='N2', 
            molecular_weight=28.0,
            partial_pressure=79000.0,  # Pa (79.0 kPa)
            concentration=0.0
        )
        
        species['CO2'] = ChemicalSpecies(
            name='Carbon Dioxide',
            formula='CO2',
            molecular_weight=44.0,
            partial_pressure=400.0,  # Pa (0.4 kPa)
            concentration=0.0
        )
        
        species['H2O'] = ChemicalSpecies(
            name='Water Vapor',
            formula='H2O',
            molecular_weight=18.0,
            partial_pressure=2500.0,  # Pa (2.5 kPa, ~45% RH at 22°C)
            concentration=0.0
        )
        
        species['CO'] = ChemicalSpecies(
            name='Carbon Monoxide',
            formula='CO',
            molecular_weight=28.0,
            partial_pressure=0.1,  # Pa (trace)
            concentration=0.0
        )
        
        species['NH3'] = ChemicalSpecies(
            name='Ammonia',
            formula='NH3',
            molecular_weight=17.0,
            partial_pressure=0.01,  # Pa (trace)
            concentration=0.0
        )
        
        species['H2'] = ChemicalSpecies(
            name='Hydrogen',
            formula='H2',
            molecular_weight=2.0,
            partial_pressure=0.5,  # Pa (trace)
            concentration=0.0
        )
        
        species['CH4'] = ChemicalSpecies(
            name='Methane',
            formula='CH4',
            molecular_weight=16.0,
            partial_pressure=1.8,  # Pa (trace)
            concentration=0.0
        )
        
        # Update concentrations from partial pressures
        self._update_concentrations_from_pressure()
        
        return species
    
    def _load_reaction_database(self) -> List[ChemicalReaction]:
        """Load chemical reaction database."""
        
        reactions = []
        
        # Sabatier reaction (CO2 + 4H2 → CH4 + 2H2O)
        reactions.append(ChemicalReaction(
            reactants={'CO2': 1, 'H2': 4},
            products={'CH4': 1, 'H2O': 2},
            rate_constant=1e-6,  # m³/mol·s
            activation_energy=50000.0  # J/mol
        ))
        
        # Water-gas shift reaction (CO + H2O → CO2 + H2)
        reactions.append(ChemicalReaction(
            reactants={'CO': 1, 'H2O': 1},
            products={'CO2': 1, 'H2': 1},
            rate_constant=1e-4,
            activation_energy=30000.0
        ))
        
        # Methane combustion (CH4 + 2O2 → CO2 + 2H2O)
        reactions.append(ChemicalReaction(
            reactants={'CH4': 1, 'O2': 2},
            products={'CO2': 1, 'H2O': 2},
            rate_constant=1e-3,
            activation_energy=80000.0
        ))
        
        # Partial methane oxidation (2CH4 + 3O2 → 2CO + 4H2O)
        reactions.append(ChemicalReaction(
            reactants={'CH4': 2, 'O2': 3},
            products={'CO': 2, 'H2O': 4},
            rate_constant=1e-5,
            activation_energy=70000.0
        ))
        
        # Hydrogen combustion (2H2 + O2 → 2H2O)
        reactions.append(ChemicalReaction(
            reactants={'H2': 2, 'O2': 1},
            products={'H2O': 2},
            rate_constant=1e-2,
            activation_energy=40000.0
        ))
        
        # CO oxidation (2CO + O2 → 2CO2)
        reactions.append(ChemicalReaction(
            reactants={'CO': 2, 'O2': 1},
            products={'CO2': 2},
            rate_constant=1e-4,
            activation_energy=60000.0
        ))
        
        return reactions
    
    def _update_concentrations_from_pressure(self):
        """Update molar concentrations from partial pressures."""
        
        for species in self.species.values():
            # Ideal gas law: C = P / (RT)
            species.concentration = species.partial_pressure / (self.R * self.temperature)
    
    def _update_pressures_from_concentration(self):
        """Update partial pressures from molar concentrations."""
        
        total_pressure = 0.0
        
        for species in self.species.values():
            # Ideal gas law: P = CRT
            species.partial_pressure = species.concentration * self.R * self.temperature
            total_pressure += species.partial_pressure
        
        self.pressure = total_pressure
    
    def step(self,
             dt: float,
             o2_generation_rate: float,
             co2_scrubbing_rate: float,
             crew_metabolism: Dict[str, float],
             temperature: Optional[float] = None) -> Dict[str, Any]:
        """
        Advance chemistry simulation by one timestep.
        
        Args:
            dt: Timestep in seconds
            o2_generation_rate: O2 generation rate (0-1 normalized)
            co2_scrubbing_rate: CO2 scrubbing rate (0-1 normalized)
            crew_metabolism: Crew metabolic rates
            temperature: Optional temperature update
            
        Returns:
            Updated atmospheric composition and metrics
        """
        
        if temperature is not None:
            self.temperature = temperature
        
        # Apply life support system effects
        self._apply_life_support_systems(dt, o2_generation_rate, co2_scrubbing_rate)
        
        # Apply crew metabolism
        self._apply_crew_metabolism(dt, crew_metabolism)
        
        # Solve chemical reactions
        self._solve_chemical_kinetics(dt)
        
        # Update pressures from concentrations
        self._update_pressures_from_concentration()
        
        # Update time
        self.current_time += dt
        
        return {
            'o2_pressure': self.species['O2'].partial_pressure / 1000.0,  # kPa
            'co2_pressure': self.species['CO2'].partial_pressure / 1000.0,  # kPa
            'n2_pressure': self.species['N2'].partial_pressure / 1000.0,  # kPa
            'total_pressure': self.pressure / 1000.0,  # kPa
            'h2o_pressure': self.species['H2O'].partial_pressure / 1000.0,  # kPa
            'trace_species': self._get_trace_species_summary(),
            'reaction_rates': dict(self.reaction_rates),
            'chemical_balance': self._compute_chemical_balance()
        }
    
    def _apply_life_support_systems(self, dt: float, o2_gen_rate: float, co2_scrub_rate: float):
        """Apply life support system effects on atmospheric composition."""
        
        # Oxygen generation (electrolysis of water)
        # 2H2O → 2H2 + O2
        max_o2_generation = 2.0 / 86400.0  # kg/day → kg/s (max capacity)
        actual_o2_generation = o2_gen_rate * max_o2_generation  # kg/s
        
        # Convert to molar generation rate
        o2_molar_rate = actual_o2_generation / 0.032  # mol/s (MW of O2 = 32 g/mol)
        h2_molar_rate = 2 * o2_molar_rate  # Stoichiometry from electrolysis
        
        # Update concentrations
        self.species['O2'].concentration += (o2_molar_rate * dt) / self.volume
        self.species['H2'].concentration += (h2_molar_rate * dt) / self.volume
        
        # Water consumption for electrolysis
        h2o_consumed = 2 * o2_molar_rate  # mol/s
        self.species['H2O'].concentration -= (h2o_consumed * dt) / self.volume
        
        # CO2 scrubbing (absorption)
        max_co2_scrubbing = 1.5 / 86400.0  # kg/day → kg/s (max capacity)
        actual_co2_scrubbing = co2_scrub_rate * max_co2_scrubbing  # kg/s
        
        # Convert to molar scrubbing rate
        co2_molar_rate = actual_co2_scrubbing / 0.044  # mol/s (MW of CO2 = 44 g/mol)
        
        # Remove CO2
        co2_removed = (co2_molar_rate * dt) / self.volume
        self.species['CO2'].concentration = max(0.0, self.species['CO2'].concentration - co2_removed)
    
    def _apply_crew_metabolism(self, dt: float, crew_metabolism: Dict[str, float]):
        """Apply crew metabolic effects."""
        
        # Crew oxygen consumption
        o2_consumption_rate = crew_metabolism.get('o2_consumption_rate', 0.0)  # kg/s
        o2_molar_consumption = o2_consumption_rate / 0.032  # mol/s
        
        # Crew CO2 production  
        co2_production_rate = crew_metabolism.get('co2_production_rate', 0.0)  # kg/s
        co2_molar_production = co2_production_rate / 0.044  # mol/s
        
        # Crew water vapor production (breathing, sweating)
        h2o_production_rate = crew_metabolism.get('water_vapor_rate', 0.0)  # kg/s
        h2o_molar_production = h2o_production_rate / 0.018  # mol/s
        
        # Update concentrations
        self.species['O2'].concentration -= (o2_molar_consumption * dt) / self.volume
        self.species['CO2'].concentration += (co2_molar_production * dt) / self.volume
        self.species['H2O'].concentration += (h2o_molar_production * dt) / self.volume
        
        # Ensure non-negative concentrations
        for species in self.species.values():
            species.concentration = max(0.0, species.concentration)
    
    def _solve_chemical_kinetics(self, dt: float):
        """Solve chemical reaction kinetics."""
        
        # Reset reaction rates
        self.reaction_rates.clear()
        
        for i, reaction in enumerate(self.reactions):
            # Compute reaction rate
            rate = self._compute_reaction_rate(reaction)
            
            if rate > 1e-12:  # Only process significant reactions
                # Apply reaction stoichiometry
                self._apply_reaction_stoichiometry(reaction, rate, dt)
                
                # Store reaction rate for analysis
                self.reaction_rates[f'reaction_{i}'] = rate
    
    def _compute_reaction_rate(self, reaction: ChemicalReaction) -> float:
        """Compute reaction rate using chemical kinetics."""
        
        # Arrhenius equation: k = A * exp(-Ea / RT)
        if reaction.temperature_dependent:
            k_temp = reaction.rate_constant * np.exp(-reaction.activation_energy / (self.R * self.temperature))
        else:
            k_temp = reaction.rate_constant
        
        # Rate law (assuming elementary reactions)
        rate = k_temp
        
        # Multiply by reactant concentrations
        for species, stoich in reaction.reactants.items():
            if species in self.species:
                concentration = self.species[species].concentration
                rate *= concentration ** stoich
            else:
                rate = 0.0  # Species not present
                break
        
        return rate
    
    def _apply_reaction_stoichiometry(self, reaction: ChemicalReaction, rate: float, dt: float):
        """Apply reaction stoichiometry to species concentrations."""
        
        # Compute extent of reaction
        extent = rate * dt
        
        # Limit extent to prevent negative concentrations
        for species, stoich in reaction.reactants.items():
            if species in self.species:
                max_extent = self.species[species].concentration / stoich
                extent = min(extent, max_extent)
        
        # Apply stoichiometry
        # Consume reactants
        for species, stoich in reaction.reactants.items():
            if species in self.species:
                self.species[species].concentration -= stoich * extent
        
        # Produce products
        for species, stoich in reaction.products.items():
            if species in self.species:
                self.species[species].concentration += stoich * extent
            else:
                # Create new species if not present
                self.species[species] = ChemicalSpecies(
                    name=species,
                    formula=species,
                    molecular_weight=self._get_molecular_weight(species),
                    partial_pressure=0.0,
                    concentration=stoich * extent
                )
    
    def _get_molecular_weight(self, species: str) -> float:
        """Get molecular weight for species."""
        
        molecular_weights = {
            'O2': 32.0,
            'N2': 28.0,
            'CO2': 44.0,
            'H2O': 18.0,
            'CO': 28.0,
            'NH3': 17.0,
            'H2': 2.0,
            'CH4': 16.0,
            'C2H4': 28.0,
            'NO': 30.0,
            'NO2': 46.0,
        }
        
        return molecular_weights.get(species, 30.0)  # Default MW
    
    def _get_trace_species_summary(self) -> Dict[str, float]:
        """Get summary of trace species concentrations."""
        
        trace_species = {}
        
        for name, species in self.species.items():
            if name not in ['O2', 'N2', 'CO2', 'H2O']:
                if species.partial_pressure > 0.01:  # Pa threshold
                    trace_species[name] = species.partial_pressure
        
        return trace_species
    
    def _compute_chemical_balance(self) -> Dict[str, float]:
        """Compute chemical balance metrics."""
        
        # Oxygen balance
        o2_supply_rate = self.reaction_rates.get('o2_generation', 0.0)
        o2_consumption_rate = self.reaction_rates.get('o2_consumption', 0.0)
        o2_balance = o2_supply_rate - o2_consumption_rate
        
        # Carbon balance
        co2_production_rate = self.reaction_rates.get('co2_production', 0.0)  
        co2_removal_rate = self.reaction_rates.get('co2_scrubbing', 0.0)
        c_balance = co2_production_rate - co2_removal_rate
        
        # Water balance
        h2o_production_rate = self.reaction_rates.get('h2o_production', 0.0)
        h2o_consumption_rate = self.reaction_rates.get('h2o_consumption', 0.0)
        h2o_balance = h2o_production_rate - h2o_consumption_rate
        
        return {
            'oxygen_balance': o2_balance,
            'carbon_balance': c_balance,
            'water_balance': h2o_balance,
            'total_moles': sum(species.concentration * self.volume for species in self.species.values()),
            'chemical_stability': self._assess_chemical_stability()
        }
    
    def _assess_chemical_stability(self) -> float:
        """Assess overall chemical stability of atmosphere."""
        
        # Check if major species are within acceptable ranges
        o2_stable = 16.0 <= self.species['O2'].partial_pressure / 1000.0 <= 25.0  # kPa
        co2_stable = self.species['CO2'].partial_pressure / 1000.0 <= 1.0  # kPa
        
        # Check for toxic species
        toxic_threshold_exceeded = False
        toxic_species = ['CO', 'NH3', 'NO', 'NO2']
        
        for species_name in toxic_species:
            if species_name in self.species:
                if species_name == 'CO' and self.species[species_name].partial_pressure > 100.0:  # Pa
                    toxic_threshold_exceeded = True
                elif species_name == 'NH3' and self.species[species_name].partial_pressure > 50.0:  # Pa
                    toxic_threshold_exceeded = True
        
        # Compute stability score
        if o2_stable and co2_stable and not toxic_threshold_exceeded:
            stability = 1.0
        elif o2_stable and co2_stable:
            stability = 0.8  # Acceptable but with trace contaminants
        elif o2_stable or co2_stable:
            stability = 0.5  # Partial stability
        else:
            stability = 0.0  # Unstable atmosphere
        
        return stability
    
    def get_atmospheric_composition(self) -> Dict[str, Dict[str, float]]:
        """Get complete atmospheric composition."""
        
        composition = {}
        
        for name, species in self.species.items():
            composition[name] = {
                'partial_pressure_kPa': species.partial_pressure / 1000.0,
                'concentration_mol_m3': species.concentration,
                'mole_fraction': species.partial_pressure / self.pressure if self.pressure > 0 else 0.0,
                'mass_fraction': self._compute_mass_fraction(species)
            }
        
        return composition
    
    def _compute_mass_fraction(self, species: ChemicalSpecies) -> float:
        """Compute mass fraction of species."""
        
        # Mass = concentration * molecular_weight * volume
        species_mass = species.concentration * species.molecular_weight * self.volume  # g
        
        # Total mass
        total_mass = sum(
            spec.concentration * spec.molecular_weight * self.volume 
            for spec in self.species.values()
        )
        
        if total_mass > 0:
            return species_mass / total_mass
        else:
            return 0.0
    
    def detect_contamination(self, threshold_ppm: float = 10.0) -> Dict[str, float]:
        """Detect atmospheric contamination above threshold."""
        
        contamination = {}
        
        # Convert threshold from ppm to partial pressure (Pa)
        threshold_pa = threshold_ppm * 1e-6 * self.pressure
        
        contaminant_species = ['CO', 'NH3', 'CH4', 'NO', 'NO2']
        
        for species_name in contaminant_species:
            if species_name in self.species:
                pressure_pa = self.species[species_name].partial_pressure
                
                if pressure_pa > threshold_pa:
                    contamination[species_name] = pressure_pa * 1e6 / self.pressure  # ppm
        
        return contamination
    
    def simulate_leak(self, leak_rate: float, duration: float) -> Dict[str, Any]:
        """Simulate atmospheric leak scenario."""
        
        results = {
            'pressure_loss': [],
            'composition_changes': [],
            'time_to_critical': None
        }
        
        initial_pressure = self.pressure
        dt = 60.0  # 1 minute timesteps
        time = 0.0
        
        while time < duration:
            # Apply leak (proportional to current pressure)
            leak_fraction = leak_rate * dt / self.volume  # fraction per timestep
            
            # Reduce all species proportionally
            for species in self.species.values():
                species.concentration *= (1.0 - leak_fraction)
            
            # Update pressures
            self._update_pressures_from_concentration()
            
            # Store results
            results['pressure_loss'].append(self.pressure / 1000.0)  # kPa
            results['composition_changes'].append(self.get_atmospheric_composition())
            
            # Check for critical pressure
            if self.pressure < 50000.0 and results['time_to_critical'] is None:  # 50 kPa
                results['time_to_critical'] = time
            
            time += dt
        
        logger.info(f"Leak simulation completed: {initial_pressure/1000:.1f} → {self.pressure/1000:.1f} kPa")
        
        return results