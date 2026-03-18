import numpy as np
import os
import toml
import h5py

import rust_mucus as rmc

from .config import Config
from .topology import Topology
from .utils import get_path, get_number_of_frames, Filetypes #, get_forces_cell_linked_virial_cuda, flatten_bond_list
from pathlib import Path
from tqdm import tqdm

from time import time
from copy import deepcopy 

# TODO:
# - remove distance writing option (deprecated)
# - write an integrator function and call it during step()
# - create seperate step function that is called during simulate() and can be used for single stepping 
# - handle I/0 properly: 
#    - use context manager for h5py files
#    - create Writer class that can be called during step() 
# - get rid of unnecessary null block in __init__
# - get rid of the active particle handling demo code

class System:
    
    def __init__(self, config: Config,  topology: Topology = None):

        self.config = config
        self.topology = topology if topology is not None else Topology(self.config)
        
        self.timestep               = None
        self.chunksize              = None
        self.n_particles            = None
        self.particle_indices       = None
        self.B_debye                = None    # move into topology    
        self.box_length             = None
        self.positions              = None
        self.forces                 = None
        self.energy                 = None  # not used
        
        # TODO THESE MUST ONLY BE DEFINED IF write is set to true respectively
        self.traj_chunk             = None
        self.force_chunk            = None
        self.dist_chunk             = None
        
        # TODO IMPLEMENT THIS PROPERLY
        self.mobility_list          = None
        
        self.bond_list              = None
        self.cell_index             = None
        self.neighbor_cells_idx     = None
        self.head_array             = None
        self.list_array             = None
        
        self.n_cells                = None
        self.cell_length            = None
        self.n_neighbor_cells       = None
        
        self.active_particle_present    = None
        self.active_particle_indices    = None
        self.n_active_particles         = None
        
        self.virial                 = None                    
        
        self.setup()
        
    #! TODO TODO TODO TODO TODO TODO
    # ADD "use forces" TO TOPOLOGY  
      
    def setup(self):
        
        self.n_particles        = self.config.n_particles
        self.particle_indices   = np.arange(self.n_particles)
        self.box_length         = self.config.lbox
        self.timestep           = self.config.timestep                                
        
        self.traj_chunk      = np.zeros((self.config.chunksize, self.n_particles, 3))
        self.force_chunk     = np.zeros((self.config.chunksize, self.n_particles, 3))
        
        if self.config.write_distances:
            self.dist_chunk      = np.zeros((self.config.chunksize, self.n_particles, self.n_particles), dtype=np.float64)
        else:
            self.dist_chunk      = np.zeros((self.config.chunksize, 1, 1), dtype=np.float64)
        
        self.forces          = np.zeros((self.n_particles, 3), dtype=np.float64)
        
        # create bond list
        self.bond_list = [[] for _ in range(self.n_particles)]
        for i, j in self.topology.bonds:
            self.bond_list[i].append(j)
        
        # TODO implement this properly
        self.B_debye         = np.sqrt(self.config.c_S)*self.config.r0_nm/10 # from the relationship in the Hansing thesis [c_S] = mM
        self.mobility_list   = np.array([self.topology.mobility[i] for i in self.topology.tags], ndmin=2).reshape(-1, 1)
        
        # TODO delete this nonsense
        # calculate debye cutoff from salt concentration
        if self.config.cutoff_debye == None:
            self.get_cutoff_debye()
        
        # TODO at this stage of the project using pbc is necessary for the simulations to work
        # check for pbc
        if self.config.pbc == True: 
            
            if self.config.cutoff_pbc is None:
                # NOTE here the cutoff of the force with the longest range is used
                cutoff = np.max((self.config.cutoff_debye, self.config.cutoff_LJ))
                # minimal possible cutoff is 1.5*r0
                # otherwise nn calculation breaks
                if cutoff < 4:
                    if self.config.lbox < 8:
                        raise ValueError("Box length is too small for cutoff = 4")
                    cutoff = 4
                self.config.cutoff_pbc = cutoff
        
        # load positions
        self.set_positions(self.topology.positions)
        
        # Define all necessary cell-linked list arrays
        # NOTE the following only works if pbc is used
        # TODO write a seperate function to initialize the cell linked list arrays 
        
        # calculate the number of cells in each direction
        self.n_cells = int(self.box_length/self.config.cutoff_pbc)
        self.cell_length = float(self.box_length/self.n_cells)
        
        # get the indices of the neighboring cells and self with shape (n_neighbor_cells + 1, 3)
        self.neighbor_cells_idx = np.array(np.indices((3, 3, 3), dtype=int).reshape(3, -1).T - 1).astype(int)
        self.n_neighbor_cells = self.neighbor_cells_idx.shape[0]
        
        self.apply_pbc()
        
        self.head_array = -np.ones((self.n_cells, self.n_cells, self.n_cells), dtype=int)
        self.list_array = -np.ones(self.n_particles, dtype=int)
        
        rmc.update_linked_list(
            self.list_array,
            self.head_array,
            self.positions,
            self.cell_length
        )
    
        self.active_particle_present = self.topology.tag_active_particle is not None

        if self.active_particle_present:
            self.active_particle_indices = np.where(self.topology.tags == self.topology.tag_active_particle)[0]
            self.n_active_particles = len(self.active_particle_indices)
            
        self.virial = np.zeros((3,3), dtype=float)
        if self.config.calc_virial:
            self.write_virial = True
        else:
            self.write_virial = False
            
        if self.config.stride_virial is None:
            self.config.stride_virial = self.config.stride
            
    
    # TODO MOVE THIS TO config: something like config.print()
    def print_sim_info(self):
        """
        print the config
        """
        
        # this is done so the version is printed but the class variable is not updated
        cfg = deepcopy(self.config)
        
        # print everything but bonds
        output = str(cfg).split("bonds")[0]
        output = output.replace(" ", "\n")
        output = output.replace("=", " = ")
        print(output)
        return
    
    def set_timestep(self, timestep):
        
        self.timestep = timestep
        
        return
    
    def set_positions(self, pos):
        
        self.positions = np.array(pos, dtype=np.float64)
        
        return
    
    def set_cutoff(self, cutoff, ctype = "pbc"):
        
        if ctype.lower() == "pbc":
            self.config.cutoff_pbc = cutoff
        elif ctype.lower() == "lj":
            self.cutoff_LJ = cutoff
        elif ctype.lower() == "debye":
            self.config.cutoff_debye = cutoff
        else:
            raise TypeError(f"Cutoff type \'{ctype:s}\' does not exist.")
        
        return
    
    # TODO MOVE THIS TO UTILS
    def get_cutoff_debye(self, eps=1):
        """
        use the maximum charge in the config to determine the debey force cutoff
        
        the distance, where the force is smaller than the treshold eps is the debey cutoff
        """
        r = 0
        dr = 0.05
        force = 9999
        # TODO CHANGE TO MAX DISPLACEMENT = 0.1 
        while np.max(force) > eps:
            r += dr
            force = np.max(self.topology.q_particle)**2*self.config.lB_debye*(1+self.B_debye*r)*np.exp(-self.B_debye*r)/r**2
            
        self.config.cutoff_debye = r
        
        return 
    
    def force_Random(self):
        """
        Gaussian random Force with a per particle standard deviation of sqrt(6 mu_0) w.r.t. its absolute value
        """
        
        # since the std of the foce should be sqrt(6*mu) but the std of the absolute randn vector is sqrt(3)
        # the std used here is sqrt(2*mu)
        
        # TODO THIS WILL BREAK WHEN MOBILITY IS CHANGED TO SHAPE (ntags)
        return np.sqrt(2*self.timestep*self.mobility_list)*np.random.randn(self.n_particles, 3)
    
    def force_Random_Correlated(self, step):
        
        force = step*np.random.randn(self.n_active_particles, 3)
        
        return force
    
    def apply_pbc(self):
        """
        Repositions all atoms outside of the box to the other end of the box
        """
        
        # calculate modulo L
        self.positions = np.mod(self.positions, self.box_length)
        
        return
    
    def simulate(self):
        """
        Simulates the overdamped Langevin dynamics of the system with the defined forcefield using forward Euler.
        """
        
        n_frames = get_number_of_frames(self.config)
        n_frames_virial = int(self.config.steps/self.config.stride_virial)
        # create datasets
        
        # save initial pos
        if self.config.write_traj==True:
            fh5_traj = h5py.File(get_path(self.config, filetype=Filetypes.Trajectory), 'w-')
            fh5_traj.create_dataset(Filetypes.Trajectory.key, shape=(n_frames, self.n_particles, 3), dtype="float32")
        
        # define flag for distance writing
        if self.config.write_distances:
            write_distances = True
            fh5_distances = h5py.File(get_path(self.config, filetype=Filetypes.Distances), 'w-')
            fh5_distances.create_dataset(Filetypes.Distances.key, shape=(n_frames, self.n_particles, self.n_particles), dtype="float16")
        else:    
            write_distances = False
        
        if self.config.write_forces:
            fh5_forces = h5py.File(get_path(self.config, filetype=Filetypes.Forces), 'w-')
            fh5_forces.create_dataset(Filetypes.Forces.key, shape=(n_frames, self.n_particles, 3), dtype="float32")
        
        if self.config.calc_virial:
            calc_virial = True
            fh5_virial = h5py.File(get_path(self.config, filetype=Filetypes.Virial), 'w-')
            fh5_virial.create_dataset(Filetypes.Virial.key, shape=(n_frames_virial, 3, 3), dtype="float32")
        else:
            calc_virial = False
            
        idx_chunk = 0 
        idx_traj = 0
        idx_virial = 0
        
        # flatten bond_list (for cuda implementation)
        # bond_offsets, bond_neighbors = flatten_bond_list(self.bond_list)
        
        # testing purposes: 
        print("force cutoff: config.cutoff_pbc", self.config.cutoff_pbc)
        
        print(f"\nStarting simulation with {self.config.steps} steps.")
        t_start = time()
        for step in tqdm(range(self.config.steps)):
            
            # apply periodic boundary conditions (0, L) x (0, L) x (0, L)
            self.apply_pbc()
            
            rmc.update_linked_list(
                self.list_array,
                self.head_array,
                self.positions,
                self.cell_length
            )
            
            # TODO hack: remove later
            if step%self.config.stride_virial==0 and self.config.calc_virial:
                calc_virial = True 
            
            # calculate forces
            self.forces.fill(0)
            
            # rmc.get_forces_cell_linked_test(
            rmc.get_forces_cell_linked_virial(
                self.positions,
                self.topology.tags,
                self.bond_list,
                self.topology.force_constant_nn,
                self.topology.r0_bond,
                self.topology.sigma_lj,
                self.topology.epsilon_lj,
                self.topology.q_particle,
                self.config.lB_debye,
                self.B_debye,
                self.forces,
                self.dist_chunk[idx_chunk],
                self.box_length,
                self.config.cutoff_pbc**2,
                self.n_particles,
                3,                                              # number of spacial dimensions
                write_distances,                                #! deprecated
                self.config.use_pot_bond,                       # use bond force
                self.config.use_pot_WCA,                        # use LJ force
                self.config.use_pot_debye,                      # use Debye force
                self.neighbor_cells_idx,
                self.head_array,
                self.list_array,
                self.n_cells,
                self.n_neighbor_cells,
                calc_virial,
                self.virial
            )
            
            # get_forces_cell_linked_virial_cuda(
            #     self.positions,
            #     self.topology.tags,
            #     bond_offsets,
            #     bond_neighbors,
            #     self.topology.force_constant_nn,
            #     self.topology.r0_bond,
            #     self.topology.sigma_lj,
            #     self.topology.epsilon_lj,
            #     self.topology.q_particle,
            #     self.config.lB_debye,
            #     self.B_debye,
            #     self.forces,
            #     self.dist_chunk[idx_chunk],
            #     self.box_length,
            #     self.config.cutoff_pbc**2,
            #     self.n_particles,
            #     3,                                              # number of spacial dimensions
            #     write_distances,                                #! deprecated
            #     self.config.use_pot_bond,                       # use bond force
            #     self.config.use_pot_WCA,                        # use LJ force
            #     self.config.use_pot_debye,                      # use Debye force
            #     self.neighbor_cells_idx,
            #     self.head_array,
            #     self.list_array,
            #     self.n_cells,
            #     self.n_neighbor_cells,
            #     calc_virial,
            #     self.virial)
            
            # reset distance flag until next stride
            # TODO remove this
            write_distances = False
            
            # TODO implement virial chunk
            if step%self.config.stride_virial==0 and self.config.calc_virial:
                fh5_virial[Filetypes.Virial.key][idx_virial] = self.virial
                self.virial.fill(0.0)
                calc_virial = False
                idx_virial += 1
            
            if step%self.config.stride==0:  
              
                
                if self.config.write_traj:    
                    self.traj_chunk[idx_chunk] = self.positions
                
                if self.config.write_forces:
                    self.force_chunk[idx_chunk] = self.forces
                    
                if self.config.write_distances:
                    write_distances = True
                
                
                idx_chunk += 1
                
                if idx_chunk == self.config.chunksize:
                    if self.config.write_traj:
                        fh5_traj["trajectory"][idx_traj:idx_traj+self.config.chunksize] = self.traj_chunk
                    
                    if self.config.write_forces:    
                        fh5_forces["forces"][idx_traj:idx_traj+self.config.chunksize] = self.force_chunk
                        
                    if self.config.write_distances:
                        fh5_distances["distances"][idx_traj:idx_traj+self.config.chunksize] = self.dist_chunk
                    
                    idx_traj += self.config.chunksize
                    idx_chunk = 0

            
            # integrate                                     # TODO implement mobility in forces
            self.positions = self.positions + self.timestep*self.mobility_list*self.forces + self.force_Random()
            
            
            if self.active_particle_present:
                self.positions[self.active_particle_indices] += self.force_Random_Correlated(step)
            
        t_end = time()
        
        self.config.simulation_time = t_end - t_start
        
        # fill rest of trajectory in case the steps//stride is not a multiple of the chunksize
        if idx_traj!= n_frames:
            if self.config.write_traj:
                fh5_traj["trajectory"][idx_traj:] = self.traj_chunk[:idx_chunk]
            if self.config.write_forces:
                fh5_forces["forces"][idx_traj:] = self.force_chunk[:idx_chunk]
            if self.config.write_distances:
                fh5_distances["distances"][idx_traj:] = self.dist_chunk[:idx_chunk]
        
        if self.config.write_traj:        
            fh5_traj.close()
        if self.config.write_forces: 
            fh5_forces.close()
        if self.config.write_distances:    
            fh5_distances.close()
        if self.config.calc_virial:
            fh5_virial.close()
        
        # save config
        self.config.save()

        return
    