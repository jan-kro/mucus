import numpy as np
import os
import toml
import h5py

import rust_mucus as rmc

from .config import Config
from .topology import Topology
from .utils import get_path, get_number_of_frames
from pathlib import Path
from tqdm import tqdm

from time import time
from copy import deepcopy 

class System:
    
    def __init__(self, config: Config):

        self.config = config
        self.topology = Topology(self.config)
        
        self.timestep               = None
        self.chunksize              = None
        self.n_particles            = None
        self.B_debye                = None        
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
        
        self.cell_index             = None
        self.neighbor_cells_idx     = None
        self.head_array             = None
        self.list_array             = None
        self.n_cells                = None
        self.cell_length            = None
        self.n_neighbor_cells       = None
        
        self.setup()
        
    #! TODO TODO TODO TODO TODO TODO
    # ADD "use forces" TO TOPOLOGY  
      
    def setup(self):
        
        self.n_particles     = self.config.n_particles
        self.box_length      = self.config.lbox
        self.timestep        = self.config.timestep
        self.lB_debye        = self.config.lB_debye # units of beed radii                                       
        
        self.traj_chunk      = np.zeros((self.config.chunksize, self.n_particles, 3))
        self.force_chunk     = np.zeros((self.config.chunksize, self.n_particles, 3))
        self.dist_chunk      = np.zeros((self.config.chunksize, self.n_particles, self.n_particles))
        
        self.forces          = np.zeros((self.n_particles, 3), dtype=np.float64)
        
        # TODO implement this properly
        self.B_debye         = np.sqrt(self.config.c_S)*self.config.r0_nm/10 # from the relationship in the Hansing thesis [c_S] = mM
        self.mobility_list   = np.array([self.topology.mobility[i] for i in self.topology.tags], ndmin=2).reshape(-1, 1)
        
        #TODO delete this nonsense
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
        
        # calculate the number of cells in each direction
        self.n_cells = int(self.box_length/self.config.cutoff_pbc)
        self.cell_length = self.box_length/self.n_cells
        
        # get the indices of the neighboring cells and self with shape (n_neighbor_cells + 1, 3)
        self.neighbor_cells_idx = np.indices((3, 3, 3), dtype=np.int16).reshape(3, -1).T - 1
        self.n_neighbor_cells = self.neighbor_cells_idx.shape[0]
        
        self.apply_pbc()
        
        self.head_array = -np.ones((self.n_cells, self.n_cells, self.n_cells), dtype=np.int16)
        self.list_array = -np.ones(self.n_particles, dtype=np.int16)
        
        self.update_linked_list()

    def update_linked_list(self):
        """
        updates the list and head array for the current positions
        """
        
        self.list_array.fill(-1)
        self.head_array.fill(-1)
        
        # get cell index for each particle
        self.cell_index = np.floor(self.positions/self.cell_length).astype(np.int16)
        
        for i, cell_idx in enumerate(self.cell_index):
            self.list_array[i] = self.head_array[cell_idx[0], cell_idx[1], cell_idx[2]]
            self.head_array[cell_idx[0], cell_idx[1], cell_idx[2]] = i
    
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
        
        self.positions = pos
        
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
    
    def apply_pbc(self):
        """
        Repositions all atoms outside of the box to the other end of the box
        """
        
        # calculate modulo L
        self.positions = np.mod(self.positions, self.box_length)
        
        return
    
    def simulate(self, steps=None):
        """
        Simulates the overdamped langevin dynamics of the system with the defined forcefield using forward Euler.
        """
        
        if steps == None:
            steps = self.config.steps
        
        t_start = time()
        
        n_frames = get_number_of_frames(self.config)
        
        # save initial pos
        if self.config.write_traj==True:
            self.traj_chunk[0] = self.positions
            #fh5_results.create_dataset("trajectory", shape=(n_frames, self.n_particles, 3), dtype="float16")
            fh5_traj = h5py.File(get_path(self.config, filetype='trajectory'), 'w-')
            fh5_traj.create_dataset("trajectory", shape=(n_frames, self.n_particles, 3), dtype="float32")
        
        # define flag for distance writing
        if self.config.write_distances:
            write_distances = True
            fh5_distances = h5py.File(get_path(self.config, filetype='distances'), 'w-')
            fh5_distances.create_dataset("distances", shape=(n_frames, self.n_particles, self.n_particles), dtype="float16")
        else:    
            write_distances = False
        
        rmc.get_forces_cell_linked(
                self.positions,
                self.topology.tags,
                self.topology.bond_table,
                self.topology.force_constant_nn,
                self.topology.r0_bond,
                self.topology.sigma_lj,
                self.topology.epsilon_lj,
                self.topology.q_particle,
                self.config.lB_debye,
                self.B_debye,
                self.forces,
                self.dist_chunk[0],
                self.box_length,
                self.config.cutoff_pbc**2,
                self.n_particles,
                3,                          # number of dimensions
                write_distances,
                True,                       # use bond force
                True,                       # use LJ force
                False,                      # use Debye force
                self.neighbor_cells_idx,
                self.head_array,
                self.list_array,
                self.n_cells,
                self.n_neighbor_cells
            )
            
        if self.config.write_forces==True:
            # rmc.get_forces(
            #     self.positions,
            #     self.topology.tags,
            #     self.topology.bond_table,
            #     self.topology.force_constant_nn,
            #     self.topology.r0_bond,
            #     self.topology.sigma_lj,
            #     self.topology.epsilon_lj,
            #     self.topology.q_particle,
            #     self.config.lB_debye,
            #     self.B_debye,
            #     self.forces,
            #     self.dist_chunk[0],
            #     self.box_length,
            #     self.config.cutoff_pbc**2,
            #     self.n_particles,
            #     3,
            #     False,
            #     True,
            #     True,
            #     False
            # )
            
            self.force_chunk[0] = self.forces
            fh5_forces = h5py.File(get_path(self.config, filetype='forces'), 'w-')
            fh5_forces.create_dataset("forces", shape=(n_frames, self.n_particles, 3), dtype="float32")
        
        idx_chunk = 1 # because traj_chunk[0] is already the initial position
        idx_traj = 0
        
        print(f"\nStarting simulation with {steps} steps.")
        for step in tqdm(range(1, steps)):
            
            # calculate forces
            self.forces.fill(0)
            
            self.update_linked_list()
            
            rmc.get_forces_cell_linked(
                self.positions,
                self.topology.tags,
                self.topology.bond_table,
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
                3,                          # number of dimensions
                write_distances,
                True,                       # use bond force
                True,                       # use LJ force
                False,                      # use Debye force
                self.neighbor_cells_idx,
                self.head_array,
                self.list_array,
                self.n_cells,
                self.n_neighbor_cells
            )
            
            # reset distance flag until next stride
            write_distances = False
            
            # integrate                                     # TODO implement mobility in forces
            self.positions = self.positions + self.timestep*self.mobility_list*self.forces + self.force_Random()
            
            # apply periodic boundary conditions (0, L) x (0, L) x (0, L)
            self.apply_pbc()

            
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
        
        # save config
        self.config.save()

        return