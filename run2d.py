import torch
import math
import matplotlib.pyplot as plt
from torch import nn
import numpy as np
from tqdm import tqdm
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from torch import nn
from torch.nn import functional as F

class BC_2D:
    def __init__(self, left, right, up, down):
        """
        Args:
            left, right, up, down: (alpha, beta, f(t))
        """
        # alpha*u + beta*u_x + gamma*u_y = f(t)
        self.left_alpha, self.left_beta, self.left_func = left
        self.right_alpha, self.right_beta, self.right_func = right
        self.up_alpha, self.up_beta, self.up_func = up
        self.down_alpha, self.down_beta, self.down_func = down

    def sanity_check(self):
        left = self.left_alpha == 0 and self.left_beta == 0
        right = self.right_alpha == 0 and self.right_beta == 0
        up = self.up_alpha == 0 and self.up_beta == 0
        down = self.down_alpha == 0 and self.down_beta == 0
        if not (left and right and up and down):
            raise ValueError('Check the boundary conditions. You cannot have alpha and beta be 0 zt the same time.')

    @torch.compile(fullgraph=True)
    def apply(self, simu):
        gamma_left = self.left_beta / simu.dx
        gamma_right = self.right_beta / simu.dx
        gamma_up = self.up_beta / simu.dx
        gamma_down = self.down_beta / simu.dx

        simu.grid[1:-1,0] = (self.left_func(simu.x_coord_tensor[0,:], simu.y_coord_tensor[:,0], simu.cur_time) - gamma_left * simu.grid[1:-1,1]) / (self.left_alpha - gamma_left)


        # Right boundary
        simu.grid[1:-1,-1] = (self.right_func(simu.x_coord_tensor[0,:], simu.y_coord_tensor[:,0], simu.cur_time) + gamma_right * simu.grid[1:-1,-2]) / (self.right_alpha + gamma_right)

        # Left boundary
        simu.grid[0,1:-1] = (self.up_func(simu.x_coord_tensor[0,:], simu.y_coord_tensor[:,0], simu.cur_time) - gamma_up * simu.grid[1,1:-1]) / (self.up_alpha - gamma_up)

        # Down boundary
        simu.grid[-1,1:-1] = (self.down_func(simu.x_coord_tensor[0,:], simu.y_coord_tensor[:,0], simu.cur_time) + gamma_down * simu.grid[-2,1:-1]) / (self.down_alpha + gamma_down)
        
class ContConduct:
    def __init__(self, c_func):
        self.c_func = c_func
        self.map = None

    @torch.compile
    def make_conduct_map(self, simu):
        # Initialize conduct map
        self.map = torch.zeros(simu.grid.shape[0], simu.grid.shape[1], device=simu.device, dtype=simu.dtype)

        # Get coordinates
        x_coord = torch.arange(simu.x_grid, requires_grad=False, device=simu.device).expand(simu.y_grid, simu.x_grid) * simu.dx
        y_coord = torch.arange(simu.y_grid, requires_grad=False, device=simu.device).unsqueeze(1).expand(simu.y_grid, simu.x_grid) * simu.dx

        # Apply conductivity for interior
        self.map[1:-1,1:-1] = self.c_func(x_coord, y_coord)

        # Apply conductivity for boundary
        self.map[0,:] = self.map[1,:] # up
        self.map[-1,:] = self.map[-2,:] # down
        self.map[:,0] = self.map[:,1] # left
        self.map[:,-1] = self.map[:,-2] # right

        # Compute harmonic mean conductivity
        # left
        self.map_left = 2 * self.map[1:-1,1:-1] * self.map[0:-2, 1:-1] / (self.map[1:-1,1:-1] + self.map[0:-2, 1:-1])

        # right
        self.map_right = 2 * self.map[1:-1,1:-1] * self.map[2:, 1:-1] / (self.map[1:-1,1:-1] + self.map[2, 1:-1])

        # up
        self.map_up = 2 * self.map[1:-1,1:-1] * self.map[1:-1, 0:-2] / (self.map[1:-1,1:-1] + self.map[1:-1, 0:-2])

        # down
        self.map_down = 2 * self.map[1:-1,1:-1] * self.map[1:-1, 2:] / (self.map[1:-1,1:-1] + self.map[1:-1, 2:])

        self.merge_map = torch.stack([
            self.map_left,
            self.map_right,
            self.map_up,
            self.map_down
        ], dim=0).unsqueeze(0)
        
    def sanity_check(self, simu):
        max_conduct = torch.max(self.map)
        factor = simu.dt * max_conduct * 2 / simu.dx**2
        if factor > 0.5:
            raise ValueError(f'Improper setting for time steps and grid steps. The factor is {factor} and unstability will occur! Consider decrease the time step or increase the grid step.')

class Heat2dSimu:
    def __init__(self, map_shape, dx, total_time, tstep, bc, ic, c, plot_step, Q=0, device='cpu', do_progress_bar=True, dtype=torch.float32):
        """
        Args:
            map_shape (tuple): Physical size of the 2D domain.
            step (float): Step size of *interior* points (excluding boundaries).
            total_time (float): End time for the simulation.
            tstep (int): Step size of time.
            bc (iterable): Boundary condition with 4 elements. Order: up, r down, left, right.
            ic (callable): Function for initial condition.
            c (float): Diffusion coefficient.
            plot_step (int): How often (in steps) to plot the solution.
            device (str): 'cpu' or 'cuda', which device to use for Tensor operations.
        """
        self.grid = None
        self.grid_bach = None
        self.map_shape = map_shape
        self.dx = dx
        self.total_time = total_time
        self.dt = tstep
        self.bc = bc
        self.ic = ic
        self.c = c

        # Sanity check
        self.sanity_check()

        self.plot_step = plot_step
        self.do_progress_bar = do_progress_bar
        self.Q = self.make_heat_source_func(Q)
        self.device = device
        self.dtype = dtype
        self.conv = None
        self.decide_computation_mode()
        self.cur_time = 0
        



        # Check device
        if torch.cuda.is_available() and device != 'cpu':
            self.device = device
            print('Your simulation will be performed based on CUDA.')
        else:
            self.device = 'cpu'
            print('Your simulation will be performed based on CPU.')

        self.make_grid()

        # Useful preload data
        self.x_coord_tensor = torch.arange(self.x_grid, requires_grad=False, device=self.device).expand(self.y_grid, self.x_grid) * self.dx
        self.y_coord_tensor = torch.arange(self.y_grid, requires_grad=False, device=self.device).unsqueeze(1).expand(self.y_grid, self.x_grid) * self.dx

        # Some initialization
        self.set_ic()
        if isinstance(self.c, ContConduct):
            self.c.make_conduct_map(self)
            

        
    def sanity_check(self):
        # Check conductivity
        if isinstance(self.c, ContConduct):
            self.c.sanity_check(self)
        else:
            factor = self.dt * c * 2 / self.dx**2
            if factor > 0.5:
                raise ValueError(f'Improper setting for time steps and grid steps. The factor is {factor} and unstability will occur! Consider decrease the time step or increase the grid step.')
        
        # Check dt size setting
        if self.dt > self.total_time/2:
            raise ValueError('The time step is too big.')
        
        # Check dx size setting
        if self.dt > self.total_time/3:
            raise ValueError('The grid step is too big.')

    def make_heat_source_func(self, Q):
        if callable(Q):
            return torch.compile(Q)
        else:
            def func(x, y, t):
                return Q
            return torch.compile(func)

    def set_ic(self):
        # print(self.grid[1:-1,1:-1].shape)
        self.grid[1:-1,1:-1] = self.ic(self.x_coord_tensor, self.y_coord_tensor)

    def set_bc(self):
        self.bc.apply(self)

    def make_grid(self):
        # Get size of grid
        self.x_grid = math.ceil(self.map_shape[1] / self.dx)
        self.y_grid = math.ceil(self.map_shape[0] / self.dx)
        self.grid = torch.zeros(self.y_grid+2, self.x_grid+2, dtype=self.dtype, device=self.device)

        # For convenience, prevent overhead for unsqueeze
        self.grid_ch = self.grid.unsqueeze(0).unsqueeze(0).expand(1,1,-1,-1)

    def make_conv_core_continuous(self):
        self.conv = nn.Conv2d(1, 4, kernel_size=(3,3), bias=False, device=self.device, dtype=self.dtype)
        dt_dx2 = self.dt / (self.dx**2)
        kernel = torch.tensor([
            [[ [0,0,0], [1,-1,0], [0,0,0] ]],
            [[ [0,0,0], [0,-1,1], [0,0,0] ]],
            [[ [0,1,0], [0,-1,0], [0,0,0] ]],
            [[ [0,0,0], [0,-1,0], [0,1,0] ]]
        ], device=self.device, dtype=self.dtype) * dt_dx2

        with torch.no_grad():
            self.conv.weight[:] = kernel

    # def make_conv_core_continuous(self):
    #     self.conv = [nn.Conv2d(1, 1, kernel_size=(3,3), bias=False, device=self.device, dtype=self.dtype) for i in range(4)]
    #     dt_dx2 = self.dt / (self.dx ** 2)
    #     kernel = [ [[[ [0,0,0], [1,-1,0], [0,0,0] ]]],
    #                [[[ [0,0,0], [0,-1,1], [0,0,0] ]]],
    #                [[[ [0,1,0], [0,-1,0], [0,0,0] ]]],
    #                [[[ [0,0,0], [0,-1,0], [0,1,0] ]]],]
    #     with torch.no_grad():
    #         for i in range(4):
    #             self.conv[i].weight[:] = torch.tensor(kernel[i], device=self.device, dtype=self.dtype) * dt_dx2

    
    def make_conv_core_const(self):
        self.conv = nn.Conv2d(1, 1, kernel_size=(3,3), bias=False, device=self.device, dtype=self.dtype)
        dt_dx2 = self.dt / (self.dx**2)
        kernel = torch.tensor([
            [[ [0,1,0], [1,-4,1], [0,1,0] ]],
        ], device=self.device, dtype=self.dtype) * dt_dx2 * self.c

        with torch.no_grad():
            self.conv.weight[:] = kernel

    def decide_computation_mode(self):
        if isinstance(self.c, ContConduct):
            self.update = self.update_continuous
            self.make_conv_core_continuous()
        else:
            self.update = self.update_const
            self.make_conv_core_const()

    @torch.compile
    def update_continuous(self):
        with torch.inference_mode():
            diff_map = self.conv(self.grid_ch)
            diff = torch.sum(diff_map * self.c.merge_map, dim=1, keepdim=True) + self.Q(self.x_coord_tensor, self.y_coord_tensor, self.cur_time) * self.dt
            self.grid_ch[:, :, 1:-1, 1:-1] += diff

    # @torch.compile
    # def update_continuous(self):
    #     with torch.inference_mode():
    #         diff0 = self.conv[0](self.grid_ch)
    #         diff1 = self.conv[1](self.grid_ch)
    #         diff2 = self.conv[2](self.grid_ch)
    #         diff3 = self.conv[3](self.grid_ch)
    #
    #         self.grid_ch[:, :, 1:-1, 1:-1] += diff0 + diff1 + diff2 + diff3 + self.Q(self.x_coord_tensor, self.y_coord_tensor, self.cur_time) * self.dt

    @torch.compile
    def update_const(self):
        with torch.inference_mode():
            diff = self.conv(self.grid_ch)
            self.grid_ch[:, :, 1:-1, 1:-1] += diff + self.Q(self.x_coord_tensor, self.y_coord_tensor, self.cur_time) * self.dt
            # self.add_const_diff(diff)

    # @torch.compile
    # def add_const_diff(self, diff):
    #     with torch.inference_mode():
    #         self.grid_ch[:, :, 1:-1, 1:-1] += diff + self.Q(self.x_coord_tensor, self.y_coord_tensor,
    #                                                         self.cur_time) * self.dt

    # @torch.compile
    def start(self):
        saved = []
        append = saved.append
        cur_max = -float('inf')
        cur_min = float('inf')
        with torch.inference_mode():
            for step in tqdm(range( int(self.total_time/self.dt) ),disable=False):
                self.set_bc()
                self.update()
                self.cur_time += self.dt

                if step % self.plot_step == 0:
                    copied = self.grid[1:-1,1:-1].clone().to('cpu')
                    if self.dtype == torch.bfloat16:
                        copied = copied.to(dtype=torch.float32)
                    append(copied)

                    this_max = torch.max(self.grid[1:-1,1:-1])
                    if cur_max < this_max:
                        cur_max = this_max

                    this_min = torch.min(self.grid[1:-1,1:-1])
                    if cur_min > this_min:
                        cur_min = this_min
        
        # Append the very final result
                  
        copied = self.grid[1:-1,1:-1].clone().to('cpu')
        
        if self.dtype == torch.bfloat16:
            copied = copied.to(dtype=torch.float32)
        append(copied)
        # copied#.to('cpu')
        fig, axis = plt.subplots()
        
        
        pcm = axis.pcolormesh(self.grid.to(dtype=torch.float32).cpu().numpy()[1:-1,1:-1], cmap=plt.cm.jet,
                              vmin=float(cur_min), vmax=float(cur_max))
        plt.colorbar(pcm, ax=axis)
        axis.set_xlabel('x grids')
        axis.set_ylabel('y grids')
        
        
        
        for i, data in enumerate(saved):
            pcm.set_array(data.numpy())
            axis.set_title(f'Distribution at t={i * self.plot_step * self.dt:.4f}')
            plt.pause(0.01)

        plt.show()



if __name__ == '__main__':
    map_shape=(torch.pi, torch.pi)
    dx = 0.005
    total_time=1
    dt=0.0000125
    # dt = 0.05


    def ic(x,y):
        return torch.sin(x) * torch.sin(y)
        # return torch.sin(x)
        # return 0
    c=0.5
    plot_step=100
    
    def Q(x,y,t):
        # res = -torch.sin(5*x) * torch.sin(5*y) * torch.cos( torch.sqrt( (x-torch.pi/2)**2 + (y-torch.pi/2)**2) *4) * 5
        # if t < 0.5:
        #     return res
        # return -res
        return -torch.sin(5*x) * torch.sin(5*y) * torch.cos( torch.sqrt( (x-torch.pi/2)**2 + (y-torch.pi/2)**2) *4) * 5 * math.sin(t*torch.pi*8)
   
    factor = dt*c*2/dx**2
    print(factor)
    
    
    def func(x,y):
        return 0.5
    con = ContConduct(func)
    
   
    def func2(x,y,t):
        # return (torch.where(y < torch.pi/5, -1, 1) + torch.where(y > 2*torch.pi/5, -1, 1) - 1
        #         + torch.where(y < 3*torch.pi/5, -1, 1) + torch.where(y > 4*torch.pi/5, -1, 1) - 1)
        return (torch.where(y < torch.pi/4, 0, 1) + torch.where(y > 3*torch.pi/4, 0, 1) - 1)# * math.sin(4*torch.pi*t)
        # return 0
    
    
    def func3(x,y,t):
        # return (torch.where(x < torch.pi/5, -1, 1) + torch.where(x > 2*torch.pi/5, -1, 1) - 1 +
        #         torch.where(x < 3*torch.pi/5, -1, 1) + torch.where(x > 4*torch.pi/5, -1, 1) - 1)
        return (torch.where(y < torch.pi/4, 0, 1) + torch.where(y > 3*torch.pi/4, 0, 1) - 1)# * math.sin(4*torch.pi*t)
        # return 0
        
    bc= BC_2D((1,0,func2),(1,0,func2),(1,0,func3),(1,0,func3))
    test = Heat2dSimu(map_shape, dx, total_time, dt, bc, ic, c, plot_step, Q, device='cuda', do_progress_bar=True, dtype=torch.float64)
    # con.make_conduct_map(test)
    # con.map_right.shape
    # torch.backends.cudnn.benchmark = True
    test.start()