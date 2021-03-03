# [Built-in modules]

# [3rd party modules]
import numpy as np
import time
import pickle

from modules.utils        import my_print
import matplotlib.pyplot as plt

try:
    import mujoco_py as mjPy

except ImportError as e:
    raise error.DependencyNotInstalled( "{}. (HINT: you need to install mujoco_py, \
                                             and also perform the setup instructions here: \
                                             https://github.com/openai/mujoco-py/.)".format( e ) )

# Added
try:
    import sympy as sp
    from sympy.utilities.lambdify import lambdify, implemented_function

except ImportError as e:
    raise error.DependencyNotInstalled( "{}. (HINT: you need to install sympy, \
                                             Simply type pip3 install sympy. \
                                             Sympy is necessary for building ZFT Calculation)".format( e ) )

# [Local modules]


class Controller( ):
    """
        Description:
        -----------
            Parent class for the controllers
    """


    def __init__( self, mjModel, mjData ):
        """

        """
        self.mjModel        = mjModel
        self.mjData         = mjData
        self.ctrl_par_names = None


    def set_ctrl_par( self, **kwargs ):
        """
            Setting the control parameters

            Each controllers have their own controller parameters names (ctrl_par_names),

            This method function will become handy when we want to modify, or set the control parameters.

        """
        if kwargs is not None:
            for args in kwargs:
                if args in self.ctrl_par_names:
                    setattr( self, args, kwargs[ args ] )
                else:
                    pass

    def input_calc( self, start_time, current_time ):
        """
            Calculating the torque input
        """
        raise NotImplementedError                                               # Adding this NotImplementedError will force the child class to override parent's methods.


class ImpedanceController( Controller ):
    """
        Description:
        ----------
            Class for an Impedance Controller
            First order impedance controller with gravity compenation

    """

    def __init__( self, mjModel, mjData ):

        super().__init__( mjModel, mjData )

        self.act_names      = mjModel.actuator_names                            # The names of the actuators, all the names end with "TorqueMotor" (Refer to xml model files)
        self.n_act          = len( mjModel.actuator_names )                     # The number of actuators, 2 for 2D model and 4 for 3D model
        self.idx_act        = np.arange( 0, self.n_act )                        # The idx array of the actuators, this is useful for self.input_calc method

        # Controller uses first-order impedance controller. Hence the position/velocity of the ZFT(Zero-torque trajectory) must be defined
        self.ZFT_func_pos   = None
        self.ZFT_func_vel   = None

        self.ctrl_par_names = [ "K", "B" ]                                      # Useful for self.set_ctrl_par method
        self.t_sym = sp.symbols( 't' )                                          # time symbol of the equation


    def set_ZFT( self, trajectory ):
        """
            Description:
            ----------
                Setting the ZFT (Zero-torque trajectory, strictly speaking it should be ZTT, but ZFT is much popular usage :)
                This method is only called once "before" running the simulation, and "after" the self.mov_parameters are well-defined

        """

        # Lambdify the functions
        # [TIP] This is necessary for computation Speed!
        self.ZFT_func_pos = lambdify( self.t_sym, trajectory )
        self.ZFT_func_vel = lambdify( self.t_sym, sp.diff( trajectory, self.t_sym ) )

        self.x0 = self.ZFT_func_pos( 0 )

    def get_ZFT( self, time ):

        x0  = np.array( self.ZFT_func_pos( time ) )
        # dx0 = np.array( self.ZFT_func_vel( t ) )

        return x0

    def input_calc( self, start_time, current_time ):


        self.x  = self.mjData.qpos[ 0 ]
        self.x0 = self.get_ZFT( current_time )

        # tau_imp = np.dot( self.K, self.x0 - self.x )
        tau_imp = self.x0
        print( self.x0 )

        return self.mjData.ctrl, self.idx_act, tau_imp



if __name__ == "__main__":
    pass
