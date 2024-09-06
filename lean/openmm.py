from openmmtools.integrators import ThermostatedIntegrator

class PathProbabilityIntegrator(ThermostatedIntegrator):
    """Abstract base class for path probability integrators.
    These integrators track the path probability ratio which is required in stochastic normalizing flows.

    Parameters
    ----------
    temperature : float or unit.Quantity
        Temperature in kelvin.
    stepsize : float or unit.Quantity
        Step size in picoseconds.

    Attributes
    ----------
    ratio : float
        The logarithmic path probability ratio summed over all steps taken during the previous invocation of `step`.
    """

    def __init__(self, temperature, stepsize):
        super(PathProbabilityIntegrator, self).__init__(temperature, stepsize)
        self.addGlobalVariable("log_path_probability_ratio", 0.0)

    @property
    def ratio(self):
        return self.getGlobalVariableByName("log_path_probability_ratio")

    @ratio.setter
    def ratio(self, value):
        self.setGlobalVariableByName("log_path_probability_ratio", value)

    def step(self, n_steps):
        """Propagate the system using the integrator.
        This method returns the current log path probability ratio and resets it to 0.0 afterwards.

        Parameters
        ----------
        n_steps : int
            The number of steps

        Returns
        -------
        ratio : float
            The logarithmic path probability ratio summed over n_steps steps.
        """
        self.ratio = 0.0
        super().step(n_steps)
        ratio = self.ratio
        return ratio

class OverdampedLangevinIntegrator(PathProbabilityIntegrator):
    """Overdamped Langevin Dynamics"""

    def __init__(self, temperature, friction_coeff, stepsize):
        super().__init__(temperature, stepsize)
        # variable definitions
        self.addGlobalVariable("gamma", friction_coeff)
        self.addPerDofVariable("w", 0)
        self.addPerDofVariable("w_", 0)
        self.addPerDofVariable("epsilon", 0)
        self.addPerDofVariable("f_old", 0)
        self.addPerDofVariable("x_old", 0)

        # propagation
        self.addUpdateContextState()
        self.addComputeTemperatureDependentConstants({"epsilon": "dt/gamma/m"})
        self.addComputePerDof("w", "gaussian")
        self.addComputePerDof("f_old", "f")
        self.addComputePerDof("x_old", "x")
        self.addComputePerDof(
            "x", "x+epsilon*f + sqrt(2*epsilon*kT)*w"
        )  # position update
        self.addComputePerDof(
            "w_", "sqrt(epsilon/2/kT) * (- f_old - f) - w"
        )  # backward noise
        self.addConstrainPositions()
        self.addComputePerDof("v", "(x-x_old)/dt")
        self.addConstrainVelocities()

        # update logarithmic path probability ratio
        self.addComputeSum("wsquare", "w*w")
        self.addComputeSum("w_square", "w_*w_")
        self.addComputeGlobal(
            "log_path_probability_ratio",
            "log_path_probability_ratio-0.5*(w_square - wsquare)",
        )
        
        
    